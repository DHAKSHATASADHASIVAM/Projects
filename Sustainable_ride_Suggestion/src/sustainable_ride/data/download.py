"""Acquisition of the public source datasets.

Two real datasets underpin this project:

* **NYC TLC yellow taxi trip records** -- roughly 3 million trips per month,
  with metered distance, fare and timestamps. Since 2017 the TLC has published
  pickup and dropoff *zones* rather than coordinates for privacy reasons, so
  we also pull the zone shapefile and reduce each of the 263 zones to its
  centroid. That introduces a real quantisation error, quantified in the EDA
  notebook, and it is the main accuracy limit on the taxi models.

* **Citi Bike system data** -- every ride, with station coordinates at both
  ends. These are true point coordinates, so the bike model is not subject to
  the same quantisation.

Downloads are resumable-ish (an existing file of plausible size is left alone)
and everything lands under ``data/raw``, which is gitignored.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ..config import load_config, resolve_path

logger = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MiB


def _download(url: str, dest: Path, description: str, min_bytes: int = 1024) -> Path:
    """Stream ``url`` to ``dest``, skipping the work if it is already there."""
    if dest.exists() and dest.stat().st_size > min_bytes:
        logger.info("%s already present (%.1f MB), skipping download",
                    description, dest.stat().st_size / 1e6)
        return dest

    logger.info("Downloading %s from %s", description, url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        last_report = 0
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if total and written - last_report > 50 * CHUNK:
                    last_report = written
                    logger.info("  %s: %.0f%% (%.0f/%.0f MB)", description,
                                100 * written / total, written / 1e6, total / 1e6)

    tmp.replace(dest)
    logger.info("%s downloaded: %.1f MB", description, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# Taxi
# ---------------------------------------------------------------------------

def download_taxi_trips(month: str | None = None) -> Path:
    """Fetch one month of TLC yellow taxi Parquet records."""
    cfg = load_config()
    month = month or cfg.get_path("data.month", "2024-01")
    url = cfg.get_path("data.taxi.url_template").format(month=month)
    dest = resolve_path("raw", f"yellow_tripdata_{month}.parquet")
    return _download(url, dest, f"TLC yellow taxi {month}", min_bytes=1_000_000)


def download_zone_lookup() -> Path:
    """Fetch the taxi zone lookup table (zone id -> borough, service zone)."""
    cfg = load_config()
    url = cfg.get_path("data.taxi.zone_lookup_url")
    dest = resolve_path("reference", "taxi_zone_lookup.csv")
    return _download(url, dest, "TLC taxi zone lookup")


def build_zone_centroids(force: bool = False) -> pd.DataFrame:
    """Reduce the taxi zone polygons to WGS84 centroids.

    The shapefile is projected in EPSG:2263 (NY State Plane, Long Island, feet).
    Centroids are computed in that planar CRS -- which is the correct order of
    operations, since averaging degrees of longitude is only approximately
    meaningful -- and then reprojected to WGS84.

    The result is cached to ``data/reference/taxi_zone_centroids.csv`` and is
    small enough to commit, so downstream users need neither the shapefile nor
    a projection library.
    """
    out_path = resolve_path("reference", "taxi_zone_centroids.csv")
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    cfg = load_config()
    url = cfg.get_path("data.taxi.zone_shapefile_url")
    zip_path = resolve_path("raw", "taxi_zones.zip")
    _download(url, zip_path, "TLC taxi zone shapefile")

    try:
        import shapefile  # pyshp
        from shapely.geometry import shape
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Building zone centroids needs `pyshp` and `shapely`. "
            "Install them with: pip install pyshp shapely"
        ) from exc

    extract_dir = resolve_path("interim", "taxi_zones")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # The archive nests its members in a taxi_zones/ folder, so search recursively.
    shp_files = sorted(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found inside {zip_path}")

    reader = shapefile.Reader(str(shp_files[0]))
    field_names = [f[0] for f in reader.fields[1:]]

    rows = []
    for record, geom in zip(reader.records(), reader.shapes()):
        attrs = dict(zip(field_names, record))
        centroid = shape(geom.__geo_interface__).centroid
        rows.append({
            "LocationID": int(attrs.get("LocationID", attrs.get("OBJECTID", 0))),
            "zone": attrs.get("zone", ""),
            "borough": attrs.get("borough", ""),
            "x_2263": centroid.x,
            "y_2263": centroid.y,
        })
    reader.close()

    df = pd.DataFrame(rows)
    lat, lon = _reproject_2263_to_wgs84(df["x_2263"].to_numpy(), df["y_2263"].to_numpy())
    df["latitude"] = lat
    df["longitude"] = lon
    df = df.drop(columns=["x_2263", "y_2263"])

    df.to_csv(out_path, index=False)
    logger.info("Wrote %d zone centroids to %s", len(df), out_path)
    return df


def _reproject_2263_to_wgs84(x, y):
    """EPSG:2263 (feet) -> EPSG:4326, using pyproj when available.

    An affine fallback is provided so the pipeline still runs without pyproj.
    Over the ~40 km span of the five boroughs the Lambert Conformal Conic
    projection used by EPSG:2263 is near-linear, so a first-order fit is
    accurate to roughly 100 m -- small against the several-hundred-metre error
    already introduced by collapsing a zone polygon to a single point, but it
    is an approximation and the log says so.
    """
    try:
        from pyproj import Transformer
    except ImportError:
        logger.warning(
            "pyproj not installed; using an affine approximation for the "
            "EPSG:2263 -> WGS84 conversion (accurate to ~100 m). "
            "Install pyproj for an exact reprojection."
        )
        import numpy as np
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        # EPSG:2263 grid origin: false easting 984_250 ft at 74 deg W, and
        # 40deg 10' N at y = 0. Scale factors are degrees per survey foot,
        # evaluated at the latitude of Manhattan.
        lat = 40.166667 + y * 2.7433e-6
        lon = -74.0 + (x - 984_250.0) * 3.6083e-6
        return lat, lon

    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


# ---------------------------------------------------------------------------
# Citi Bike
# ---------------------------------------------------------------------------

def download_bike_trips(month: str | None = None) -> Path:
    """Fetch one month of Citi Bike ride data.

    Citi Bike's filenames have changed several times (and one historical month
    is misspelled ``citbike``), so each configured pattern is tried in turn.
    """
    cfg = load_config()
    month = month or cfg.get_path("data.month", "2024-01")
    ym = month.replace("-", "")
    dest = resolve_path("raw", f"citibike_{ym}.zip")

    if dest.exists() and dest.stat().st_size > 1_000_000:
        logger.info("Citi Bike %s already present, skipping download", month)
        return dest

    errors = []
    for template in cfg.get_path("data.bike.url_templates", []):
        url = template.format(ym=ym)
        try:
            head = requests.head(url, timeout=30, allow_redirects=True)
            if head.status_code != 200:
                errors.append(f"{url} -> HTTP {head.status_code}")
                continue
            return _download(url, dest, f"Citi Bike {month}", min_bytes=1_000_000)
        except requests.RequestException as exc:
            errors.append(f"{url} -> {exc}")

    raise FileNotFoundError(
        "Could not locate Citi Bike data for {}. Tried:\n  {}".format(
            month, "\n  ".join(errors))
    )


def read_bike_zip(path: Path, sample_size: int | None = None,
                  seed: int = 42) -> pd.DataFrame:
    """Read the CSVs inside a Citi Bike archive into one frame.

    Recent archives nest a folder of monthly part-files, and include a
    ``__MACOSX`` directory that must be skipped.
    """
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as zf:
        members = [
            n for n in zf.namelist()
            if n.lower().endswith(".csv") and "__MACOSX" not in n
            and not Path(n).name.startswith(".")
        ]
        if not members:
            raise ValueError(f"No CSV members found inside {path}")

        logger.info("Reading %d CSV part(s) from %s", len(members), path.name)
        for name in members:
            with zf.open(name) as fh:
                frames.append(pd.read_csv(io.BytesIO(fh.read()), low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    logger.info("Citi Bike raw rows: %s", f"{len(df):,}")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        logger.info("Sampled down to %d rows", len(df))
    return df


def download_all(month: str | None = None) -> dict[str, Path]:
    """Fetch every source dataset. Roughly 400 MB for the default month."""
    return {
        "taxi": download_taxi_trips(month),
        "zone_lookup": download_zone_lookup(),
        "bike": download_bike_trips(month),
    }
