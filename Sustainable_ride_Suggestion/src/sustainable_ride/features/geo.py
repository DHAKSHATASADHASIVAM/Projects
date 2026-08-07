"""Geographic primitives.

Vectorised where it matters -- these run over hundreds of thousands of trip
records during feature building, so the array paths avoid Python-level loops.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088

# Bounding box for the NYC service area. Used to discard GPS glitches that put
# a pickup in the Atlantic or in New Jersey's farmland.
NYC_BOUNDS = {
    "lat_min": 40.45,
    "lat_max": 41.00,
    "lon_min": -74.30,
    "lon_max": -73.65,
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres.

    Accepts scalars or numpy arrays; returns the matching type.
    """
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype="float64"))
                              for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # arcsin of a clipped value: floating point can push `a` a hair above 1.
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing from point 1 to point 2, in degrees [0, 360).

    Direction of travel is a genuinely useful feature in Manhattan: the avenue
    grid means north-south trips flow very differently from cross-town ones.
    """
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype="float64"))
                              for x in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def manhattan_km(lat1, lon1, lat2, lon2):
    """L1 distance along lines of latitude and longitude.

    A crude stand-in for grid-constrained travel: you cannot cut diagonally
    through a city block. Complements the haversine feature rather than
    replacing it -- the model learns which regime applies where.
    """
    lat_leg = haversine_km(lat1, lon1, lat2, lon1)
    lon_leg = haversine_km(lat2, lon1, lat2, lon2)
    return lat_leg + lon_leg


def within_nyc(lat, lon) -> np.ndarray:
    """Boolean mask for coordinates inside the NYC service bounding box."""
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    return (
        (lat >= NYC_BOUNDS["lat_min"])
        & (lat <= NYC_BOUNDS["lat_max"])
        & (lon >= NYC_BOUNDS["lon_min"])
        & (lon <= NYC_BOUNDS["lon_max"])
    )


def validate_coordinate(lat: float, lon: float, name: str = "coordinate") -> None:
    """Raise ``ValueError`` if a coordinate is not a usable NYC-area point."""
    if lat is None or lon is None:
        raise ValueError(f"{name}: latitude and longitude are both required")
    if not (-90.0 <= float(lat) <= 90.0):
        raise ValueError(f"{name}: latitude {lat} is outside [-90, 90]")
    if not (-180.0 <= float(lon) <= 180.0):
        raise ValueError(f"{name}: longitude {lon} is outside [-180, 180]")
    if not bool(within_nyc(lat, lon)):
        raise ValueError(
            f"{name}: ({lat}, {lon}) falls outside the NYC service area. "
            f"The models are trained on NYC trips and would extrapolate wildly "
            f"elsewhere."
        )
