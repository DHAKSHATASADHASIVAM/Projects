"""Command line interface -- the whole pipeline, reproducible from a clean clone.

    python -m sustainable_ride.cli download    # fetch the public datasets
    python -m sustainable_ride.cli prepare     # clean into a common schema
    python -m sustainable_ride.cli train       # fit and persist the models
    python -m sustainable_ride.cli evaluate    # metrics and figures
    python -m sustainable_ride.cli recommend   # one-off recommendation
    python -m sustainable_ride.cli serve       # start the API
    python -m sustainable_ride.cli pipeline    # everything, in order
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty at DEBUG and never say anything we need.
    for noisy in ("urllib3", "matplotlib", "PIL", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_download(args) -> int:
    from .data.download import build_zone_centroids, download_all

    paths = download_all(month=args.month)
    for name, path in paths.items():
        print(f"  {name:12} {path}")
    centroids = build_zone_centroids()
    print(f"  {'centroids':12} {len(centroids)} taxi zones")
    return 0


def cmd_prepare(args) -> int:
    from .data.preprocess import build_processed_datasets, fit_circuity

    paths = build_processed_datasets(use_synthetic=args.synthetic, month=args.month)
    for name, path in paths.items():
        print(f"  {name:12} {path}")

    if not args.synthetic:
        factors = fit_circuity()
        print(f"  circuity     {factors['factors']} "
              f"(fitted on {factors['n_trips']:,} metered trips)")
    return 0


def cmd_train(args) -> int:
    from .models.train import train_all

    models = train_all()
    print()
    print(f"  {'model':16} {'MAE':>8} {'RMSE':>8} {'R2':>7} {'MAPE':>8}")
    print(f"  {'-' * 16} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 8}")
    for name, model in models.items():
        m = model.metrics
        print(f"  {name:16} {m['mae']:8.2f} {m['rmse']:8.2f} "
              f"{m['r2']:7.3f} {m['mape']:7.1f}%")
    return 0


def cmd_evaluate(args) -> int:
    from .models.evaluate import run_evaluation

    report = run_evaluation(make_figures=not args.no_figures)
    print(json.dumps(report["summary"], indent=2))
    return 0


def cmd_recommend(args) -> int:
    from .recommender import RideRecommender

    recommender = RideRecommender()
    departure = datetime.fromisoformat(args.departure) if args.departure else datetime.now()

    result = recommender.recommend(
        origin=(args.origin_lat, args.origin_lon),
        destination=(args.dest_lat, args.dest_lon),
        departure=departure,
        weights={"cost": args.w_cost, "time": args.w_time, "co2": args.w_co2},
        passenger_count=args.passengers,
        rain_probability=args.rain,
        accessibility_required=args.accessible,
        include_geometry=False,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0

    print()
    print(f"  Trip      ({args.origin_lat:.4f}, {args.origin_lon:.4f}) -> "
          f"({args.dest_lat:.4f}, {args.dest_lon:.4f})")
    print(f"  Departing {departure:%Y-%m-%d %H:%M}")
    print(f"  Routing   {result.routing_provider}")
    print(f"  Weights   cost {args.w_cost:.2f} / time {args.w_time:.2f} / "
          f"CO2 {args.w_co2:.2f}")
    print()
    header = (f"  {'':2} {'mode':22} {'time':>8} {'cost':>9} {'CO2':>10} "
              f"{'dist':>8} {'score':>7}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for option in result.options:
        if not option.feasible:
            marker = " x"
            score = "  --"
        else:
            marker = " *" if option is result.best else "  "
            score = f"{option.score:7.3f}"
        print(f"{marker} {option.label:22} {option.duration_min:7.0f}m "
              f"{option.cost_usd:8.2f}$ {option.co2_grams:9.0f}g "
              f"{option.distance_km:7.2f}km {score}"
              + ("  [frontier]" if option.on_pareto_frontier else ""))
        for reason in option.infeasible_reasons:
            print(f"       ! {reason}")

    print()
    print("  " + result.narrative.replace(". ", ".\n  "))
    print()
    return 0


def cmd_serve(args) -> int:
    from .api.main import run

    print(f"  API on http://{args.host}:{args.port}   (docs at /docs)")
    run(host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_pipeline(args) -> int:
    """Run the full pipeline from raw data to trained, evaluated models."""
    steps = []
    if not args.synthetic:
        steps.append(("Downloading public datasets", cmd_download))
    steps += [
        ("Cleaning and preparing", cmd_prepare),
        ("Training models", cmd_train),
        ("Evaluating", cmd_evaluate),
    ]

    for index, (title, func) in enumerate(steps, start=1):
        print(f"\n{'=' * 70}\n[{index}/{len(steps)}] {title}\n{'=' * 70}")
        code = func(args)
        if code != 0:
            print(f"\nPipeline stopped: '{title}' returned {code}", file=sys.stderr)
            return code

    print(f"\n{'=' * 70}\nPipeline complete.")
    print("  Try:  python -m sustainable_ride.cli serve")
    print("   or:  streamlit run app/streamlit_app.py")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sustainable_ride",
        description="Sustainable Ride Suggestion -- multi-modal trip recommender.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the public datasets (~420 MB)")
    p.add_argument("--month", default=None, help="YYYY-MM (default from config)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("prepare", help="clean raw data into the common schema")
    p.add_argument("--month", default=None)
    p.add_argument("--synthetic", action="store_true",
                   help="use the synthetic generator instead of real data")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="fit and persist the models")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="metrics and figures")
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("recommend", help="one-off recommendation")
    p.add_argument("--origin-lat", type=float, default=40.7580)
    p.add_argument("--origin-lon", type=float, default=-73.9855)
    p.add_argument("--dest-lat", type=float, default=40.7061)
    p.add_argument("--dest-lon", type=float, default=-73.9969)
    p.add_argument("--departure", default=None, help="ISO 8601; default now")
    p.add_argument("--w-cost", type=float, default=0.34)
    p.add_argument("--w-time", type=float, default=0.33)
    p.add_argument("--w-co2", type=float, default=0.33)
    p.add_argument("--passengers", type=int, default=1)
    p.add_argument("--rain", type=float, default=0.0, help="0-1 rain probability")
    p.add_argument("--accessible", action="store_true",
                   help="require step-free modes")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("serve", help="start the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("pipeline", help="run everything end to end")
    p.add_argument("--month", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        print("\nRun with -v for the full traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
