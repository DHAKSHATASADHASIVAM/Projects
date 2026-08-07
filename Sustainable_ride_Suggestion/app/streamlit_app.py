"""Streamlit demo for the Sustainable Ride Suggestion recommender.

Run with:  streamlit run app/streamlit_app.py

The interface is built around the weight sliders, because the interesting thing
about this problem is not any single recommendation -- it is watching the answer
change as you shift what you care about. A user who drags the CO2 slider up and
sees the recommendation flip from taxi to bike has learned something about the
trade-off that a static answer could never convey.
"""

from __future__ import annotations

import sys
from datetime import datetime, time as dtime
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the package importable when run via `streamlit run` from the repo root.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sustainable_ride.emissions import factor_provenance, humanise_co2  # noqa: E402
from sustainable_ride.models.registry import ModelsNotTrainedError  # noqa: E402
from sustainable_ride.recommender import RideRecommender  # noqa: E402
from sustainable_ride.viz import INK, MODE_COLORS, MODE_ICONS  # noqa: E402

st.set_page_config(
    page_title="Sustainable Ride Suggestion",
    page_icon="🚲",
    layout="wide",
)

# Well-known NYC locations, so a visitor can get a result without knowing
# coordinates or hunting on a map.
LANDMARKS = {
    "Times Square": (40.7580, -73.9855),
    "Empire State Building": (40.7484, -73.9857),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Penn Station": (40.7506, -73.9935),
    "Wall Street": (40.7061, -73.9969),
    "Central Park (S)": (40.7660, -73.9773),
    "Columbia University": (40.8075, -73.9626),
    "Brooklyn Bridge": (40.7061, -73.9969),
    "Williamsburg": (40.7143, -73.9613),
    "Prospect Park": (40.6602, -73.9690),
    "Yankee Stadium": (40.8296, -73.9262),
    "LaGuardia Airport": (40.7769, -73.8740),
    "JFK Airport": (40.6413, -73.7781),
}


@st.cache_resource(show_spinner="Loading trained models...")
def load_recommender() -> RideRecommender:
    return RideRecommender()


def coordinate_picker(label: str, default_landmark: str, key: str):
    """Landmark dropdown with a custom-coordinate escape hatch."""
    st.markdown(f"**{label}**")
    choice = st.selectbox(
        "Location", list(LANDMARKS) + ["Custom coordinates..."],
        index=list(LANDMARKS).index(default_landmark), key=f"{key}_select",
        label_visibility="collapsed",
    )
    if choice == "Custom coordinates...":
        col_a, col_b = st.columns(2)
        lat = col_a.number_input("Latitude", 40.45, 41.00, 40.7580,
                                 format="%.4f", key=f"{key}_lat")
        lon = col_b.number_input("Longitude", -74.30, -73.65, -73.9855,
                                 format="%.4f", key=f"{key}_lon")
        return (lat, lon)
    return LANDMARKS[choice]


def render_option_card(option, is_best: bool) -> None:
    """One mode's result, as a bordered card."""
    colour = MODE_COLORS.get(option.mode, INK["muted"])
    icon = MODE_ICONS.get(option.mode, "")

    with st.container(border=True):
        header, badge = st.columns([3, 2])
        header.markdown(
            f"<span style='color:{colour};font-size:1.05rem;font-weight:600'>"
            f"{icon} {option.label}</span>",
            unsafe_allow_html=True,
        )
        if not option.feasible:
            badge.markdown(
                "<div style='text-align:right;color:#d03b3b;font-weight:600'>"
                "Not available</div>", unsafe_allow_html=True)
        elif is_best:
            badge.markdown(
                "<div style='text-align:right;color:#0ca30c;font-weight:600'>"
                "Recommended</div>", unsafe_allow_html=True)
        elif option.on_pareto_frontier:
            badge.markdown(
                f"<div style='text-align:right;color:{INK['secondary']}'>"
                "Also defensible</div>", unsafe_allow_html=True)

        # Values are labelled directly rather than read off a colour, which is
        # what lets the palette's low-contrast slot stay legible.
        m1, m2, m3 = st.columns(3)
        m1.metric("Time", f"{option.duration_min:.0f} min")
        m2.metric("Cost", f"${option.cost_usd:.2f}")
        m3.metric("CO₂e", f"{option.co2_grams:,.0f} g")

        st.caption(
            f"{option.distance_km:.1f} km · {option.vehicle_duration_min:.0f} min "
            f"riding + {option.access_egress_min:.0f} min access/egress"
        )

        for reason in option.infeasible_reasons:
            st.error(reason, icon="🚫")
        for warning in option.warnings:
            st.warning(warning, icon="⚠️")


def comparison_figure(options):
    """Three small horizontal bar panels, one per objective.

    Direct value labels on every bar, so the comparison never depends on
    reading a length against a gridline -- and so the palette's lower-contrast
    slot stays legible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sustainable_ride.viz import apply_matplotlib_style

    apply_matplotlib_style()

    labels = [o.label for o in options]
    colours = [MODE_COLORS.get(o.mode, INK["muted"]) for o in options]
    panels = [
        ("Travel time", [o.duration_min for o in options], "{:.0f} min"),
        ("Cost", [o.cost_usd for o in options], "${:.2f}"),
        ("CO₂e", [o.co2_grams for o in options], "{:,.0f} g"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 0.85 * len(options) + 1.4))
    y = range(len(options))

    for index_panel, (ax, (title, values, fmt)) in enumerate(zip(axes, panels)):
        ax.barh(list(y), values, height=0.5, color=colours, zorder=3)
        ax.set_yticks(list(y))
        # Only the leftmost panel carries the mode names; repeating them three
        # times would triple the ink for no added information.
        ax.set_yticklabels(labels if index_panel == 0 else [""] * len(labels),
                           fontsize=8.5, color=INK["secondary"])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5, color=INK["secondary"], pad=8)
        ax.grid(axis="x", zorder=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(values) * 1.28 if max(values) > 0 else 1)
        ax.tick_params(axis="x", labelsize=7.5)

        for index, value in enumerate(values):
            ax.text(value + max(values) * 0.03, index, fmt.format(value),
                    va="center", ha="left", fontsize=8.5,
                    color=INK["primary"], zorder=4)

    fig.tight_layout()
    return fig


def main() -> None:
    st.title("Sustainable Ride Suggestion")
    st.caption(
        "Predicts travel time, cost and lifecycle CO₂ for taxi, bike and "
        "e-scooter trips in New York City — then ranks them against what you "
        "actually care about. Trained on NYC TLC yellow-cab records and Citi "
        "Bike system data."
    )

    try:
        recommender = load_recommender()
    except ModelsNotTrainedError as exc:
        st.error(str(exc), icon="🚫")
        st.info("Run `python -m sustainable_ride.cli pipeline` to build the models.")
        st.stop()

    # -- sidebar: the query ------------------------------------------------
    with st.sidebar:
        st.header("Your trip")
        origin = coordinate_picker("From", "Times Square", "origin")
        destination = coordinate_picker("To", "Wall Street", "dest")

        st.divider()
        st.header("When")
        date = st.date_input("Date", datetime(2024, 1, 15))
        clock = st.time_input("Departure time", dtime(17, 30))
        departure = datetime.combine(date, clock)

        st.divider()
        st.header("What matters to you")
        st.caption("Drag these and watch the recommendation change.")
        w_cost = st.slider("Cost", 0.0, 1.0, 0.34, 0.01)
        w_time = st.slider("Travel time", 0.0, 1.0, 0.33, 0.01)
        w_co2 = st.slider("CO₂ emissions", 0.0, 1.0, 0.33, 0.01)
        if w_cost + w_time + w_co2 == 0:
            st.error("At least one priority must be above zero.")
            st.stop()

        st.divider()
        st.header("Conditions")
        passengers = st.number_input("Passengers", 1, 6, 1,
                                     help="Taxi emissions are split across occupants.")
        rain = st.slider("Chance of rain", 0.0, 1.0, 0.0, 0.05,
                         help="At 60% or above, exposed modes are ruled out.")
        accessible = st.checkbox("Step-free access required")

    if origin == destination:
        st.warning("Choose two different locations.")
        st.stop()

    try:
        result = recommender.recommend(
            origin=origin, destination=destination, departure=departure,
            weights={"cost": w_cost, "time": w_time, "co2": w_co2},
            passenger_count=int(passengers), rain_probability=float(rain),
            accessibility_required=bool(accessible), include_geometry=True,
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        st.stop()

    # -- headline ----------------------------------------------------------
    if result.best:
        st.success(result.narrative, icon="✅")
    else:
        st.error(result.narrative, icon="🚫")

    st.subheader("Your options")
    for option, column in zip(result.options, st.columns(len(result.options))):
        with column:
            render_option_card(option, is_best=(option is result.best))

    feasible = [o for o in result.options if o.feasible]

    # -- comparison --------------------------------------------------------
    if len(feasible) > 1:
        st.subheader("How the options compare")
        st.caption(
            "Three separate panels, never one shared scale — minutes, dollars "
            "and grams are different units, and plotting them on a common axis "
            "would be meaningless."
        )
        st.pyplot(comparison_figure(feasible), use_container_width=True)

        frame = pd.DataFrame({
            "Mode": [o.label for o in feasible],
            "Time (min)": [round(o.duration_min, 1) for o in feasible],
            "Cost (USD)": [o.cost_usd for o in feasible],
            "CO₂e (g)": [round(o.co2_grams) for o in feasible],
            "Distance (km)": [o.distance_km for o in feasible],
            "On Pareto frontier": ["yes" if o.on_pareto_frontier else "no"
                                   for o in feasible],
        }).set_index("Mode")
        st.caption("**Full comparison**")
        st.dataframe(frame, width="stretch")

    # -- map ---------------------------------------------------------------
    geometry_options = [o for o in feasible if o.geometry]
    if geometry_options:
        st.subheader("Route")
        if result.options[0].route_is_estimate:
            st.info(
                "Routes are straight-line estimates scaled by a circuity factor "
                "fitted from taxi meter readings. Add an `ORS_API_KEY` to `.env` "
                "for real road geometry.",
                icon="ℹ️",
            )
        points = pd.DataFrame(
            [{"lat": origin[0], "lon": origin[1]},
             {"lat": destination[0], "lon": destination[1]}]
        )
        st.map(points, size=40, zoom=12)

    # -- provenance --------------------------------------------------------
    with st.expander("Where these numbers come from"):
        st.markdown(f"**Routing:** {result.routing_provider}")
        st.markdown("**Per-mode estimate sources**")
        for option in result.options:
            confidence = {"high": "🟢", "medium": "🟡", "low": "🟠"}.get(
                option.duration_confidence, "⚪")
            st.markdown(
                f"- {confidence} **{option.label}** — {option.duration_source} "
                f"Cost: {option.cost_source}"
            )

        st.markdown("**Emission factors**")
        st.dataframe(
            pd.DataFrame(factor_provenance())[
                ["mode", "basis", "g_per_km", "deadhead_multiplier", "source"]
            ],
            width="stretch", hide_index=True,
        )
        st.caption(
            "Lifecycle basis: manufacturing and fleet operations are included, "
            "not just tailpipe emissions. Reporting operational emissions alone "
            "would make an e-scooter look about 27× cleaner than a taxi; on a "
            "lifecycle basis the advantage is 6.3×. Still a large win — but a "
            "different claim, and the defensible one."
        )

    if result.best and result.best.co2_saved_vs_taxi_g > 0:
        st.caption(
            f"Choosing {result.best.label} over a taxi avoids "
            f"{result.best.co2_saved_vs_taxi_g:,.0f} g CO₂e — roughly "
            f"{humanise_co2(result.best.co2_saved_vs_taxi_g)}."
        )


if __name__ == "__main__":
    main()
