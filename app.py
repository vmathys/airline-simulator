
# app.py
# Airline Flight Economics Simulator
# ----------------------------------
# A Streamlit app to configure a single flight and analyze CASM, RASM, yield, and profitability.
# The app supports route and aircraft selection, cabin seat mix, and fare inputs.
# It enforces a simple floor space constraint and optional extra crew cost when seat thresholds are exceeded.
#
# Author: (You)
# Date: 2026-01-18

import math
from typing import Dict, Any, Tuple, List

import streamlit as st
import pandas as pd
import altair as alt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Airline Flight Economics Simulator",
    layout="wide",
    page_icon="✈️",
)

# -------------------------------
# Constants and Reference Data
# -------------------------------

# Load factor fixed at 85% (spec)
LOAD_FACTOR = 0.85

# Ancillary revenue per passenger (optional assumption)
ANCILLARY_REVENUE_PER_PAX = 25.0

# Seat space multipliers (floor space units per seat)
SPACE_FIRST = 3.5
SPACE_PREMIUM = 2.0
SPACE_ECONOMY = 1.0

# Supported routes
ROUTES = [
    {"name": "NYC–DEN", "distance_miles": 1600, "airport_fees_per_flight": 5000.0},
    {"name": "NYC–ATL", "distance_miles": 760,  "airport_fees_per_flight": 3500.0},
    {"name": "NYC–ORD", "distance_miles": 740,  "airport_fees_per_flight": 3500.0},
    {"name": "NYC–MIA", "distance_miles": 1090, "airport_fees_per_flight": 4000.0},
]

# Supported aircraft types
AIRCRAFT = [
    {
        "name": "A321neo",
        "max_floor_space_units": 220.0,
        "base_fuel_burn_gallons": 4500.0,   # one-way, at this route distance (used as given)
        "fuel_cost_per_gallon": 2.50,
        "base_crew_cost": 8000.0,
        "extra_crew_threshold_seats": 180,
        "extra_crew_cost": 2000.0,
        "maintenance_cost_per_flight": 6000.0,
    },
    {
        "name": "A220-300",
        "max_floor_space_units": 135.0,
        "base_fuel_burn_gallons": 3000.0,
        "fuel_cost_per_gallon": 2.50,
        "base_crew_cost": 6000.0,
        "extra_crew_threshold_seats": 120,
        "extra_crew_cost": 1500.0,
        "maintenance_cost_per_flight": 4000.0,
    },
    {
        "name": "737 MAX 8",
        "max_floor_space_units": 175.0,
        "base_fuel_burn_gallons": 4200.0,
        "fuel_cost_per_gallon": 2.50,
        "base_crew_cost": 7500.0,
        "extra_crew_threshold_seats": 170,
        "extra_crew_cost": 1800.0,
        "maintenance_cost_per_flight": 5500.0,
    },
]


# -------------------------------
# Helper Functions
# -------------------------------

def get_route_by_name(name: str) -> Dict[str, Any]:
    for r in ROUTES:
        if r["name"] == name:
            return r
    raise ValueError(f"Route '{name}' not found.")


def get_aircraft_by_name(name: str) -> Dict[str, Any]:
    for a in AIRCRAFT:
        if a["name"] == name:
            return a
    raise ValueError(f"Aircraft '{name}' not found.")


def validate_configuration(
    seats_first: int,
    seats_premium: int,
    seats_economy: int,
    aircraft: Dict[str, Any],
) -> Tuple[bool, List[str], float, float]:
    """
    Validate seat counts and floor space constraint for the selected aircraft.
    Returns:
        (is_valid, errors, floor_space_used, floor_space_max)
    """
    errors = []
    # Non-negative and integer counts are handled by Streamlit inputs,
    # but we enforce again for safety.
    if seats_first < 0 or seats_premium < 0 or seats_economy < 0:
        errors.append("Seat counts must be non-negative integers.")
    total_seats = seats_first + seats_premium + seats_economy
    if total_seats <= 0:
        errors.append("Total seats must be greater than 0.")

    # Floor space use
    floor_space_used = (
        seats_first * SPACE_FIRST +
        seats_premium * SPACE_PREMIUM +
        seats_economy * SPACE_ECONOMY
    )
    floor_space_max = float(aircraft["max_floor_space_units"])

    if floor_space_used > floor_space_max + 1e-9:
        errors.append(
            f"Seat layout exceeds aircraft floor space. "
            f"Used {floor_space_used:.1f} vs max {floor_space_max:.1f} units."
        )

    return (len(errors) == 0, errors, floor_space_used, floor_space_max)


def compute_flight_economics(
    route: Dict[str, Any],
    aircraft: Dict[str, Any],
    seats_first: int,
    seats_premium: int,
    seats_economy: int,
    fare_first: float,
    fare_premium: float,
    fare_economy: float,
    load_factor: float = LOAD_FACTOR,
    ancillary_per_pax: float = ANCILLARY_REVENUE_PER_PAX,
) -> Dict[str, Any]:
    """
    Compute all key outputs for the configured flight.

    Definitions:
    - ASMs (Available Seat Miles): total_seats * distance
    - RPMs (Revenue Passenger Miles): total_pax * distance
    - Yield ($/RPM): passenger_revenue / RPMs
    - CASM ($/ASM): operating_cost / ASMs
    - RASM ($/ASM): total_revenue / ASMs
    - Profit = total_revenue - operating_cost
    - Profit Margin = Profit / total_revenue
    """

    distance = float(route["distance_miles"])
    airport_fees = float(route["airport_fees_per_flight"])

    total_seats = int(seats_first + seats_premium + seats_economy)

    # Passengers by cabin (fixed load factor)
    pax_F = seats_first * load_factor
    pax_P = seats_premium * load_factor
    pax_Y = seats_economy * load_factor
    total_pax = pax_F + pax_P + pax_Y

    # Revenues
    rev_F = pax_F * fare_first
    rev_P = pax_P * fare_premium
    rev_Y = pax_Y * fare_economy
    passenger_revenue = rev_F + rev_P + rev_Y
    ancillary_revenue = total_pax * ancillary_per_pax
    total_revenue = passenger_revenue + ancillary_revenue

    # Fuel cost (as provided per aircraft, one-way)
    fuel_cost = float(aircraft["base_fuel_burn_gallons"]) * float(aircraft["fuel_cost_per_gallon"])

    # Crew cost (extra crew if total seats exceed aircraft threshold)
    crew_cost = float(aircraft["base_crew_cost"])
    if total_seats > int(aircraft["extra_crew_threshold_seats"]):
        crew_cost += float(aircraft["extra_crew_cost"])

    # Maintenance
    maintenance_cost = float(aircraft["maintenance_cost_per_flight"])

    # Total operating cost
    operating_cost = fuel_cost + crew_cost + maintenance_cost + airport_fees

    # Productivity metrics
    ASMs = total_seats * distance  # Available Seat Miles
    RPMs = total_pax * distance    # Revenue Passenger Miles

    # Guard against division by zero (edge cases)
    yield_per_rpm = passenger_revenue / RPMs if RPMs > 0 else 0.0  # $/RPM
    casm = operating_cost / ASMs if ASMs > 0 else 0.0              # $/ASM
    rasm = total_revenue / ASMs if ASMs > 0 else 0.0               # $/ASM

    # Profitability
    profit = total_revenue - operating_cost
    profit_margin = (profit / total_revenue) if total_revenue > 0 else 0.0

    return {
        # Inputs echo
        "route_name": route["name"],
        "aircraft_name": aircraft["name"],
        "distance_miles": distance,
        "airport_fees": airport_fees,
        "seats_first": seats_first,
        "seats_premium": seats_premium,
        "seats_economy": seats_economy,
        "fare_first": fare_first,
        "fare_premium": fare_premium,
        "fare_economy": fare_economy,
        "load_factor": load_factor,
        # Pax breakdown
        "pax_first": pax_F,
        "pax_premium": pax_P,
        "pax_economy": pax_Y,
        "total_pax": total_pax,
        # Revenues
        "rev_first": rev_F,
        "rev_premium": rev_P,
        "rev_economy": rev_Y,
        "passenger_revenue": passenger_revenue,
        "ancillary_revenue": ancillary_revenue,
        "total_revenue": total_revenue,
        # Costs
        "fuel_cost": fuel_cost,
        "crew_cost": crew_cost,
        "maintenance_cost": maintenance_cost,
        "operating_cost": operating_cost,
        # Capacity / utilization
        "total_seats": total_seats,
        "ASMs": ASMs,
        "RPMs": RPMs,
        # Unit metrics
        "yield_per_rpm": yield_per_rpm,   # $ per RPM
        "casm": casm,                     # $ per ASM
        "rasm": rasm,                     # $ per ASM
        # Profitability
        "profit": profit,
        "profit_margin": profit_margin,
    }


def cents_per_mile(value_dollars_per_mile: float) -> float:
    """Convert $/mile to cents/mile for display."""
    return value_dollars_per_mile * 100.0


def money(value: float) -> str:
    """Format as dollars with commas."""
    return f"${value:,.0f}"


def money_precise(value: float, decimals: int = 2) -> str:
    return f"${value:,.{decimals}f}"


# -------------------------------
# Session State for Scenario Storage (Optional Feature)
# -------------------------------
if "scenario_history" not in st.session_state:
    st.session_state["scenario_history"] = []  # list of dicts


# -------------------------------
# Sidebar: Inputs
# -------------------------------
st.sidebar.title("✈️ Flight Configuration")

# Route selection
route_names = [r["name"] for r in ROUTES]
route_name = st.sidebar.selectbox("Select Route", options=route_names, index=0)
selected_route = get_route_by_name(route_name)

# Aircraft selection
aircraft_names = [a["name"] for a in AIRCRAFT]
aircraft_name = st.sidebar.selectbox("Select Aircraft", options=aircraft_names, index=0)
selected_aircraft = get_aircraft_by_name(aircraft_name)

st.sidebar.markdown("---")
st.sidebar.subheader("Cabin Seat Configuration")

# Defaults chosen to fit on A321neo by default:
# 8F (28 units), 18P (36 units), 140Y (140 units) => 204 units <= 220
seats_first = st.sidebar.number_input("First Class Seats", min_value=0, value=8, step=1, format="%d")
seats_premium = st.sidebar.number_input("Premium Economy Seats", min_value=0, value=18, step=1, format="%d")
seats_economy = st.sidebar.number_input("Economy Seats", min_value=0, value=140, step=1, format="%d")

st.sidebar.markdown("---")
st.sidebar.subheader("Average One-Way Fares")

fare_first = st.sidebar.number_input("First Fare ($)", min_value=500.0, max_value=1500.0, value=900.0, step=25.0)
fare_premium = st.sidebar.number_input("Premium Fare ($)", min_value=250.0, max_value=750.0, value=400.0, step=10.0)
fare_economy = st.sidebar.number_input("Economy Fare ($)", min_value=99.0, max_value=350.0, value=180.0, step=5.0)

st.sidebar.info("Load factor is fixed at 85% across all cabins.")


# -------------------------------
# Main Area: Summary & Validation
# -------------------------------
st.title("Airline Flight Economics Simulator")
st.caption("Explore how route, aircraft, cabin mix, and fares affect CASM, RASM, yield, and profitability (one-way flight).")

col_summary1, col_summary2, col_summary3 = st.columns([1.5, 1.5, 1.5])
with col_summary1:
    st.write(f"**Route:** {selected_route['name']} ({selected_route['distance_miles']} miles)")
with col_summary2:
    st.write(f"**Aircraft:** {selected_aircraft['name']} (Max floor space: {selected_aircraft['max_floor_space_units']:.0f})")
with col_summary3:
    fs_used = seats_first*SPACE_FIRST + seats_premium*SPACE_PREAMIUM if False else None  # placeholder safety; replaced immediately below

# Validate configuration and compute floor space usage
is_valid, errors, floor_space_used, floor_space_max = validate_configuration(
    seats_first=seats_first,
    seats_premium=seats_premium,
    seats_economy=seats_economy,
    aircraft=selected_aircraft,
)

# Display floor space utilization
util_pct = (floor_space_used / floor_space_max * 100.0) if floor_space_max > 0 else 0.0
util_note = f"**Floor Space Used:** {floor_space_used:.1f} / {floor_space_max:.1f} units ({util_pct:.1f}%)"
st.write(util_note)
if is_valid:
    if util_pct >= 95.0:
        st.warning("You're close to the aircraft's floor space limit.")
    else:
        st.success("Seat layout fits within the aircraft's floor space constraint.")
else:
    for err in errors:
        st.error(err)

# Early exit if invalid
if not is_valid:
    st.stop()

# -------------------------------
# Compute Results
# -------------------------------
results = compute_flight_economics(
    route=selected_route,
    aircraft=selected_aircraft,
    seats_first=seats_first,
    seats_premium=seats_premium,
    seats_economy=seats_economy,
    fare_first=fare_first,
    fare_premium=fare_premium,
    fare_economy=fare_economy,
)

# -------------------------------
# Metrics Row
# -------------------------------
st.markdown("### Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

profit_str = money(results["profit"])
casm_cents = cents_per_mile(results["casm"])
rasm_cents = cents_per_mile(results["rasm"])
yield_cents = cents_per_mile(results["yield_per_rpm"])

col1.metric("Profit", profit_str)
col2.metric("CASM (¢/ASM)", f"{casm_cents:.2f}")
col3.metric("RASM (¢/ASM)", f"{rasm_cents:.2f}")
col4.metric("Yield (¢/RPM)", f"{yield_cents:.2f}")
col5.metric("Load Factor", f"{int(LOAD_FACTOR*100)}%")

# -------------------------------
# Detailed Tables
# -------------------------------
st.markdown("### Detailed Breakdown")

# Seats and Passengers by Cabin
seats_pax_df = pd.DataFrame({
    "Cabin": ["First", "Premium", "Economy", "Total"],
    "Seats": [
        results["seats_first"],
        results["seats_premium"],
        results["seats_economy"],
        results["total_seats"],
    ],
    "Passengers (LF 85%)": [
        results["pax_first"],
        results["pax_premium"],
        results["pax_economy"],
        results["total_pax"],
    ],
    "Passenger Revenue ($)": [
        results["rev_first"],
        results["rev_premium"],
        results["rev_economy"],
        results["passenger_revenue"],
    ],
})
st.dataframe(
    seats_pax_df.style.format({
        "Seats": "{:,.0f}",
        "Passengers (LF 85%)": "{:,.1f}",
        "Passenger Revenue ($)": "${:,.0f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# Costs by category and in total
costs_df = pd.DataFrame({
    "Cost Category": ["Fuel", "Crew", "Maintenance", "Airport Fees", "Total Operating Cost"],
    "Amount ($)": [
        results["fuel_cost"],
        results["crew_cost"],
        results["maintenance_cost"],
        results["airport_fees"],
        results["operating_cost"],
    ],
})
st.dataframe(
    costs_df.style.format({"Amount ($)": "${:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)

# Additional operational metrics
st.markdown("#### Operational Metrics")
ops_df = pd.DataFrame({
    "Metric": ["Distance (miles)", "ASMs", "RPMs", "Ancillary Revenue ($)", "Total Revenue ($)", "Profit Margin"],
    "Value": [
        results["distance_miles"],
        results["ASMs"],
        results["RPMs"],
        results["ancillary_revenue"],
        results["total_revenue"],
        f"{results['profit_margin']*100:.1f}%",
    ],
})
st.dataframe(
    ops_df.style.format({
        "Value": lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and "%" not in str(x) and "$" not in str(x) else x
    }),
    use_container_width=True,
    hide_index=True,
)

# -------------------------------
# Charts
# -------------------------------
st.markdown("### Charts")

# 1) CASM vs RASM (cents per ASM)
cr_df = pd.DataFrame({
    "Metric": ["CASM", "RASM"],
    "Cents_per_ASM": [casm_cents, rasm_cents],
})
cr_chart = (
    alt.Chart(cr_df)
    .mark_bar()
    .encode(
        x=alt.X("Metric:N", title="Metric"),
        y=alt.Y("Cents_per_ASM:Q", title="Cents per ASM"),
        color=alt.Color("Metric:N", legend=None),
        tooltip=["Metric", alt.Tooltip("Cents_per_ASM:Q", format=".2f")]
    )
    .properties(title="CASM vs RASM (¢ per ASM)", width=400, height=300)
)
st.altair_chart(cr_chart, use_container_width=True)

# Optional: Profit indicator as a separate bar
profit_chart_df = pd.DataFrame({
    "Measure": ["Profit ($)"],
    "Value": [results["profit"]],
})
profit_chart = (
    alt.Chart(profit_chart_df)
    .mark_bar(color="seagreen" if results["profit"] >= 0 else "crimson")
    .encode(
        x=alt.X("Measure:N", title=""),
        y=alt.Y("Value:Q", title="Dollars"),
        tooltip=[alt.Tooltip("Value:Q", title="Profit ($)", format=",.0f")],
    )
    .properties(title="Profit (one-way)", width=400, height=300)
)
st.altair_chart(profit_chart, use_container_width=True)

# -------------------------------
# Scenario History (Optional Feature)
# -------------------------------
st.markdown("### Scenario Comparison (Optional)")

# Button to save current scenario
if st.button("💾 Save Scenario"):
    st.session_state["scenario_history"].append({
        "Route": results["route_name"],
        "Aircraft": results["aircraft_name"],
        "F": results["seats_first"],
        "P": results["seats_premium"],
        "Y": results["seats_economy"],
        "Fare_F": results["fare_first"],
        "Fare_P": results["fare_premium"],
        "Fare_Y": results["fare_economy"],
        "CASM_cents": casm_cents,
        "RASM_cents": rasm_cents,
        "Yield_cents": yield_cents,
        "Profit": results["profit"],
        "Profit_Margin_%": results["profit_margin"] * 100.0,
    })
    st.success("Scenario saved!")

if len(st.session_state["scenario_history"]) > 0:
    hist_df = pd.DataFrame(st.session_state["scenario_history"])
    st.dataframe(
        hist_df.style.format({
            "Fare_F": "${:,.0f}",
            "Fare_P": "${:,.0f}",
            "Fare_Y": "${:,.0f}",
            "CASM_cents": "{:,.2f}",
            "RASM_cents": "{:,.2f}",
            "Yield_cents": "{:,.2f}",
            "Profit": "${:,.0f}",
            "Profit_Margin_%": "{:,.1f}%",
        }),
        use_container_width=True
    )

    # Comparison chart: Profit by scenario index
    hist_df_plot = hist_df.reset_index().rename(columns={"index": "Scenario #"})
    profit_history_chart = (
        alt.Chart(hist_df_plot)
        .mark_bar()
        .encode(
            x=alt.X("Scenario #:O", title="Scenario #"),
            y=alt.Y("Profit:Q", title="Profit ($)"),
            color=alt.Color("Aircraft:N", title="Aircraft"),
            tooltip=[
                alt.Tooltip("Scenario #:O"),
                alt.Tooltip("Route:N"),
                alt.Tooltip("Aircraft:N"),
                alt.Tooltip("Profit:Q", format=",.0f"),
                alt.Tooltip("CASM_cents:Q", title="CASM (¢)", format=".2f"),
                alt.Tooltip("RASM_cents:Q", title="RASM (¢)", format=".2f"),
            ],
        )
        .properties(title="Saved Scenarios: Profit Comparison", height=350)
    )
    st.altair_chart(profit_history_chart, use_container_width=True)
else:
    st.info("Click **Save Scenario** to add the current configuration to the comparison history.")

# -------------------------------
# Footer / Teaching Notes
# -------------------------------
with st.expander("Notes and Definitions"):
    st.markdown(
        """
- **ASMs (Available Seat Miles)** = Total Seats × Route Distance  
- **RPMs (Revenue Passenger Miles)** = Total Passengers × Route Distance  
- **Yield ($/RPM)** = Passenger Revenue ÷ RPMs (excludes ancillary revenue)  
- **CASM ($/ASM)** = Operating Cost ÷ ASMs  
- **RASM ($/ASM)** = Total Revenue ÷ ASMs (includes ancillary revenue)  
- **Profit** = Total Revenue − Operating Cost  
- **Profit Margin** = Profit ÷ Total Revenue

The model assumes a fixed **85% load factor** in each cabin, and a simple floor space constraint
based on seat-type multipliers (F=3.5, P=2.0, Y=1.0). Crew costs may increase when total seats exceed
an aircraft-specific threshold. Fuel burn is treated as a fixed one-way figure per aircraft for the purposes
of this workshop.
        """
    )
