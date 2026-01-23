# app.py
# Airline Flight Economics Simulator
# ----------------------------------
# A Streamlit app to configure a single flight and analyze CASM, RASM, yield, and profitability.
# Supports route & aircraft selection, cabin seat mix, and fare inputs.
# Enforces a simple floor space constraint and optional extra crew cost when seat thresholds are exceeded.
#
# Updates (2026-01-20):
# - Added top-of-page intro + expandable definitions + “save scenarios” callout
# - Fuel now scales with route distance (stage length)
# - Added optional “demand cap” demo + price warnings when fares look high vs route benchmarks
# - Added simple automated commentary comparing saved scenarios

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

# Fuel scaling assumptions
FUEL_REF_DISTANCE_MILES = 1600.0   # interpret base_fuel_burn_gallons as burn for ~1600-mile stage
FUEL_DISTANCE_EXPONENT = 1.03      # slight nonlinearity; set to 1.0 for purely linear scaling

# Fare benchmarks (workshop-friendly, NOT “real market fares”)
FARE_BENCHMARKS = {
    # Baselines are workshop-friendly (not “true market fares”).
    # Economy anchored to observed/advertised floors; premium/first derived from NYC–DEN ratios.
    "NYC–DEN": {"Y": 120, "P": 255, "F": 700},  # from your observed shopping
    "NYC–ATL": {"Y": 110, "P": 280,  "F": 600},  # econ floors ~25–28; benchmark slightly above floor
    "NYC–ORD": {"Y": 80, "P": 150,  "F": 330},  # econ floor ~33
    "NYC–MIA": {"Y": 100, "P": 210, "F": 720},  # econ floor ~28 (often volatile); benchmark above floor
}
FARE_WARN_MULTIPLIER = {"Y": 1.5, "P": 1.45, "F": 1.35}

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
        "base_fuel_burn_gallons": 4500.0,   # ~1600-mile stage (reference)
        "fuel_cost_per_gallon": 2.50,
        "base_crew_cost": 8000.0,
        "extra_crew_threshold_seats": 180,
        "extra_crew_cost": 2000.0,
        "maintenance_cost_per_flight": 6000.0,
    },
    {
        "name": "A220-300",
        "max_floor_space_units": 135.0,
        "base_fuel_burn_gallons": 3000.0,   # ~1600-mile stage (reference)
        "fuel_cost_per_gallon": 2.50,
        "base_crew_cost": 6000.0,
        "extra_crew_threshold_seats": 120,
        "extra_crew_cost": 1500.0,
        "maintenance_cost_per_flight": 4000.0,
    },
    {
        "name": "737 MAX 8",
        "max_floor_space_units": 175.0,
        "base_fuel_burn_gallons": 4200.0,   # ~1600-mile stage (reference)
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

    if seats_first < 0 or seats_premium < 0 or seats_economy < 0:
        errors.append("Seat counts must be non-negative integers.")

    total_seats = seats_first + seats_premium + seats_economy
    if total_seats <= 0:
        errors.append("Total seats must be greater than 0.")

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


def demand_factor(fare: float, baseline: float, elasticity: float, floor: float) -> float:
    """
    Returns a multiplier in (floor..1]. If fare <= baseline -> 1.0.
    If fare > baseline -> declines as (baseline/fare)^elasticity.
    This is a workshop-friendly “demand cap” proxy (not a true demand model).
    """
    if baseline <= 0:
        return 1.0
    if fare <= baseline:
        return 1.0
    return max(floor, (baseline / fare) ** elasticity)


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
    apply_demand_cap: bool = False,
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

    # Demand cap (optional)
    bench = FARE_BENCHMARKS.get(route["name"], {"Y": fare_economy, "P": fare_premium, "F": fare_first})
    df_Y = demand_factor(fare_economy, bench["Y"], elasticity=0.9, floor=0.55)
    df_P = demand_factor(fare_premium, bench["P"], elasticity=0.75, floor=0.60)
    df_F = demand_factor(fare_first,   bench["F"], elasticity=0.60, floor=0.65)
    if not apply_demand_cap:
        df_Y = df_P = df_F = 1.0

    # Passengers by cabin (fixed load factor × optional demand cap)
    pax_F = seats_first * load_factor * df_F
    pax_P = seats_premium * load_factor * df_P
    pax_Y = seats_economy * load_factor * df_Y
    total_pax = pax_F + pax_P + pax_Y

    # Revenues
    rev_F = pax_F * fare_first
    rev_P = pax_P * fare_premium
    rev_Y = pax_Y * fare_economy
    passenger_revenue = rev_F + rev_P + rev_Y
    ancillary_revenue = total_pax * ancillary_per_pax
    total_revenue = passenger_revenue + ancillary_revenue

    # Fuel burn scales with distance
    base_burn = float(aircraft["base_fuel_burn_gallons"])
    fuel_price = float(aircraft["fuel_cost_per_gallon"])
    distance_factor = (distance / FUEL_REF_DISTANCE_MILES) ** FUEL_DISTANCE_EXPONENT
    fuel_burn_gallons = base_burn * distance_factor
    fuel_cost = fuel_burn_gallons * fuel_price

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

    # Unit metrics
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
        "apply_demand_cap": apply_demand_cap,
        "demand_factor_first": df_F,
        "demand_factor_premium": df_P,
        "demand_factor_economy": df_Y,

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
        "fuel_burn_gallons": fuel_burn_gallons,
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


def scenario_insight(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """
    Compare scenario b vs a and explain profit delta with a simple driver + suggestion.
    """
    dp = float(b.get("Profit", 0.0) - a.get("Profit", 0.0))
    direction = "higher" if dp >= 0 else "lower"

    d_rev = float(b.get("Total_Revenue", 0.0) - a.get("Total_Revenue", 0.0))
    d_cost = float(b.get("Operating_Cost", 0.0) - a.get("Operating_Cost", 0.0))

    # Biggest cost swing
    cost_drivers = []
    for k, label in [("Fuel_Cost", "fuel"), ("Crew_Cost", "crew"), ("Maint_Cost", "maintenance"), ("Airport_Fees", "airport fees")]:
        if k in a and k in b:
            delta = float(b[k] - a[k])
            cost_drivers.append((abs(delta), delta, label))
    cost_drivers.sort(reverse=True, key=lambda x: x[0])

    driver_line = ""
    if cost_drivers and cost_drivers[0][0] >= 250:  # only mention if meaningful
        _, d, label = cost_drivers[0]
        driver_line = f" Biggest cost swing: **{label}** ({'+' if d >= 0 else ''}{d:,.0f})."

    # Suggestions (simple heuristics)
    suggestions = []
    if float(b.get("CASM_cents", 0)) > float(a.get("CASM_cents", 0)) + 0.05:
        suggestions.append("Try lowering unit cost (right-size aircraft, improve seat-space efficiency, or reduce per-flight cost items).")
    if float(b.get("RASM_cents", 0)) + 0.05 < float(a.get("RASM_cents", 0)):
        suggestions.append("Try lifting unit revenue (seat mix, smarter pricing, or higher ancillaries).")
    if float(b.get("Fare_Y", 0)) > float(a.get("Fare_Y", 0)) and dp < 0:
        suggestions.append("If pricing is constraining demand, test a lower economy fare or shift seats toward premium cabins.")
    if not suggestions:
        suggestions.append("To push profit further, experiment with cabin mix + fares while watching the RASM–CASM spread.")

    return (
        f"Scenario **{b.get('Route','')} / {b.get('Aircraft','')}** is **{direction} profit** by "
        f"{dp:,.0f} versus the comparison scenario. Revenue changed by {d_rev:,.0f} and operating cost changed by {d_cost:,.0f}."
        f"{driver_line} Next step: {suggestions[0]}"
    )


# -------------------------------
# Session State for Scenario Storage
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

# Defaults chosen to fit on A321neo:
# 8F (28 units), 18P (36 units), 140Y (140 units) => 204 units <= 220
seats_first = st.sidebar.number_input("First Class Seats", min_value=0, value=8, step=1, format="%d")
seats_premium = st.sidebar.number_input("Premium Economy Seats", min_value=0, value=18, step=1, format="%d")
seats_economy = st.sidebar.number_input("Economy Seats", min_value=0, value=140, step=1, format="%d")

st.sidebar.markdown("---")
st.sidebar.subheader("Average One-Way Fares")

bench = FARE_BENCHMARKS.get(selected_route["name"], {"Y": 80, "P": 170, "F": 600})

# Dynamic bounds (workshop-friendly)
y_min = max(25.0, round(bench["Y"] * 0.5))
y_max = min(400.0, round(bench["Y"] * 3.0))

p_min = max(60.0, round(bench["P"] * 0.5))
p_max = min(600.0, round(bench["P"] * 2.5))

f_min = max(180.0, round(bench["F"] * 0.5))
f_max = min(1500.0, round(bench["F"] * 2.0))

fare_economy = st.sidebar.number_input(
    "Economy Fare ($)",
    min_value=float(y_min),
    max_value=float(y_max),
    value=float(bench["Y"]),
    step=5.0
)

fare_premium = st.sidebar.number_input(
    "Premium Fare ($)",
    min_value=float(p_min),
    max_value=float(p_max),
    value=float(bench["P"]),
    step=10.0
)

fare_first = st.sidebar.number_input(
    "First Fare ($)",
    min_value=float(f_min),
    max_value=float(f_max),
    value=float(bench["F"]),
    step=25.0
)


st.sidebar.info("Load factor is fixed at 85% across all cabins (baseline assumption).")

st.sidebar.markdown("---")
st.sidebar.subheader("Demand realism (optional)")
apply_demand_cap = st.sidebar.checkbox(
    "Apply demand cap when fares are high (demo)",
    value=False,
    help="If enabled, expected sold seats per cabin are reduced when fares exceed a route benchmark."
)


# -------------------------------
# Main Area: Title + Intro + Definitions
# -------------------------------
st.title("Airline Flight Economics Simulator")
st.caption("Explore how route, aircraft, cabin mix, and fares affect CASM, RASM, yield, and profitability (one-way flight).")

st.markdown("""
Welcome! Configure a flight in the sidebar, then review the economics below.
This is a **teaching model** designed for workshops—directionally right, not airline-finance perfect.
""")


st.info("You can save configurations at the bottom of the page to compare scenarios side-by-side.")
st.caption(
    "Benchmark fares are calibrated to **publicly observed one-way prices approximately ~3 weeks from today** "
    "for these routes, then rounded and simplified for workshop use (they are directional, not exact)."
)
st.caption(
    "Premium here is a stand-in for full-fare economy / extra-legroom access, not a true premium cabin."
)
st.markdown("---")


# -------------------------------
# Summary & Validation
# -------------------------------
col_summary1, col_summary2, col_summary3, col_summary4 = st.columns([1.4, 1.4, 1.2, 1.2])
with col_summary1:
    st.write(f"**Route:** {selected_route['name']} ({selected_route['distance_miles']} miles)")
with col_summary2:
    st.write(f"**Aircraft:** {selected_aircraft['name']} (Max floor space: {selected_aircraft['max_floor_space_units']:.0f})")
with col_summary3:
    st.write(f"**Seat Count:** {seats_first + seats_premium + seats_economy}")
with col_summary4:
    st.write(f"**Demand cap demo:** {'On' if apply_demand_cap else 'Off'}")

# Validate configuration and compute floor space usage
is_valid, errors, floor_space_used, floor_space_max = validate_configuration(
    seats_first=seats_first,
    seats_premium=seats_premium,
    seats_economy=seats_economy,
    aircraft=selected_aircraft,
)

# Display floor space utilization
util_pct = (floor_space_used / floor_space_max * 100.0) if floor_space_max > 0 else 0.0
st.write(f"**Floor Space Used:** {floor_space_used:.1f} / {floor_space_max:.1f} units ({util_pct:.1f}%)")

if is_valid:
    if util_pct >= 95.0:
        st.warning("You're close to the aircraft's floor space limit.")
    else:
        st.success("Seat layout fits within the aircraft's floor space constraint.")
else:
    for err in errors:
        st.error(err)

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
    apply_demand_cap=apply_demand_cap,
)

# Price guardrails / warnings
bench = FARE_BENCHMARKS.get(results["route_name"])
if bench:
    warnings = []
    if results["fare_economy"] > bench["Y"] * FARE_WARN_MULTIPLIER["Y"]:
        warnings.append("Economy fare looks **high** vs the route benchmark — would you really sell a full cabin at that price?")
    if results["fare_premium"] > bench["P"] * FARE_WARN_MULTIPLIER["P"]:
        warnings.append("Premium fare looks **high** vs the route benchmark — consider whether demand would soften.")
    if results["fare_first"] > bench["F"] * FARE_WARN_MULTIPLIER["F"]:
        warnings.append("First fare looks **high** vs the route benchmark — are there enough premium travelers on this route?")

    for w in warnings:
        st.warning(w)

    if (not apply_demand_cap) and len(warnings) > 0:
        st.info("If you want the model to *demonstrate* softer demand, enable **Demand cap when fares are high (demo)** in the sidebar.")


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
col5.metric("Load Factor (base)", f"{int(LOAD_FACTOR*100)}%")


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
    "Passengers (LF × demand cap)": [
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
        "Passengers (LF × demand cap)": "{:,.1f}",
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

# Operational metrics
st.markdown("#### Operational Metrics")
ops_df = pd.DataFrame({
    "Metric": [
        "Distance (miles)",
        "Fuel burn (gallons, scaled)",
        "ASMs",
        "RPMs",
        "Ancillary Revenue ($)",
        "Total Revenue ($)",
        "Profit Margin",
        "Demand factor (First)",
        "Demand factor (Premium)",
        "Demand factor (Economy)",
    ],
    "Value": [
        results["distance_miles"],
        results["fuel_burn_gallons"],
        results["ASMs"],
        results["RPMs"],
        results["ancillary_revenue"],
        results["total_revenue"],
        f"{results['profit_margin']*100:.1f}%",
        results["demand_factor_first"],
        results["demand_factor_premium"],
        results["demand_factor_economy"],
    ],
})

def _fmt_ops(metric: str, value: Any) -> str:
    if isinstance(value, str):
        return value
    if "Revenue" in metric or "Total Revenue" in metric:
        return f"${value:,.0f}"
    if "Fuel burn" in metric:
        return f"{value:,.0f}"
    if metric in ("ASMs", "RPMs", "Distance (miles)"):
        return f"{value:,.0f}"
    if "Demand factor" in metric:
        return f"{value:.2f}"
    return str(value)

ops_df["Value"] = [_fmt_ops(m, v) for m, v in zip(ops_df["Metric"], ops_df["Value"])]

st.dataframe(
    ops_df,
    use_container_width=True,
    hide_index=True,
)

# -------------------------------
# Scenario History + Comparison
# -------------------------------
st.markdown("### Scenario Comparison")

st.success("When you're ready, save this configuration below to compare it against others.")

if st.button("💾 Save this scenario to comparison table"):
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

        # Extra drivers for commentary
        "Total_Revenue": results["total_revenue"],
        "Operating_Cost": results["operating_cost"],
        "Fuel_Cost": results["fuel_cost"],
        "Crew_Cost": results["crew_cost"],
        "Maint_Cost": results["maintenance_cost"],
        "Airport_Fees": results["airport_fees"],
        "Total_Seats": results["total_seats"],
        "Distance": results["distance_miles"],
        "DemandCap_On": results["apply_demand_cap"],
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
            "Total_Revenue": "${:,.0f}",
            "Operating_Cost": "${:,.0f}",
            "Fuel_Cost": "${:,.0f}",
            "Crew_Cost": "${:,.0f}",
            "Maint_Cost": "${:,.0f}",
            "Airport_Fees": "${:,.0f}",
            "Distance": "{:,.0f}",
        }),
        use_container_width=True
    )

    # Profit by scenario index
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

    # Automated commentary (compare last 2 scenarios)
    st.markdown("#### Scenario commentary")
    if len(hist_df) >= 2:
        a = hist_df.iloc[-2].to_dict()
        b = hist_df.iloc[-1].to_dict()
        st.write(scenario_insight(a, b))
    else:
        st.caption("Save at least two scenarios to see automated commentary on why profit changed.")
else:
    st.info("Click **Save this scenario** to add the current configuration to the comparison history.")


# -------------------------------
# Footer / Teaching Notes
# -------------------------------

with st.expander("Key terms (click to expand)", expanded=False):
    st.markdown(
        """
- **ASM (Available Seat Mile):** Seats flown × distance  
- **RPM (Revenue Passenger Mile):** Paying passengers × distance  
- **Yield ($/RPM):** Passenger revenue ÷ RPMs (excludes ancillaries)  
- **CASM ($/ASM):** Operating cost ÷ ASMs  
- **RASM ($/ASM):** Total revenue (incl. ancillaries) ÷ ASMs  
- **Profit:** Total revenue − operating cost  
- **Profit margin:** Profit ÷ total revenue
        """
    )

with st.expander("Available routes & aircraft (click to expand)", expanded=False):

    # Routes table
    routes_df = pd.DataFrame([
        {
            "Route": r["name"],
            "Distance (mi)": r["distance_miles"],
            "Airport fees / flight ($)": r["airport_fees_per_flight"],
        }
        for r in ROUTES
    ])

    st.markdown("**Routes in this simulator**")
    st.dataframe(
        routes_df.style.format({
            "Distance (mi)": "{:,.0f}",
            "Airport fees / flight ($)": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Aircraft table (show seat capacity proxy + fuel at reference distance)
    aircraft_df = pd.DataFrame([
        {
            "Aircraft": a["name"],
            "Max floor space (units)": a["max_floor_space_units"],
            "Reference fuel burn (gal @ ~1600 mi)": a["base_fuel_burn_gallons"],
            "Fuel price ($/gal)": a["fuel_cost_per_gallon"],
            "Implied fuel cost @ ref ($)": a["base_fuel_burn_gallons"] * a["fuel_cost_per_gallon"],
        }
        for a in AIRCRAFT
    ])

    st.markdown("**Aircraft options (simplified)**")
    st.dataframe(
        aircraft_df.style.format({
            "Max floor space (units)": "{:,.0f}",
            "Reference fuel burn (gal @ ~1600 mi)": "{:,.0f}",
            "Fuel price ($/gal)": "${:,.2f}",
            "Implied fuel cost @ ref ($)": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Fuel burn scales with distance in the model (using the reference burn around a ~1600-mile stage length). "
        "Floor space is a simplified proxy for aircraft size/capacity."
    )

with st.expander("Notes (assumptions + model boundaries)"):
    st.markdown(
        f"""
- **Load factor** is fixed at **{int(LOAD_FACTOR*100)}%** as a baseline assumption.  
- **Demand cap (optional):** When enabled, sold seats per cabin are reduced if fares exceed a simple benchmark
  using a basic elasticity curve. This is meant to illustrate the idea that *pricing impacts volume*.
- **Fuel scaling:** base fuel burn is treated as a ~{FUEL_REF_DISTANCE_MILES:.0f}-mile reference and scaled with distance.
- **Floor space constraint:** a simplified proxy using seat-type multipliers (F=3.5, P=2.0, Y=1.0).
- **Crew thresholds:** adds incremental crew cost above an aircraft-specific seat count.
        """
    )
