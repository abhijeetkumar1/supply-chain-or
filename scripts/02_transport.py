"""Day 1, exercise 2: Which factory should ship to which DC?

Script version of notebooks/02_transport.ipynb. Same steps, printed to the terminal:
    Part A - ship directly from factories to DCs (the "transportation problem")
        1. The story and data
        2. Can supply cover demand?
        3. Solve by hand: cheapest route first
        4. Let a solver do it, and see why it beats the hand rule
        5. Shadow prices: what one more pair at each DC costs, and what one more
           pair of factory capacity saves
        6. Reduced costs: how much cheaper an unused route must get to be used
    Part B - add a hub (the "transshipment problem")
        7. Add a West Coast hub that factories can ship through
        8. What is more hub capacity worth? (Plotly chart)

Run: .venv/bin/python scripts/02_transport.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pulp
from plotly.subplots import make_subplots

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

OUT = Path(__file__).resolve().parent.parent / "outputs"

# 1. The story ----------------------------------------------------------------
# Three factories in Asia make the same shoe. Five US distribution centers (DCs)
# each need a set number of pairs this week. Shipping cost depends on the route.
# Which factory should send how many pairs to which DC, at the lowest total cost?

FACTORIES = ["Vietnam", "Indonesia", "China"]
DCS = ["Los Angeles", "Portland", "Memphis", "New Jersey", "Chicago"]

supply = {"Vietnam": 50_000, "Indonesia": 30_000, "China": 25_000}  # pairs per week
demand = {"Los Angeles": 20_000, "Portland": 15_000, "Memphis": 25_000,
          "New Jersey": 25_000, "Chicago": 15_000}                  # pairs per week

ship_cost = pd.DataFrame(  # $ per pair, factory -> DC directly
    [[2.0, 2.3, 3.4, 3.9, 3.5],
     [2.2, 2.5, 3.3, 3.6, 3.4],
     [1.8, 2.0, 3.6, 4.2, 3.7]],
    index=FACTORIES, columns=DCS,
)

# Part B: a hub at the Port of Long Beach. Ocean freight to the hub, then rail/truck out.
HUB = "Long Beach hub"
to_hub = {"Vietnam": 1.4, "Indonesia": 1.5, "China": 1.2}  # $ per pair, factory -> hub
from_hub = {"Los Angeles": 0.3, "Portland": 0.6, "Memphis": 1.3,
            "New Jersey": 1.8, "Chicago": 1.5}              # $ per pair, hub -> DC
hub_capacity = 30_000                                        # pairs per week


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def var_name(*parts):
    return "_".join(parts).replace(" ", "_")


def solve(use_hub=False, hub_capacity=hub_capacity):
    """Min-cost shipping plan. With use_hub, factories may also ship via the hub."""
    model = pulp.LpProblem("shipping", pulp.LpMinimize)

    # Decision variables: pairs shipped on each route
    direct = {(f, d): model.add_variable(var_name("direct", f, d), lowBound=0) for f in FACTORIES for d in DCS}
    into_hub = {f: model.add_variable(var_name("to_hub", f), lowBound=0) for f in FACTORIES} if use_hub else {}
    out_of_hub = {d: model.add_variable(var_name("from_hub", d), lowBound=0) for d in DCS} if use_hub else {}

    # Objective: total shipping cost
    model += (pulp.lpSum(ship_cost.loc[f, d] * direct[f, d] for f in FACTORIES for d in DCS)
              + pulp.lpSum(to_hub[f] * into_hub[f] for f in into_hub)
              + pulp.lpSum(from_hub[d] * out_of_hub[d] for d in out_of_hub))

    # Constraint 1: a factory can't ship more than it makes
    for f in FACTORIES:
        model += pulp.lpSum(direct[f, d] for d in DCS) + into_hub.get(f, 0) <= supply[f], var_name("supply", f)

    # Constraint 2: every DC gets at least what it needs
    for d in DCS:
        model += pulp.lpSum(direct[f, d] for f in FACTORIES) + out_of_hub.get(d, 0) >= demand[d], var_name("demand", d)

    if use_hub:
        # Constraint 3: the hub stores nothing, so pairs in = pairs out
        model += pulp.lpSum(into_hub.values()) == pulp.lpSum(out_of_hub.values()), "hub_in_equals_out"
        # Constraint 4: the hub can only handle so many pairs
        model += pulp.lpSum(into_hub.values()) <= hub_capacity, "hub_capacity"

    stats = model.solve(pulp.HiGHS(msg=False))
    assert stats.status == pulp.LpSolveStatus.Optimal, stats.status
    return model, direct, into_hub, out_of_hub


def cheapest_route_first():
    """Hand rule: repeatedly ship as much as possible on the cheapest route still open."""
    left_supply, left_demand = dict(supply), dict(demand)
    plan = pd.DataFrame(0.0, index=FACTORIES, columns=DCS)
    for f, d in ship_cost.stack().sort_values().index:
        qty = min(left_supply[f], left_demand[d])
        if qty > 0:
            plan.loc[f, d] = qty
            left_supply[f] -= qty
            left_demand[d] -= qty
    return plan


def plan_table(direct):
    return pd.DataFrame({d: {f: direct[f, d].varValue for f in FACTORIES} for d in DCS})


if __name__ == "__main__":
    # ------------------------------------------------------------- Part A
    section("1. The data")
    print("Shipping cost, $ per pair:")
    print(ship_cost)
    print("\nSupply (pairs/week):", supply)
    print("Demand (pairs/week):", demand)

    section("2. Can supply cover demand?")
    print(f"Total supply {sum(supply.values()):,} vs total demand {sum(demand.values()):,}: "
          f"{sum(supply.values()) - sum(demand.values()):,} pairs spare. The problem is which routes to use.")

    section("3. Solve by hand: cheapest route first")
    hand = cheapest_route_first()
    hand_cost = (hand * ship_cost).to_numpy().sum()
    print(hand)
    print(f"\nCost by hand: ${hand_cost:,.0f}")

    section("4. Let the solver do it")
    model_a, direct_a, _, _ = solve()
    cost_a = pulp.value(model_a.objective)
    print(plan_table(direct_a))
    print(f"\nCost by hand: ${hand_cost:,.0f}   solver: ${cost_a:,.0f}   "
          f"solver saves ${hand_cost - cost_a:,.0f} per week")
    print("Why: the hand rule grabbed China -> LA ($1.80) first, which later forced Vietnam -> New Jersey\n"
          "at $3.90. The solver saves Indonesia, the cheapest source for New Jersey ($3.60), for New Jersey.")

    section("5. Shadow prices")
    why = {
        "demand_Los_Angeles": "Vietnam has spare pairs: Vietnam -> LA = $2.00",
        "demand_Portland":    "China -> Portland $2.00, but China is full, so move one China pair off LA "
                              "and cover LA from Vietnam (+$0.20) = $2.20",
        "demand_Memphis":     "Vietnam has spare pairs: Vietnam -> Memphis = $3.40",
        "demand_New_Jersey":  "Indonesia -> NJ $3.60, but Indonesia is full, so move one Indonesia pair off "
                              "Chicago and cover Chicago from Vietnam (+$0.10) = $3.70",
        "demand_Chicago":     "Vietnam has spare pairs: Vietnam -> Chicago = $3.50",
        "supply_Vietnam":     "Already has 5,000 spare pairs, so one more saves nothing",
        "supply_Indonesia":   "One more pair to Chicago ($3.40) replaces Vietnam -> Chicago ($3.50): saves $0.10",
        "supply_China":       "One more pair to LA ($1.80) replaces Vietnam -> LA ($2.00): saves $0.20",
    }
    print(pd.DataFrame(
        [{"constraint": c.name, "left over": round(c.slack, 0) + 0.0,
          "$ per extra pair": round(abs(c.pi), 2), "why": why[c.name]} for c in model_a.constraints()]
    ).set_index("constraint"))
    print("\nDemand rows: what serving 1 more pair at that DC would cost.\n"
          "Supply rows: what 1 more pair of factory capacity would save.")

    section("6. Reduced costs: unused routes")
    unused = [{"route": f"{f} -> {d}", "cost now": ship_cost.loc[f, d],
               "must get cheaper by": round(v.dj, 2),
               "worth using below": round(ship_cost.loc[f, d] - v.dj, 2)}
              for (f, d), v in direct_a.items() if v.varValue < 1e-6]
    print(pd.DataFrame(unused).set_index("route").sort_values("must get cheaper by"))

    # ------------------------------------------------------------- Part B
    section("7. Add a hub at Long Beach")
    via_hub = pd.DataFrame({d: {f: to_hub[f] + from_hub[d] for f in FACTORIES} for d in DCS})
    print("Cost via the hub, $ per pair:")
    print(via_hub.round(2))
    print("\nSaving vs shipping direct (positive = hub is cheaper):")
    print((ship_cost - via_hub).round(2))

    model_b, direct_b, into_hub_b, out_of_hub_b = solve(use_hub=True)
    cost_b = pulp.value(model_b.objective)
    print("\nDirect shipments:")
    print(plan_table(direct_b))
    print("\nInto hub:", {f: round(v.varValue) for f, v in into_hub_b.items()})
    print("Out of hub:", {d: round(v.varValue) for d, v in out_of_hub_b.items()})
    hub_value = abs(model_b.get_constraint_by_name("hub_capacity").pi)
    print(f"\nCost without hub: ${cost_a:,.0f}   with hub: ${cost_b:,.0f}   saves ${cost_a - cost_b:,.0f} per week")
    print(f"Hub is full ({hub_capacity:,} pairs). One more pair of hub capacity saves ${hub_value:.2f}:\n"
          "Vietnam has spare pairs, and Vietnam -> hub -> Chicago ($2.90) beats Vietnam -> Chicago direct ($3.50).")

    section("8. What is more hub capacity worth?")
    capacities = list(range(0, 105_001, 500))
    costs, values = [], []
    for cap in capacities:
        m_c, *_ = solve(use_hub=True, hub_capacity=cap)
        costs.append(pulp.value(m_c.objective))
        values.append(abs(m_c.get_constraint_by_name("hub_capacity").pi))
    # Exactly at a kink the solver may report either side's value; smooth those single points
    for i in range(1, len(values) - 1):
        if abs(values[i] - values[i - 1]) > 1e-6 and abs(values[i] - values[i + 1]) > 1e-6:
            values[i] = values[i + 1]
    breaks = [capacities[i] for i in range(1, len(capacities)) if abs(values[i] - values[i - 1]) > 1e-6]
    for lo, hi in zip([capacities[0]] + breaks, breaks + [capacities[-1]]):
        print(f"  {lo:>7,}-{hi:<7,} pairs: one more pair of hub capacity saves ${values[capacities.index(lo)]:.2f}")

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Weekly shipping cost vs hub capacity", "Saving from one more pair of hub capacity"))
    fig.add_trace(go.Scatter(x=capacities, y=costs, mode="lines",
                             hovertemplate="%{x:,} pairs<br>cost $%{y:,.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=capacities, y=values, mode="lines", line_shape="hv",
                             hovertemplate="%{x:,} pairs<br>saves $%{y:.2f} per pair<extra></extra>"), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=hub_capacity, line_dash="dot", line_color="gray", row=1, col=col)
    fig.update_xaxes(title_text="Hub capacity (pairs per week)", tickformat=",")
    fig.update_yaxes(title_text="Shipping cost", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="$ saved per extra pair", tickprefix="$", row=1, col=2)
    fig.update_layout(height=430, width=1000, showlegend=False, template="plotly_white")
    OUT.mkdir(exist_ok=True)
    fig.write_html(OUT / "02_hub_capacity.html")
    print(f"\nChart saved to {OUT / '02_hub_capacity.html'}")
