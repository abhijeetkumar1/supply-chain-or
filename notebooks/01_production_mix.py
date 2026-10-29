"""Day 1, exercise 1: How many shoes should the factory make?

Script version of 01_production_mix.ipynb. Same steps, printed to the terminal:
    1. The story and data
    2. Why we can't make everything
    3. Solve by hand: spend stitching hours on the best $ per stitching hour
    4. Let a solver (PuLP + HiGHS) do it and check it agrees
    5. Slack and shadow prices, with the reason for each
    6. Prove the shadow prices: add 1 hour and re-solve
    7. Sweep stitching hours and chart what an extra hour is worth (Plotly)
    8. Where the hand rule breaks (1,100 stitching hours)

Run: .venv/bin/python notebooks/01_production_mix.py
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
# A factory makes three kinds of shoes. Each pair needs time on three machines,
# each machine has limited hours per week, and the market only buys so many of
# each shoe. How many pairs of each should we make to earn the most profit?

PRODUCTS = ["running", "basketball", "lifestyle"]
MACHINES = ["cutting", "stitching", "assembly"]

profit = {"running": 30, "basketball": 45, "lifestyle": 22}            # $ per pair
max_sales = {"running": 400, "basketball": 250, "lifestyle": 500}      # pairs we can sell
hours_available = {"cutting": 600, "stitching": 900, "assembly": 700}  # hours per week
hours_per_pair = {
    "cutting":   {"running": 0.6, "basketball": 0.9, "lifestyle": 0.4},
    "stitching": {"running": 1.0, "basketball": 1.5, "lifestyle": 0.8},
    "assembly":  {"running": 0.7, "basketball": 1.2, "lifestyle": 0.5},
}


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def solve(hours_available):
    """The LP: choose pairs of each shoe to maximize profit within machine hours and sales limits."""
    model = pulp.LpProblem("shoe_factory", pulp.LpMaximize)

    # Decision variables: pairs of each shoe (can't be negative)
    x = {p: model.add_variable(f"x_{p}", lowBound=0) for p in PRODUCTS}

    # Objective: total profit
    model += pulp.lpSum(profit[p] * x[p] for p in PRODUCTS)

    # Constraint 1: each machine's hours used <= hours available
    for m in MACHINES:
        model += pulp.lpSum(hours_per_pair[m][p] * x[p] for p in PRODUCTS) <= hours_available[m], f"hours_{m}"

    # Constraint 2: don't make more than we can sell
    for p in PRODUCTS:
        model += x[p] <= max_sales[p], f"sales_{p}"

    stats = model.solve(pulp.HiGHS(msg=False))
    assert stats.status == pulp.LpSolveStatus.Optimal, stats.status
    return model, x


def hand_rule(hours_available, ranking):
    """Fill shoes in ranking order, each as far as sales and every machine's leftover hours allow."""
    left = dict(hours_available)
    plan = {}
    for p in ranking:
        can_make = min(left[m] / hours_per_pair[m][p] for m in MACHINES)
        plan[p] = min(max_sales[p], can_make)
        for m in MACHINES:
            left[m] -= plan[p] * hours_per_pair[m][p]
    return plan


def total_profit(plan):
    return sum(profit[p] * plan[p] for p in PRODUCTS)


if __name__ == "__main__":
    section("1. The data")
    table = pd.DataFrame(hours_per_pair).T
    table["hours available"] = pd.Series(hours_available)
    table.loc["profit per pair ($)"] = [profit[p] for p in PRODUCTS] + [None]
    table.loc["most we can sell"] = [max_sales[p] for p in PRODUCTS] + [None]
    print(table)

    section("2. Why not make everything we can sell?")
    need = {m: sum(hours_per_pair[m][p] * max_sales[p] for p in PRODUCTS) for m in MACHINES}
    check = pd.DataFrame({"hours needed": need, "hours available": hours_available})
    check["short by"] = (check["hours needed"] - check["hours available"]).clip(lower=0)
    print(check)
    print("\nStitching is short by the most, so it is the bottleneck.")

    section("3. Solve by hand: best $ per stitching hour first")
    per_hour = pd.Series({p: profit[p] / hours_per_pair["stitching"][p] for p in PRODUCTS})
    ranking = list(per_hour.sort_values(ascending=False).index)
    print(per_hour.sort_values(ascending=False).rename("$ per stitching hour"))
    hand_plan = hand_rule(hours_available, ranking)
    hand_profit = total_profit(hand_plan)
    print(f"\nHand plan: { {p: round(v, 2) for p, v in hand_plan.items()} }")
    print(f"Profit by hand: ${hand_profit:,.2f}")

    section("4. Let the solver do it")
    model, x = solve(hours_available)
    solver_profit = pulp.value(model.objective)
    print(pd.DataFrame({"by hand": hand_plan, "solver": {p: x[p].varValue for p in PRODUCTS}}))
    print(f"\nProfit by hand: ${hand_profit:,.2f}   solver: ${solver_profit:,.2f}")

    section("5. Slack and shadow prices")
    why = {
        "hours_cutting":    "Has spare hours, so one more hour adds nothing",
        "hours_stitching":  "Bottleneck. 1 more hour = 1/0.8 = 1.25 more lifestyle pairs x $22 = $27.50",
        "hours_assembly":   "Has spare hours, so one more hour adds nothing",
        "sales_running":    "+$30 per pair, minus 1 stitching hour taken from lifestyle (-$27.50) = +$2.50",
        "sales_basketball": "+$45 per pair, minus 1.5 stitching hours taken from lifestyle (-$41.25) = +$3.75",
        "sales_lifestyle":  "We aren't selling all we could anyway, so more demand adds nothing",
    }
    print(pd.DataFrame(
        [{"constraint": c.name, "slack": round(c.slack, 2) + 0.0, "shadow price ($)": round(c.pi, 2),
          "why": why[c.name]} for c in model.constraints()]
    ).set_index("constraint"))

    section("6. Prove the shadow prices: +1 hour, re-solve")
    rows = []
    for m in MACHINES:
        model_plus, _ = solve(dict(hours_available, **{m: hours_available[m] + 1}))
        rows.append({"machine": m,
                     "shadow price said": model.get_constraint_by_name(f"hours_{m}").pi,
                     "profit actually went up by": pulp.value(model_plus.objective) - solver_profit})
    print(pd.DataFrame(rows).set_index("machine").round(2))

    section("7. What if we had a lot more stitching time?")
    hours = list(range(600, 1301, 5))
    profits, values = [], []
    for h in hours:
        m_h, _ = solve(dict(hours_available, stitching=h))
        profits.append(pulp.value(m_h.objective))
        values.append(m_h.get_constraint_by_name("hours_stitching").pi)
    breaks = [hours[i] for i in range(1, len(hours)) if abs(values[i] - values[i - 1]) > 1e-6]
    for lo, hi in zip([hours[0]] + breaks, breaks + [hours[-1]]):
        print(f"  {lo:>5}-{hi:<5} h: extra hour worth ${values[hours.index(lo)]:.2f}")

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Total profit vs stitching hours", "What one extra stitching hour is worth"))
    fig.add_trace(go.Scatter(x=hours, y=profits, mode="lines",
                             hovertemplate="%{x} h<br>profit $%{y:,.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=values, mode="lines", line_shape="hv",
                             hovertemplate="%{x} h<br>$%{y:.2f} per extra hour<extra></extra>"), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=hours_available["stitching"], line_dash="dot", line_color="gray", row=1, col=col)
        for b in breaks:
            fig.add_vline(x=b, line_width=1, line_color="lightgray", row=1, col=col)
    fig.update_xaxes(title_text="Stitching hours per week")
    fig.update_yaxes(title_text="Profit", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="$ per extra hour", tickprefix="$", row=1, col=2)
    fig.update_layout(height=430, width=1000, showlegend=False, template="plotly_white")
    OUT.mkdir(exist_ok=True)
    fig.write_html(OUT / "01_stitching_sweep.html")
    print(f"\nChart saved to {OUT / '01_stitching_sweep.html'}")

    section("8. Where the hand rule breaks: 1,100 stitching hours")
    big = dict(hours_available, stitching=1100)
    hand_plan_2 = hand_rule(big, ranking)
    model_2, x_2 = solve(big)
    compare = pd.DataFrame({"hand rule": hand_plan_2, "solver": {p: x_2[p].varValue for p in PRODUCTS}}).round(1)
    compare.loc["profit ($)"] = [total_profit(hand_plan_2), pulp.value(model_2.objective)]
    print(compare)
    print(f"\nSolver earns ${pulp.value(model_2.objective) - total_profit(hand_plan_2):,.0f} more. "
          "Assembly is now the bottleneck, so the solver ranks by $ per assembly hour\n"
          "(lifestyle $44.00 > running $42.86 > basketball $37.50) and makes fewer basketball shoes.")
