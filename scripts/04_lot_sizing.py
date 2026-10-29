"""Day 2, exercise 4: When should the factory run the production line?

Script version of notebooks/04_lot_sizing.ipynb. Same steps, printed to the terminal:
    1. The story and data: a 12-week demand forecast, setup cost, holding cost
    2. Two hand plans: run every week vs. big batches
    3. Inventory balance: how the weeks link together
    4. The MIP: when to run, how much to make, what to carry (Plotly chart)
    5. Big-M: why M should be as small as safely possible
    6. Change the setup cost and watch the plan change (Plotly chart)

Run: .venv/bin/python scripts/04_lot_sizing.py
"""

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pulp
from plotly.subplots import make_subplots

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

OUT = Path(__file__).resolve().parent.parent / "outputs"

# 1. The story ----------------------------------------------------------------
# A factory makes one running shoe. We have a 12-week demand forecast. Every week
# we run the line we pay a setup cost (changeover, cleaning, test pairs). Every
# pair left in the warehouse at the end of a week costs a holding cost. The line
# can make at most 3,000 pairs a week. When should we run it, and how much?

WEEKS = list(range(1, 13))
forecast = dict(zip(WEEKS, [1200, 900, 1500, 800, 600, 1800, 2200, 1000, 700, 1300, 1600, 900]))  # pairs
setup_cost = 2_000     # $ each week the line runs
holding_cost = 0.50    # $ per pair left in stock at the end of a week
line_capacity = 3_000  # pairs per week


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def plan_table(make, demand=forecast):
    """Walk the inventory balance week by week: end stock = start stock + make - demand."""
    rows, stock = [], 0
    for t in WEEKS:
        start = stock
        stock = start + make[t] - demand[t]
        rows.append({"week": t, "start stock": start, "make": make[t], "demand": demand[t], "end stock": stock})
    return pd.DataFrame(rows).set_index("week")


def plan_cost(make, setup_cost=setup_cost):
    table = plan_table(make)
    assert (table["end stock"] >= 0).all(), "plan runs out of stock"
    runs = sum(1 for t in WEEKS if make[t] > 0)
    return {"runs": runs, "setup $": runs * setup_cost, "holding $": holding_cost * table["end stock"].sum(),
            "total $": runs * setup_cost + holding_cost * table["end stock"].sum()}


def big_batches():
    """Hand rule: whenever stock can't cover this week, run the line flat out."""
    make, stock = {}, 0
    remaining = sum(forecast.values())
    for t in WEEKS:
        if stock < forecast[t]:
            make[t] = min(line_capacity, remaining - stock)  # don't make more than is still needed
        else:
            make[t] = 0
        stock += make[t] - forecast[t]
        remaining -= forecast[t]
    return make


def solve(setup_cost=setup_cost, big_m="capacity", relax=False):
    """Lot sizing MIP. big_m: 'huge' (1,000,000), 'capacity', or 'tight' (min of capacity, demand still to come)."""
    model = pulp.LpProblem("lot_sizing", pulp.LpMinimize)

    # Decision variables
    run = {t: model.add_variable(f"run_{t}", lowBound=0, upBound=1,
                                 cat="Continuous" if relax else "Binary") for t in WEEKS}  # 1 = run the line
    make = {t: model.add_variable(f"make_{t}", lowBound=0) for t in WEEKS}                # pairs made
    stock = {t: model.add_variable(f"stock_{t}", lowBound=0) for t in WEEKS}              # pairs left at week end

    # Objective: setups + holding
    model += pulp.lpSum(setup_cost * run[t] + holding_cost * stock[t] for t in WEEKS)

    for t in WEEKS:
        # Inventory balance: last week's stock + what we make - demand = this week's stock
        previous = stock[t - 1] if t > 1 else 0
        model += previous + make[t] - forecast[t] == stock[t], f"balance_{t}"

        # Big-M: can only make pairs in a week the line runs
        M = {"huge": 1_000_000,
             "capacity": line_capacity,
             "tight": min(line_capacity, sum(forecast[u] for u in WEEKS if u >= t))}[big_m]
        model += make[t] <= M * run[t], f"run_link_{t}"

        # Line capacity
        model += make[t] <= line_capacity, f"capacity_{t}"

    stats = model.solve(pulp.HiGHS(msg=False))
    assert stats.status == pulp.LpSolveStatus.Optimal, stats.status
    return model, run, make, stock


def plan_chart(make, title):
    table = plan_table(make)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=WEEKS, y=table["demand"], name="demand (forecast)", marker_color="#BAB0AC",
                         hovertemplate="week %{x}<br>demand %{y:,}<extra></extra>"))
    fig.add_trace(go.Bar(x=WEEKS, y=table["make"], name="make", marker_color="#4C78A8",
                         hovertemplate="week %{x}<br>make %{y:,}<extra></extra>"))
    fig.add_trace(go.Scatter(x=WEEKS, y=table["end stock"], name="end-of-week stock", mode="lines+markers",
                             line=dict(color="#F58518", width=3),
                             hovertemplate="week %{x}<br>stock %{y:,}<extra></extra>"))
    fig.add_hline(y=line_capacity, line_dash="dot", line_color="gray", annotation_text="line capacity")
    fig.update_layout(title=title, barmode="group", xaxis=dict(title="week", dtick=1), yaxis_title="pairs",
                      height=430, width=1000, template="plotly_white", legend=dict(orientation="h", y=1.1))
    return fig


if __name__ == "__main__":
    section("1. The data")
    print("Forecast (pairs):", forecast)
    print(f"Total {sum(forecast.values()):,} pairs. Setup ${setup_cost:,}/run, holding ${holding_cost}/pair/week, "
          f"capacity {line_capacity:,}/week")

    section("2. Two hand plans")
    every_week = dict(forecast)
    batches = big_batches()
    hand = pd.DataFrame({"run every week": plan_cost(every_week), "big batches": plan_cost(batches)}).T
    print(hand)

    section("3. Inventory balance for the big-batches plan")
    print(plan_table(batches))

    section("4. The MIP")
    model, run, make, stock = solve()
    best_make = {t: round(make[t].varValue) for t in WEEKS}
    print(plan_table(best_make))
    print(pd.Series(plan_cost(best_make)))
    OUT.mkdir(exist_ok=True)
    plan_chart(best_make, f"Best plan: ${pulp.value(model.objective):,.0f}").write_html(OUT / "04_best_plan.html")

    section("5. Big-M: LP relaxation bound for three choices of M")
    mip = pulp.value(model.objective)
    rows = []
    for big_m in ["huge", "capacity", "tight"]:
        relaxed, *_ = solve(big_m=big_m, relax=True)  # keep the model alive: PuLP 4 frees it otherwise
        exact, *_ = solve(big_m=big_m)
        lp = pulp.value(relaxed.objective)
        rows.append({"M": big_m, "LP relaxation $": round(lp), "real best $": round(pulp.value(exact.objective)),
                     "gap %": round(100 * (mip - lp) / mip, 1)})
    print(pd.DataFrame(rows).set_index("M"))

    section("6. Change the setup cost")
    rows = []
    for k in [0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]:
        m_k, _, make_k, stock_k = solve(setup_cost=k)
        mk = {t: round(make_k[t].varValue) for t in WEEKS}
        rows.append({"setup $": k, "runs": sum(1 for t in WEEKS if mk[t] > 0),
                     "avg stock (pairs)": round(sum(stock_k[t].varValue for t in WEEKS) / len(WEEKS)),
                     "total $": round(pulp.value(m_k.objective))})
    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False))
    print(f"\nFewest possible runs: {sum(forecast.values()):,} / {line_capacity:,} = "
          f"{sum(forecast.values()) / line_capacity:.2f}, so at least {math.ceil(sum(forecast.values()) / line_capacity)}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=sweep["setup $"], y=sweep["runs"], name="line runs", mode="lines+markers",
                             line_shape="hv"), secondary_y=False)
    fig.add_trace(go.Scatter(x=sweep["setup $"], y=sweep["avg stock (pairs)"], name="average stock",
                             mode="lines+markers"), secondary_y=True)
    fig.update_xaxes(title_text="setup cost ($ per run)", tickprefix="$")
    fig.update_yaxes(title_text="runs in 12 weeks", secondary_y=False)
    fig.update_yaxes(title_text="average stock (pairs)", secondary_y=True)
    fig.update_layout(title="Higher setup cost: fewer runs, more stock", height=430, width=1000,
                      template="plotly_white", legend=dict(orientation="h", y=1.1))
    fig.write_html(OUT / "04_setup_sweep.html")
    print(f"\nCharts saved to {OUT / '04_best_plan.html'} and {OUT / '04_setup_sweep.html'}")
