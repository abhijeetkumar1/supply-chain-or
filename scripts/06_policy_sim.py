"""Day 3, exercise 6: When should a store reorder, and how much? The (s, S) policy.

Script version of notebooks/06_policy_sim.ipynb. Same steps, printed to the terminal:
    1. The story and data: one M5 item, a year of real daily demand
    2. The (s, S) policy and the simulator
    3. Set s and S with formulas: safety stock + EOQ
    4. Simulate the real test year, then 200 resampled years: cycle service level vs fill rate (Plotly)
    5. Grid search over s and S: the cost vs fill-rate frontier (Plotly)
    6. Lead-time uncertainty: what the safety stock formula's second term buys (Plotly)

Needs data/processed/m5_ca1_daily.csv: run .venv/bin/python data/prepare_m5.py first.
Run: .venv/bin/python scripts/06_policy_sim.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
daily = pd.read_csv(ROOT / "data" / "processed" / "m5_ca1_daily.csv")

ITEM = "HOUSEHOLD_1_517"
series = daily[daily.item_id == ITEM].sort_values("date")
demand_all = series["units"].to_numpy()
history = demand_all[-728:-364]  # the year we learn from
test = demand_all[-364:]         # the year we simulate on (the policy never sees it in advance)

# Costs (assumptions, per unit of this item)
UNIT_COST = 10.0                       # $ we pay per unit
HOLDING_PER_DAY = 0.25 * UNIT_COST / 365  # 25% a year to carry stock: capital, space, damage
ORDER_COST = 10.0                      # $ per order: admin, truck slot, receiving
LEAD_TIME = 4                          # days from order to delivery
TARGET_CSL = 0.95                      # aim: no stockout in 95% of replenishment cycles


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def simulate(s, S, demand, lead_time=LEAD_TIME, lead_time_sampler=None, seed=0):
    """Daily (s, S) policy with lost sales.

    Each day: receive deliveries due today, sell what we can (unmet demand is lost),
    then if inventory position (on hand + on order) <= s, order up to S.
    """
    rng = np.random.default_rng(seed)
    on_hand, pipeline = S, []  # pipeline: list of (arrival_day, quantity)
    held = sold = asked = orders = 0
    cycles = cycles_with_stockout = 0
    stockout_this_cycle = False

    for day, d in enumerate(demand):
        arriving = [q for t, q in pipeline if t == day]
        if arriving:
            on_hand += sum(arriving)
            pipeline = [(t, q) for t, q in pipeline if t != day]
            cycles += 1
            cycles_with_stockout += stockout_this_cycle
            stockout_this_cycle = False

        sale = min(on_hand, d)
        on_hand -= sale
        sold += sale
        asked += d
        stockout_this_cycle |= d > sale

        position = on_hand + sum(q for _, q in pipeline)
        if position <= s:
            lt = lead_time if lead_time_sampler is None else lead_time_sampler(rng)
            pipeline.append((day + lt, S - position))
            orders += 1
        held += on_hand

    holding, ordering = HOLDING_PER_DAY * held, ORDER_COST * orders
    return {"fill rate": sold / asked, "cycle service level": 1 - cycles_with_stockout / max(cycles, 1),
            "avg on hand": held / len(demand), "orders": orders,
            "holding $": holding, "ordering $": ordering, "total $": holding + ordering}


def formula_policy(mu, sigma, lead_mean=LEAD_TIME, lead_sd=0.0, csl=TARGET_CSL):
    """s = expected demand over the lead time + safety stock; S = s + EOQ."""
    z = norm.ppf(csl)
    safety_stock = z * math.sqrt(lead_mean * sigma ** 2 + mu ** 2 * lead_sd ** 2)
    eoq = math.sqrt(2 * mu * ORDER_COST / HOLDING_PER_DAY)
    s = mu * lead_mean + safety_stock
    return {"z": z, "safety stock": safety_stock, "EOQ": eoq, "s": round(s), "S": round(s + eoq)}


def resampled_year(seed, weeks=test.reshape(-1, 7)):
    """A plausible alternative year: 52 whole weeks drawn at random (with replacement) from the test year."""
    rng = np.random.default_rng(seed)
    return weeks[rng.integers(0, len(weeks), len(weeks))].ravel()


def average_over_years(s, S, n_years, **kwargs):
    runs = [simulate(s, S, resampled_year(k), seed=k, **kwargs) for k in range(n_years)]
    return pd.DataFrame(runs).mean().to_dict()


def frontier(grid):
    """Cheapest policy at or above each fill rate: the efficient frontier."""
    g = grid.sort_values("fill rate", ascending=False)
    g = g[g["total $"] == g["total $"].cummin()]
    return g.sort_values("fill rate")


if __name__ == "__main__":
    section("1. The data")
    mu, sigma = history.mean(), history.std()
    print(f"{ITEM}: learning year mean {mu:.2f}/day, sd {sigma:.2f}; test year mean {test.mean():.2f}/day")
    print(f"Holding ${HOLDING_PER_DAY:.4f}/unit/day, order ${ORDER_COST:.0f}, lead time {LEAD_TIME} days")

    section("3. Formula policy")
    policy = formula_policy(mu, sigma)
    print({k: round(v, 2) for k, v in policy.items()})

    section("4. Simulate the test year, then 200 resampled years")
    result = simulate(policy["s"], policy["S"], test)
    print("Real test year:", {k: round(v, 3) for k, v in result.items()})
    rows = []
    for csl in [0.80, 0.90, 0.95, 0.99]:
        pol = formula_policy(mu, sigma, csl=csl)
        avg = average_over_years(pol["s"], pol["S"], 200)
        rows.append({"target CSL": csl, "s": pol["s"], "S": pol["S"], "achieved CSL": avg["cycle service level"],
                     "fill rate": avg["fill rate"], "total $": avg["total $"]})
    service = pd.DataFrame(rows).set_index("target CSL")
    print(service.round(3))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{c:.0%}" for c in service.index], y=100 * service["achieved CSL"], name="cycle service level"))
    fig.add_trace(go.Bar(x=[f"{c:.0%}" for c in service.index], y=100 * service["fill rate"], name="fill rate"))
    fig.update_layout(title="Same policies, two ways to score them (average of 200 years)", barmode="group",
                      xaxis_title="target cycle service level used to set s", yaxis_title="%", yaxis_range=[70, 100.5],
                      height=430, width=900, template="plotly_white")
    OUT.mkdir(exist_ok=True)
    fig.write_html(OUT / "06_csl_vs_fill.html")

    section("5. Grid search over s and S (each policy averaged over 20 resampled years)")
    grid = pd.DataFrame([{"s": s, "S": S, **average_over_years(s, S, 20)}
                         for s in range(10, 71, 4) for S in range(60, 361, 20) if S > s])
    formula_avg = average_over_years(policy["s"], policy["S"], 20)
    front = frontier(grid)
    for target in [0.95, 0.98, 0.99, 0.995]:
        row = grid[grid["fill rate"] >= target].nsmallest(1, "total $").iloc[0]
        print(f"cheapest policy with fill rate >= {target:.1%}: s={row.s:.0f}, S={row.S:.0f}, "
              f"fill {row['fill rate']:.3f}, total ${row['total $']:.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=100 * grid["fill rate"], y=grid["total $"], mode="markers", name="every (s, S) tried",
                             marker=dict(size=5, color="#BAB0AC"), customdata=grid[["s", "S"]],
                             hovertemplate="s=%{customdata[0]}, S=%{customdata[1]}<br>fill %{x:.1f}%<br>$%{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=100 * front["fill rate"], y=front["total $"], mode="lines+markers", name="frontier",
                             line=dict(color="#4C78A8", width=3), customdata=front[["s", "S"]],
                             hovertemplate="s=%{customdata[0]}, S=%{customdata[1]}<br>fill %{x:.1f}%<br>$%{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[100 * formula_avg["fill rate"]], y=[formula_avg["total $"]], mode="markers", name="formula policy",
                             marker=dict(size=14, symbol="star", color="#E45756")))
    fig.update_layout(title="Yearly cost vs fill rate for every (s, S)", xaxis_title="fill rate (%)",
                      yaxis_title="holding + ordering cost per year ($)", xaxis_range=[85, 100.5],
                      height=480, width=950, template="plotly_white")
    fig.write_html(OUT / "06_frontier.html")

    section("6. Lead-time uncertainty")
    lead_options = np.array([2, 3, 4, 5, 6, 8, 10])
    sampler = lambda rng: int(rng.choice(lead_options))
    lead_mean, lead_sd = lead_options.mean(), lead_options.std()
    print(f"Lead time is now 2-10 days: mean {lead_mean:.2f}, sd {lead_sd:.2f}")
    rows = []
    for name, sd_used in [("safety stock ignores lead-time spread", 0.0), ("safety stock includes it", lead_sd)]:
        pol = formula_policy(mu, sigma, lead_mean=lead_mean, lead_sd=sd_used)
        runs = pd.DataFrame([simulate(pol["s"], pol["S"], resampled_year(k), lead_time_sampler=sampler, seed=k)
                             for k in range(200)])
        rows.append({"policy": name, "safety stock": round(pol["safety stock"]), "s": pol["s"], "S": pol["S"],
                     "fill rate": runs["fill rate"].mean(), "cycle service level": runs["cycle service level"].mean(),
                     "total $": runs["total $"].mean()})
    lead = pd.DataFrame(rows).set_index("policy")
    print(lead.round(3))
    print(f"\nCharts saved to {OUT / '06_csl_vs_fill.html'} and {OUT / '06_frontier.html'}")
