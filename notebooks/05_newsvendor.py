"""Day 3, exercise 5: How much should we order when demand is uncertain? (the newsvendor)

Script version of 05_newsvendor.ipynb. Same steps, printed to the terminal:
    1. The story: order before you know demand; leftovers cost money, so do lost sales
    2. One item by hand: try every order quantity against 26 weeks of possible demand
    3. The critical ratio: why the best order is a quantile of the forecast
    4. Newsvendor cost = pinball loss: the link to quantile forecasting
    5. Backtest on 50 real M5 items x 52 weeks: order the mean vs order the quantile
    6. Change the margins: when does the quantile beat the mean, and by how much? (Plotly)

Needs data/processed/m5_ca1_weekly.csv: run .venv/bin/python data/prepare_m5.py first.
Run: .venv/bin/python notebooks/05_newsvendor.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
weekly = pd.read_csv(ROOT / "data" / "processed" / "m5_ca1_weekly.csv")

# Margins, as a share of the selling price (Nike-like: high margin, some markdown value)
COST_SHARE = 0.40     # we pay 40% of the price for each unit
SALVAGE_SHARE = 0.20  # a leftover unit is cleared at 20% of the price
HISTORY_WEEKS = 26    # the "forecast" looks at the last 26 weeks
TEST_WEEKS = 52       # backtest on the last 52 weeks

units = weekly.pivot(index="wm_yr_wk", columns="item_id", values="units")
price = weekly.pivot(index="wm_yr_wk", columns="item_id", values="price")
WEEKS = list(units.index)
ITEMS = list(units.columns)
category = weekly.drop_duplicates("item_id").set_index("item_id")["cat_id"]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def costs_per_unit(p, cost_share=COST_SHARE, salvage_share=SALVAGE_SHARE):
    """Underage cost cu (a lost sale loses the margin) and overage cost co (a leftover loses cost - salvage)."""
    cu = p * (1 - cost_share)
    co = p * (cost_share - salvage_share)
    return cu, co


def newsvendor_cost(order, demand, cu, co):
    """Cost of ordering `order` when `demand` turns up: co per leftover unit, cu per unit short."""
    return co * np.maximum(order - demand, 0) + cu * np.maximum(demand - order, 0)


def pinball_loss(forecast, actual, tau):
    """The quantile (pinball) loss used to train and score quantile forecasts."""
    return np.where(actual >= forecast, tau * (actual - forecast), (1 - tau) * (forecast - actual))


# The "forecast": a stand-in for your LightGBM / TimesFM model. Point forecast = mean of the
# last 26 weeks; quantile forecast = that quantile of the last 26 weeks. Swap in real model output here.
def forecast_mean(history):
    return history.mean()


def forecast_quantile(history, q):
    return history.quantile(q)


def backtest(critical_ratio=None, cost_share=COST_SHARE, salvage_share=SALVAGE_SHARE):
    """Order each of the 50 items every week for the last 52 weeks, two ways. Returns one row per item-week."""
    rows = []
    for t in WEEKS[-TEST_WEEKS:]:
        i = WEEKS.index(t)
        history = units.loc[WEEKS[i - HISTORY_WEEKS:i]]
        demand = units.loc[t]
        cu, co = costs_per_unit(price.loc[t], cost_share, salvage_share)
        if critical_ratio is not None:  # override margins: keep cu + co = price, set the ratio directly
            cu, co = price.loc[t] * critical_ratio, price.loc[t] * (1 - critical_ratio)
        cr = float((cu / (cu + co)).iloc[0])
        for plan, order in [("order the mean", forecast_mean(history)),
                            (f"order the {cr:.0%} quantile", forecast_quantile(history, cr))]:
            order = order.round()
            rows.append(pd.DataFrame({
                "plan": plan, "week": t, "item": ITEMS, "category": category[ITEMS].to_numpy(),
                "order": order.to_numpy(), "demand": demand.to_numpy(),
                "cost": newsvendor_cost(order, demand, cu, co).to_numpy(),
                "sold": np.minimum(order, demand).to_numpy(),
                "leftover": np.maximum(order - demand, 0).to_numpy(),
                "ran out": (demand > order).to_numpy(),
            }))
    return pd.concat(rows, ignore_index=True)


def summarize(results):
    g = results.groupby("plan", sort=False).agg(
        cost=("cost", "sum"), sold=("sold", "sum"), demand=("demand", "sum"),
        leftover_units=("leftover", "sum"), weeks_ran_out=("ran out", "mean"))
    g["fill rate"] = g["sold"] / g["demand"]
    return g[["cost", "fill rate", "weeks_ran_out", "leftover_units"]]


if __name__ == "__main__":
    section("2. One item by hand")
    item = "HOUSEHOLD_1_517"
    last = WEEKS[-TEST_WEEKS - 1]
    history = units.loc[WEEKS[WEEKS.index(last) - HISTORY_WEEKS + 1:WEEKS.index(last) + 1], item]
    p = price.loc[last, item]
    cu, co = costs_per_unit(p)
    print(f"{item}: price ${p:.2f}, cu = ${cu:.2f} per unit short, co = ${co:.2f} per unit left over")
    print("Last 26 weeks of demand:", sorted(history.astype(int).tolist()))
    tries = pd.DataFrame([{"order": q, "avg cost $": newsvendor_cost(q, history, cu, co).mean()}
                          for q in range(int(history.min()), int(history.max()) + 1)])
    best = tries.loc[tries["avg cost $"].idxmin()]
    mean_q = round(history.mean())
    print(f"Best order: {best['order']:.0f} (avg cost ${best['avg cost $']:.2f}). "
          f"Ordering the mean ({mean_q}) averages ${newsvendor_cost(mean_q, history, cu, co).mean():.2f}")

    section("3. The critical ratio")
    cr = cu / (cu + co)
    print(f"critical ratio = cu / (cu + co) = {cu:.2f} / ({cu:.2f} + {co:.2f}) = {cr:.2f}")
    print(f"{cr:.0%} quantile of the 26 weeks: {history.quantile(cr):.1f}  (best order by trying: {best['order']:.0f})")

    section("4. Newsvendor cost = (cu + co) x pinball loss at tau = critical ratio")
    q = history.quantile(cr)
    print(f"avg newsvendor cost ${newsvendor_cost(q, history, cu, co).mean():.4f}  vs  "
          f"(cu + co) x avg pinball ${(cu + co) * pinball_loss(q, history, cr).mean():.4f}")

    section("5. Backtest: 50 items x 52 weeks")
    results = backtest()
    summary = summarize(results)
    print(summary.round(3))
    saving = 1 - summary["cost"].iloc[1] / summary["cost"].iloc[0]
    print(f"\nOrdering the quantile cuts cost by {saving:.1%}")
    by_cat = results.groupby(["category", "plan"], sort=False)["cost"].sum().unstack()
    print((1 - by_cat.iloc[:, 1] / by_cat.iloc[:, 0]).rename("saving by category").round(3))

    section("6. Change the margins: saving vs critical ratio")
    rows = []
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]:
        s = summarize(backtest(critical_ratio=ratio))
        rows.append({"critical ratio": ratio, "saving vs mean": 1 - s["cost"].iloc[1] / s["cost"].iloc[0],
                     "fill rate (quantile)": s["fill rate"].iloc[1], "fill rate (mean)": s["fill rate"].iloc[0]})
    sweep = pd.DataFrame(rows)
    print(sweep.round(3).to_string(index=False))

    fig = go.Figure(go.Scatter(x=sweep["critical ratio"], y=100 * sweep["saving vs mean"], mode="lines+markers",
                               hovertemplate="critical ratio %{x}<br>saving %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(title="Cost saved by ordering the critical-ratio quantile instead of the mean",
                      xaxis_title="critical ratio cu / (cu + co)", yaxis_title="cost saved (%)",
                      height=430, width=900, template="plotly_white")
    OUT.mkdir(exist_ok=True)
    fig.write_html(OUT / "05_cr_sweep.html")
    print(f"\nChart saved to {OUT / '05_cr_sweep.html'}")
