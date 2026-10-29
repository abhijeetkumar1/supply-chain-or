"""Day 2, exercise 3: Which DCs should Nike India open?

Script version of 03_facility_location.ipynb. Same steps, printed to the terminal:
    1. The story and data: 5 candidate DC sites, 20 cities
    2. Why this isn't an LP: open/closed is a yes/no choice
    3. Solve by hand: try every combination of DCs (31 of them)
    4. The MIP: binary "open" variables and the linking (big-M) constraint
    5. LP relaxation: let "open" be a fraction, get a lower bound
    6. Branch and bound, by hand, on this problem
    7. A tighter formulation: same answer, far less searching
    8. The same model in OR-Tools (SCIP)
    9. Scaling: solve time and MIP gap as the problem grows (Plotly)

Run: .venv/bin/python notebooks/03_facility_location.py
"""

import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pulp
from ortools.linear_solver import pywraplp
from plotly.subplots import make_subplots

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

OUT = Path(__file__).resolve().parent.parent / "outputs"

# 1. The story ----------------------------------------------------------------
# Nike India wants a DC network. There are 5 candidate sites. An open DC costs a
# fixed amount every week (rent, staff) whether it ships 1 pair or 30,000, and it
# can only handle so many pairs. Shipping a pair costs Rs 5 handling plus
# Rs 0.05 per road-km. Which sites should open, and which DC serves each city?

cities = {  # name: (latitude, longitude, pairs needed per week)
    "Delhi": (28.61, 77.21, 9000), "Mumbai": (19.08, 72.88, 9000), "Bengaluru": (12.97, 77.59, 7000),
    "Kolkata": (22.57, 88.36, 4500), "Chennai": (13.08, 80.27, 5000), "Hyderabad": (17.39, 78.49, 5000),
    "Ahmedabad": (23.02, 72.57, 3500), "Pune": (18.52, 73.86, 4000), "Jaipur": (26.91, 75.79, 2000),
    "Lucknow": (26.85, 80.95, 2000), "Chandigarh": (30.73, 76.78, 1500), "Kochi": (9.93, 76.27, 1500),
    "Bhubaneswar": (20.30, 85.82, 1200), "Guwahati": (26.14, 91.74, 1000), "Indore": (22.72, 75.86, 1500),
    "Nagpur": (21.15, 79.09, 1200), "Patna": (25.59, 85.14, 1300), "Surat": (21.17, 72.83, 2000),
    "Coimbatore": (11.02, 76.96, 1200), "Visakhapatnam": (17.69, 83.22, 1100),
}
CITIES = list(cities)
demand = {c: cities[c][2] for c in CITIES}

SITES = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Nagpur"]
fixed_cost = {"Delhi": 600_000, "Mumbai": 700_000, "Bengaluru": 550_000,
              "Kolkata": 400_000, "Nagpur": 300_000}                     # Rs per week if open
capacity = {"Delhi": 30_000, "Mumbai": 30_000, "Bengaluru": 25_000,
            "Kolkata": 20_000, "Nagpur": 35_000}                         # pairs per week


def road_km(a, b):
    """Straight-line (great-circle) distance x 1.25, a common rule of thumb for road distance."""
    lat1, lon1, _ = cities[a]
    lat2, lon2, _ = cities[b]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 1.25 * 2 * 6371 * math.asin(math.sqrt(h))


ship_cost = pd.DataFrame({c: {s: 5 + 0.05 * road_km(s, c) for s in SITES} for c in CITIES})  # Rs per pair


def lakh(rupees):
    return f"Rs {rupees / 1e5:,.1f} lakh"


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def shipping_only(open_sites):
    """Day 1's transportation problem: cheapest shipping from a fixed set of open DCs."""
    if sum(capacity[s] for s in open_sites) < sum(demand.values()):
        return None  # not enough capacity to serve everyone
    model = pulp.LpProblem("shipping", pulp.LpMinimize)
    ship = {(s, c): model.add_variable(f"ship_{s}_{c}", lowBound=0) for s in open_sites for c in CITIES}
    model += pulp.lpSum(ship_cost.loc[s, c] * ship[s, c] for s, c in ship)
    for c in CITIES:
        model += pulp.lpSum(ship[s, c] for s in open_sites) >= demand[c]
    for s in open_sites:
        model += pulp.lpSum(ship[s, c] for c in CITIES) <= capacity[s]
    model.solve(pulp.HiGHS(msg=False))
    return pulp.value(model.objective)


def solve_mip(relax=False, tight=False, fixed=None):
    """Facility location. relax: 'open' may be fractional. tight: add ship <= demand x open.
    fixed: {site: 0 or 1} to force some sites closed/open (used by branch and bound)."""
    model = pulp.LpProblem("facility_location", pulp.LpMinimize)

    # Decision variables
    is_open = {s: model.add_variable(f"open_{s}", lowBound=0, upBound=1,
                                     cat="Continuous" if relax else "Binary") for s in SITES}
    ship = {(s, c): model.add_variable(f"ship_{s}_{c}", lowBound=0) for s in SITES for c in CITIES}

    # Objective: fixed cost of open DCs + shipping cost
    model += (pulp.lpSum(fixed_cost[s] * is_open[s] for s in SITES)
              + pulp.lpSum(ship_cost.loc[s, c] * ship[s, c] for s in SITES for c in CITIES))

    # Every city gets what it needs
    for c in CITIES:
        model += pulp.lpSum(ship[s, c] for s in SITES) >= demand[c], f"demand_{c}"

    # Linking constraint: a closed DC ships nothing; an open one ships up to its capacity
    for s in SITES:
        model += pulp.lpSum(ship[s, c] for c in CITIES) <= capacity[s] * is_open[s], f"capacity_{s}"

    # Tighter version: a DC can't send a city more than that city needs, and nothing if closed
    if tight:
        for s in SITES:
            for c in CITIES:
                model += ship[s, c] <= demand[c] * is_open[s]

    for s, v in (fixed or {}).items():
        model += is_open[s] == v

    stats = model.solve(pulp.HiGHS(msg=False))
    return stats, model, is_open, ship


def branch_and_bound(tight=False):
    """Plain depth-first branch and bound on the 'open' variables, using LP relaxations."""
    best = {"cost": math.inf, "open": None}
    log = []

    def visit(fixed, depth):
        stats, model, is_open, _ = solve_mip(relax=True, tight=tight, fixed=fixed)
        name = ", ".join(f"{s}={v}" for s, v in fixed.items()) or "(start: nothing fixed)"
        if stats.status != pulp.LpSolveStatus.Optimal:
            log.append({"depth": depth, "fixed so far": name, "LP bound": None, "what happened": "no feasible plan, stop"})
            return
        bound = pulp.value(model.objective)
        values = {s: is_open[s].varValue for s in SITES}
        if bound >= best["cost"] - 1e-6:
            log.append({"depth": depth, "fixed so far": name, "LP bound": bound,
                        "what happened": f"bound >= best so far ({lakh(best['cost'])}), prune"})
            return
        fractional = [s for s in SITES if 1e-6 < values[s] < 1 - 1e-6]
        if not fractional:
            best.update(cost=bound, open=[s for s in SITES if values[s] > 0.5])
            log.append({"depth": depth, "fixed so far": name, "LP bound": bound,
                        "what happened": "all whole numbers: new best plan"})
            return
        pick = max(fractional, key=lambda s: min(values[s], 1 - values[s]))
        log.append({"depth": depth, "fixed so far": name, "LP bound": bound,
                    "what happened": f"{pick} is {values[pick]:.2f} open, branch on it"})
        for v in (1, 0):
            visit({**fixed, pick: v}, depth + 1)

    visit({}, 0)
    return best, pd.DataFrame(log)


def solve_ortools(n_sites=None, n_cities=None, seed=0, time_limit_s=None):
    """Facility location in OR-Tools with SCIP. With n_sites/n_cities, builds a random instance."""
    if n_sites is None:
        sites, places, dem, cap, fix = SITES, CITIES, demand, capacity, fixed_cost
        cost = {(s, c): ship_cost.loc[s, c] for s in SITES for c in CITIES}
    else:
        rng = np.random.default_rng(seed)
        site_xy, city_xy = rng.uniform(0, 100, (n_sites, 2)), rng.uniform(0, 100, (n_cities, 2))
        sites, places = range(n_sites), range(n_cities)
        dem = dict(enumerate(rng.integers(500, 3000, n_cities).tolist()))
        avg = sum(dem.values()) / n_sites
        cap = dict(enumerate(rng.integers(int(2 * avg), int(4 * avg), n_sites).tolist()))
        fix = dict(enumerate(rng.integers(200_000, 600_000, n_sites).tolist()))
        dist = np.linalg.norm(site_xy[:, None, :] - city_xy[None, :, :], axis=2)
        cost = {(s, c): 5 + 0.5 * float(dist[s, c]) for s in sites for c in places}

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if time_limit_s:
        solver.SetTimeLimit(int(time_limit_s * 1000))

    is_open = {s: solver.BoolVar(f"open_{s}") for s in sites}
    ship = {(s, c): solver.NumVar(0, solver.infinity(), f"ship_{s}_{c}") for s in sites for c in places}

    solver.Minimize(sum(fix[s] * is_open[s] for s in sites) + sum(cost[k] * ship[k] for k in ship))
    for c in places:
        solver.Add(sum(ship[s, c] for s in sites) >= dem[c])
    for s in sites:
        solver.Add(sum(ship[s, c] for c in places) <= cap[s] * is_open[s])

    start = time.time()
    status = solver.Solve()
    seconds = time.time() - start
    best, bound = solver.Objective().Value(), solver.Objective().BestBound()
    return {
        "status": "optimal" if status == pywraplp.Solver.OPTIMAL else "stopped at time limit",
        "cost": best, "bound": bound, "gap %": 100 * (best - bound) / best, "seconds": seconds,
        "open": [s for s in sites if is_open[s].solution_value() > 0.5],
    }


if __name__ == "__main__":
    section("1. The data")
    sites_table = pd.DataFrame({"fixed cost per week": {s: lakh(v) for s, v in fixed_cost.items()},
                                "capacity (pairs/week)": capacity})
    print(sites_table)
    print(f"\nTotal demand: {sum(demand.values()):,} pairs/week across {len(CITIES)} cities")
    print("\nShipping cost, Rs per pair (rows = DC site, columns = city):")
    print(ship_cost.round(0).astype(int))

    section("3. Solve by hand: try every combination of open DCs")
    rows = []
    for k in range(1, len(SITES) + 1):
        for combo in itertools.combinations(SITES, k):
            shipping = shipping_only(combo)
            fixed = sum(fixed_cost[s] for s in combo)
            rows.append({"open DCs": " + ".join(combo), "fixed": fixed, "shipping": shipping,
                         "total": None if shipping is None else fixed + shipping})
    combos = pd.DataFrame(rows)
    feasible = combos.dropna().sort_values("total").reset_index(drop=True)
    print(f"{len(combos)} combinations, {len(feasible)} have enough capacity. Best 6:")
    print(feasible.head(6).assign(**{k: feasible.head(6)[k].map(lakh) for k in ["fixed", "shipping", "total"]}))

    section("4. The MIP (PuLP + HiGHS)")
    stats, model, is_open, ship = solve_mip()
    mip_cost = pulp.value(model.objective)
    mip_open = [s for s in SITES if is_open[s].varValue > 0.5]
    print(f"Open: {mip_open}   cost {lakh(mip_cost)}   (brute force best: {feasible.loc[0, 'open DCs']})")
    served = {c: max(SITES, key=lambda s: ship[s, c].varValue) for c in CITIES}
    print("Each city is served by:", served)

    section("5. LP relaxation: 'open' may be a fraction")
    _, model_lp, open_lp, ship_lp = solve_mip(relax=True)
    lp_bound = pulp.value(model_lp.objective)
    print(pd.DataFrame({
        "open (fraction)": {s: round(open_lp[s].varValue, 3) for s in SITES},
        "pairs shipped": {s: round(sum(ship_lp[s, c].varValue for c in CITIES)) for s in SITES},
        "capacity": capacity,
    }))
    print(f"\nLP relaxation cost {lakh(lp_bound)} vs real best {lakh(mip_cost)}: "
          f"gap {100 * (mip_cost - lp_bound) / mip_cost:.1f}%")

    section("6. Branch and bound, by hand")
    best, log = branch_and_bound()
    log["LP bound"] = log["LP bound"].map(lambda v: "" if pd.isna(v) else lakh(v))
    print(log.to_string(index=False))
    print(f"\n{len(log)} LP relaxations solved. Best: {best['open']}  {lakh(best['cost'])}")

    section("7. A tighter formulation")
    _, model_tight_lp, open_tight, _ = solve_mip(relax=True, tight=True)
    print("LP relaxation 'open' values:", {s: round(open_tight[s].varValue, 3) for s in SITES})
    print(f"LP relaxation cost {lakh(pulp.value(model_tight_lp.objective))} (real best {lakh(mip_cost)})")
    best_t, log_t = branch_and_bound(tight=True)
    print(f"Branch and bound with the tighter model: {len(log_t)} LP relaxation(s) instead of {len(log)}")

    section("8. Same model in OR-Tools (SCIP)")
    ort = solve_ortools()
    print(f"OR-Tools/SCIP: open {ort['open']}, cost {lakh(ort['cost'])}, {ort['seconds']:.2f}s")

    section("9. Scaling: solve time and MIP gap as the problem grows (20 s time limit)")
    sizes = [(10, 50), (20, 100), (30, 150), (40, 200), (60, 300), (80, 400)]
    runs = []
    for n_sites, n_cities in sizes:
        r = solve_ortools(n_sites, n_cities, time_limit_s=20)
        runs.append({"sites": n_sites, "cities": n_cities, "yes/no choices": n_sites,
                     "seconds": round(r["seconds"], 2), "gap %": round(r["gap %"], 2), "status": r["status"]})
        print(runs[-1])
    runs = pd.DataFrame(runs)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Solve time", "MIP gap when the solver stopped"))
    labels = [f"{s} sites x {c} cities" for s, c in sizes]
    fig.add_trace(go.Bar(x=labels, y=runs["seconds"], hovertemplate="%{x}<br>%{y:.1f} s<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=runs["gap %"], hovertemplate="%{x}<br>gap %{y:.2f}%<extra></extra>"), row=1, col=2)
    fig.add_hline(y=20, line_dash="dot", annotation_text="time limit", row=1, col=1)
    fig.update_yaxes(title_text="seconds", row=1, col=1)
    fig.update_yaxes(title_text="gap (%)", row=1, col=2)
    fig.update_layout(height=430, width=1000, showlegend=False, template="plotly_white")
    OUT.mkdir(exist_ok=True)
    fig.write_html(OUT / "03_scaling.html")
    print(f"\nChart saved to {OUT / '03_scaling.html'}")
