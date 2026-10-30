"""Day 2, exercise 3b: the same DC problem in four different tools.

Script version of notebooks/03b_facility_location_solvers.ipynb. Same steps, printed to the terminal:
    1. Modeling layer vs solver: what PuLP actually is
    2. PuLP + HiGHS (the version from exercise 3, written out again for comparison)
    3. Pyomo + HiGHS: the same model, built from sets and rules
    4. OR-Tools CP-SAT: a different kind of solver that only understands whole numbers
    5. SciPy milp: no modeling layer, you build the matrix yourself (Plotly picture of it)
    6. Side by side: same answer, different code

Run: .venv/bin/python scripts/03b_facility_location_solvers.py
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pulp
import pyomo.environ as pyo
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp

pd.set_option("display.width", 250)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# Reuse exercise 3's data, so every tool below solves exactly the same problem.
_spec = importlib.util.spec_from_file_location("fl", ROOT / "scripts" / "03_facility_location.py")
fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fl)
SITES, CITIES = fl.SITES, fl.CITIES
demand, capacity, fixed_cost, ship_cost, lakh = fl.demand, fl.capacity, fl.fixed_cost, fl.ship_cost, fl.lakh


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# 2. PuLP + HiGHS --------------------------------------------------------------
def solve_pulp():
    m = pulp.LpProblem("facility_location", pulp.LpMinimize)
    is_open = {s: m.add_variable(f"open_{s}", cat="Binary") for s in SITES}
    ship = {(s, c): m.add_variable(f"ship_{s}_{c}", lowBound=0) for s in SITES for c in CITIES}

    m += (pulp.lpSum(fixed_cost[s] * is_open[s] for s in SITES)
          + pulp.lpSum(ship_cost.loc[s, c] * ship[s, c] for s in SITES for c in CITIES))
    for c in CITIES:
        m += pulp.lpSum(ship[s, c] for s in SITES) >= demand[c]
    for s in SITES:
        m += pulp.lpSum(ship[s, c] for c in CITIES) <= capacity[s] * is_open[s]

    start = time.time()
    m.solve(pulp.HiGHS(msg=False))
    return {"cost": pulp.value(m.objective), "open": [s for s in SITES if is_open[s].varValue > 0.5],
            "seconds": time.time() - start}


# 3. Pyomo + HiGHS -------------------------------------------------------------
def solve_pyomo(solver_name="highs"):
    m = pyo.ConcreteModel()
    m.SITES = pyo.Set(initialize=SITES)
    m.CITIES = pyo.Set(initialize=CITIES)

    m.is_open = pyo.Var(m.SITES, domain=pyo.Binary)
    m.ship = pyo.Var(m.SITES, m.CITIES, domain=pyo.NonNegativeReals)

    m.cost = pyo.Objective(
        expr=sum(fixed_cost[s] * m.is_open[s] for s in m.SITES)
        + sum(ship_cost.loc[s, c] * m.ship[s, c] for s in m.SITES for c in m.CITIES),
        sense=pyo.minimize,
    )

    # A "rule" is a function Pyomo calls once per index to build one constraint.
    def meet_demand(m, c):
        return sum(m.ship[s, c] for s in m.SITES) >= demand[c]

    def link_capacity(m, s):
        return sum(m.ship[s, c] for c in m.CITIES) <= capacity[s] * m.is_open[s]

    m.meet_demand = pyo.Constraint(m.CITIES, rule=meet_demand)
    m.link_capacity = pyo.Constraint(m.SITES, rule=link_capacity)

    start = time.time()
    pyo.SolverFactory(solver_name).solve(m)  # swap "highs" for "gurobi" or "cplex" if licensed
    return {"cost": pyo.value(m.cost), "open": [s for s in SITES if pyo.value(m.is_open[s]) > 0.5],
            "seconds": time.time() - start}


# 4. OR-Tools CP-SAT -----------------------------------------------------------
def solve_cpsat():
    """CP-SAT only takes whole numbers: costs in paise, shipments in whole pairs."""
    paise = {(s, c): round(100 * ship_cost.loc[s, c]) for s in SITES for c in CITIES}

    m = cp_model.CpModel()
    is_open = {s: m.new_bool_var(f"open_{s}") for s in SITES}
    ship = {(s, c): m.new_int_var(0, demand[c], f"ship_{s}_{c}") for s in SITES for c in CITIES}

    m.minimize(sum(100 * fixed_cost[s] * is_open[s] for s in SITES)
               + sum(paise[k] * ship[k] for k in ship))
    for c in CITIES:
        m.add(sum(ship[s, c] for s in SITES) >= demand[c])
    for s in SITES:
        m.add(sum(ship[s, c] for c in CITIES) <= capacity[s] * is_open[s])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_workers = 8
    start = time.time()
    status = solver.solve(m)
    seconds = time.time() - start
    # Re-price the plan in exact rupees so it is comparable with the others
    cost = (sum(fixed_cost[s] * solver.value(is_open[s]) for s in SITES)
            + sum(ship_cost.loc[k] * solver.value(ship[k]) for k in ship))
    return {"cost": float(cost), "open": [s for s in SITES if solver.value(is_open[s])], "seconds": seconds,
            "status": solver.status_name(status)}


# 5. SciPy milp ----------------------------------------------------------------
def build_matrix():
    """Write the model as numbers only: minimise c.x subject to lo <= A x <= hi.

    Column order: [open_Delhi, ..., open_Nagpur, ship_Delhi_Delhi, ship_Delhi_Mumbai, ...]
    Row order:    20 demand rows (one per city), then 5 capacity rows (one per site)."""
    n_s, n_c = len(SITES), len(CITIES)
    col_open = {s: i for i, s in enumerate(SITES)}
    col_ship = {(s, c): n_s + i * n_c + j for i, s in enumerate(SITES) for j, c in enumerate(CITIES)}
    n_vars = n_s + n_s * n_c

    c_vec = np.zeros(n_vars)
    for s in SITES:
        c_vec[col_open[s]] = fixed_cost[s]
    for (s, c), j in col_ship.items():
        c_vec[j] = ship_cost.loc[s, c]

    A = np.zeros((n_c + n_s, n_vars))
    lo, hi = np.zeros(n_c + n_s), np.zeros(n_c + n_s)
    for r, c in enumerate(CITIES):            # sum_s ship[s,c] >= demand[c]
        for s in SITES:
            A[r, col_ship[s, c]] = 1
        lo[r], hi[r] = demand[c], np.inf
    for k, s in enumerate(SITES):             # sum_c ship[s,c] - capacity[s] * open[s] <= 0
        r = n_c + k
        for c in CITIES:
            A[r, col_ship[s, c]] = 1
        A[r, col_open[s]] = -capacity[s]
        lo[r], hi[r] = -np.inf, 0

    integrality = np.zeros(n_vars)
    integrality[:n_s] = 1                                   # the "open" columns are whole numbers...
    upper = np.full(n_vars, np.inf)
    upper[:n_s] = 1                                         # ...between 0 and 1, so yes/no
    return c_vec, A, lo, hi, integrality, Bounds(0, upper), col_open, col_ship


def solve_scipy():
    c_vec, A, lo, hi, integrality, bounds, col_open, _ = build_matrix()
    start = time.time()
    res = milp(c_vec, constraints=LinearConstraint(A, lo, hi), integrality=integrality, bounds=bounds)
    return {"cost": res.fun, "open": [s for s in SITES if res.x[col_open[s]] > 0.5],
            "seconds": time.time() - start}


def matrix_figure():
    _, A, *_ = build_matrix()
    cols = [f"open {s}" for s in SITES] + [f"ship {s}→{c}" for s in SITES for c in CITIES]
    rows = [f"demand {c}" for c in CITIES] + [f"capacity {s}" for s in SITES]
    shown = np.where(A == 0, np.nan, np.sign(A))           # blank = 0, blue = +1, red = -capacity
    fig = go.Figure(go.Heatmap(
        z=shown, x=cols, y=rows, customdata=A, colorscale=[[0, "#c0392b"], [1, "#2e6fba"]],
        zmin=-1, zmax=1, showscale=False, xgap=1, ygap=1,
        hovertemplate="row: %{y}<br>column: %{x}<br>coefficient: %{customdata:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(showticklabels=False, title_text=f"{A.shape[1]} columns (5 open + 100 ship variables)")
    fig.update_yaxes(autorange="reversed", tickfont_size=9)
    fig.update_layout(title=f"The whole model as a {A.shape[0]} x {A.shape[1]} matrix "
                            f"({np.count_nonzero(A)} non-zero cells out of {A.size})",
                      height=520, width=1000, template="plotly_white")
    return fig


if __name__ == "__main__":
    section("1. Modeling layer vs solver")
    print("Modeling layer = how you write the model in Python (PuLP, Pyomo, OR-Tools, or raw SciPy matrices).")
    print("Solver         = the engine that finds the answer (HiGHS, CBC, SCIP, CP-SAT, Gurobi, CPLEX).")
    print("PuLP 4.2 in this repo uses HiGHS by default. Available here:", pulp.listSolvers(onlyAvailable=True))

    results = {}
    section("2. PuLP + HiGHS")
    results["PuLP + HiGHS"] = solve_pulp()
    print(results["PuLP + HiGHS"])

    section("3. Pyomo + HiGHS")
    results["Pyomo + HiGHS"] = solve_pyomo()
    print(results["Pyomo + HiGHS"])

    section("4. OR-Tools CP-SAT (whole numbers only)")
    results["OR-Tools CP-SAT"] = solve_cpsat()
    print(results["OR-Tools CP-SAT"])

    section("5. SciPy milp (you build the matrix)")
    results["SciPy milp (HiGHS)"] = solve_scipy()
    print(results["SciPy milp (HiGHS)"])
    fig = matrix_figure()
    OUT.mkdir(exist_ok=True)
    fig.write_html(OUT / "03b_matrix.html")
    print(f"Matrix picture saved to {OUT / '03b_matrix.html'}")

    section("6. Side by side")
    table = pd.DataFrame({
        name: {"open DCs": " + ".join(r["open"]), "cost": lakh(r["cost"]), "seconds": round(r["seconds"], 3)}
        for name, r in results.items()
    }).T
    print(table)
