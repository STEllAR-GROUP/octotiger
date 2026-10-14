#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Compare transferred SSP history and RK3 restart under joint mesh/time refinement.

The reference is analytic continuum cell averages, INCLUDING current-state
remap error. This is a standalone scalar experiment, not an Octo-Tiger runtime
or SSP regression test. Default upwind3 has no FE positivity guarantee.
"""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np

# Python puts this script's directory on the import path when run by filename.
from history_remap_accuracy import coefficients, multistep, remap, rk3


CASES = ("none", "refine", "coarsen", "fixed-cadence", "every8", "shifted-every8",
         "refine-halved")
STRATEGIES = ("transfer", "restart")
ERRORS = ("l1", "linf", "max_l1", "max_linf", "event_l1", "event_linf",
          "remap_l1", "remap_linf")
FINAL_TIME = 0.5
FINEST_CFL = 0.2


def continuum(cells, time, origin=0.0):
    """Exact averages of the periodic advected smooth initial profile."""
    x = origin + (np.arange(cells) + 0.5) / cells - time
    return (2 + np.sinc(1 / cells) * np.sin(2 * np.pi * x)
            + 0.2 * np.sinc(3 / cells) * np.cos(6 * np.pi * x))


def new_run(cells, strategy):
    state = continuum(cells, 0)
    return {"state": state, "previous": None, "previous_rhs": None,
            "previous_dt": None, "previous_cells": cells, "strategy": strategy,
            "origin": 0.0, "regrids": 0, "rhs_evaluations": 0,
            "max_l1": 0.0, "max_linf": 0.0, "event_l1": 0.0, "event_linf": 0.0,
            "remap_l1": 0.0, "remap_linf": 0.0, "mass_error": 0.0,
            "remap_mass_error": 0.0, "minimum_state": float(state.min()),
            "maximum_euler_courant": 0.0}


def observe(run, time, event=False, projection=False):
    state = run["state"]
    error = abs(state - continuum(len(state), time, run["origin"]))
    l1, linf = float(error.mean()), float(error.max())
    run["l1"], run["linf"] = l1, linf
    run["max_l1"] = max(run["max_l1"], l1)
    run["max_linf"] = max(run["max_linf"], linf)
    if event:
        run["event_l1"] = max(run["event_l1"], l1)
        run["event_linf"] = max(run["event_linf"], linf)
    if projection:
        run["remap_l1"] = max(run["remap_l1"], l1)
        run["remap_linf"] = max(run["remap_linf"], linf)
    run["mass_error"] = max(run["mass_error"], abs(float(state.mean()) - 2))
    run["minimum_state"] = min(run["minimum_state"], float(state.min()))


def apply_remap(run, event, prolongation, shift, time):
    if event == "coarsen" and shift:
        # Change which fine-cell pairs are restricted. The new coarse grid is
        # translated by one fine cell, so aligned projection cancellation is
        # not needed for convergence. All fields share that same translation.
        run["origin"] = (run["origin"] + 1 / len(run["state"])) % 1
        for field in ("state", "previous", "previous_rhs"):
            if run[field] is not None:
                run[field] = np.roll(run[field], -1)
    mass = float(run["state"].mean())
    run["state"] = remap(run["state"], event, prolongation)
    run["remap_mass_error"] = max(
        run["remap_mass_error"], abs(float(run["state"].mean()) - mass))
    if run["strategy"] == "restart":
        run["previous"] = run["previous_rhs"] = None
    else:
        reference = run["previous"]
        run["previous"] = remap(reference, event, prolongation)
        # Freeze the old state's minmod choices for its old derivative.
        run["previous_rhs"] = remap(run["previous_rhs"], event, prolongation,
                                    limiter_reference=reference)
    run["regrids"] += 1
    observe(run, time, projection=True)


def advance(run, dt, operator):
    state = run["state"]
    if run["previous"] is None:
        result, derivative = rk3(state, dt, operator)
        courant = dt * len(state)
        run["rhs_evaluations"] += 3
    else:
        coeff = coefficients(run["previous_dt"] / dt)
        result, derivative = multistep(
            state, run["previous"], run["previous_rhs"], dt, coeff, operator)
        courant = max(dt * coeff.v * len(state),
                      dt * coeff.v5 * run["previous_cells"])
        run["rhs_evaluations"] += 2
    run["maximum_euler_courant"] = max(run["maximum_euler_courant"], courant)
    run["previous"], run["previous_rhs"] = state, derivative
    run["previous_dt"], run["previous_cells"] = dt, len(state)
    run["state"] = result


def event_at(step, cells, case, current_cells):
    if step == 0:
        return None
    if case in ("refine", "coarsen") and step == 5 * cells // 2:
        return case
    if case == "refine-halved" and step == 5 * cells // 4:
        return "refine"
    if ((case == "fixed-cadence" and step % cells == 0)
            or (case in ("every8", "shifted-every8") and step % 8 == 0)):
        return "refine" if current_cells == cells else "coarsen"
    return None


def solve_pair(cells, operator, prolongation, case):
    initial_cells = 2 * cells if case == "coarsen" else cells
    runs = [new_run(initial_cells, strategy) for strategy in STRATEGIES]
    steps = 15 * cells // 4 if case == "refine-halved" else 5 * cells
    time = maximum_difference = 0.0
    for step in range(steps):
        event = event_at(step, cells, case, len(runs[0]["state"]))
        if event:
            for run in runs:
                apply_remap(run, event, prolongation, case == "shifted-every8", time)
            maximum_difference = max(maximum_difference,
                                     float(np.mean(abs(runs[0]["state"] - runs[1]["state"]))))
        # Keep finest-grid CFL=0.2, with fixed dt through the ordinary events.
        # The separate halved case uses CFL=0.2 on both its coarse and fine
        # intervals, so its first new-mesh step actually has alpha=2.
        coarse_interval = case == "refine-halved" and step < 5 * cells // 4
        dt = FINEST_CFL / (cells if coarse_interval else 2 * cells)
        for run in runs:
            advance(run, dt, operator)
        time += dt
        for run in runs:
            observe(run, time, event=event is not None)
        maximum_difference = max(maximum_difference,
                                 float(np.mean(abs(runs[0]["state"] - runs[1]["state"]))))
    if abs(time - FINAL_TIME) > 1e-11:
        raise RuntimeError("The timestep schedule missed its prescribed final time")
    difference = float(np.mean(abs(runs[0]["state"] - runs[1]["state"])))
    records = []
    for run in runs:
        record = {key: value for key, value in run.items()
                  if key not in ("state", "previous", "previous_rhs",
                                 "previous_dt", "previous_cells")}
        record.update(coarse_cells=cells, final_cells=len(run["state"]), final_time=time,
                      operator=operator, prolongation=prolongation, case=case,
                      steps=steps, transfer_restart_difference_l1=difference,
                      max_transfer_restart_difference_l1=maximum_difference)
        records.append(record)
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("upwind1", "upwind3"), default="upwind3",
                        help="upwind3 is a smooth accuracy control, not an SSP spatial operator")
    parser.add_argument("--prolongations", choices=("linear", "minmod"), nargs="+",
                        default=["linear", "minmod"])
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128],
                        help="increasing coarse-grid counts, multiples of16")
    parser.add_argument("--all-rows", action="store_true", help="print every resolution")
    parser.add_argument("--json", type=Path, help="write all resolutions and diagnostics")
    args = parser.parse_args()
    if (len(args.grids) < 2 or args.grids != sorted(set(args.grids))
            or any(cells < 16 or cells % 16 for cells in args.grids)):
        parser.error("--grids must contain at least two increasing multiples of16")
    if len(set(args.cases)) != len(args.cases) or len(set(args.prolongations)) != len(args.prolongations):
        parser.error("case and prolongation selections must not contain duplicates")
    return args


def main():
    args = parse_args()
    records = []
    for prolongation in args.prolongations:
        for case in args.cases:
            previous = {}
            for cells in args.grids:
                for record in solve_pair(cells, args.operator, prolongation, case):
                    strategy = record["strategy"]
                    old = previous.get(strategy)
                    for error in ERRORS:
                        record[error + "_order"] = None
                        if old is not None and record[error] > 0 and old[error] > 0:
                            record[error + "_order"] = math.log(old[error] / record[error]) / math.log(
                                cells / old["coarse_cells"])
                    previous[strategy] = record
                    records.append(record)
    print(f"operator={args.operator} final_time={FINAL_TIME} finest_CFL={FINEST_CFL}")
    print("Reference: analytic continuum cell averages, including current-state remap error.")
    print("Max-time errors include immediately after remapping and every outgoing step.")
    print(f"{'remap':6} {'case':15} {'history':8} {'N':>5} {'final L1':>12} {'order':>6} "
          f"{'max-time L1':>12} {'order':>6} {'event L1':>12} {'order':>6}")
    def order_text(value):
        return "-" if value is None else f"{value:.3f}"
    for r in records:
        if args.all_rows or r["coarse_cells"] == args.grids[-1]:
            print(f"{r['prolongation']:6} {r['case']:15} {r['strategy']:8} {r['coarse_cells']:5d} "
                  f"{r['l1']:12.5e} {order_text(r['l1_order']):>6} "
                  f"{r['max_l1']:12.5e} {order_text(r['max_l1_order']):>6} "
                  f"{r['event_l1']:12.5e} {order_text(r['event_l1_order']):>6}")
    print(f"Maximum mean drift: {max(r['mass_error'] for r in records):.6e}")
    print(f"Minimum evolved state: {min(r['minimum_state'] for r in records):.6e}")
    print("This is a scalar joint-convergence comparison, not proof of AMR positivity or SSP.")
    if args.json is not None:
        settings = vars(args).copy()
        settings.pop("json")
        args.json.write_text(json.dumps({"settings": settings, "results": records}, indent=2) + "\n")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
