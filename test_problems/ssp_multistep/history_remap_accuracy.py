#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Measure SSP history-remap accuracy against exact semidiscrete propagation.

This is a standalone experiment, not an Octo-Tiger runtime test. Separate
refinement/coarsening cases avoid cancellation between successive mesh changes.
Default cases use an FE-monotone upwind operator and run on one NumPy thread.
"""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

# Set before importing NumPy, including when inherited environment settings
# would otherwise permit more threads.
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np


@dataclass(frozen=True)
class Coefficients:
    c1: float
    c2: float
    c4: float
    c5: float
    v: float
    v5: float


def coefficients(alpha):
    """The production c3=0 family; this experiment only needs alpha=1 or 2."""
    v = alpha / (math.sqrt(alpha * alpha + 2) + alpha - 2)
    c2 = (v / (alpha + v)) ** 2
    c4 = ((alpha + 1) / alpha * (alpha + v) / (alpha + 2 * v)) ** 2 / (2 * v)
    c5 = 1 - c4
    derivative_weight = -(alpha + 1) * (
        2 * alpha * alpha * v - 2 * alpha * alpha
        + 5 * alpha * v * v - 6 * alpha * v - 3 * v * v
    ) / (2 * alpha * alpha * (alpha + 2 * v) ** 2)
    return Coefficients(1 - c2, c2, c4, c5, v, derivative_weight / c5)


def initial(cells):
    """Exact cell averages of 2 + sin(2*pi*x) + 0.2*cos(6*pi*x)."""
    centers = (np.arange(cells) + 0.5) / cells
    return (2 + np.sinc(1 / cells) * np.sin(2 * np.pi * centers)
            + 0.2 * np.sinc(3 / cells) * np.cos(6 * np.pi * centers))


def rhs(state, spatial_operator):
    if spatial_operator == "upwind1":
        return -len(state) * (state - np.roll(state, 1))
    face = (-np.roll(state, 1) + 5 * state + 2 * np.roll(state, -1)) / 6
    return -len(state) * (face - np.roll(face, 1))


def exact(state, elapsed, spatial_operator):
    """Exact evolution of the selected fixed-mesh linear ODE, up to roundoff."""
    cells = len(state)
    theta = 2 * np.pi * np.fft.fftfreq(cells)
    symbol = -cells * (1 - np.exp(-1j * theta))
    if spatial_operator == "upwind3":
        symbol *= (-np.exp(-1j * theta) + 5 + 2 * np.exp(1j * theta)) / 6
    return np.fft.ifft(np.fft.fft(state) * np.exp(elapsed * symbol)).real


def remap(state, event, prolongation, limiter_reference=None):
    if event == "coarsen":
        return (state[::2] + state[1::2]) / 2
    if prolongation == "linear":
        slope = (np.roll(state, -1) - np.roll(state, 1)) / 2
    else:
        reference = state if limiter_reference is None else limiter_reference
        left_difference = reference - np.roll(reference, 1)
        right_difference = np.roll(reference, -1) - reference
        monotone = left_difference * right_difference > 0
        use_left = monotone & (np.abs(left_difference) <= np.abs(right_difference))
        use_right = monotone & ~use_left
        # Reuse these choices for the historical derivative: the past state
        # and past RHS are transferred by a common frozen linear operator P.
        slope = np.where(use_left, state - np.roll(state, 1),
                         np.where(use_right, np.roll(state, -1) - state, 0))
    result = np.empty(2 * len(state))
    result[::2] = state - slope / 4
    result[1::2] = state + slope / 4
    return result


def rk3(state, dt, spatial_operator):
    derivative = rhs(state, spatial_operator)
    stage = state + dt * derivative
    stage = 0.75 * state + 0.25 * (stage + dt * rhs(stage, spatial_operator))
    result = (state + 2 * (stage + dt * rhs(stage, spatial_operator))) / 3
    return result, derivative


def multistep(state, previous, previous_rhs, dt, coeff, spatial_operator):
    derivative = rhs(state, spatial_operator)
    stage = coeff.c1 * (state + coeff.v * dt * derivative) + coeff.c2 * previous
    result = (coeff.c4 * (stage + coeff.v * dt * rhs(stage, spatial_operator))
              + coeff.c5 * (previous + coeff.v5 * dt * previous_rhs))
    return result, derivative


def solve(args, steps, event, strategy):
    state = initial(args.cells)
    reference = state.copy()
    previous = previous_rhs = previous_dt = previous_cells = None
    maximum_mass_error = maximum_remap_mass_error = maximum_euler_ratio = 0.0
    minimum_state = float(np.min(state))
    # All cases run for two intervals, including the unchanged-mesh control.
    for segment in range(2):
        segment_steps = steps
        if segment == 1 and event == "refine" and args.halve_on_refinement:
            segment_steps *= 2
        dt = args.interval / segment_steps
        reference = exact(reference, args.interval, args.spatial_operator)
        for _ in range(segment_steps):
            old_state = state
            if previous is None:
                state, derivative = rk3(state, dt, args.spatial_operator)
                maximum_euler_ratio = max(maximum_euler_ratio, dt * len(state))
            else:
                coeff = coefficients(previous_dt / dt)
                state, derivative = multistep(
                    state, previous, previous_rhs, dt, coeff, args.spatial_operator)
                maximum_euler_ratio = max(maximum_euler_ratio,
                                          dt * coeff.v * len(state),
                                          dt * coeff.v5 * previous_cells)
            previous, previous_rhs = old_state, derivative
            previous_dt, previous_cells = dt, len(state)
            maximum_mass_error = max(maximum_mass_error, abs(float(np.mean(state)) - 2))
            minimum_state = min(minimum_state, float(np.min(state)))
        if segment == 0 and event != "none":
            before_remap = float(np.mean(state))
            state = remap(state, event, args.prolongation)
            reference = remap(reference, event, args.prolongation)
            maximum_remap_mass_error = max(
                maximum_remap_mass_error, abs(float(np.mean(state)) - before_remap))
            if strategy == "restart":
                previous = previous_rhs = None
            else:
                limiter_reference = previous
                previous = remap(previous, event, args.prolongation)
                previous_rhs = remap(previous_rhs, event, args.prolongation,
                                     limiter_reference=limiter_reference)
    error = np.abs(state - reference)
    return {"event": event, "strategy": strategy, "steps_per_initial_interval": steps,
            "initial_dt": args.interval / steps, "l1_error": float(np.mean(error)),
            "linf_error": float(np.max(error)), "mass_error": maximum_mass_error,
            "remap_mass_error": maximum_remap_mass_error, "minimum_state": minimum_state,
            "maximum_euler_courant": maximum_euler_ratio}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", dest="spatial_operator", choices=("upwind1", "upwind3"),
                        default="upwind1", help="upwind1 is FE monotone for dt/dx <= 1")
    parser.add_argument("--prolongation", choices=("minmod", "linear"), default="minmod")
    parser.add_argument("--cells", type=int, default=32, help="initial cell count")
    parser.add_argument("--steps", type=int, nargs="+", default=[64, 128, 256, 512, 1024],
                        help="successively increasing step counts per initial interval")
    parser.add_argument("--interval", type=float, default=0.125,
                        help="each case ends at twice this time, regridding once at this time")
    parser.add_argument("--events", choices=("none", "refine", "coarsen"), nargs="+",
                        default=["none", "refine", "coarsen"],
                        help="independent cases, each with at most one regrid")
    parser.add_argument("--halve-on-refinement", action="store_true",
                        help="halve dt after refining (alpha=2 at the transition)")
    parser.add_argument("--json", type=Path, help="optional detailed JSON output path")
    args = parser.parse_args()
    if args.cells < 8 or args.cells % 2:
        parser.error("--cells must be an even integer of at least 8")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    if any(step < 2 for step in args.steps) or args.steps != sorted(set(args.steps)):
        parser.error("--steps must be distinct increasing integers of at least 2")
    return args


def main():
    args = parse_args()
    records = []
    for event in args.events:
        for strategy in ("transfer", "restart"):
            last = None
            for steps in args.steps:
                record = solve(args, steps, event, strategy)
                record["observed_order"] = None
                if last is not None and record["l1_error"] > 0 and last["l1_error"] > 0:
                    record["observed_order"] = math.log(last["l1_error"] / record["l1_error"]) / math.log(
                        steps / last["steps_per_initial_interval"])
                records.append(record)
                last = record
    print(f"operator={args.spatial_operator} prolongation={args.prolongation} "
          f"initial_cells={args.cells} regrid_time={args.interval:g} "
          f"halve_on_refinement={args.halve_on_refinement}")
    print("Errors are against exact semidiscrete evolution with the same current-state remap.")
    print(f"{'event':8} {'history':9} {'steps':>6} {'L1 error':>13} {'order':>7}")
    for record in records:
        order = record["observed_order"]
        order_text = "-" if order is None else f"{order:.4f}"
        print(f"{record['event']:8} {record['strategy']:9} "
              f"{record['steps_per_initial_interval']:6d} {record['l1_error']:13.6e} {order_text:>7}")
    print(f"Maximum mean drift: {max(r['mass_error'] for r in records):.6e}")
    print(f"Maximum remap mean drift: {max(r['remap_mass_error'] for r in records):.6e}")
    print(f"Minimum evolved state: {min(r['minimum_state'] for r in records):.6e}")
    if args.spatial_operator == "upwind1":
        ratio = max(r["maximum_euler_courant"] for r in records)
        print(f"Maximum Euler Courant number in time-step decomposition: {ratio:.6f} (bound 1)")
    else:
        print("upwind3 isolates higher-order spatial accuracy; it has no FE positivity guarantee.")
    print("This experiment reports observed order; it is not an Octo-Tiger runtime regression test.")
    if args.json is not None:
        settings = vars(args).copy()
        settings.pop("json")
        args.json.write_text(json.dumps({"settings": settings, "results": records}, indent=2) + "\n")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
