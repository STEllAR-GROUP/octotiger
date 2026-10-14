#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Research prototype for admissible SSP history reconstruction after remapping.

Periodic scalar finite-volume advection/Burgers only; no Octo-Tiger production
code is changed. NumPy and SciPy are required. Every RHS input and every
forward-Euler probe bound is checked. The default suite takes a few seconds.
"""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Statistics:
    rhs_calls: int = 0
    probe_rhs_calls: int = 0
    accepted_rk3_steps: int = 0
    accepted_multistep_steps: int = 0
    regrids: int = 0
    limited_regrids: int = 0
    declined_reconstructions: int = 0
    minimum_theta: float = 1.0
    minimum_rhs_input: float = math.inf
    minimum_history: float = math.inf
    minimum_history_euler: float = math.inf
    minimum_evolved: float = math.inf
    maximum_euler_ratio: float = 0.0
    maximum_probe_euler_ratio: float = 0.0
    maximum_probe_extrapolation: float = 0.0


def require_admissible(state, label):
    if not np.all(np.isfinite(state)) or np.min(state) < 0:
        raise ArithmeticError(f"{label} is not finite and nonnegative: min={np.min(state)}")


def raw_rhs(state, equation):
    flux = state if equation == "advection" else 0.5 * state * state
    return -len(state) * (flux - np.roll(flux, 1))


def fe_limit(state, equation):
    speed = 1.0 if equation == "advection" else float(np.max(state))
    return math.inf if speed == 0 else 1 / (len(state) * speed)


def checked_rhs(state, equation, stats, euler_duration, *, probe=False):
    # Check BEFORE calling the operator. This scalar admissible set is not a
    # replacement for hydrodynamic density/internal-energy/species constraints.
    require_admissible(state, "RHS input")
    ratio = euler_duration / fe_limit(state, equation)
    if not math.isfinite(ratio) or ratio > 1 + 16 * np.finfo(float).eps:
        raise ArithmeticError(f"Forward-Euler bound exceeded: ratio={ratio}")
    stats.minimum_rhs_input = min(stats.minimum_rhs_input, float(np.min(state)))
    stats.maximum_euler_ratio = max(stats.maximum_euler_ratio, ratio)
    stats.rhs_calls += 1
    if probe:
        stats.probe_rhs_calls += 1
        stats.maximum_probe_euler_ratio = max(stats.maximum_probe_euler_ratio, ratio)
    result = raw_rhs(state, equation)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("Nonfinite RHS")
    return result


def rk3(state, dt, equation, stats, *, probe=False):
    first_rhs = checked_rhs(state, equation, stats, dt, probe=probe)
    stage = state + dt * first_rhs
    second_rhs = checked_rhs(stage, equation, stats, dt, probe=probe)
    stage = 0.75 * state + 0.25 * (stage + dt * second_rhs)
    third_rhs = checked_rhs(stage, equation, stats, dt, probe=probe)
    result = (state + 2 * (stage + dt * third_rhs)) / 3
    require_admissible(result, "RK3 endpoint")
    return result, first_rhs


def coefficients(alpha):
    # Equivalent to the production c3=0 family. This experiment uses alpha=1
    # or 2; cancellation-free evaluation for extreme ratios is not needed.
    v = alpha / (math.sqrt(alpha * alpha + 2) + alpha - 2)
    c2 = (v / (alpha + v)) ** 2
    c4 = ((alpha + 1) / alpha * (alpha + v) / (alpha + 2 * v)) ** 2 / (2 * v)
    c5 = 1 - c4
    weight = -(alpha + 1) * (
        2 * alpha * alpha * v - 2 * alpha * alpha
        + 5 * alpha * v * v - 6 * alpha * v - 3 * v * v
    ) / (2 * alpha * alpha * (alpha + 2 * v) ** 2)
    return 1 - c2, c2, c4, c5, v, weight / c5


def remap(state, event, *, constant=False):
    if event == "coarsen":
        return (state[::2] + state[1::2]) / 2
    if constant:
        return np.repeat(state, 2)
    left = state - np.roll(state, 1)
    right = np.roll(state, -1) - state
    slope = np.where(left * right > 0, np.sign(left) * np.minimum(abs(left), abs(right)), 0)
    result = np.empty(2 * len(state))
    result[::2] = state - slope / 4
    result[1::2] = state + slope / 4
    return result


def limited_pair(base_past, base_euler, target_past, target_euler):
    """One GLOBAL theta preserves the mean of each conservative history array."""
    require_admissible(base_past, "Baseline historical state")
    require_admissible(base_euler, "Baseline historical Euler point")
    if not np.all(np.isfinite(target_past)) or not np.all(np.isfinite(target_euler)):
        return base_past, base_euler, 0.0
    theta = 1.0
    for base, target in ((base_past, target_past), (base_euler, target_euler)):
        negative = target < 0
        if np.any(negative):
            theta = min(theta, float(np.min(base[negative] / (base[negative] - target[negative]))))
    theta = max(0.0, min(theta, 1.0))
    if theta < 1:
        theta *= 1 - 64 * np.finfo(float).eps
    past = base_past + theta * (target_past - base_past)
    euler = base_euler + theta * (target_euler - base_euler)
    require_admissible(past, "Limited historical state")
    require_admissible(euler, "Limited historical Euler point")
    return past, euler, theta


def reconstruct_history(current, old_past, old_rhs, event, previous_dt, next_dt, equation, stats):
    _, _, _, _, _, v5 = coefficients(previous_dt / next_dt)
    historical_euler_duration = next_dt * v5
    # Piecewise-constant prolongation and averaging restriction are positive,
    # conservative linear maps. The base pair uses the ACTUAL old trajectory.
    base_past = remap(old_past, event, constant=True)
    base_euler = remap(old_past + historical_euler_duration * old_rhs, event, constant=True)
    require_admissible(base_past, "Remapped baseline past")
    require_admissible(base_euler, "Remapped baseline Euler point")
    require_admissible(current, "Remapped current state")

    # The probe advances safely FORWARD, while only its dense polynomial is
    # extrapolated backward. No RHS is evaluated on synthetic past states.
    delta = min(previous_dt / 2, 0.8 * fe_limit(current, equation))
    if not delta > 0 or previous_dt / delta > 16:
        # A bounded research fallback: retain the admissible transferred pair.
        # This can lose temporal accuracy; it is explicitly counted below.
        stats.declined_reconstructions += 1
        return base_past, base_euler, None
    stats.maximum_probe_extrapolation = max(stats.maximum_probe_extrapolation, previous_dt / delta)
    endpoint, current_rhs = rk3(current, delta, equation, stats, probe=True)
    endpoint_rhs = checked_rhs(endpoint, equation, stats, delta, probe=True)
    x = -previous_dt / delta
    target_past = ((2 * x**3 - 3 * x*x + 1) * current
                   + (x**3 - 2 * x*x + x) * delta * current_rhs
                   + (-2 * x**3 + 3 * x*x) * endpoint
                   + (x**3 - x*x) * delta * endpoint_rhs)
    target_rhs = ((6 * x*x - 6 * x) * (current - endpoint) / delta
                  + (3 * x*x - 4 * x + 1) * current_rhs
                  + (3 * x*x - 2 * x) * endpoint_rhs)
    target_euler = target_past + historical_euler_duration * target_rhs
    past, euler, theta = limited_pair(base_past, base_euler, target_past, target_euler)
    stats.minimum_theta = min(stats.minimum_theta, theta)
    stats.limited_regrids += int(theta < 1)
    return past, euler, current_rhs


def initial(cells, equation, front):
    centers = (np.arange(cells) + 0.5) / cells
    if front:
        return np.where(centers < 0.5, 1.0, 1e-12)
    if equation == "burgers":
        return (1 + 0.2 * np.sinc(1 / cells) * np.sin(2 * np.pi * centers)
                + 0.05 * np.sinc(3 / cells) * np.cos(6 * np.pi * centers))
    return (2 + np.sinc(1 / cells) * np.sin(2 * np.pi * centers)
            + 0.2 * np.sinc(3 / cells) * np.cos(6 * np.pi * centers))


def schedule(steps, interval, event):
    if event == "frequent":
        # Alternate refine/coarsen every eight accepted steps. Increasing the
        # temporal resolution also increases the number of regrids.
        return [(8, interval / steps, "refine" if i % 2 == 0 else "coarsen")
                for i in range(steps // 4)]
    return [(steps, interval / steps, "refine" if event == "half_dt" else event),
            (2 * steps if event == "half_dt" else steps,
             interval / (2 * steps if event == "half_dt" else steps), "none")]


def reference_solution(cells, equation, front, segments):
    state = initial(cells, equation, front)
    for index, (steps, dt, event) in enumerate(segments):
        duration = steps * dt
        if equation == "advection":
            theta = 2 * np.pi * np.fft.fftfreq(len(state))
            symbol = -len(state) * (1 - np.exp(-1j * theta))
            state = np.fft.ifft(np.fft.fft(state) * np.exp(duration * symbol)).real
        else:
            result = solve_ivp(lambda _, u: raw_rhs(u, equation), (0, duration), state,
                               method="DOP853", rtol=3e-14, atol=3e-14)
            if not result.success:
                raise RuntimeError(result.message)
            state = result.y[:, -1]
        if index + 1 < len(segments) and event != "none":
            state = remap(state, event)
    return state


def solve_case(cells, steps, interval, equation, event, front, strategy, reference):
    state = initial(cells, equation, front)
    initial_mean = float(np.mean(state))
    stats = Statistics()
    previous = previous_rhs = previous_dt = pending_euler = cached_current_rhs = None
    maximum_mean_error = maximum_remap_mean_error = 0.0
    segments = schedule(steps, interval, event)
    for index, (segment_steps, dt, change) in enumerate(segments):
        for _ in range(segment_steps):
            before = state
            if previous is None:
                state, first_rhs = rk3(state, dt, equation, stats)
                stats.accepted_rk3_steps += 1
            else:
                c1, c2, c4, c5, v, v5 = coefficients(previous_dt / dt)
                euler = previous + dt * v5 * previous_rhs if pending_euler is None else pending_euler
                require_admissible(previous, "Historical state before multistep")
                require_admissible(euler, "Historical Euler point before multistep")
                stats.minimum_history = min(stats.minimum_history, float(np.min(previous)))
                stats.minimum_history_euler = min(stats.minimum_history_euler, float(np.min(euler)))
                if cached_current_rhs is None:
                    first_rhs = checked_rhs(state, equation, stats, dt * v)
                else:
                    # Reuse the probe's F(current), but still verify the actual
                    # multistep Euler duration, which differs from probe delta.
                    require_admissible(state, "Cached RHS state")
                    ratio = dt * v / fe_limit(state, equation)
                    if ratio > 1 + 16 * np.finfo(float).eps:
                        raise ArithmeticError("Cached current RHS violates the new Euler bound")
                    stats.maximum_euler_ratio = max(stats.maximum_euler_ratio, ratio)
                    first_rhs = cached_current_rhs
                stage = c1 * (state + dt * v * first_rhs) + c2 * previous
                stage_rhs = checked_rhs(stage, equation, stats, dt * v)
                state = c4 * (stage + dt * v * stage_rhs) + c5 * euler
                require_admissible(state, "Multistep endpoint")
                stats.accepted_multistep_steps += 1
            previous, previous_rhs, previous_dt = before, first_rhs, dt
            pending_euler = cached_current_rhs = None
            maximum_mean_error = max(maximum_mean_error, abs(float(np.mean(state)) - initial_mean))
            stats.minimum_evolved = min(stats.minimum_evolved, float(np.min(state)))
        if index + 1 < len(segments) and change != "none":
            stats.regrids += 1
            mean_before = float(np.mean(state))
            state = remap(state, change)
            require_admissible(state, "Current-state spatial remap")
            maximum_remap_mean_error = max(maximum_remap_mean_error, abs(float(np.mean(state)) - mean_before))
            if strategy == "restart":
                previous = previous_rhs = None
            else:
                previous, pending_euler, cached_current_rhs = reconstruct_history(
                    state, previous, previous_rhs, change, previous_dt,
                    segments[index + 1][1], equation, stats)
                previous_rhs = None  # pending_euler is used directly on the next step.
    error = abs(state - reference)
    result = dict(vars(stats))
    result.update(equation=equation, event=event, front=front, strategy=strategy,
                  initial_cells=cells, steps=steps, dt=interval / steps,
                  l1_error=float(np.mean(error)), linf_error=float(np.max(error)),
                  maximum_mean_error=maximum_mean_error,
                  maximum_remap_mean_error=maximum_remap_mean_error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--steps", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--interval", type=float, default=0.125)
    parser.add_argument("--equations", nargs="+", choices=("advection", "burgers"),
                        default=["advection", "burgers"])
    parser.add_argument("--events", nargs="+", choices=("none", "refine", "coarsen", "frequent", "half_dt"),
                        default=["none", "refine", "coarsen", "frequent", "half_dt"])
    parser.add_argument("--skip-front", action="store_true")
    parser.add_argument("--front-steps", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--json", type=Path, help="optional detailed output JSON")
    args = parser.parse_args()
    if args.cells < 8 or args.cells % 2:
        parser.error("--cells must be even and at least 8")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    for values in (args.steps, args.front_steps):
        if any(value < 8 or value % 4 for value in values) or values != sorted(set(values)):
            parser.error("step counts must be increasing distinct multiples of four, at least 8")
    started = time.monotonic()
    records = []
    cases = [(equation, event, False, args.cells, args.steps)
             for equation in args.equations for event in args.events]
    if not args.skip_front:
        cases += [("advection", "frequent", True, 32, args.front_steps)]
    for equation, event, front, cells, step_counts in cases:
        prior = {}
        for steps in step_counts:
            reference = reference_solution(cells, equation, front, schedule(steps, args.interval, event))
            for strategy in ("reconstruct", "restart"):
                result = solve_case(cells, steps, args.interval, equation, event, front, strategy, reference)
                result["observed_order"] = None
                if strategy in prior and result["l1_error"] > 0:
                    previous = prior[strategy]
                    result["observed_order"] = math.log(previous["l1_error"] / result["l1_error"]) / math.log(steps / previous["steps"])
                prior[strategy] = result
                records.append(result)
    print("Scalar research prototype: forward SSP probes + Hermite history + conservative pair limiter")
    print("Errors use the same prescribed current-state remaps in the reference.")
    print(f"{'equation':9} {'event':8} {'profile':7} {'method':11} {'steps':>5} {'L1 error':>12} {'order':>6} {'theta':>7} {'RHS':>7} {'RK3':>5}")
    for result in records:
        order = result["observed_order"]
        order_text = "-" if order is None else f"{order:.3f}"
        print(f"{result['equation']:9} {result['event']:8} {'front' if result['front'] else 'smooth':7} "
              f"{result['strategy']:11} {result['steps']:5d} {result['l1_error']:12.5e} "
              f"{order_text:>6} {result['minimum_theta']:7.4f} {result['rhs_calls']:7d} {result['accepted_rk3_steps']:5d}")
    print(f"Maximum mean error: {max(r['maximum_mean_error'] for r in records):.5e}")
    print(f"Minimum RHS input: {min(r['minimum_rhs_input'] for r in records):.5e}")
    print(f"Minimum historical state/Euler point: {min(r['minimum_history'] for r in records):.5e} / "
          f"{min(r['minimum_history_euler'] for r in records):.5e}")
    print(f"Maximum probe Euler ratio: {max(r['maximum_probe_euler_ratio'] for r in records):.5f}")
    print(f"Elapsed: {time.monotonic() - started:.2f} s; all RHS inputs and Euler bounds checked")
    print("A reconstructed transition normally costs five RHS calls versus three for RK3 restart.")
    print("Only the initial accepted step is RK3 in reconstruction runs; probes still perform RK3 work.")
    print("These scalar checks do not establish general hydrodynamic SSP, energy admissibility, or AMR accuracy.")
    if args.json:
        settings = vars(args).copy()
        settings.pop("json")
        args.json.write_text(json.dumps({"settings": settings, "results": records}, indent=2) + "\n")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
