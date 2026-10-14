#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Compare SSP multistep and ordinary RK3 on an exactly advected density pulse.

Default: density-based AMR, levels 1:3, CFL 0.4, t=0.8, no gravity/contact
steepening/hard timestep cap. Use --mesh uniform for the fixed-grid control.
Runs sequentially with one HPX worker and records accuracy and actual RHS work.
"""

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import tempfile

from amr_accuracy import MeshReader, analyze_run, decode_node, np, require, run_binary


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
STEP = re.compile(rf"^(\d+)[ \t]+({NUMBER})[ \t]+({NUMBER})[ \t]+({NUMBER})[ \t]+({NUMBER})[ \t]*$", re.M)


def printed_roundoff(token):
    """Half of the last printed decimal place, plus float conversion roundoff."""
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    quantum = 10.0 ** (int(exponent or "0") - decimals)
    bound = 0.5 * quantum + math.ulp(float(token))
    require(math.isfinite(bound), "Unsupported printed-number precision")
    return bound


def step_statistics(log, count, final_time):
    records = sorted((int(n), t, dt, elapsed)
                     for n, t, dt, elapsed, _ in STEP.findall(log))
    rows = [(n, float(t), float(dt), float(elapsed)) for n, t, dt, elapsed in records]
    require([row[0] for row in rows] == list(range(1, count + 1)),
            "Accepted-step log is incomplete or duplicated")
    require(count > 0 and math.isfinite(final_time) and final_time > 0, "Invalid final step metadata")
    previous = previous_error = cumulative_dt = cumulative_error = 0.0
    dt_errors, clock_errors = [], []
    for (_, time, dt, elapsed), (_, time_token, dt_token, _) in zip(rows, records):
        require(all(math.isfinite(value) for value in (time, dt, elapsed)), "Nonfinite step record")
        # Adjacent positive timesteps can round to the same printed time.
        require(dt > 0 and elapsed >= 0 and time >= previous, "Invalid accepted-step record")
        time_error, dt_error = printed_roundoff(time_token), printed_roundoff(dt_token)
        # One floating-point time += dt operation occurs in the solver. Use
        # the upper printed-time endpoint to cover a possible binary exponent
        # transition as well as subtraction/conversion roundoff in this check.
        clock_error = math.ulp(abs(time) + time_error)
        increment = time - previous
        increment_error = (time_error + previous_error + dt_error + clock_error
                           + math.ulp(increment) + math.ulp(increment - dt))
        require(abs(increment - dt) <= increment_error,
                "Logged step duration disagrees with successive times")
        cumulative_dt += dt
        cumulative_error += dt_error + clock_error + math.ulp(cumulative_dt)
        require(abs(cumulative_dt - time) <= cumulative_error + time_error
                + math.ulp(cumulative_dt - time),
                "Logged cumulative timesteps disagree with the printed time")
        dt_errors.append(dt_error)
        clock_errors.append(clock_error)
        previous, previous_error = time, time_error
    # Silo carries full-precision time. Allow two ulps for its optional
    # conversion to physical units and back, not a relative tolerance tied to
    # the duration of the run. fsum avoids accumulating analysis-side error.
    final_error = 2 * math.ulp(final_time)
    require(abs(previous - final_time) <= previous_error + final_error
            + math.ulp(previous - final_time), "Final step time mismatch")
    total_dt = math.fsum(row[2] for row in rows)
    sum_error = math.fsum(dt_errors) + math.fsum(clock_errors) + math.ulp(total_dt) + final_error
    require(abs(total_dt - final_time) <= sum_error + math.ulp(total_dt - final_time),
            "Logged timestep sum does not cover the simulated interval")
    return {"accepted_steps": count, "mean_dt": final_time / count,
            "median_dt": statistics.median(row[2] for row in rows),
            "minimum_dt": min(row[2] for row in rows), "maximum_dt": max(row[2] for row in rows),
            "summed_step_elapsed_seconds": math.fsum(row[3] for row in rows),
            "log_precision": "seven significant digits; bounds inferred from each printed decimal place",
            "logged_timestep_sum": total_dt, "logged_timestep_sum_absolute_tolerance": sum_error,
            "rows": rows}


def command_options(path):
    command = json.loads(path.read_text())
    return {item.split("=", 1)[0]: item.split("=", 1)[1] for item in command[1:] if "=" in item}


def check_command(args, directory, integrator):
    options = command_options(directory / "command.json")
    expected = {"--problem": "AMR_ADVECTION", "--time_integrator": integrator,
                "--unigrid": "off" if args.mesh == "amr" else "on",
                "--min_level": str(args.minimum_level), "--max_level": str(args.level),
                "--gravity": "off", "--cdisc_detect": "off", "--hpx:threads": "1",
                "--cfl": str(args.cfl), "--stop_time": str(args.stop_time),
                "--odt": str(args.output_dt), "--refinement_floor": str(args.refinement_floor)}
    for key, value in expected.items():
        require(options.get(key) == value, f"Unexpected saved option {key}: {options.get(key)}")
    require(float(options.get("--hard_dt", -1)) <= 0, "Benchmark imposed a hard timestep cap")
    return json.loads((directory / "command.json").read_text())[1:]


def check_initial_state(result):
    initial = next((row for row in result["snapshot_errors"] if row["step"] == 0), None)
    require(initial is not None and initial["time"] == 0, "Missing initial snapshot")
    require(initial["density_l1"] < 1e-13 and initial["density_linf"] < 1e-12,
            "Initial pulse differs from its exact cell-average reference")
    return initial


def check_uniform_mesh(result, level):
    initial = check_initial_state(result)
    leaves = result["final"]["leaf_ids"]
    require(len(leaves) == 8**level and all(decode_node(node)[0] == level for node in leaves),
            "Expected a complete uniform mesh at the requested level")
    require(not result["topology_changes"], "Uniform benchmark changed topology")
    require(all(row["leaf_ids"] == leaves for row in result["snapshots"]), "Saved leaf inventories changed")
    require(all(row["node_count"] == sum(8**depth for depth in range(level + 1))
                for row in result["snapshots"]), "Incomplete uniform tree inventory")
    require(all(row["cells"] == initial["cells"] for row in result["snapshot_errors"]),
            "Uniform benchmark changed the number of cells")
    return initial


def check_amr_mesh(result, minimum_level, maximum_level):
    initial = check_initial_state(result)
    histograms = []
    for snapshot in result["snapshots"]:
        levels = [decode_node(node)[0] for node in snapshot["leaf_ids"]]
        require(all(minimum_level <= level <= maximum_level for level in levels),
                "Leaf outside the requested AMR level range")
        require(max(levels) == maximum_level, "Pulse did not retain the requested finest level")
        require(min(levels) < maximum_level, "AMR benchmark became a uniform finest grid")
        histograms.append({"step": snapshot["step"], "time": snapshot["time"],
                           "leaf_blocks_by_level": {str(level): levels.count(level) for level in sorted(set(levels))}})
    result["leaf_level_histograms"] = histograms
    return initial


def check_density_refinement(reader, result, maximum_level, threshold):
    """Verify density tagging and 2:1 nesting in each saved post-regrid mesh."""
    checks = []
    for snapshot in result["snapshots"]:
        levels, lows, highs = [], [], []
        dense_cells = coarse_dense_cells = 0
        coarse_maximum = None
        for node, _, fields in reader.meshes(snapshot["path"], field_names=("rho_1",)):
            level, xyz = decode_node(node)
            scale = 1 << (maximum_level - level)
            levels.append(level)
            lows.append([value * scale for value in xyz])
            highs.append([(value + 1) * scale for value in xyz])
            density = fields["rho_1"]
            dense = int(np.count_nonzero(density > threshold))
            dense_cells += dense
            if level < maximum_level:
                coarse_dense_cells += dense
                value = float(np.max(density))
                coarse_maximum = value if coarse_maximum is None else max(coarse_maximum, value)
        require(dense_cells > 0, "No density-tagged pulse cells remain")
        require(coarse_dense_cells == 0, "Density-tagged pulse cells are below the maximum level")
        lo, hi, lev = np.array(lows), np.array(highs), np.array(levels)
        touching = np.all((lo[:, None, :] <= hi[None, :, :]) &
                          (lo[None, :, :] <= hi[:, None, :]), axis=2)
        differences = abs(lev[:, None] - lev[None, :])
        maximum_jump = int(np.max(differences[touching]))
        require(maximum_jump <= 1, "AMR mesh violates 2:1 neighbor nesting")
        checks.append({"step": snapshot["step"], "density_tagged_cells": dense_cells,
                       "coarse_density_tagged_cells": coarse_dense_cells,
                       "maximum_coarse_density": coarse_maximum,
                       "maximum_neighbor_level_difference": maximum_jump})
    return checks


def rhs_work(result, log, batch_steps, adaptive):
    """Count interior-cell RHS evaluations on the mesh used by each step.

    Snapshot k follows regridding: it supplies steps k+1 onward. TS at a batch
    endpoint precedes regridding and independently records that batch's mesh.
    """
    count = result["final"]["step"]
    cell_counts = {item["path"]: item["cells"] for item in result["snapshot_errors"]}
    states = {}
    cells_per_leaf = None
    for snapshot in result["snapshots"]:
        state = dict(snapshot, cells=cell_counts[snapshot["path"]])
        require(state["leaf_count"] > 0 and state["cells"] % state["leaf_count"] == 0,
                "Invalid cells per leaf")
        block_cells = state["cells"] // state["leaf_count"]
        require(block_cells > 0 and (cells_per_leaf is None or cells_per_leaf == block_cells),
                "Cells per leaf changed")
        cells_per_leaf = block_cells
        if state["step"] in states:
            old = states[state["step"]]
            require(old["leaf_ids"] == state["leaf_ids"] and old["cells"] == state["cells"],
                    "Conflicting snapshots at one step")
        states[state["step"]] = state
    boundaries = list(range(0, count, batch_steps)) + [count]
    required = boundaries if adaptive else [0, count]
    require(all(step in states for step in required), "Missing mesh snapshot at a batch boundary")
    stages = [stage for stage, _ in result["stages"]] if result["mode"] != "rk3" else [3] * count
    require(len(stages) == count, "RHS stage count does not cover every step")
    batches = [(int(step), int(nodes), int(leaves)) for step, nodes, leaves in
               re.findall(r"^TS (\d+)::[^\n]*ngrids: (\d+), leafs: (\d+),", log, re.M)]
    require([row[0] for row in batches] == boundaries[1:], "Missing or duplicate batch inventory log")
    intervals = []
    for start, (end, nodes, leaves) in zip(boundaries[:-1], batches):
        state = states[start] if adaptive else states[0]
        require(nodes == state["node_count"] and leaves == state["leaf_count"],
                "Batch inventory disagrees with its preceding post-regrid snapshot")
        stage_count = sum(stages[start:end])
        intervals.append({"first_step": start + 1, "last_step": end,
                          "mesh_snapshot_step": state["step"], "leaf_count": leaves,
                          "cells": state["cells"], "rhs_evaluations": stage_count,
                          "cell_rhs_evaluations": stage_count * state["cells"]})
    total = sum(item["cell_rhs_evaluations"] for item in intervals)
    return {"cell_rhs_evaluations": total,
            "cell_rhs_per_simulated_time": total / result["final"]["time"],
            "cells_per_leaf": cells_per_leaf, "mesh_work_intervals": intervals}


def check_initial_fields(reader, left, right):
    for a, b in zip(reader.meshes(left), reader.meshes(right)):
        require(a[0] == b[0], "Initial leaf order differs")
        require(all(np.array_equal(x, y) for x, y in zip(a[1], b[1])), "Initial coordinates differ")
        require(all(np.array_equal(a[2][key], b[2][key]) for key in a[2]), "Initial fields differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir", type=Path)
    output.add_argument("--work-directory", type=Path, help="create a fresh result directory below this path")
    parser.add_argument("--mesh", choices=("amr", "uniform"), default="amr")
    parser.add_argument("--level", "--max-level", dest="level", type=int, default=3)
    parser.add_argument("--min-level", type=int, default=1, help="coarsest allowed AMR level")
    parser.add_argument("--refinement-floor", type=float, default=1.005,
                        help="refine density above this value up to the maximum level; background density is 1")
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--stop-time", type=float, default=0.8)
    parser.add_argument("--max-error-ratio", type=float, default=1.05,
                        help="maximum SSP/RK3 final L1 and Linf error ratio (default: 1.05)")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per application run")
    parser.add_argument("--silo-library", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.level < 0 or not (math.isfinite(args.cfl) and args.cfl > 0):
        parser.error("--level must be nonnegative and --cfl must be finite and positive")
    if args.mesh == "amr" and not (0 <= args.min_level < args.level):
        parser.error("AMR requires 0 <= --min-level < --max-level")
    if not (math.isfinite(args.refinement_floor) and 1 < args.refinement_floor < 1.2):
        parser.error("--refinement-floor must lie between background density 1 and peak density 1.2")
    if not (1 <= 2 / args.cfl + 0.5 < 2**31):
        parser.error("--cfl must yield a positive representable refinement batch size")
    if not (math.isfinite(args.stop_time) and 0 < args.stop_time <= 1.15):
        parser.error("--stop-time must be in (0, 1.15] to retain the exact pulse within the domain")
    if not (math.isfinite(args.max_error_ratio) and args.max_error_ratio >= 1):
        parser.error("--max-error-ratio must be finite and at least 1")
    if args.work_directory:
        if args.analyze_only:
            parser.error("--analyze-only requires --output-dir")
        args.work_directory.mkdir(parents=True, exist_ok=True)
        args.output_dir = Path(tempfile.mkdtemp(prefix="ssp-rhs-efficiency-", dir=args.work_directory))
    args.output_dir = args.output_dir.resolve()
    args.executable = args.executable.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.minimum_level = args.min_level if args.mesh == "amr" else args.level
    args.output_dt = 0.001 if args.mesh == "amr" else 2 * args.stop_time
    args.contact_steepening = "off"
    reader = MeshReader(args.silo_library)
    report = {"problem": "advected smooth density pulse", "settings": {
        "mesh": args.mesh, "minimum_level": args.minimum_level, "maximum_level": args.level,
        "refinement_floor": args.refinement_floor, "cfl": args.cfl, "stop_time": args.stop_time,
        "max_error_ratio": args.max_error_ratio, "hpx_workers": 1,
        "contact_steepening": "off", "gravity": "off",
        "output": "every regrid boundary" if args.mesh == "amr" else "initial and final",
        "comparison": "SSP_MULTISTEP versus ordinary RK3"},
        "theoretical_steady_rhs_ratio": 2 / (3 * (math.sqrt(3) - 1)), "runs": [], "comparison": None}
    initial_paths = []
    commands = []
    for mode, integrator in (("transfer", "SSP_MULTISTEP"), ("rk3", "RK3")):
        directory = args.output_dir / mode
        if not args.analyze_only:
            print(f"Running {integrator}: {args.mesh} levels {args.minimum_level}:{args.level}, "
                  f"CFL {args.cfl}, t={args.stop_time}", flush=True)
            run_binary(args, args.executable, directory, args.minimum_level, args.level, integrator,
                       uniform_grid=args.mesh == "uniform", output_dt=args.output_dt)
        commands.append(check_command(args, directory, integrator))
        result = analyze_run(reader, directory, mode, require_topology=args.mesh == "amr",
                             require_post_change_steps=args.mesh == "amr")
        initial = (check_amr_mesh(result, args.minimum_level, args.level) if args.mesh == "amr"
                   else check_uniform_mesh(result, args.level))
        initial_paths.append(initial["path"])
        if args.mesh == "amr":
            result["density_refinement_checks"] = check_density_refinement(
                reader, result, args.level, args.refinement_floor)
        final = result["final"]
        require(math.isclose(final["time"], args.stop_time, rel_tol=1e-11, abs_tol=1e-12),
                "Application did not reach the requested final time")
        result["step_statistics"] = step_statistics((directory / "run.log").read_text(), final["step"], final["time"])
        result["rhs_evaluations"] = (sum(stage for stage, _ in result["stages"])
                                     if mode == "transfer" else 3 * final["step"])
        result["rhs_per_simulated_time"] = result["rhs_evaluations"] / final["time"]
        result.update(rhs_work(result, (directory / "run.log").read_text(),
                               int(2 / args.cfl + 0.5), args.mesh == "amr"))
        result["execution"] = json.loads((directory / "execution.json").read_text())
        report["runs"].append(result)
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{integrator}: {final['step']} steps, {result['rhs_evaluations']} RHS, "
              f"cell-RHS={result['cell_rhs_evaluations']}, "
              f"L1={final['density_l1']:.8e}, Linf={final['density_linf']:.8e}", flush=True)
    ssp, rk3 = report["runs"]
    initial_meshes = [next(row for row in run["snapshots"] if row["step"] == 0) for run in report["runs"]]
    require(initial_meshes[0]["leaf_ids"] == initial_meshes[1]["leaf_ids"], "Initial meshes differ")
    final_mesh_equal = ssp["final"]["leaf_ids"] == rk3["final"]["leaf_ids"]
    if args.mesh == "uniform":
        require(final_mesh_equal, "Uniform runs used different final meshes")
    require(ssp["execution"]["executable_sha256"] == rk3["execution"]["executable_sha256"],
            "The two runs used different executable hashes")
    commands = [[item for item in command if not item.startswith("--time_integrator=")]
                for command in commands]
    require(commands[0] == commands[1], "Application options differ beyond the integrator")
    check_initial_fields(reader, *initial_paths)
    rhs_ratio = ssp["rhs_per_simulated_time"] / rk3["rhs_per_simulated_time"]
    cell_ratio = ssp["cell_rhs_per_simulated_time"] / rk3["cell_rhs_per_simulated_time"]
    errors = {}
    comparable = True
    for norm in ("l1", "linf"):
        actual, reference = ssp["final"][f"density_{norm}"], rk3["final"][f"density_{norm}"]
        errors[norm] = actual / reference if reference > 0 else (1.0 if actual == 0 else None)
        comparable = comparable and actual <= args.max_error_ratio * reference
    report["comparison"] = {"rhs_ratio_ssp_over_rk3": rhs_ratio, "rhs_reduction_percent": 100 * (1 - rhs_ratio),
        "density_error_ratios_ssp_over_rk3": errors, "accuracy_comparable": comparable,
        "rhs_reduced": rhs_ratio < 1, "identical_initial_fields": True, "final_mesh_equal": final_mesh_equal,
        "cell_rhs_ratio_ssp_over_rk3": cell_ratio, "cell_rhs_reduction_percent": 100 * (1 - cell_ratio),
        "cell_rhs_reduced": cell_ratio < 1,
        "summed_step_time_ratio_ssp_over_rk3": ssp["step_statistics"]["summed_step_elapsed_seconds"] / rk3["step_statistics"]["summed_step_elapsed_seconds"],
        "whole_process_time_ratio_ssp_over_rk3": ssp["execution"]["elapsed_seconds"] / rk3["execution"]["elapsed_seconds"],
        "timing_scope": "One serial pair; process timing includes initialization, diagnostics, regrids and unequal checkpoint counts and payloads. Step timing is root-reported.",
        "passed": comparable and rhs_ratio < 1 and cell_ratio < 1}
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"RHS reduction: {100 * (1-rhs_ratio):.3f}%; interior-cell RHS reduction: "
          f"{100 * (1-cell_ratio):.3f}%; SSP/RK3 error ratios: {errors}")
    print(f"Results: {args.output_dir / 'results.json'}")
    require(report["comparison"]["passed"], "Efficiency or accuracy acceptance failed; see saved comparison")
    print("PASS: fewer RHS and interior-cell RHS evaluations per simulated time with comparable final density errors")


if __name__ == "__main__":
    main()
