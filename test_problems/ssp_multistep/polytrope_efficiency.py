#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Compare SSP multistep and ordinary RK3 on a drifting n=1.5 polytrope.

Runs one CPU application at a time with one HPX worker. The default star drifts
from x=-0.3 to x=0.3 in code time 3 through density-based AMR levels 1:3.
Measures admissibility, conservation, topology and actual RHS work. The final
density L1 error against the point-sampled translated continuum polytrope must
be within 5% of ordinary RK3 by default; this is not a temporal convergence test.
"""

import argparse
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

from amr_accuracy import MeshReader, decode_node, np, require
from amr_gravity import GravityMeshReader, analyze_run, energy_comparisons, make_command
from rhs_efficiency import rhs_work, step_statistics


INITIAL_FIELDS = ("rho_1", "rho_2", "sx", "sy", "sz", "egas", "tau", "pot",
                  "lx", "ly", "lz", "gx", "gy", "gz")
INTEGRATORS = {"transfer": "SSP_MULTISTEP", "rk3": "RK3"}
PROJECT_LIBRARIES = ("libhpx_octolib.so", "libhpx_hydrolib.so",
                     "libhpx_optionslib.so", "libhpx_debuglib.so")
RUNTIME_MANIFEST_SCOPE = ("Executable and the four project shared libraries resolved by ldd "
                          "in the application's working directory and inherited environment. "
                          "System and third-party dependency libraries are outside this manifest.")
GRAVITY_WORK_SCOPE = ("Two gravity passes per time-integration RHS stage: DRHODT and RHO. "
                      "Excludes initialization and regrid gravity solves. Interior-cell RHS "
                      "work excludes ghost cells, communications and regrid/transfer work; "
                      "it is not a gravity-operation or wall-clock cost model.")


def benchmark_command(args, mode):
    return make_command(args, args.executable, INTEGRATORS[mode]) + [
        "--star_n=1.5", "--star_rho_center=1", "--star_rmax=0.3333333333333333",
        "--star_alpha=0.09123047097730642", "--star_dr=0.0003255208333333333",
        "--cdisc_detect=off", f"--theta={args.theta}"]


def executable_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_manifest(executable, directory):
    """Resolve actual project-library load paths, including LD_LIBRARY_PATH."""
    completed = subprocess.run(["ldd", str(executable)], cwd=directory, capture_output=True,
                               text=True, timeout=30, check=False)
    require(completed.returncode == 0,
            f"Cannot resolve runtime libraries with ldd: {completed.stderr.strip()}")
    libraries = {}
    for line in completed.stdout.splitlines():
        match = re.match(r"^\s*(\S+)\s+=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)\s*$", line)
        if match is None or match[1] not in PROJECT_LIBRARIES:
            continue
        name, loaded_path = match.groups()
        path = Path(loaded_path)
        if not path.is_absolute():
            path = directory / path
        path = path.resolve()
        require(path.is_file(), f"Resolved project library does not exist: {path}")
        require(name not in libraries, f"Duplicate ldd entry for {name}")
        libraries[name] = {"path": str(path), "sha256": executable_digest(path)}
    require(set(libraries) == set(PROJECT_LIBRARIES),
            f"ldd did not resolve all project libraries: {sorted(set(PROJECT_LIBRARIES) - set(libraries))}")
    return {"scope": RUNTIME_MANIFEST_SCOPE,
            "executable": {"path": str(executable.resolve()), "sha256": executable_digest(executable)},
            "project_shared_libraries": dict(sorted(libraries.items()))}


def check_runtime_manifest(executable, directory, execution):
    before = execution.get("runtime_manifest_before")
    after = execution.get("runtime_manifest_after")
    if before is None and after is None:
        # Earlier launchers recorded only the executable. A present-day ldd
        # result cannot establish which libraries those earlier runs loaded.
        execution["runtime_manifest_status"] = (
            "unavailable: launcher did not record before/after runtime manifests; "
            "only executable identity is verified")
        execution["runtime_manifest_verified"] = False
        return
    require(before is not None and after is not None, "Incomplete saved runtime provenance")
    require(before == after, "Executable or project libraries changed during the application run")
    require(before == runtime_manifest(executable, directory),
            "Current executable or resolved project libraries differ from the saved runtime manifest")
    execution["runtime_manifest_status"] = "verified: saved before/after and current manifests agree"
    execution["runtime_manifest_verified"] = True


def run_application(args, directory, mode):
    require(not directory.exists(), f"Refusing to overwrite run directory {directory}")
    directory.mkdir(parents=True)
    command = benchmark_command(args, mode)
    before = runtime_manifest(args.executable, directory)
    digest = before["executable"]["sha256"]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    record = {"executable": str(args.executable), "executable_sha256": digest,
              "runtime_manifest_before": before, "runtime_manifest_after": None,
              "returncode": None, "timed_out": False}
    try:
        with (directory / "run.log").open("w") as log:
            completed = subprocess.run(command, cwd=directory, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
        record["returncode"] = completed.returncode
    except subprocess.TimeoutExpired:
        record["timed_out"] = True
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        try:
            record["runtime_manifest_after"] = runtime_manifest(args.executable, directory)
            record["runtime_manifest_unchanged"] = before == record["runtime_manifest_after"]
        except (OSError, subprocess.SubprocessError, AssertionError) as error:
            record["runtime_manifest_error"] = str(error)
            record["runtime_manifest_unchanged"] = False
        (directory / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    require(not record["timed_out"], f"Application timed out; inspect {directory / 'run.log'}")
    require(record["returncode"] == 0, f"Application failed; inspect {directory / 'run.log'}")
    require(record["runtime_manifest_unchanged"],
            "Runtime changed or could not be verified after the run; inspect execution.json")
    require((directory / "final.silo").is_file(), f"Missing final checkpoint in {directory}")


def equivalent_argument(actual, expected):
    """Accept equivalent numeric spelling while retaining option order/repeats."""
    if actual == expected:
        return True
    left, separator, left_value = actual.partition("=")
    right, _, right_value = expected.partition("=")
    if not separator or left != right:
        return False
    try:
        return float(left_value) == float(right_value)
    except ValueError:
        return False


def check_provenance(args, directory, mode):
    command = json.loads((directory / "command.json").read_text())
    expected = benchmark_command(args, mode)
    require(len(command) == len(expected), "Saved command has unexpected or missing options")
    require(Path(command[0]).resolve() == args.executable, "Saved executable path differs")
    require(all(equivalent_argument(a, b) for a, b in zip(command[1:], expected[1:])),
            f"Saved command differs from requested benchmark settings in {directory}")
    execution = json.loads((directory / "execution.json").read_text())
    require(execution.get("returncode") == 0 and not execution.get("timed_out", False),
            f"Application did not complete successfully in {directory}")
    require(execution.get("executable_sha256") == executable_digest(args.executable),
            "Saved executable hash differs from the supplied executable")
    check_runtime_manifest(args.executable, directory, execution)
    elapsed = execution.get("elapsed_seconds")
    require(isinstance(elapsed, (float, int)) and math.isfinite(elapsed) and elapsed > 0,
            "Missing or invalid application elapsed time")
    return command, execution


def check_mesh(reader, result, minimum_level, maximum_level, threshold):
    """Audit total-density tagging and touching-neighbor 2:1 nesting."""
    histograms, checks = [], []
    for snapshot in result["snapshots"]:
        levels, lows, highs = [], [], []
        dense_cells = coarse_dense_cells = criterion_violations = 0
        coarse_maximum = None
        for node, _, fields in reader.meshes(snapshot["path"], field_names=("rho_1", "rho_2")):
            level, xyz = decode_node(node)
            require(minimum_level <= level <= maximum_level, "Leaf outside requested AMR level range")
            scale = 1 << (maximum_level - level)
            levels.append(level)
            lows.append([value * scale for value in xyz])
            highs.append([(value + 1) * scale for value in xyz])
            density = fields["rho_1"] + fields["rho_2"]
            require(np.all(np.isfinite(density)) and np.min(density) > 0,
                    "Nonfinite or nonpositive density in refinement audit")
            dense = int(np.count_nonzero(density > threshold))
            dense_cells += dense
            if level < maximum_level:
                coarse_dense_cells += dense
                # refine_test_moving_star lowers the desired-level threshold
                # by eight for every level below the maximum target level.
                local_threshold = threshold / 8 ** (maximum_level - level - 1)
                criterion_violations += int(np.count_nonzero(density > local_threshold))
                value = float(np.max(density))
                coarse_maximum = value if coarse_maximum is None else max(coarse_maximum, value)
        require(levels and max(levels) == maximum_level, "Star does not reach requested finest level")
        require(min(levels) < maximum_level, "AMR benchmark became a uniform finest grid")
        require(dense_cells > 0, "No density-tagged stellar cells remain")
        require(coarse_dense_cells == 0, "Density-tagged stellar cells lie below maximum level")
        require(criterion_violations == 0, "Saved mesh does not satisfy the stellar density criterion")
        lo, hi, lev = np.array(lows), np.array(highs), np.array(levels)
        touching = np.all((lo[:, None, :] <= hi[None, :, :]) &
                          (lo[None, :, :] <= hi[:, None, :]), axis=2)
        maximum_jump = int(np.max(abs(lev[:, None] - lev[None, :])[touching]))
        require(maximum_jump <= 1, "AMR mesh violates 2:1 face/edge/corner nesting")
        histograms.append({"step": snapshot["step"], "time": snapshot["time"],
                           "leaf_blocks_by_level": {str(level): levels.count(level)
                                                    for level in sorted(set(levels))}})
        checks.append({"step": snapshot["step"], "time": snapshot["time"],
                       "density_tagged_cells": dense_cells,
                       "coarse_density_tagged_cells": coarse_dense_cells,
                       "density_criterion_violating_cells": criterion_violations,
                       "maximum_coarse_density": coarse_maximum,
                       "maximum_neighbor_level_difference": maximum_jump})
    result["leaf_level_histograms"] = histograms
    result["density_refinement_checks"] = checks


def check_initial_fields(reader, left, right):
    require(left["step"] == right["step"] == 0 and left["time"] == right["time"] == 0,
            "Missing time-zero initial snapshots")
    require(left["leaf_ids"] == right["leaf_ids"] and left["node_count"] == right["node_count"],
            "Initial AMR meshes differ")
    require(left["code_units"] == right["code_units"], "Initial unit conversions differ")
    # Compare raw stored arrays: this includes angular momentum and acceleration
    # fields, which GravityMeshReader does not need for its physical diagnostics.
    first = MeshReader.meshes(reader, left["path"], field_names=INITIAL_FIELDS)
    second = MeshReader.meshes(reader, right["path"], field_names=INITIAL_FIELDS)
    for a, b in zip_longest(first, second):
        require(a is not None and b is not None, "Initial leaf counts differ")
        require(a[0] == b[0], "Initial leaf order differs")
        require(all(np.array_equal(x, y) for x, y in zip(a[1], b[1])), "Initial coordinates differ")
        require(all(np.array_equal(a[2][field], b[2][field]) for field in INITIAL_FIELDS),
                "Initial hydro or gravity fields differ")


def compare_runs(reader, runs, commands):
    by_mode = {result["mode"]: result for result in runs}
    if "transfer" not in by_mode or "rk3" not in by_mode:
        return None
    ssp, rk3 = by_mode["transfer"], by_mode["rk3"]
    require(ssp["execution"]["executable_sha256"] == rk3["execution"]["executable_sha256"],
            "Compared runs used different executables")
    verified_runtime = all(run["execution"].get("runtime_manifest_verified", False) for run in (ssp, rk3))
    if verified_runtime:
        require(ssp["execution"]["runtime_manifest_before"] == rk3["execution"]["runtime_manifest_before"],
                "Compared runs loaded different executable/project-library manifests")
    runtime_status = ("verified identical executable and project-library manifests across both runs"
                      if verified_runtime else
                      "unavailable: at least one launcher lacks before/after runtime manifests; "
                      "only executable identity is verified across the pair")
    settings = [[option for option in commands[mode] if not option.startswith("--time_integrator=")]
                for mode in ("transfer", "rk3")]
    require(settings[0] == settings[1], "Compared application options differ beyond integrator")
    initial = [next((row for row in result["snapshots"] if row["step"] == 0), None)
               for result in (ssp, rk3)]
    require(all(row is not None for row in initial), "Missing initial snapshots")
    check_initial_fields(reader, *initial)
    rhs_ratio = ssp["rhs_per_simulated_time"] / rk3["rhs_per_simulated_time"]
    cell_ratio = ssp["cell_rhs_per_simulated_time"] / rk3["cell_rhs_per_simulated_time"]
    reference_step_time = rk3["step_statistics"]["summed_step_elapsed_seconds"]
    return {"rhs_ratio_ssp_over_rk3": rhs_ratio, "rhs_reduction_percent": 100 * (1 - rhs_ratio),
            "cell_rhs_ratio_ssp_over_rk3": cell_ratio, "cell_rhs_reduction_percent": 100 * (1 - cell_ratio),
            "rhs_reduced": rhs_ratio < 1, "cell_rhs_reduced": cell_ratio < 1,
            "integration_gravity_pass_ratio_ssp_over_rk3": rhs_ratio,
            "identical_initial_fields": True, "identical_initial_mesh": True,
            "initial_fields_compared": list(INITIAL_FIELDS),
            "identical_options_except_integrator": True, "identical_executable_hash": True,
            "identical_project_runtime_manifest": True if verified_runtime else None,
            "runtime_manifest_comparison": runtime_status,
            "final_mesh_equal": ssp["final"]["leaf_ids"] == rk3["final"]["leaf_ids"],
            "summed_step_time_ratio_ssp_over_rk3":
                ssp["step_statistics"]["summed_step_elapsed_seconds"] / reference_step_time
                if reference_step_time > 0 else None,
            "whole_process_time_ratio_ssp_over_rk3":
                ssp["execution"]["elapsed_seconds"] / rk3["execution"]["elapsed_seconds"],
            "timing_scope": "One sequential pair; process timing includes initialization, diagnostics, "
                            "AMR, and unequal checkpoint counts/payloads. Step timing is root-reported.",
            "work_reduction_passed": rhs_ratio < 1 and cell_ratio < 1}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir", type=Path)
    output.add_argument("--work-directory", type=Path, help="create a fresh result directory below this path")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=tuple(INTEGRATORS), default=["transfer", "rk3"])
    parser.add_argument("--min-level", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument("--center", nargs=3, type=float, default=[-0.3, 0, 0])
    parser.add_argument("--velocity", nargs=3, type=float, default=[0.2, 0, 0])
    parser.add_argument("--stop-time", type=float, default=3.0)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--star-rho-out", type=float, default=1e-6)
    parser.add_argument("--star-egas-out", type=float, default=1e-8)
    parser.add_argument("--refinement-floor", type=float, default=0.001)
    parser.add_argument("--theta", type=float, default=0.35, help="gravity opening parameter")
    parser.add_argument("--dual-energy-switch", type=float, default=0.001)
    parser.add_argument("--mass-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-error-ratio", type=float, default=1.05,
                        help="maximum SSP/RK3 final density L1-per-ideal-star-mass ratio")
    parser.add_argument("--timeout", type=float, default=7200, help="seconds per application run")
    parser.add_argument("--silo-library", type=Path)
    parser.add_argument("--allow-no-topology-change", action="store_true", help="allow short exploratory pilots")
    args = parser.parse_args()
    if not (0 <= args.min_level < args.max_level <= 20):
        parser.error("require 0 <= --min-level < --max-level <= 20")
    for name in ("stop_time", "cfl", "star_rho_out", "star_egas_out", "refinement_floor", "theta",
                 "dual_energy_switch", "mass_tolerance", "timeout"):
        value = getattr(args, name)
        if not (math.isfinite(value) and value > 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not (1 <= 2 / args.cfl + 0.5 < 2**31):
        parser.error("--cfl must yield a positive representable refinement batch size")
    if not args.star_rho_out < args.refinement_floor < 1:
        parser.error("require atmosphere density < refinement threshold < central density 1")
    if not (0 < args.theta < 1 and 0 < args.dual_energy_switch < 1):
        parser.error("--theta and --dual-energy-switch must lie strictly between 0 and 1")
    if not (math.isfinite(args.max_error_ratio) and args.max_error_ratio >= 1):
        parser.error("--max-error-ratio must be finite and at least 1")
    if not all(math.isfinite(value) for value in args.center + args.velocity):
        parser.error("--center and --velocity must be finite")
    radius = 1 / 3
    if any(max(abs(position), abs(position + speed * args.stop_time)) + radius >= 1
           for position, speed in zip(args.center, args.velocity)):
        parser.error("the nominal stellar sphere must remain strictly inside [-1,1]^3 throughout the run")
    if len(args.modes) != len(set(args.modes)):
        parser.error("--modes cannot contain duplicates")
    args.executable = args.executable.resolve()
    if not args.executable.is_file() or not os.access(args.executable, os.X_OK):
        parser.error("--executable must name an existing executable file")
    if args.work_directory:
        if args.analyze_only:
            parser.error("--analyze-only requires --output-dir")
        args.work_directory.mkdir(parents=True, exist_ok=True)
        args.output_dir = Path(tempfile.mkdtemp(prefix="ssp-polytrope-efficiency-", dir=args.work_directory))
    args.output_dir = args.output_dir.resolve()
    if args.analyze_only and not args.output_dir.is_dir():
        parser.error("--analyze-only requires an existing output directory")
    args.hard_dt = -1.0
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = GravityMeshReader(args.silo_library)
    settings = ("min_level", "max_level", "center", "velocity", "stop_time", "cfl", "hard_dt",
                "star_rho_out", "star_egas_out", "refinement_floor", "theta", "dual_energy_switch",
                "mass_tolerance", "allow_no_topology_change", "max_error_ratio")
    report = {"problem": "MOVING_STAR", "settings": {name: getattr(args, name) for name in settings},
              "polytrope": {"n": 1.5, "gamma": 5 / 3, "G": 1, "central_density": 1,
                            "radius": 1 / 3, "alpha": 0.09123047097730642,
                            "lane_emden_dr": 0.0003255208333333333},
              "validation_scope": "Admissibility, mass/species conservation, AMR tagging/nesting, "
                                  "measured RHS work, and final density L1 versus a point-sampled "
                                  "translated continuum reference; no temporal convergence claim.",
              "gravity_work_scope": GRAVITY_WORK_SCOPE, "runs": [], "comparison": None}
    commands = {}
    for mode in args.modes:
        directory = args.output_dir / mode
        if not args.analyze_only:
            print(f"Running {INTEGRATORS[mode]}: AMR levels {args.min_level}:{args.max_level}, "
                  f"CFL {args.cfl}, code time {args.stop_time}", flush=True)
            run_application(args, directory, mode)
        commands[mode], execution = check_provenance(args, directory, mode)
        result = analyze_run(args, reader, directory, mode)
        require(math.isclose(result["final"]["time"], args.stop_time,
                             rel_tol=1e-11, abs_tol=1e-12),
                "Final checkpoint did not reach the requested code time")
        check_mesh(reader, result, args.min_level, args.max_level, args.refinement_floor)
        log = (directory / "run.log").read_text()
        # rhs_work consumes this generic adapter name; gravity diagnostics use
        # snapshot_diagnostics because there is no exact pulse reference here.
        adapter = dict(result, snapshot_errors=result["snapshot_diagnostics"])
        result.update(rhs_work(adapter, log, int(2 / args.cfl + 0.5), adaptive=True))
        result["step_statistics"] = step_statistics(log, result["final"]["step"], result["final"]["time"])
        result["rhs_evaluations"] = (3 * result["final"]["step"] if mode == "rk3"
                                     else sum(stage for stage, _ in result["stages"]))
        result["rhs_per_simulated_time"] = result["rhs_evaluations"] / result["final"]["time"]
        result["integration_gravity_passes"] = 2 * result["rhs_evaluations"]
        result["integration_gravity_passes_per_simulated_time"] = 2 * result["rhs_per_simulated_time"]
        result["execution"] = execution
        report["runs"].append(result)
        report["energy_comparisons"] = energy_comparisons(report["runs"])
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{mode}: {result['final']['step']} steps, {result['rhs_evaluations']} RHS, "
              f"{result['cell_rhs_evaluations']} interior-cell RHS, "
              f"refined/coarsened parents {result['refined_parent_count']}/{result['coarsened_parent_count']}",
              flush=True)
    report["comparison"] = compare_runs(reader, report["runs"], commands)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    from polytrope_diagnostics import analyze_results
    equilibrium = analyze_results(args.output_dir / "results.json",
                                  source_root=Path(__file__).resolve().parents[2],
                                  silo_library=args.silo_library)
    report["equilibrium_analysis"] = str(args.output_dir / "equilibrium-analysis.json")
    if report["comparison"] is not None:
        by_mode = {run["mode"]: run for run in equilibrium["runs"]}
        key = "density_l1_per_ideal_star_mass"
        actual = by_mode["transfer"]["summary"]["final"][key]
        reference = by_mode["rk3"]["summary"]["final"][key]
        require(math.isfinite(actual) and math.isfinite(reference) and actual >= 0 and reference >= 0,
                "Invalid density L1 errors from equilibrium analysis")
        comparable = actual <= args.max_error_ratio * reference
        comparison = report["comparison"]
        comparison.update(final_density_l1_per_ideal_star_mass_ssp=actual,
                          final_density_l1_per_ideal_star_mass_rk3=reference,
                          final_density_l1_ratio_ssp_over_rk3=actual / reference if reference > 0
                          else (1.0 if actual == 0 else None),
                          maximum_density_error_ratio=args.max_error_ratio,
                          accuracy_comparable=comparable,
                          accuracy_reference="Point-sampled translated continuum Lane-Emden polytrope; "
                                             "includes spatial, initialization, AMR, atmosphere and gravity errors.",
                          passed=comparison["work_reduction_passed"] and comparable)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # The comparison above adds derived metadata after the diagnostic pass.
    # Keep its provenance tied to the final, user-visible results file.
    equilibrium["source_results_sha256"] = executable_digest(args.output_dir / "results.json")
    (args.output_dir / "equilibrium-analysis.json").write_text(
        json.dumps(equilibrium, indent=2, allow_nan=False) + "\n")
    print(f"Results: {args.output_dir / 'results.json'}", flush=True)
    if report["comparison"] is not None:
        comparison = report["comparison"]
        print(f"RHS reduction: {comparison['rhs_reduction_percent']:.3f}%; interior-cell RHS reduction: "
              f"{comparison['cell_rhs_reduction_percent']:.3f}%", flush=True)
        print(f"Final density L1 SSP/RK3 ratio: {comparison['final_density_l1_ratio_ssp_over_rk3']}", flush=True)
        require(comparison["passed"], "Work-reduction or density-error acceptance failed; see saved comparison")
        print("PASS: fewer RHS and interior-cell RHS evaluations per simulated time. "
              "Final density L1 error meets the declared RK3 comparison tolerance.")
    else:
        print("Single-mode diagnostics complete; no SSP/RK3 work comparison was performed.")


if __name__ == "__main__":
    main()
