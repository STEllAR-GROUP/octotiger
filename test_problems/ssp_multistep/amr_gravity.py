#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Validate real moving-star hydro/gravity AMR and exact SSP-history restart.

This is a coupling, topology, conservation and restart test, not an analytic
convergence test. Each simulation uses one HPX worker and CPU kernels. A baseline
executable must match the transfer executable except for RK3 restart on regrid.
The RK3 control uses the transfer executable with time_integrator=RK3. Defaults
use a moderate atmosphere to exercise coupling and moving refinement.

Example (runs real applications; choose an otherwise idle CPU allocation):
  python3 amr_gravity.py --executable /path/to/transfer-wrapper \
      --baseline-executable /path/to/baseline-wrapper --output-dir /tmp/star-amr
Use --analyze-only with the same arguments to inspect existing runs.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np

from amr_accuracy import MeshReader, ancestors, require
from silo_snapshot import compare_checkpoints

GAMMA = 5 / 3
N_FIELDS = 12
FIELDS = ("rho_1", "rho_2", "sx", "sy", "sz", "egas", "tau", "pot")


class GravityMeshReader(MeshReader):
    # Gravity chooses G=1 and derives the time unit, so even command-line unit
    # inputs of one do not make the visualization fields/time code quantities.
    def code_units(self, path):
        names = ("code_to_g", "code_to_cm", "code_to_s")
        values = self.read(path, names)
        units = {name: float(self.reals(values[name])[0]) for name in names}
        require(all(math.isfinite(value) and value > 0 for value in units.values()),
                f"Invalid code units in {path}")
        return units

    def inventory(self, path):
        result = super().inventory(path)
        units = self.code_units(path)
        result["cgs_time"] = result["time"]
        result["time"] /= units["code_to_s"]
        result["code_units"] = units
        return result

    def meshes(self, path, field_names=FIELDS):
        units = self.code_units(path)
        mass, length, duration = (units[name] for name in ("code_to_g", "code_to_cm", "code_to_s"))
        density = mass / length ** 3
        momentum = mass / (duration * length ** 2)
        energy = mass / (duration ** 2 * length)
        factors = {"rho_1": density, "rho_2": density, "sx": momentum, "sy": momentum,
                   "sz": momentum, "egas": energy, "pot": energy, "tau": energy ** (1 / GAMMA)}
        for node, coordinates, fields in super().meshes(path, field_names):
            yield node, [axis / length for axis in coordinates], {
                name: values / factors[name] for name, values in fields.items()}


def stage_sequence(directory):
    return [(int(stage), float(alpha)) for stage, alpha in re.findall(
        r"SSP_MULTISTEP stages=(\d+) alpha=([^\s]+)", (directory / "run.log").read_text())]


def analyze_snapshot(reader, path, dual_energy_switch):
    result = reader.inventory(path)
    minima = {name: math.inf for name in ("density", "tau", "pressure", "conservative_thermal", "species_1", "species_2", "potential_density")}
    volume = mass = energy = negative_thermal_volume = dual_energy_volume = 0.0
    species = np.zeros(2)
    first_moment = np.zeros(3)
    cells = negative_thermal_cells = dual_energy_cells = 0
    for _, xyz, fields in reader.meshes(path, field_names=FIELDS):
        for name, values in fields.items():
            require(np.all(np.isfinite(values)), f"Nonfinite {name} in {path}")
        x, y, z = xyz
        weights = np.diff(z)[:, None, None] * np.diff(y)[None, :, None] * np.diff(x)[None, None, :]
        density = fields["rho_1"] + fields["rho_2"]
        require(np.min(density) > 0, f"Nonpositive density in {path}")
        require(np.min(fields["tau"]) > 0, f"Nonpositive tau in {path}")
        require(np.min(fields["rho_1"]) >= 0 and np.min(fields["rho_2"]) >= 0,
                f"Negative species density in {path}")
        thermal = fields["egas"] - 0.5 * sum(fields[name] ** 2 for name in ("sx", "sy", "sz")) / density
        # Match IDEAL to_prim in octotiger/unitiger/physics_impl.hpp. Cold-star
        # conservative thermal energy can be negative while tau supplies pressure.
        use_tau = thermal <= dual_energy_switch * fields["egas"]
        pressure = (GAMMA - 1) * np.where(use_tau, fields["tau"] ** GAMMA, thermal)
        require(np.all(np.isfinite(pressure)) and np.min(pressure) > 0,
                f"Nonfinite/nonpositive physical pressure in {path}")
        quantities = {"density": density, "tau": fields["tau"], "pressure": pressure,
                      "conservative_thermal": thermal, "species_1": fields["rho_1"],
                      "species_2": fields["rho_2"], "potential_density": fields["pot"]}
        for name, values in quantities.items():
            minima[name] = min(minima[name], float(np.min(values)))
        volume += float(np.sum(weights))
        mass += float(np.sum(weights * density))
        energy += float(np.sum(weights * (fields["egas"] + 0.5 * fields["pot"])))
        for index in range(2):
            species[index] += np.sum(weights * fields[f"rho_{index + 1}"])
        centers = (0.5 * (x[:-1] + x[1:])[None, None, :],
                   0.5 * (y[:-1] + y[1:])[None, :, None],
                   0.5 * (z[:-1] + z[1:])[:, None, None])
        for axis in range(3):
            first_moment[axis] += np.sum(weights * fields["rho_1"] * centers[axis])
        negative_thermal_cells += int(np.count_nonzero(thermal < 0))
        negative_thermal_volume += float(np.sum(weights * (thermal < 0)))
        dual_energy_cells += int(np.count_nonzero(use_tau))
        dual_energy_volume += float(np.sum(weights * use_tau))
        cells += density.size
    require(math.isclose(volume, 8, rel_tol=1e-12), f"Leaf meshes do not cover volume 8: {volume}")
    require(minima["potential_density"] < 0, f"No nonzero attractive gravitational potential in {path}")
    result.update(cells=cells, minima=minima, volume=volume, mass=mass,
                  total_energy=energy, species_mass=species.tolist(),
                  star_centroid=(first_moment / species[0]).tolist(),
                  negative_conservative_thermal_cells=negative_thermal_cells,
                  negative_conservative_thermal_volume=negative_thermal_volume,
                  dual_energy_cells=dual_energy_cells, dual_energy_volume=dual_energy_volume)
    return result


def topology_changes(snapshots):
    changes = []
    for old, new in zip(snapshots, snapshots[1:]):
        old_leaves, new_leaves = set(old["leaf_ids"]), set(new["leaf_ids"])
        if old_leaves != new_leaves:
            changes.append({"time": new["time"], "step": new["step"], "checkpoint": new["path"],
                            "added_leaf_ids": sorted(new_leaves - old_leaves),
                            "removed_leaf_ids": sorted(old_leaves - new_leaves),
                            "refined_parent_ids": sorted(old_leaves & ancestors(new_leaves)),
                            "coarsened_parent_ids": sorted(new_leaves & ancestors(old_leaves))})
    return changes


def conservation_diagnostics(directory, tolerance, snapshots):
    sums = np.atleast_2d(np.loadtxt(directory / "sums.dat"))
    require(sums.shape[1] == 1 + 2 * N_FIELDS + 3, "Expected 28 columns for two-species diagnostics")
    require(np.all(np.isfinite(sums)), "Nonfinite conservation diagnostics")
    require(len(sums) > 1 and np.all(np.diff(sums[:, 0]) >= 0), "Invalid diagnostic time sequence")
    total_mass_scale = max(abs(sums[0, 1]), np.finfo(float).tiny)
    result = {}
    for name, field in (("mass", 0), ("total_energy", 1), ("species_1", 10), ("species_2", 11)):
        # grid_sum + cumulative grid_out, including gravity for egas: grid_sum
        # includes 0.5*pot and grid_out includes the gravitational energy outflow.
        ledger, outflow = sums[:, 1 + 2 * field], sums[:, 2 + 2 * field]
        maximum = float(np.max(abs(ledger - ledger[0])))
        result[name] = {"initial": float(ledger[0]), "final": float(ledger[-1]),
                        "signed_final_drift": float(ledger[-1] - ledger[0]),
                        "maximum_absolute_drift": maximum,
                        "maximum_relative_drift": maximum / max(abs(float(ledger[0])), np.finfo(float).tiny),
                        "maximum_drift_per_initial_total_mass": maximum / total_mass_scale,
                        "final_boundary_outflow": float(outflow[-1])}
        if name != "total_energy":
            # Normalize both species to total mass: the almost-vacuum ambient
            # species starts tiny, and must not create a misleading denominator.
            require(maximum / total_mass_scale <= tolerance,
                    f"{name} plus outflow drift / initial mass exceeds {tolerance}: {maximum / total_mass_scale}")
    # Visualization fields were converted back to code units by GravityMeshReader.
    # Add the small outflow columns to full-precision cell integrals, rather than
    # relying only on rounded large ledger totals from the text diagnostics.
    matched = []
    for snapshot in snapshots:
        row = sums[int(np.argmin(abs(sums[:, 0] - snapshot["time"])))]
        require(math.isclose(float(row[0]), snapshot["time"], rel_tol=1e-11, abs_tol=1e-12),
                f"No conservation row matching snapshot time {snapshot['time']}")
        matched.append({"step": snapshot["step"], "time": snapshot["time"],
                        "diagnostic_time": float(row[0]), "mass_outflow": float(row[2]),
                        "total_energy_outflow": float(row[4]), "species_1_outflow": float(row[22]),
                        "species_2_outflow": float(row[24])})
    mesh_fields = {}
    precise_mass_scale = max(abs(snapshots[0]["mass"] + matched[0]["mass_outflow"]), np.finfo(float).tiny)
    for name in ("mass", "total_energy", "species_1", "species_2"):
        if name.startswith("species_"):
            raw = np.array([snapshot["species_mass"][int(name[-1]) - 1] for snapshot in snapshots])
        else:
            raw = np.array([snapshot[name] for snapshot in snapshots])
        ledger = raw + np.array([row[name + "_outflow"] for row in matched])
        mesh_fields[name] = {}
        for description, values in (("mesh", raw), ("mesh_plus_outflow", ledger)):
            maximum = float(np.max(abs(values - values[0])))
            mesh_fields[name][description] = {
                "initial": float(values[0]), "final": float(values[-1]),
                "signed_final_drift": float(values[-1] - values[0]), "maximum_absolute_drift": maximum,
                "maximum_relative_drift": maximum / max(abs(float(values[0])), np.finfo(float).tiny),
                "maximum_drift_per_initial_total_mass": maximum / precise_mass_scale}
        if name != "total_energy":
            require(mesh_fields[name]["mesh_plus_outflow"]["maximum_drift_per_initial_total_mass"] <= tolerance,
                    f"Silo mesh {name} plus matched boundary outflow is not conserved")
    return {"first_time": float(sums[0, 0]), "last_time": float(sums[-1, 0]),
            "rows": len(sums), "mass_tolerance": tolerance, "fields": result,
            "units": "code units from each snapshot's stored conversion metadata",
            "timestamp_relative_tolerance": 1e-11, "timestamp_absolute_tolerance": 1e-12,
            "matched_diagnostics": matched, "mesh_fields": mesh_fields}


def analyze_run(args, reader, directory, mode):
    stages = stage_sequence(directory)
    paths = list(directory.glob("X.*.silo")) + [directory / "final.silo"]
    snapshots = sorted((reader.inventory(path) for path in paths), key=lambda item: (item["step"], item["path"]))
    changes = topology_changes(snapshots)
    if not args.allow_no_topology_change:
        require(any(item["refined_parent_ids"] for item in changes), f"No actual refinement in {directory}")
        require(any(item["coarsened_parent_ids"] for item in changes), f"No actual coarsening in {directory}")
        final_step = next(item["step"] for item in snapshots
                          if item["path"] == str((directory / "final.silo").resolve()))
        for direction in ("refined_parent_ids", "coarsened_parent_ids"):
            require(any(item[direction] and item["step"] < final_step for item in changes),
                    f"No {direction} event followed by an accepted step in {directory}")
    if mode == "transfer":
        require(stages and stages[0][0] == 3, f"Missing initial RK3 in {directory}")
        require(sum(stage == 3 for stage, _ in stages) == 1, f"Expected exactly one RK3 in {directory}")
        require(all(stage == 2 for stage, _ in stages[1:]), f"Unexpected stage count in {directory}")
    elif mode == "baseline":
        require(sum(stage == 3 for stage, _ in stages) > 1, "Baseline did not exercise RK3 rebootstrap")
    else:
        require(not stages, "RK3 control unexpectedly used the SSP multistep path")
    series = [analyze_snapshot(reader, item["path"], args.dual_energy_switch) for item in snapshots]
    final = next(item for item in series if item["path"] == str((directory / "final.silo").resolve()))
    require(math.isclose(final["time"], args.stop_time, rel_tol=1e-11, abs_tol=1e-12),
            f"Run did not reach stop_time: {directory}")
    if mode in ("transfer", "baseline"):
        require(len(stages) == final["step"], "Stage log does not cover every accepted step")
    return {"directory": str(directory.resolve()), "mode": mode, "final": final,
            "snapshots": snapshots, "snapshot_diagnostics": series, "topology_changes": changes,
            "refined_parent_count": sum(len(item["refined_parent_ids"]) for item in changes),
            "coarsened_parent_count": sum(len(item["coarsened_parent_ids"]) for item in changes),
            "minima": {name: min(item["minima"][name] for item in series) for name in final["minima"]},
            "maximum_negative_thermal_cells": max(item["negative_conservative_thermal_cells"] for item in series),
            "maximum_negative_thermal_volume": max(item["negative_conservative_thermal_volume"] for item in series),
            "maximum_dual_energy_volume": max(item["dual_energy_volume"] for item in series),
            "conservation": conservation_diagnostics(directory, args.mass_tolerance, series),
            "rk3_steps": sum(stage == 3 for stage, _ in stages) if mode != "rk3" else final["step"],
            "multistep_steps": sum(stage == 2 for stage, _ in stages), "stages": stages}


def make_command(args, executable, integrator, restart=None):
    command = [str(executable), "--hpx:threads=1", "--hpx:ini=hpx.threadpools.io_pool_size=1",
               "--hpx:ini=hpx.threadpools.timer_pool_size=1", "--hpx:ini=hpx.threadpools.parcel_pool_size=1",
               "--hpx:ini=hpx.parcel.tcp.enable=0", "--problem=MOVING_STAR", "--eos=IDEAL",
               "--gravity=on", "--hydro=on", "--n_species=2", "--omega=0", "--xscale=1",
               "--code_to_g=1", "--code_to_cm=1", "--code_to_s=1", "--inflow_bc=on",
               "--reflect_bc=off", "--unigrid=off", "--rotating_star_amr=off",
               f"--min_level={args.min_level}", f"--max_level={args.max_level}",
               f"--refinement_floor={args.refinement_floor}", f"--cfl={args.cfl}",
               f"--hard_dt={args.hard_dt}", f"--stop_time={args.stop_time}",
               f"--star_rho_out={args.star_rho_out}", f"--star_egas_out={args.star_egas_out}",
               f"--dual_energy_sw1={args.dual_energy_switch}", "--disable_output=off",
               "--disable_analytic=on", "--disable_diagnostics=off", "--odt=0.001",
               "--silo_num_groups=1", "--number_gpus=0", "--executors_per_gpu=0",
               "--hydro_host_kernel_type=LEGACY", "--hydro_device_kernel_type=OFF",
               "--multipole_host_kernel_type=LEGACY", "--multipole_device_kernel_type=OFF",
               "--monopole_host_kernel_type=LEGACY", "--monopole_device_kernel_type=OFF"]
    for axis, center, velocity in zip("xyz", args.center, args.velocity):
        command.extend((f"--star_{axis}center={center}", f"--moving_star_{axis}velocity={velocity}"))
    if integrator is not None:
        command.append(f"--time_integrator={integrator}")
    if restart is not None:
        command.append(f"--restart_filename={restart}")
    return command


def run_binary(args, executable, directory, integrator, restart=None):
    require(not directory.exists(), f"Refusing to overwrite run directory {directory}")
    directory.mkdir(parents=True)
    if restart is not None:
        destination = directory / "checkpoint.silo"
        shutil.copy2(restart, destination)
        shutil.copytree(Path(str(restart) + ".data"), Path(str(destination) + ".data"))
        restart = destination
    command = make_command(args, executable, integrator, restart)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with (directory / "run.log").open("w") as log:
        completed = subprocess.run(command, cwd=directory, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
    (directory / "execution.json").write_text(json.dumps({"returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started, "executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}, indent=2) + "\n")
    require(completed.returncode == 0, f"Octo-Tiger failed; inspect {directory / 'run.log'}")
    require((directory / "final.silo").is_file(), f"No final checkpoint in {directory}")


def restart_check(args, reader, original):
    choices = [change for change in original["topology_changes"] if change["step"] > 0 and any(
        later["step"] > change["step"] for later in original["topology_changes"])]
    require(choices, "Restart needs a post-change checkpoint before another observed topology change")
    change = choices[0]
    checkpoint = Path(change["checkpoint"])
    directory = Path(original["directory"]) / "resumed"
    if not args.analyze_only:
        # Omit time_integrator so the checkpoint must restore SSP_MULTISTEP.
        run_binary(args, args.executable, directory, None, checkpoint)
    stages = stage_sequence(directory)
    require(stages == original["stages"][change["step"]:], "Restart changed stage/ratio sequence")
    require(stages and all(stage == 2 for stage, _ in stages), "Restart inserted RK3 after transferred history")
    nodes, size = compare_checkpoints(reader, Path(original["directory"]) / "final.silo",
                                     directory / "final.silo", expected_nodes=None)
    resumed_final = analyze_snapshot(reader, directory / "final.silo", args.dual_energy_switch)
    require(resumed_final["leaf_ids"] == original["final"]["leaf_ids"], "Restart changed final topology")
    later = [item for item in original["topology_changes"] if item["step"] > change["step"]]
    return {"checkpoint": str(checkpoint), "checkpoint_step": change["step"],
            "checkpoint_topology_change": change, "later_topology_changes": later,
            "resumed_directory": str(directory), "exact_node_archives": nodes,
            "exact_payload_bytes": size, "resumed_rk3_steps": 0, "resumed_steps": len(stages)}


def energy_comparisons(runs):
    transfer = next((item for item in runs if item["mode"] == "transfer"), None)
    if transfer is None:
        return []
    candidate = transfer["conservation"]["mesh_fields"]["total_energy"]["mesh_plus_outflow"]
    result = []
    for control in runs:
        if control["mode"] == "transfer":
            continue
        reference = control["conservation"]["mesh_fields"]["total_energy"]["mesh_plus_outflow"]
        require(math.isclose(candidate["initial"], reference["initial"], rel_tol=1e-12, abs_tol=1e-14),
                "Control initial energy differs; runs do not share an initial problem")
        denominator = reference["maximum_absolute_drift"]
        result.append({"control": control["mode"],
                       "transfer_maximum_absolute_drift": candidate["maximum_absolute_drift"],
                       "control_maximum_absolute_drift": denominator,
                       "maximum_drift_ratio": candidate["maximum_absolute_drift"] / denominator if denominator else None,
                       "final_ledger_difference": candidate["final"] - reference["final"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--baseline-executable", type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir", type=Path)
    output.add_argument("--work-directory", type=Path,
                        help="create a fresh run directory below this path (repeatable CTest runs)")
    parser.add_argument("--modes", nargs="+", choices=("transfer", "baseline", "rk3"))
    parser.add_argument("--min-level", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=2)
    parser.add_argument("--center", nargs=3, type=float, default=[-0.3, -0.3, -0.3])
    parser.add_argument("--velocity", nargs=3, type=float, default=[1, 1, 1])
    parser.add_argument("--stop-time", type=float, default=0.8)
    parser.add_argument("--hard-dt", type=float, default=0.005)
    parser.add_argument("--star-rho-out", type=float, default=0.01)
    parser.add_argument("--star-egas-out", type=float, default=0.01)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--refinement-floor", type=float, default=0.05)
    parser.add_argument("--dual-energy-switch", type=float, default=0.001)
    parser.add_argument("--mass-tolerance", type=float, default=1e-10,
                        help="maximum mass/species ledger drift divided by initial total mass")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--silo-library", type=Path)
    parser.add_argument("--skip-restart", action="store_true")
    parser.add_argument("--allow-no-topology-change", action="store_true", help="exploratory runs only")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    require(0 <= args.min_level < args.max_level, "Require 0 <= min-level < max-level")
    require(args.stop_time > 0 and args.hard_dt > 0 and args.cfl > 0, "Timesteps and CFL must be positive")
    require(args.star_rho_out > 0 and args.star_egas_out > 0, "Atmosphere density and thermal energy must be positive")
    args.executable = args.executable.resolve()
    if args.work_directory is not None:
        if args.analyze_only:
            parser.error("--analyze-only requires --output-dir pointing to existing results")
        work_directory = args.work_directory.resolve()
        work_directory.mkdir(parents=True, exist_ok=True)
        args.output_dir = Path(tempfile.mkdtemp(prefix="ssp-amr-gravity-", dir=work_directory))
    else:
        args.output_dir = args.output_dir.resolve()
    if args.baseline_executable:
        args.baseline_executable = args.baseline_executable.resolve()
    args.modes = args.modes or (["transfer", "baseline", "rk3"] if args.baseline_executable else ["transfer", "rk3"])
    if "baseline" in args.modes and args.baseline_executable is None:
        parser.error("baseline mode requires --baseline-executable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = GravityMeshReader(args.silo_library)
    keys = ("min_level", "max_level", "center", "velocity", "stop_time", "hard_dt", "cfl",
            "refinement_floor", "dual_energy_switch", "mass_tolerance", "star_rho_out", "star_egas_out")
    report = {"problem": "MOVING_STAR", "validation_scope": "hydro/gravity coupling, AMR topology, conservation, positivity and exact restart; no analytic convergence claim",
              "settings": {key: getattr(args, key) for key in keys}, "runs": [], "restart": None,
              "energy_note": "Total energy includes half the potential-energy density and cumulative boundary outflow. Drift comparisons are reported, not an analytic energy-accuracy pass criterion."}
    for mode in args.modes:
        directory = args.output_dir / mode
        if not args.analyze_only:
            executable = args.baseline_executable if mode == "baseline" else args.executable
            run_binary(args, executable, directory, "RK3" if mode == "rk3" else "SSP_MULTISTEP")
        result = analyze_run(args, reader, directory, mode)
        report["runs"].append(result)
        report["energy_comparisons"] = energy_comparisons(report["runs"])
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{mode}: steps={result['final']['step']}, RK3={result['rk3_steps']}, "
              f"refined={result['refined_parent_count']}, coarsened={result['coarsened_parent_count']}, "
              f"min_rho={result['minima']['density']:.6e}, min_p={result['minima']['pressure']:.6e}, "
              f"max_energy_drift={result['conservation']['fields']['total_energy']['maximum_absolute_drift']:.6e}", flush=True)
    if not args.skip_restart and "transfer" in args.modes:
        original = next(item for item in report["runs"] if item["mode"] == "transfer")
        report["restart"] = restart_check(args, reader, original)
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Exact post-AMR restart passed: {report['restart']['exact_node_archives']} node archives", flush=True)
    print(f"Results: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
