#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Run/analyze the moving AMR_ADVECTION problem with real Octo-Tiger binaries.

Each run uses one HPX worker and CPU kernels. NumPy is used only for analysis.
The binary must contain the dedicated AMR_ADVECTION initializer. Baseline means
an otherwise matching SSP binary with regrid RK3 restart retained. Contact
steepening defaults off for this smooth-profile study; use on to reproduce the
production reconstruction default.
"""

import argparse
import ctypes as C
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

from silo_snapshot import SiloSnapshot, compare_checkpoints


class QuadMeshPrefix(C.Structure):
    # Public Silo DBquadmesh fields through dims (silo.h). C owns the full object.
    _fields_ = [("id", C.c_int), ("block_no", C.c_int), ("group_no", C.c_int),
                ("name", C.c_char_p), ("cycle", C.c_int), ("coord_sys", C.c_int),
                ("major_order", C.c_int), ("stride", C.c_int * 3),
                ("coordtype", C.c_int), ("facetype", C.c_int), ("planar", C.c_int),
                ("coords", C.c_void_p * 3), ("datatype", C.c_int), ("time", C.c_float),
                ("dtime", C.c_double), ("min_extents", C.c_float * 6),
                ("max_extents", C.c_float * 6), ("labels", C.c_char_p * 3),
                ("units", C.c_char_p * 3), ("ndims", C.c_int), ("nspace", C.c_int),
                ("nnodes", C.c_int), ("dims", C.c_int * 3)]


class QuadVarPrefix(C.Structure):
    _fields_ = [("id", C.c_int), ("name", C.c_char_p), ("units", C.c_char_p),
                ("label", C.c_char_p), ("cycle", C.c_int), ("meshid", C.c_int),
                ("vals", C.POINTER(C.c_void_p)), ("datatype", C.c_int),
                ("nels", C.c_int), ("nvals", C.c_int), ("ndims", C.c_int),
                ("dims", C.c_int * 3)]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def decode_node(node):
    xyz = [0, 0, 0]
    level = 0
    while node != 1:
        require(node > 1, "Invalid node id")
        for axis in (2, 1, 0):
            xyz[axis] = 2 * xyz[axis] + (node & 1)
            node >>= 1
        level += 1
    return level, xyz


def encode_node(level, xyz):
    node = 1
    for bit in range(level):
        for axis in range(3):
            node = 2 * node + ((xyz[axis] >> bit) & 1)
    return node


def parent_node(node):
    level, xyz = decode_node(node)
    return None if level == 0 else encode_node(level - 1, [value // 2 for value in xyz])


def ancestors(nodes):
    result = set()
    for node in nodes:
        node = parent_node(node)
        while node is not None:
            result.add(node)
            node = parent_node(node)
    return result


def leaf_nodes(nodes):
    return set(nodes) - ancestors(nodes)


class MeshReader(SiloSnapshot):
    def __init__(self, library=None):
        super().__init__(library)
        signatures = {
            "DBGetQuadmesh": ([C.c_void_p, C.c_char_p], C.POINTER(QuadMeshPrefix)),
            "DBFreeQuadmesh": ([C.c_void_p], None),
            "DBGetQuadvar": ([C.c_void_p, C.c_char_p], C.POINTER(QuadVarPrefix)),
            "DBFreeQuadvar": ([C.c_void_p], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.silo, name)
            function.argtypes = arguments
            function.restype = result

    @staticmethod
    def reals(variable):
        datatype, dimensions, contents = variable
        require(datatype in (19, 20), "Expected floating-point Silo variable")
        values = np.frombuffer(contents, dtype=np.float32 if datatype == 19 else np.float64)
        require(values.size == math.prod(dimensions), "Invalid Silo real dimensions")
        return values.copy()

    def inventory(self, path):
        values = self.read(path, ("node_list", "cgs_time", "step_count", "leaf_count"))
        nodes = self.integers(values["node_list"])
        leaves = leaf_nodes(nodes)
        require(len(leaves) == self.integers(values["leaf_count"])[0], "Decoded leaf inventory mismatch")
        return {"path": str(Path(path).resolve()), "time": float(self.reals(values["cgs_time"])[0]),
                "step": int(self.integers(values["step_count"])[0]),
                "node_count": len(nodes), "leaf_count": len(leaves), "leaf_ids": sorted(leaves)}

    @staticmethod
    def array(pointer, datatype, count):
        require(datatype in (19, 20), "Unsupported Silo mesh datatype")
        scalar = C.c_float if datatype == 19 else C.c_double
        return np.ctypeslib.as_array(C.cast(pointer, C.POINTER(scalar)), shape=(count,)).copy()

    def meshes(self, path, field_names=("rho_1", "sx", "sy", "sz", "egas", "tau")):
        inventory = self.inventory(path)
        # Harness runs one locality and explicitly requests one output group.
        data_path = Path(path).parent / (Path(path).name + ".data") / "0.silo"
        handle = self.silo.DBOpenReal(str(data_path.resolve()).encode(), 5, 1)
        require(bool(handle), f"Cannot open {data_path}")
        try:
            for node in inventory["leaf_ids"]:
                prefix = f"{node:o}/"
                mesh_pointer = self.silo.DBGetQuadmesh(handle, (prefix + "quadmesh").encode())
                require(bool(mesh_pointer), f"Missing leaf mesh {node:o}")
                try:
                    mesh = mesh_pointer.contents
                    require(mesh.ndims == 3, "Expected three-dimensional mesh")
                    coordinates = [self.array(mesh.coords[axis], mesh.datatype, mesh.dims[axis])
                                   for axis in range(3)]
                finally:
                    self.silo.DBFreeQuadmesh(mesh_pointer)
                fields = {}
                for name in field_names:
                    pointer = self.silo.DBGetQuadvar(handle, (prefix + name).encode())
                    require(bool(pointer), f"Missing {prefix + name}")
                    try:
                        variable = pointer.contents
                        require(variable.ndims == 3 and variable.nvals == 1, "Unexpected field shape")
                        dims = tuple(variable.dims)
                        require(dims == tuple(len(axis) - 1 for axis in coordinates), "Field/mesh dimension mismatch")
                        fields[name] = self.array(variable.vals[0], variable.datatype, variable.nels).reshape(dims[::-1])
                    finally:
                        self.silo.DBFreeQuadvar(pointer)
                yield node, coordinates, fields
        finally:
            self.silo.DBClose(handle)


def bump_primitive(value):
    value = np.clip(value, -1, 1)
    square = value * value
    return value * (1 + square * (-4 / 3 + square * (6 / 5 + square * (-4 / 7 + square / 9))))


def bump_average(edges, center):
    return 0.25 / np.diff(edges) * np.diff(bump_primitive((edges - center) / 0.25))


def analyze_snapshot(reader, path):
    result = reader.inventory(path)
    total_volume = density_l1 = energy_l1 = density_linf = energy_linf = 0.0
    density_min = thermal_min = math.inf
    mass = total_energy = density_excess_mass = exact_density_excess_mass = 0.0
    cells = 0
    for _, xyz, fields in reader.meshes(path):
        x, y, z = xyz
        weights = np.diff(z)[:, None, None] * np.diff(y)[None, :, None] * np.diff(x)[None, None, :]
        density_exact = (1 + 0.2 * bump_average(z, 0)[:, None, None]
                         * bump_average(y, 0)[None, :, None]
                         * bump_average(x, -0.4 + result["time"])[None, None, :])
        density = fields["rho_1"]
        energy = fields["egas"]
        for name, values in fields.items():
            require(np.all(np.isfinite(values)), f"Nonfinite {name} in {path}")
        require(np.min(density) > 0, f"Nonpositive density in {path}")
        thermal = energy - 0.5 * (fields["sx"]**2 + fields["sy"]**2 + fields["sz"]**2) / density
        density_min = min(density_min, float(np.min(density)))
        thermal_min = min(thermal_min, float(np.min(thermal)))
        error = abs(density - density_exact)
        energy_error = abs(energy - (0.015 + 0.5 * density_exact))
        density_l1 += float(np.sum(weights * error))
        energy_l1 += float(np.sum(weights * energy_error))
        density_linf = max(density_linf, float(np.max(error)))
        energy_linf = max(energy_linf, float(np.max(energy_error)))
        total_volume += float(np.sum(weights))
        mass += float(np.sum(weights * density))
        total_energy += float(np.sum(weights * energy))
        density_excess_mass += float(np.sum(weights * (density - 1.0)))
        exact_density_excess_mass += float(np.sum(weights * (density_exact - 1.0)))
        cells += density.size
    require(math.isclose(total_volume, 8, rel_tol=1e-12), f"Leaf meshes do not cover volume 8: {total_volume}")
    require(thermal_min > 0, f"Nonpositive conservative thermal energy in {path}")
    result.update(cells=cells, density_l1=density_l1 / total_volume, density_linf=density_linf,
                  energy_l1=energy_l1 / total_volume, energy_linf=energy_linf,
                  density_min=density_min, thermal_min=thermal_min, mass=mass, energy=total_energy,
                  density_excess_mass=density_excess_mass, exact_density_excess_mass=exact_density_excess_mass,
                  density_excess_mass_error=density_excess_mass - exact_density_excess_mass)
    return result


def mesh_conservation(time_series, sums):
    # Reconstruct the ledger from full-precision Silo cell integrals and the
    # small cumulative-outflow columns, avoiding rounded large totals in sums.dat.
    # Numerical tails can leave the domain although the exact bump is interior;
    # retain raw mesh drift, but apply the conservation gate to the open ledger.
    relative_time_tolerance = 1e-11
    absolute_time_tolerance = 1e-12
    matched = []
    for snapshot in time_series:
        row = sums[int(np.argmin(abs(sums[:, 0] - snapshot["time"])))]
        require(math.isclose(float(row[0]), snapshot["time"], rel_tol=relative_time_tolerance,
                             abs_tol=absolute_time_tolerance),
                f"No conservation row matching snapshot time {snapshot['time']}")
        matched.append({"step": snapshot["step"], "time": snapshot["time"],
                        "diagnostic_time": float(row[0]), "mass_outflow": float(row[2]),
                        "energy_outflow": float(row[4])})
    result = {"normalization": "max(1, abs(initial value))",
              "timestamp_relative_tolerance": relative_time_tolerance,
              "timestamp_absolute_tolerance": absolute_time_tolerance,
              "ledger_tolerance": 1e-10, "matched_diagnostics": matched, "fields": {}}
    for name in ("mass", "energy"):
        raw = np.array([snapshot[name] for snapshot in time_series])
        outflow = np.array([row[name + "_outflow"] for row in matched])
        ledger = raw + outflow
        quantities = {}
        for description, values in (("mesh", raw), ("mesh_plus_outflow", ledger)):
            absolute = float(np.max(abs(values - values[0])))
            quantities[description] = {"initial": float(values[0]), "final": float(values[-1]),
                                       "signed_final_drift": float(values[-1] - values[0]),
                                       "maximum_absolute_drift": absolute,
                                       "maximum_normalized_drift": absolute / max(1, abs(float(values[0])))}
        require(quantities["mesh_plus_outflow"]["maximum_normalized_drift"] <= result["ledger_tolerance"],
                f"Silo mesh {name} plus matched boundary outflow is not conserved: {quantities['mesh_plus_outflow']}")
        result["fields"][name] = quantities
    return result


def analyze_run(reader, directory, mode, require_topology=True, require_post_change_steps=False):
    log = (directory / "run.log").read_text()
    stages = [(int(stage), float(alpha)) for stage, alpha in
              re.findall(r"SSP_MULTISTEP stages=(\d+) alpha=([^\s]+)", log)]
    paths = list(directory.glob("X.*.silo")) + [directory / "final.silo"]
    snapshots = sorted((reader.inventory(path) for path in paths), key=lambda item: (item["step"], item["path"]))
    transitions = []
    for old, new in zip(snapshots, snapshots[1:]):
        old_leaves, new_leaves = set(old["leaf_ids"]), set(new["leaf_ids"])
        refined = sorted(old_leaves & ancestors(new_leaves))
        coarsened = sorted(new_leaves & ancestors(old_leaves))
        if old_leaves != new_leaves:
            transitions.append({"time": new["time"], "step": new["step"], "checkpoint": new["path"],
                                "added_leaf_ids": sorted(new_leaves - old_leaves),
                                "removed_leaf_ids": sorted(old_leaves - new_leaves),
                                "refined_parent_ids": refined, "coarsened_parent_ids": coarsened})
    if require_topology:
        require(any(item["refined_parent_ids"] for item in transitions), f"No actual refinement in {directory}")
        require(any(item["coarsened_parent_ids"] for item in transitions), f"No actual coarsening in {directory}")
        if require_post_change_steps:
            final_step = next(item["step"] for item in snapshots
                              if item["path"] == str((directory / "final.silo").resolve()))
            for direction in ("refined_parent_ids", "coarsened_parent_ids"):
                require(any(item[direction] and item["step"] < final_step for item in transitions),
                        f"No {direction} event followed by an accepted step in {directory}")
    if mode == "transfer":
        require(stages and stages[0][0] == 3, f"Missing initial RK3 in {directory}")
        require(sum(stage == 3 for stage, _ in stages) == 1, f"Expected only one RK3 step in {directory}")
        require(all(stage == 2 for stage, _ in stages[1:]), f"Unexpected later stage count in {directory}")
    elif mode == "baseline":
        require(sum(stage == 3 for stage, _ in stages) > 1, "Baseline did not exercise RK3 rebootstrap")
    else:
        require(not stages, "RK3 control unexpectedly used the SSP multistep path")
    sums = np.atleast_2d(np.loadtxt(directory / "sums.dat"))
    require(sums.shape[1] == 26, "Expected time + 22 field/outflow columns + 3 angular diagnostics")
    require(np.all(np.isfinite(sums)), "Nonfinite conservation diagnostics")
    drift = {}
    for name, field in (("mass", 0), ("energy", 1), ("species", 10)):
        values = sums[:, 1 + 2 * field]
        drift[name] = float(np.max(abs(values - values[0])) / max(1, abs(values[0])))
        require(drift[name] < 1e-10, f"{name} plus boundary outflow drift: {drift[name]}")
    final = analyze_snapshot(reader, directory / "final.silo")
    if mode in ("transfer", "baseline"):
        require(len(stages) == final["step"], "Stage log does not cover every accepted step")
    # Positivity is also checked at every saved physical time, not just at final.
    minima = [analyze_snapshot(reader, item["path"]) for item in snapshots if item["path"] != final["path"]]
    error_keys = ("path", "time", "step", "cells", "density_l1", "density_linf",
                  "energy_l1", "energy_linf", "density_min", "thermal_min", "mass", "energy",
                  "density_excess_mass", "exact_density_excess_mass", "density_excess_mass_error")
    time_series = sorted(minima + [final], key=lambda item: (item["step"], item["path"]))
    return {"directory": str(directory.resolve()), "mode": mode, "final": final,
            "snapshot_errors": [{key: item[key] for key in error_keys} for item in time_series],
            "maximum_density_l1": max(item["density_l1"] for item in time_series),
            "maximum_density_linf": max(item["density_linf"] for item in time_series),
            "maximum_energy_l1": max(item["energy_l1"] for item in time_series),
            "maximum_energy_linf": max(item["energy_linf"] for item in time_series),
            "conservation_drift": drift, "mesh_conservation": mesh_conservation(time_series, sums),
            "rk3_steps": sum(stage == 3 for stage, _ in stages) if mode != "rk3" else final["step"],
            "multistep_steps": sum(stage == 2 for stage, _ in stages), "stages": stages,
            "snapshots": snapshots, "topology_changes": transitions,
            "minimum_density": min([final["density_min"]] + [item["density_min"] for item in minima]),
            "minimum_thermal_energy": min([final["thermal_min"]] + [item["thermal_min"] for item in minima])}


def run_binary(args, executable, directory, minimum_level, maximum_level, integrator, restart=None,
               *, uniform_grid=False, output_dt=0.001):
    require(not directory.exists(), f"Refusing to overwrite run directory {directory}")
    directory.mkdir(parents=True)
    command = [str(executable), "--hpx:threads=1", "--hpx:ini=hpx.threadpools.io_pool_size=1",
               "--hpx:ini=hpx.threadpools.timer_pool_size=1", "--hpx:ini=hpx.threadpools.parcel_pool_size=1",
               "--hpx:ini=hpx.parcel.tcp.enable=0", "--problem=AMR_ADVECTION", "--eos=IDEAL",
               "--gravity=off", "--hydro=on", "--n_species=1", "--omega=0", "--xscale=1",
               "--code_to_g=1", "--code_to_cm=1", "--code_to_s=1", "--inflow_bc=on",
               "--reflect_bc=off", f"--unigrid={'on' if uniform_grid else 'off'}", f"--min_level={minimum_level}",
               f"--max_level={maximum_level}", f"--refinement_floor={args.refinement_floor}",
               f"--cfl={args.cfl}", f"--stop_time={args.stop_time}",
               f"--cdisc_detect={args.contact_steepening}", "--disable_output=off",
               "--disable_analytic=on", "--disable_diagnostics=off", f"--odt={output_dt}",
               "--silo_num_groups=1", "--number_gpus=0", "--executors_per_gpu=0",
               "--hydro_host_kernel_type=LEGACY", "--hydro_device_kernel_type=OFF",
               "--multipole_host_kernel_type=LEGACY", "--multipole_device_kernel_type=OFF",
               "--monopole_host_kernel_type=LEGACY", "--monopole_device_kernel_type=OFF"]
    if integrator is not None:
        command.append(f"--time_integrator={integrator}")
    if restart is not None:
        destination = directory / "checkpoint.silo"
        shutil.copy2(restart, destination)
        shutil.copytree(Path(str(restart) + ".data"), Path(str(destination) + ".data"))
        command.append(f"--restart_filename={destination}")
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


def restart_check(args, reader, original, minimum_level, maximum_level):
    final_step = original["final"]["step"]
    choices = [item for item in original["topology_changes"] if 0 < item["step"] < final_step - 2 and any(
        later["step"] > item["step"] for later in original["topology_changes"])]
    require(choices, "Restart needs a post-change checkpoint before another observed topology change")
    change = choices[0]
    checkpoint = Path(change["checkpoint"])
    directory = Path(original["directory"]) / "resumed"
    if not args.analyze_only:
        run_binary(args, args.executable, directory, minimum_level, maximum_level, None, checkpoint)
    log = (directory / "run.log").read_text()
    stages = [(int(stage), float(alpha)) for stage, alpha in
              re.findall(r"SSP_MULTISTEP stages=(\d+) alpha=([^\s]+)", log)]
    require(stages == original["stages"][change["step"]:], "Restart stage/ratio sequence changed")
    require(all(stage == 2 for stage, _ in stages), "Restart inserted RK3 after transferred history")
    nodes, size = compare_checkpoints(reader, Path(original["directory"]) / "final.silo",
                                     directory / "final.silo", expected_nodes=None)
    resumed_final = analyze_snapshot(reader, directory / "final.silo")
    require(resumed_final["leaf_ids"] == original["final"]["leaf_ids"], "Restart final topology changed")
    return {"checkpoint": str(checkpoint), "checkpoint_step": change["step"],
            "checkpoint_topology_change": change, "resumed_directory": str(directory),
            "exact_node_archives": nodes, "exact_payload_bytes": size,
            "resumed_rk3_steps": 0, "resumed_steps": len(stages)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--baseline-executable", type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir", type=Path)
    output.add_argument("--work-directory", type=Path,
                        help="create a fresh run directory below this path (repeatable CTest runs)")
    parser.add_argument("--levels", nargs="+", default=["0:2", "1:3", "2:4"])
    parser.add_argument("--modes", nargs="+", choices=("transfer", "baseline", "rk3"))
    parser.add_argument("--stop-time", type=float, default=0.8)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--refinement-floor", type=float, default=1.005)
    parser.add_argument("--contact-steepening", choices=("on", "off"), default="off",
                        help="PPM contact steepening (off for smooth convergence; on reproduces production default)")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--silo-library", type=Path)
    parser.add_argument("--skip-restart", action="store_true")
    parser.add_argument("--allow-no-topology-change", action="store_true", help="exploratory runs only")
    parser.add_argument("--require-post-change-steps", action="store_true",
                        help="require a subsequent accepted step after both refinement and coarsening")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.executable = args.executable.resolve()
    if args.work_directory is not None:
        if args.analyze_only:
            parser.error("--analyze-only requires --output-dir pointing to existing results")
        work_directory = args.work_directory.resolve()
        work_directory.mkdir(parents=True, exist_ok=True)
        args.output_dir = Path(tempfile.mkdtemp(prefix="ssp-amr-hydro-", dir=work_directory))
    else:
        args.output_dir = args.output_dir.resolve()
    if args.baseline_executable:
        args.baseline_executable = args.baseline_executable.resolve()
    args.modes = args.modes or (["transfer", "baseline", "rk3"] if args.baseline_executable else ["transfer", "rk3"])
    if "baseline" in args.modes and args.baseline_executable is None:
        parser.error("baseline mode requires --baseline-executable")
    try:
        levels = [tuple(map(int, value.split(":"))) for value in args.levels]
        require(all(len(pair) == 2 and 0 <= pair[0] < pair[1] for pair in levels), "Invalid levels")
    except (ValueError, AssertionError):
        parser.error("--levels entries must have the form MIN:MAX with 0 <= MIN < MAX")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = MeshReader(args.silo_library)
    report = {"problem": "AMR_ADVECTION", "settings": {"levels": levels, "stop_time": args.stop_time,
              "cfl": args.cfl, "refinement_floor": args.refinement_floor,
              "contact_steepening": args.contact_steepening,
              "require_post_change_steps": args.require_post_change_steps,
              "require_topology": not args.allow_no_topology_change}, "runs": [], "restart": None}
    for minimum_level, maximum_level in levels:
        for mode in args.modes:
            directory = args.output_dir / f"level-{minimum_level}-{maximum_level}" / mode
            if not args.analyze_only:
                executable = args.baseline_executable if mode == "baseline" else args.executable
                run_binary(args, executable, directory, minimum_level, maximum_level,
                           "RK3" if mode == "rk3" else "SSP_MULTISTEP")
            if args.analyze_only:
                saved_command = json.loads((directory / "command.json").read_text())
                saved_steepening = next((option.split("=", 1)[1] for option in saved_command
                                         if option.startswith("--cdisc_detect=")), "on")
                require(saved_steepening == args.contact_steepening,
                        f"Saved run used contact steepening {saved_steepening}; pass --contact-steepening {saved_steepening}")
            result = analyze_run(reader, directory, mode, not args.allow_no_topology_change,
                                 args.require_post_change_steps)
            require(math.isclose(result["final"]["time"], args.stop_time, rel_tol=1e-11, abs_tol=1e-12),
                    f"Run did not reach stop_time: {directory}")
            result.update(minimum_level=minimum_level, maximum_level=maximum_level)
            prior = [item for item in report["runs"] if item["mode"] == mode]
            result["density_l1_order"] = None
            result["maximum_density_l1_order"] = None
            if prior:
                refinement_ratio = 2 ** (maximum_level - prior[-1]["maximum_level"])
                if refinement_ratio > 1:
                    result["density_l1_order"] = math.log(prior[-1]["final"]["density_l1"] / result["final"]["density_l1"]) / math.log(refinement_ratio)
                    result["maximum_density_l1_order"] = math.log(prior[-1]["maximum_density_l1"] / result["maximum_density_l1"]) / math.log(refinement_ratio)
            report["runs"].append(result)
            (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{mode} levels {minimum_level}:{maximum_level}: L1={result['final']['density_l1']:.6e}, "
                  f"Linf={result['final']['density_linf']:.6e}, order={result['density_l1_order']}, "
                  f"peak_L1={result['maximum_density_l1']:.6e}, peak_order={result['maximum_density_l1_order']}, "
                  f"topology_changes={len(result['topology_changes'])}, RK3={result['rk3_steps']}", flush=True)
        if report["restart"] is None and not args.skip_restart and "transfer" in args.modes:
            original = next(item for item in report["runs"] if item["mode"] == "transfer")
            report["restart"] = restart_check(args, reader, original, minimum_level, maximum_level)
            (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Exact post-regrid restart passed: {report['restart']['exact_node_archives']} node archives", flush=True)
    print(f"Results: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
