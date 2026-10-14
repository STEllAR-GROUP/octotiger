#!/usr/bin/env python3
"""Exercise Euler radiation substeps inside longer hydrodynamic timesteps.

Periodic coupled waves on uniform levels zero and one check full-state restart,
physical states, and total energy/momentum conservation. The level-one case
includes parent restriction and inter-block ghost exchanges, but is not a
mixed-level AMR convergence test. A homogeneous source problem independently
compares subcycling with the identical radiation-CFL-safe Euler steps taken as
global timesteps; its hydro flux divergence vanishes. No order claim is made.

Uses the installed Silo library through test_restart's public C API reader.
Runs are sequential, each with at most two HPX workers.
"""

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from application_helpers import conservation, environment, fingerprint, require, save_report
from test_restart import CGS_C, Silo

FIELDS = ("rho_1", "egas", "tau", "sx", "sy", "sz", "er", "fx", "fy", "fz")
METADATA = ("code_to_g", "code_to_cm", "code_to_s", "n_species", "problem", "hydro",
            "radiation", "gravity", "periodic", "rad_implicit", "radiation_integrator",
            "radiation_model", "radiation_algorithm_version", "radiation_flux_semantics", "radiation_reduced_c_factor")
GAMMA = 5 / 3


def snapshot(reader, path, level, expected_time, problem):
    metadata, blocks = reader.snapshot(path, fields=FIELDS, metadata_names=METADATA,
                                        block_count=8 ** level)
    require(math.isclose(metadata["time"], expected_time, rel_tol=2e-12, abs_tol=2e-14),
            f"Unsynchronized checkpoint time in {path}: {metadata['time']} vs {expected_time}")
    require(metadata["radiation_integrator"] == metadata["radiation_model"] ==
            metadata["radiation_flux_semantics"] == 0 and metadata["radiation_algorithm_version"] == 1,
            f"Expected the original Euler integrator in {path}")
    require(metadata["hydro"] == metadata["radiation"] == metadata["rad_implicit"] ==
            metadata["periodic"] == metadata["n_species"] == 1,
            "Expected periodic coupled one-species material")
    require(metadata["gravity"] == 0 and metadata["radiation_reduced_c_factor"] == 1,
            "Expected full physical light speed without gravity")
    require(metadata["problem"] == problem, "Unexpected initial-condition problem")
    mass, length, clock = (metadata[key] for key in ("code_to_g", "code_to_cm", "code_to_s"))
    energy_unit = mass / (length * clock ** 2)
    units = {"rho_1": mass / length ** 3, "egas": energy_unit, "er": energy_unit,
             "tau": energy_unit ** (1 / GAMMA)}
    units.update({field: mass / (length ** 2 * clock) for field in ("sx", "sy", "sz")})
    units.update({field: energy_unit * CGS_C for field in ("fx", "fy", "fz")})
    state, inventory = {}, None
    for field in FIELDS:
        leaves = {key.rsplit("/", 1)[0]: values for key, values in blocks[field].items()}
        current_inventory = [(key, len(leaves[key])) for key in sorted(leaves)]
        if inventory is None:
            inventory = current_inventory
        require(current_inventory == inventory, "Different leaf layouts between fields")
        state[field] = [value / units[field] for key in sorted(leaves) for value in leaves[key]]
    state["rho"] = state.pop("rho_1")
    require(len(state["rho"]) == (8 * 2 ** level) ** 3, "This regression requires INX=8")
    for index, rho in enumerate(state["rho"]):
        require(rho > 0 and state["er"][index] > 0 and state["tau"][index] > 0,
                f"Nonpositive state in cell {index}")
        kinetic = math.fsum(state[field][index] ** 2 for field in ("sx", "sy", "sz")) / (2 * rho)
        require(state["egas"][index] - kinetic > 0, f"Nonpositive gas internal energy in cell {index}")
        require(math.hypot(*(state[field][index] for field in ("fx", "fy", "fz"))) <=
                state["er"][index] * (1 + 3e-11), f"Unrealizable radiation in cell {index}")
    return metadata, state, inventory


def compare(actual, reference, tolerance, description):
    results = {}
    for field in reference:
        require(len(actual[field]) == len(reference[field]), "Mismatched comparison meshes")
        scale = max(abs(value) for value in reference[field])
        if field in ("fx", "fy", "fz"):
            scale = max(scale, max(reference["er"]))
        elif field in ("sx", "sy", "sz"):
            scale = max(scale, max(reference["rho"]))  # rho*c_code, with c_code=1
        maximum = max(abs(a-b) for a, b in zip(actual[field], reference[field]))
        relative = maximum / max(scale, 1e-300)
        results[field] = {"linf": maximum, "normalization": scale, "normalized_linf": relative}
        require(relative < tolerance, f"{description}: {field} error {relative} exceeds {tolerance}")
    return results


def run(args, reader, root, label, level, dt, stop, problem, restart=None, start=0):
    folder = root / label
    folder.mkdir()
    steps = round(stop / dt)
    config = {
        "problem": problem, "radiation": "on", "hydro": "on", "gravity": "off",
        "radiation_integrator": "euler", "radiation_backend": "cpu", "radiation_reduced_c_factor": 1,
        "rad_implicit": "on", "periodic": "on", "omega": 0, "unigrid": "on",
        "min_level": level, "max_level": level, "xscale": 1, "n_species": 1,
        "atomic_mass": 1, "atomic_number": 1, "eos": "IDEAL", "correct_am_hydro": "off",
        "code_to_g": 1e30, "code_to_cm": CGS_C, "code_to_s": 1,
        "rad_test_chi": 1, "rad_test_amplitude": .01,
        "cfl": .25, "hard_dt": dt, "stop_time": stop, "stop_step": steps - 1,
        "odt": "1e99", "disable_output": "off", "disable_analytic": "on", "disable_diagnostics": "on",
        "hydro_host_kernel_type": "LEGACY", "hydro_device_kernel_type": "OFF",
        "monopole_device_kernel_type": "OFF", "multipole_device_kernel_type": "OFF",
        "silo_num_groups": 1, "datadir": str(folder) + os.sep,
    }
    if restart:
        config["restart_filename"] = str(restart)
    path = folder / "run.ini"
    path.write_text("".join(f"{name}={value}\n" for name, value in config.items()))
    command = [str(args.executable), f"--config_file={path}", f"--hpx:threads={args.threads}"]
    print(f"Running {label}: level={level}, hydro dt={dt}, [{start},{stop}]", flush=True)
    started = time.monotonic()
    with (folder / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=restart.parent if restart else folder,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 env=environment(None), timeout=args.timeout)
    output = (folder / "run.log").read_text(errors="replace")
    require(process.returncode == 0, f"Application failed in {folder}:\n{output[-6000:]}")
    entries = [(int(n), float(t), float(h)) for n, t, h in re.findall(
        r"Radiation Euler substeps (\d+): t=([^,]+), hydro_dt=([^\s]+)", output)]
    radiation_limit = .25 * (2 / (8 * 2 ** level)) / 3
    expected_substeps = math.ceil(dt / radiation_limit)
    if expected_substeps > 1:
        require(len(entries) == round((stop-start)/dt),
                f"Expected one subcycling record per hydro step: got {entries}")
        for index, (count, current, used_dt) in enumerate(entries):
            require(count == expected_substeps, f"Unexpected substep count {count}; expected {expected_substeps}")
            require(math.isclose(current, start + index*dt, rel_tol=2e-12, abs_tol=2e-14),
                    "Radiation macrostep start time lost synchronization")
            require(math.isclose(used_dt, dt, rel_tol=2e-12, abs_tol=2e-14), "Unexpected hydro timestep")
            require(used_dt / count <= radiation_limit * (1+1e-13), "Radiation substep exceeds streaming CFL")
    else:
        require(not entries, "The reference should take one radiation step per global step")
    problem_id = 21 if problem == "RADIATION_COUPLED_WAVE" else 13
    metadata, final, inventory = snapshot(reader, folder / "final.silo", level, stop, problem_id)
    record = {"label": label, "directory": str(folder), "level": level, "hydro_dt": dt,
              "expected_substeps": expected_substeps, "observed_macrosteps": len(entries),
              "radiation_cfl_limit": radiation_limit, "start_time": start, "final_time": metadata["time"],
              "wall_seconds": time.monotonic()-started, "command": command, "config": config}
    return record, metadata, final, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--silo-library")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--levels", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    args.executable = args.executable.resolve()
    require(args.executable.is_file(), f"Executable does not exist: {args.executable}")
    root = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="octotiger-euler-subcycling-"))
    root.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "scope": "Euler subcycling, restart and conservation; no temporal-order claim",
              "runtime": fingerprint(args.executable, None), "runs": [], "checks": []}
    report_path = root / "validation.json"
    save_report(report_path, report)
    reader = Silo(args.silo_library)
    print(f"Euler subcycling artifacts: {root}", flush=True)
    try:
        for level in args.levels:
            direct, metadata, final, inventory = run(args, reader, root, f"wave-l{level}-direct", level,
                .08, 1.28, "RADIATION_COUPLED_WAVE")
            _, initial, initial_inventory = snapshot(reader, Path(direct["directory"])/"X.0.silo", level, 0, 21)
            require(inventory == initial_inventory, "Fixed mesh changed during subcycling")
            direct["conservation"] = conservation(initial, final, metadata)
            report["runs"].append(direct)
            checkpoint, _, middle, middle_inventory = run(args, reader, root, f"wave-l{level}-checkpoint", level,
                .08, .64, "RADIATION_COUPLED_WAVE")
            resumed, restart_metadata, resumed_state, resumed_inventory = run(args, reader, root,
                f"wave-l{level}-resumed", level, .08, 1.28, "RADIATION_COUPLED_WAVE",
                Path(checkpoint["directory"])/"final.silo", .64)
            require(inventory == middle_inventory == resumed_inventory, "Restart changed the leaf inventory")
            resumed["conservation"] = conservation(middle, resumed_state, restart_metadata)
            report["runs"].extend([checkpoint, resumed])
            errors = compare(resumed_state, final, 5e-11, "Subcycling checkpoint/restart")
            require(max(abs(a-b) for a, b in zip(middle["er"], final["er"])) > 1e-9*max(final["er"]),
                    "Restart comparison did not evolve the checkpoint")
            report["checks"].append({"name": f"coupled_wave_level{level}", "passed": True,
                                      "restart_errors": errors})
            save_report(report_path, report)
        coarse, metadata, coarse_state, inventory = run(args, reader, root, "homogeneous-subcycled", 0,
            .08, 1.28, "RADIATION_COUPLING")
        fine, _, fine_state, fine_inventory = run(args, reader, root, "homogeneous-global-substeps", 0,
            .02, 1.28, "RADIATION_COUPLING")
        _, initial, _ = snapshot(reader, Path(coarse["directory"])/"X.0.silo", 0, 0, 13)
        require(inventory == fine_inventory, "Homogeneous comparison changed the mesh")
        coarse["conservation"] = conservation(initial, coarse_state, metadata)
        errors = compare(coarse_state, fine_state, 5e-11, "Repeated Euler source/transport steps")
        require(max(abs(a-b) for a, b in zip(initial["er"], coarse_state["er"])) > .01*max(initial["er"]),
                "Homogeneous source comparison did not evolve")
        report["runs"].extend([coarse, fine])
        report["checks"].append({"name": "identical_kernel_substeps_homogeneous", "passed": True,
                                  "comparison_errors": errors})
        require(fingerprint(args.executable, None) == report["runtime"], "Runtime changed during validation")
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        save_report(report_path, report)
        raise
    save_report(report_path, report)
    print(f"Euler subcycling checks passed: {report_path}", flush=True)


if __name__ == "__main__":
    main()
