#!/usr/bin/env python3
"""Short stationary FLD application, conservation, and restart checks.

Runs sequentially with one or two HPX workers. The homogeneous problem checks
two-temperature exchange; the Marshak problem checks spatial transport with
its distinct linear emission law. Neither uses an M1 analytic reference.
Pass --mixed for an additional adaptive Marshak run and an actual mixed-mesh
check. These smoke checks do not establish spatial or temporal convergence.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
import time

from application_helpers import environment, fingerprint, require, save_report
from test_restart import CGS_C, Silo


FIELDS = ("rho_1", "egas", "tau", "sx", "sy", "sz", "er", "fx", "fy", "fz")
METADATA = ("code_to_g", "code_to_cm", "code_to_s", "n_species", "problem", "hydro",
            "radiation", "gravity", "periodic", "rad_implicit", "radiation_model",
            "radiation_integrator", "radiation_options_version", "radiation_algorithm_version",
            "radiation_limiter", "radiation_flux_semantics", "radiation_reduced_c_factor",
            "fld_boundary", "fld_nonlinear_tolerance", "fld_linear_tolerance",
            "fld_max_iterations", "fld_max_linear_iterations", "leaf_count", "xscale")
GAMMA = 5 / 3
DT = .125
STEP_RE = re.compile(r"FLD accepted " + " ".join(
    name + r"=([^\s]+)" for name in
    ("t", "dt", "substeps", "retries", "nonlinear_iterations", "linear_iterations", "vcycles",
     "nonlinear_residual", "linear_residual", "boundary_energy", "conservation_error")))
STEP_KEYS = ("t", "dt", "substeps", "retries", "nonlinear_iterations", "linear_iterations", "vcycles",
             "nonlinear_residual", "linear_residual", "boundary_energy", "conservation_error")


def snapshot(reader, path, expected_time, problem, expected_settings=None):
    metadata, _ = reader.snapshot(path, fields=(), metadata_names=METADATA)
    metadata, blocks = reader.snapshot(path, fields=FIELDS, metadata_names=METADATA,
                                       block_count=int(metadata["leaf_count"]))
    require(math.isclose(metadata["time"], expected_time, rel_tol=2e-12, abs_tol=2e-14),
            f"Unexpected checkpoint time in {path}: {metadata['time']} vs {expected_time}")
    expected = {"radiation_options_version": 3, "radiation_model": 1, "radiation_integrator": 3,
                "radiation_algorithm_version": 1, "radiation_limiter": 1, "radiation_flux_semantics": 1,
                "radiation": 1, "hydro": 0, "gravity": 0, "rad_implicit": 1, "n_species": 1,
                "radiation_reduced_c_factor": 1, "fld_boundary": 0,
                "fld_nonlinear_tolerance": 1e-10, "fld_linear_tolerance": 1e-10,
                "fld_max_iterations": 50, "fld_max_linear_iterations": 200,
                "problem": 13 if problem == "RADIATION_COUPLING" else 9,
                "periodic": int(problem == "RADIATION_COUPLING")}
    if expected_settings:
        expected.update(expected_settings)
    for name, value in expected.items():
        require(metadata[name] == value, f"Checkpoint {name}: {metadata[name]} vs {value}")
    mass, length, clock = (metadata[name] for name in ("code_to_g", "code_to_cm", "code_to_s"))
    # The existing Marshak verifier explicitly normalizes c=a_r=1 and all
    # units to one; its material emission is linear rather than ideal-gas LTE.
    light_speed = 1 if problem == "MARSHAK" else CGS_C * clock / length
    require(math.isclose(light_speed, 1, rel_tol=1e-14), "Expected physical c=1 in code units")
    energy_unit = mass / (length * clock ** 2)
    units = {"rho_1": mass / length ** 3, "egas": energy_unit, "er": energy_unit,
             "tau": energy_unit ** (1 / GAMMA)}
    units.update({name: mass / (length ** 2 * clock) for name in ("sx", "sy", "sz")})
    units.update({name: energy_unit * length / clock for name in ("fx", "fy", "fz")})
    state, inventory, weights, levels = {}, None, [], set()
    for field in FIELDS:
        leaves = {key.rsplit("/", 1)[0]: values for key, values in blocks[field].items()}
        current = [(key, len(leaves[key])) for key in sorted(leaves)]
        if inventory is None:
            inventory = current
            for key, count in inventory:
                # Silo output uses octal node IDs: leading 1, one digit per level.
                digits = key.strip("/")
                require(re.fullmatch(r"1[0-7]*", digits) is not None, f"Unknown Silo leaf path {key}")
                level = len(digits) - 1
                levels.add(level)
                require(count == 8 ** 3, "This application test requires INX=8")
                dx = 2 * metadata["xscale"] / (8 * 2 ** level)
                weights.extend([dx ** 3] * count)
        require(current == inventory, "Checkpoint fields have inconsistent leaf inventories")
        state[field] = [value / units[field] for key in sorted(leaves) for value in leaves[key]]
    state["rho"] = state.pop("rho_1")
    require(math.isclose(math.fsum(weights), (2 * metadata["xscale"]) ** 3, rel_tol=1e-13),
            "Leaf volumes do not cover the domain exactly once")
    maximum_flux_ratio = 0
    for i, rho in enumerate(state["rho"]):
        energy, gas = state["er"][i], state["egas"][i]
        momentum = math.hypot(*(state[name][i] for name in ("sx", "sy", "sz")))
        flux = math.hypot(*(state[name][i] for name in ("fx", "fy", "fz")))
        require(rho > 0 and energy >= 0 and gas >= 0 and state["tau"][i] >= 0,
                f"Negative/nonpositive stationary state in cell {i}")
        require(momentum <= 1e-13 * rho * light_speed, "FLD diagnostic flux imparted material recoil")
        require(flux <= light_speed * energy * (1 + 5e-11), f"Unbounded diagnostic flux in cell {i}")
        if energy > 0:
            maximum_flux_ratio = max(maximum_flux_ratio, flux / (light_speed * energy))
        require(math.isclose(state["tau"][i] ** GAMMA, gas, rel_tol=3e-10, abs_tol=1e-300),
                "The gas energy and entropy tracer disagree")
    totals = {name: math.fsum(value * volume for value, volume in zip(state[name], weights))
              for name in ("rho", "egas", "er")}
    return {"metadata": metadata, "state": state, "inventory": inventory, "levels": sorted(levels),
            "totals": totals, "maximum_flux_ratio": maximum_flux_ratio}


def configuration(folder, problem, level, stop, mixed=False):
    config = {
        "problem": problem, "radiation": "on", "radiation_model": "fld",
        # Omitting radiation_integrator verifies the model-specific default.
        "radiation_backend": "cpu", "hydro": "off", "gravity": "off", "eos": "IDEAL",
        "omega": 0, "rad_implicit": "on", "radiation_reduced_c_factor": 1,
        "periodic": "on" if problem == "RADIATION_COUPLING" else "off",
        "fld_boundary": "reflecting", "unigrid": "off" if mixed else "on",
        "min_level": level, "max_level": level + int(mixed), "xscale": 1,
        "n_species": 1, "atomic_mass": 1, "atomic_number": 1,
        "code_to_g": 1e30, "code_to_cm": CGS_C, "code_to_s": 1,
        "cfl": .25, "hard_dt": DT, "stop_time": stop, "stop_step": round(stop / DT) - 1,
        "odt": "1e99", "disable_output": "off", "disable_analytic": "on", "disable_diagnostics": "on",
        "hydro_host_kernel_type": "LEGACY", "hydro_device_kernel_type": "OFF",
        "monopole_device_kernel_type": "OFF", "multipole_device_kernel_type": "OFF",
        "silo_num_groups": 1, "datadir": str(folder) + os.sep,
    }
    if mixed:
        config.update(gradient_refine_field="radiation_energy", gradient_refine_mode="absolute",
                      gradient_refine_threshold=.5, gradient_refine_floor=1e-20,
                      gradient_refine_buffer=0, gradient_refine_coarsen_factor=.5)
    return config


def invoke(args, root, label, problem, level, stop, restart=None, mixed=False, changes=None, error=None,
           rejection_reason="unsupported_configuration", allow_retries=False):
    folder = root / label
    folder.mkdir()
    config = configuration(folder, problem, level, stop, mixed)
    if restart:
        config.update(restart_filename=str(restart), radiation_model="m1", radiation_integrator="euler",
                      fld_boundary="vacuum", fld_nonlinear_tolerance=1e-6, fld_linear_tolerance=1e-5,
                      fld_max_iterations=7, fld_max_linear_iterations=9)
    if changes:
        config.update(changes)
    ini = folder / "run.ini"
    ini.write_text("".join(f"{name}={value}\n" for name, value in config.items()))
    command = [str(args.executable), f"--config_file={ini}", f"--hpx:threads={args.threads}"]
    print(f"Running {label}", flush=True)
    started = time.monotonic()
    with (folder / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=restart.parent if restart else folder,
                                 stdout=log, stderr=subprocess.STDOUT, env=environment(None), timeout=args.timeout)
    output = (folder / "run.log").read_text(errors="replace")
    if error:
        require(process.returncode != 0 and error in output and not (folder / "final.silo").exists(),
                f"Expected {rejection_reason} rejection failed: {label}\n{output[-6000:]}")
        if rejection_reason == "solver_exhaustion":
            require("FLD accepted" not in output,
                    "Exhausted FLD solve committed an accepted step before rejecting")
        return {"label": label, "returncode": process.returncode, "expected_error": error,
                "reason": rejection_reason, "directory": str(folder), "command": command,
                "config": config, "wall_seconds": time.monotonic() - started}
    require(process.returncode == 0, f"Application failed in {folder}:\n{output[-7000:]}")
    require("Radiation Euler substeps" not in output, "FLD called the explicit M1 radiation driver")
    steps = [dict(zip(STEP_KEYS, map(float, values))) for values in STEP_RE.findall(output)]
    start = .25 if restart else 0
    require(len(steps) == round((stop - start) / DT), f"Unexpected FLD step records: {steps}")
    for index, step in enumerate(steps):
        require(all(math.isfinite(value) for value in step.values()), "Nonfinite FLD solve diagnostic")
        require(math.isclose(step["t"], start + index * DT, rel_tol=2e-12, abs_tol=2e-14),
                f"FLD macrostep clock lost synchronization: {step}")
        require(math.isclose(step["dt"], DT, rel_tol=2e-12), "FLD timestep was reduced to the streaming CFL")
        if allow_retries:
            require(step["substeps"] >= 1 and step["retries"] >= 0,
                    "Invalid FLD accepted-substep or retry count")
        else:
            require(step["substeps"] == 1 and step["retries"] == 0, "The short FLD checks unexpectedly retried")
        require(step["nonlinear_residual"] <= 1.05e-10, "Unconverged FLD step was accepted")
    if allow_retries:
        require(any(step["retries"] > 0 for step in steps), "The forced FLD retry check never rejected an attempt")
    streaming_limit = .25 * (2 / (8 * 2 ** level)) / 3
    require(DT > 5 * streaming_limit, "The FLD check must exceed the M1 streaming timestep")
    return {"label": label, "directory": str(folder), "command": command, "config": config,
            "steps": steps, "streaming_cfl_limit": streaming_limit, "wall_seconds": time.monotonic() - started}


def conservation(initial, final, record):
    before, after = initial["totals"], final["totals"]
    boundary = math.fsum(step["boundary_energy"] for step in record["steps"])
    energy0, energy1 = before["egas"] + before["er"], after["egas"] + after["er"]
    scale = max(abs(energy0), abs(energy1), abs(boundary), 1e-300)
    error = abs(energy1 - energy0 - boundary) / scale
    require(error < 1e-8, f"FLD physical energy balance failed: {error}")
    require(math.isclose(before["rho"], after["rho"], rel_tol=2e-13), "Stationary FLD changed total mass")
    require(sum(abs(step["conservation_error"]) for step in record["steps"]) < 1e-8 * scale,
            "FLD reported an excessive nonlinear conservation residual")
    return {"initial_gas_energy": before["egas"], "final_gas_energy": after["egas"],
            "initial_radiation_energy": before["er"], "final_radiation_energy": after["er"],
            "boundary_energy": boundary, "relative_energy_error": error,
            "maximum_flux_ratio": final["maximum_flux_ratio"], "leaf_levels": final["levels"]}


def compare_restart(direct, resumed):
    require(direct["inventory"] == resumed["inventory"], "Restart changed the leaf mesh")
    result = {}
    for field, values in direct["state"].items():
        actual = resumed["state"][field]
        scale = max(max(map(abs, values)), max(map(abs, actual)), 1e-300)
        error = max(abs(a - b) for a, b in zip(values, actual)) / scale
        require(error < 2e-9, f"FLD restart field mismatch: {field}={error}")
        result[field] = error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--silo-library")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--mixed", action="store_true", help="Also test a mixed-level adaptive Marshak mesh")
    parser.add_argument("--skip-options", action="store_true", help="Skip deliberately rejected option combinations")
    args = parser.parse_args()
    args.executable = args.executable.resolve()
    require(args.executable.is_file(), f"Missing executable {args.executable}")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="octotiger-fld-application-"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"FLD application artifacts: {root}", flush=True)
    report = {"status": "running", "scope": "stationary FLD application smoke and restart",
              "runtime": fingerprint(args.executable, None), "runs": [], "rejections": []}
    report_path = root / "validation.json"
    save_report(report_path, report)
    if not args.skip_options:
        for label, changes, message in (
            ("hydro", {"hydro": "on"}, "requires stationary matter"),
            ("gravity", {"gravity": "on"}, "requires stationary matter"),
            ("reduced-c", {"radiation_reduced_c_factor": .5}, "requires radiation_reduced_c_factor=1"),
            ("explicit-source", {"rad_implicit": "off"}, "requires rad_implicit=1"),
            ("integrator", {"radiation_integrator": "euler"}, "FLD requires backward_euler"),
            ("iterations-overflow", {"fld_max_linear_iterations": 2 ** 31}, "iteration limits in [1, INT_MAX]"),
        ):
            report["rejections"].append(invoke(args, root, "reject-" + label, "RADIATION_COUPLING",
                                                0, .25, changes=changes, error=message))
    # Deliberately exhaust the collective nonlinear retry budget. Keep this
    # separate from option validation: the configuration is valid, but the
    # severely restricted solver must fail without committing a step/budget.
    # stop=.25 gives stop_step=1; the legacy driver skips stop_step=0 entirely.
    report["rejections"].append(invoke(
        args, root, "reject-solver-exhaustion", "MARSHAK", 1, .25,
        changes={"fld_max_iterations": 1, "fld_nonlinear_tolerance": 1e-16},
        error="FLD failed after 16 timestep halvings", rejection_reason="solver_exhaustion"))
    save_report(report_path, report)
    reader = Silo(args.silo_library)
    # A hard timestep plus stop_step is valid without a finite time horizon.
    # The default stop_time sentinel must not overflow timestep arithmetic.
    bounded = invoke(args, root, "coupling-stop-step", "RADIATION_COUPLING", 0, .25,
                     changes={"stop_time": sys.float_info.max})
    bounded_initial = snapshot(reader, Path(bounded["directory"]) / "X.0.silo", 0, "RADIATION_COUPLING")
    bounded_final = snapshot(reader, Path(bounded["directory"]) / "final.silo", .25, "RADIATION_COUPLING")
    bounded["balance"] = conservation(bounded_initial, bounded_final, bounded)
    report["runs"].append(bounded)
    save_report(report_path, report)
    for level in args.levels:
        prefix = f"coupling-l{level}"
        direct = invoke(args, root, prefix + "-direct", "RADIATION_COUPLING", level, .5)
        initial = snapshot(reader, Path(direct["directory"]) / "X.0.silo", 0, "RADIATION_COUPLING")
        final = snapshot(reader, Path(direct["directory"]) / "final.silo", .5, "RADIATION_COUPLING")
        require(initial["levels"] == final["levels"] == [level], "Unexpected uniform mesh level")
        direct["balance"] = conservation(initial, final, direct)
        require(final["totals"]["er"] > initial["totals"]["er"] * 1.01 and
                final["totals"]["egas"] < initial["totals"]["egas"], "The coupled thermal source did not evolve")
        require(final["maximum_flux_ratio"] < 1e-10, "Homogeneous FLD generated a spurious diagnostic flux")
        checkpoint = invoke(args, root, prefix + "-checkpoint", "RADIATION_COUPLING", level, .25)
        restart = Path(checkpoint["directory"]) / "final.silo"
        resumed = invoke(args, root, prefix + "-resumed", "RADIATION_COUPLING", level, .5, restart=restart)
        restarted = snapshot(reader, Path(resumed["directory"]) / "final.silo", .5, "RADIATION_COUPLING")
        resumed["restart_relative_errors"] = compare_restart(final, restarted)
        report["runs"].extend((direct, checkpoint, resumed))
        save_report(report_path, report)
    retry_settings = {"fld_max_iterations": 2}
    retried = invoke(args, root, "marshak-retry-l1", "MARSHAK", 1, .25,
                     changes=retry_settings, allow_retries=True)
    retry_initial = snapshot(reader, Path(retried["directory"]) / "X.0.silo", 0, "MARSHAK",
                             expected_settings=retry_settings)
    retry_final = snapshot(reader, Path(retried["directory"]) / "final.silo", .25, "MARSHAK",
                           expected_settings=retry_settings)
    require(retry_initial["levels"] == retry_final["levels"] == [1], "Retry changed the uniform leaf mesh")
    # Boundary and source budgets must include accepted substeps only. The
    # normal physical-energy balance detects duplicated rejected-attempt flux.
    retried["balance"] = conservation(retry_initial, retry_final, retried)
    report["runs"].append(retried)
    save_report(report_path, report)
    for mixed in ([False, True] if args.mixed else [False]):
        record = invoke(args, root, "marshak-mixed" if mixed else "marshak-l1", "MARSHAK", 1,
                        2 if mixed else .25, mixed=mixed)
        initial = snapshot(reader, Path(record["directory"]) / "X.0.silo", 0, "MARSHAK")
        final = snapshot(reader, Path(record["directory"]) / "final.silo", 2 if mixed else .25, "MARSHAK")
        record["balance"] = conservation(initial, final, record)
        require(final["totals"]["er"] > 0 and final["totals"]["egas"] > 0 and
                final["maximum_flux_ratio"] > 0, "Marshak transport and thermal heating did not evolve")
        require(record["balance"]["boundary_energy"] > 0, "Marshak bath did not inject energy")
        if mixed:
            require(final["levels"] == [1, 2], "Adaptive case did not produce the required mixed levels")
        else:
            require(final["levels"] == [1], "Unexpected uniform Marshak mesh")
        report["runs"].append(record)
        save_report(report_path, report)
    require(fingerprint(args.executable, None) == report["runtime"], "Runtime changed during validation")
    report["status"] = "passed"
    save_report(report_path, report)
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]),
                      "rejected_configurations": sum(item["reason"] == "unsupported_configuration"
                                                     for item in report["rejections"]),
                      "exhausted_solver_checks": sum(item["reason"] == "solver_exhaustion"
                                                     for item in report["rejections"]),
                      "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
