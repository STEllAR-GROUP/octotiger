#!/usr/bin/env python3
"""Bounded moving-FLD restart and prescribed-medium M1 scattering regressions.

All application processes run sequentially with at most two HPX workers.
The shock checks advance four one-second steps, not a benchmark final time.
The streaming checks use an explicit domain optical depth of 1e-4 and a
zero-opacity control, and inspect full-volume Silo fields and source ledgers.
"""

import argparse
import csv
import math
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

from application_helpers import environment, fingerprint, require, save_report
from test_fld_application import STEP_KEYS, STEP_RE
from test_hydro_application import check_state, read_config
from test_restart import CGS_C, Silo


FIELDS = ("rho_1", "egas", "tau", "sx", "sy", "sz", "er", "fx", "fy", "fz")
METADATA = ("code_to_g", "code_to_cm", "code_to_s", "n_species", "problem", "hydro",
            "radiation", "gravity", "periodic", "rad_implicit", "radiation_model",
            "radiation_integrator", "radiation_options_version", "radiation_algorithm_version",
            "radiation_limiter", "radiation_flux_semantics", "radiation_reduced_c_factor",
            "rad_test_chi", "leaf_count", "xscale")
HYDRO_METADATA = ("fld_hydro_unigrid", "fld_hydro_reflect_bc", "fld_hydro_cfl",
                  "fld_boundary", "fld_nonlinear_tolerance", "fld_linear_tolerance",
                  "fld_max_iterations", "fld_max_linear_iterations")


def snapshot(reader, path, level, moving=False):
    metadata, raw = reader.snapshot(path, fields=FIELDS,
                                    metadata_names=METADATA + (HYDRO_METADATA if moving else ()),
                                    block_count=8 ** level)
    fields = {name: {key.rsplit("/", 1)[0]: values for key, values in leaves.items()}
              for name, leaves in raw.items()}
    require(metadata["leaf_count"] == 8 ** level, "Checkpoint changed the uniform leaf count")
    if moving:
        check_state(metadata, fields)
    return metadata, fields


def flattened(fields, name):
    return [value for key in sorted(fields[name]) for value in fields[name][key]]


def compare_fields(expected, actual, tolerance=2e-9):
    errors = {}
    for name in FIELDS:
        require(expected[name].keys() == actual[name].keys(), f"Restart changed {name} leaf inventory")
        first, second = flattened(expected, name), flattened(actual, name)
        require(len(first) == len(second), f"Restart changed {name} array size")
        scale = max(max(map(abs, first)), max(map(abs, second)), 1e-300)
        errors[name] = max(abs(a - b) for a, b in zip(first, second)) / scale
        require(errors[name] <= tolerance, f"Restart changed {name}: {errors[name]}")
    return errors


def invoke(args, root, label, config, restart=None):
    folder = root / label
    folder.mkdir()
    config = dict(config, datadir=str(folder) + os.sep)
    if restart:
        config.update(restart_filename=str(restart), radiation_model="m1", radiation_integrator="euler",
                      hydro="off", unigrid="off", reflect_bc="on", cfl=.11,
                      fld_boundary="vacuum", fld_nonlinear_tolerance=1e-6,
                      fld_linear_tolerance=1e-5, fld_max_iterations=7, fld_max_linear_iterations=9)
    path = folder / "run.ini"
    path.write_text("".join(f"{key}={value}\n" for key, value in config.items()))
    command = [str(args.executable), f"--config_file={path}", f"--hpx:threads={args.threads}"]
    print(f"Running {label}", flush=True)
    started = time.monotonic()
    with (folder / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=restart.parent if restart else folder,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                                 env=environment(None))
    output = (folder / "run.log").read_text(errors="replace")
    require(process.returncode == 0, f"Application failed in {folder}:\n{output[-7000:]}")
    require((folder / "final.silo").is_file(), f"Missing final checkpoint in {folder}")
    return {"label": label, "directory": str(folder), "command": command, "config": config,
            "wall_seconds": time.monotonic() - started, "output": output}


def common(level, dt, stop):
    return {"radiation_backend": "cpu", "gravity": "off", "eos": "IDEAL", "omega": 0,
            "radiation_reduced_c_factor": 1, "unigrid": "on", "min_level": level, "max_level": level,
            "n_species": 1, "atomic_mass": 1, "atomic_number": 1, "hard_dt": dt, "stop_time": stop,
            "stop_step": round(stop / dt) - 1, "odt": "1e99", "disable_output": "off",
            "disable_analytic": "on", "disable_diagnostics": "on", "silo_num_groups": 1,
            "hydro_host_kernel_type": "LEGACY", "hydro_device_kernel_type": "OFF",
            "monopole_device_kernel_type": "OFF", "multipole_device_kernel_type": "OFF"}


def moving_metadata(metadata, expected_time):
    expected = {"radiation_options_version": 3, "radiation_model": 1, "radiation_integrator": 3,
                "radiation_algorithm_version": 2, "radiation_flux_semantics": 2, "radiation_limiter": 1,
                "hydro": 1, "radiation": 1, "gravity": 0, "periodic": 0, "rad_implicit": 1,
                "radiation_reduced_c_factor": 1, "fld_hydro_unigrid": 1,
                "fld_hydro_reflect_bc": 0, "fld_hydro_cfl": .27, "fld_boundary": 0,
                "fld_nonlinear_tolerance": 1e-10, "fld_linear_tolerance": 1e-10,
                "fld_max_iterations": 50, "fld_max_linear_iterations": 200}
    for name, value in expected.items():
        require(metadata[name] == value, f"Checkpoint {name}: {metadata[name]} != {value}")
    require(math.isclose(metadata["time"], expected_time, rel_tol=1e-13), "Wrong resumed physical time")


def run_moving(args, root, reader, level):
    template = Path(__file__).resolve().parents[2] / "radiation_results/configs/subcritical_shock.ini"
    base = read_config(template)
    base.update(common(level, 1., 4.))
    base.update(radiation_model="fld", radiation_integrator="backward_euler", cfl=.27,
                reflect_bc="off", fld_boundary="reflecting")
    direct = invoke(args, root, f"l{level}-fld-direct", base)
    partial = invoke(args, root, f"l{level}-fld-checkpoint", dict(base, stop_time=2., stop_step=1))
    resumed = invoke(args, root, f"l{level}-fld-resumed", base,
                     restart=Path(partial["directory"]) / "final.silo")
    snapshots = {}
    for record, start, end in ((direct, 0., 4.), (partial, 0., 2.), (resumed, 2., 4.)):
        steps = [dict(zip(STEP_KEYS, map(float, row))) for row in STEP_RE.findall(record.pop("output"))]
        require(len(steps) == int(end - start), f"Wrong accepted step count: {steps}")
        for index, step in enumerate(steps):
            require(step["t"] == start + index and step["dt"] == 1., "Wrong moving FLD step clock")
            require(step["nonlinear_residual"] <= 1.05e-10, "Unconverged moving FLD step")
        record["steps"] = steps
        metadata, fields = snapshot(reader, Path(record["directory"]) / "final.silo", level, moving=True)
        moving_metadata(metadata, end)
        record["checkpoint_metadata"] = metadata
        snapshots[record["label"]] = fields
    direct_fields, partial_fields, resumed_fields = (snapshots[row["label"]] for row in (direct, partial, resumed))
    resumed["restart_relative_errors"] = compare_fields(direct_fields, resumed_fields)
    changed = max(abs(a - b) for a, b in zip(flattened(partial_fields, "rho_1"),
                                            flattened(resumed_fields, "rho_1")))
    require(changed > 1e-12 * max(flattened(partial_fields, "rho_1")), "Restart did not evolve the shock")
    return [direct, partial, resumed]


def run_streaming(args, root, reader, problem):
    level, dt, stop = 0, .005, .02
    results = []
    for chi in (0., 5e-5):
        config = common(level, dt, stop)
        config.update(problem=problem, radiation="on", radiation_model="m1", radiation_integrator="euler",
                      hydro="off", periodic="on", rad_implicit="on" if chi else "off", xscale=1,
                      rad_test_chi=chi, rad_test_width=.2, rad_test_background=1, rad_test_amplitude=.01,
                      rad_test_luminosity=.01, code_to_g=CGS_C ** 3, code_to_cm=CGS_C, code_to_s=1, cfl=.4)
        label = problem.lower().replace("radiation_", "") + ("-thin" if chi else "-vacuum")
        record = invoke(args, root, label, config)
        record.pop("output")
        folder = Path(record["directory"])
        initial_meta, initial = snapshot(reader, folder / "X.0.silo", level)
        metadata, final = snapshot(reader, folder / "final.silo", level)
        with (folder / "radiation-conservation.csv").open() as stream:
            ledger = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        require(initial_meta["time"] == 0 and 0 < metadata["time"] <= stop and len(ledger) == 5 and
                math.isclose(metadata["time"], ledger[-1]["t"], rel_tol=1e-12),
                "Streaming smoke must execute four steps and match its saved physical clock")
        require(metadata["radiation_model"] == 0 and metadata["radiation_algorithm_version"] == 1 and
                metadata["radiation_flux_semantics"] == 0 and metadata["rad_test_chi"] == chi,
                "M1 scattering check changed the model or opacity")
        for name in ("rho_1", "egas", "tau", "sx", "sy", "sz"):
            require(flattened(initial, name) == flattened(final, name),
                    f"Prescribed-medium scattering changed material {name}")
        totals0 = [math.fsum(flattened(initial, name)) for name in ("fx", "fy", "fz")]
        totals1 = [math.fsum(flattened(final, name)) for name in ("fx", "fy", "fz")]
        ratio = math.hypot(*totals1) / math.hypot(*totals0)
        durations = [right["t"] - left["t"] for left, right in zip(ledger, ledger[1:])]
        require(all(0 < duration <= dt for duration in durations), "Invalid M1 timestep")
        expected = math.prod(1 / (1 + chi * duration) for duration in durations)
        require(abs(ratio - expected) < 5e-10,
                f"Integrated flux did not receive the prescribed scattering source: {ratio} vs {expected}")
        energy_error = abs(math.fsum(flattened(final, "er")) / math.fsum(flattened(initial, "er")) - 1)
        require(energy_error < 1e-11, "Pure periodic scattering changed integrated radiation energy")
        source = [ledger[-1][name] - ledger[0][name] for name in ("fx_source", "fy_source", "fz_source")]
        require(math.hypot(*source) > 0 if chi else math.hypot(*source) == 0,
                "Scattering source ledger does not distinguish positive opacity from vacuum")
        record["checks"] = {"opacity": chi, "domain_optical_depth": 2 * chi,
                            "integrated_flux_ratio": ratio, "backward_euler_source_ratio": expected,
                            "physical_time": metadata["time"], "step_durations": durations,
                            "relative_radiation_energy_error": energy_error, "source_ledger": source,
                            "material_fields_bitwise_unchanged": True}
        results.append(record)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--silo-library")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--skip-moving", action="store_true")
    args = parser.parse_args()
    require(not (args.skip_moving and args.skip_streaming), "At least one regression group must run")
    args.executable = args.executable.resolve()
    require(args.executable.is_file(), "Missing OctoTiger executable")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="octotiger-fld-hydro-restart-"))
    root.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scope": "four-step moving FLD restart and thin M1 scattering",
              "selected_checks": {"moving_restart_levels": [] if args.skip_moving else args.levels,
                                  "streaming_scattering": not args.skip_streaming},
              "runtime": fingerprint(args.executable, None), "runs": []}
    report_path = root / "validation.json"
    save_report(report_path, report)
    reader = Silo(args.silo_library)
    try:
        if not args.skip_moving:
            for level in args.levels:
                report["runs"].extend(run_moving(args, root, reader, level))
                save_report(report_path, report)
        if not args.skip_streaming:
            for problem in ("RADIATION_STREAMING_WAVE", "RADIATION_STREAMING_FRONT"):
                report["runs"].extend(run_streaming(args, root, reader, problem))
                save_report(report_path, report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save_report(report_path, report)
    print(f"Selected application regression checks passed: {report_path}", flush=True)


if __name__ == "__main__":
    main()
