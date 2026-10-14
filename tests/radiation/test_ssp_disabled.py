#!/usr/bin/env python3
"""Keep radiation on its existing RK3 hydro path when SSP is requested.

Small sequential CPU runs compare all stored hydro/radiation fields for CLI
and config requests, and for restarts that restore radiation over radiation=off.
A modified exact hydro checkpoint is rejected before any raw-node shard exists.
"""

import argparse
import ctypes as ct
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tempfile
import time

from application_helpers import environment, fingerprint, require, save_report
from test_fld_hydro_restart import common
from test_hydro_application import read_config
from test_restart import CGS_C, Silo

WARNING = "SSP_MULTISTEP is disabled when radiation is enabled; using RK3"
EXACT_ERROR = "SSP_MULTISTEP exact checkpoints do not support radiation"
FIELDS = ("rho_1", "egas", "tau", "sx", "sy", "sz", "lx", "ly", "lz", "pot",
          "er", "fx", "fy", "fz")
METADATA = ("code_to_g", "code_to_cm", "code_to_s", "n_species", "problem", "hydro",
            "radiation", "gravity", "periodic", "rad_implicit", "radiation_model",
            "radiation_integrator", "radiation_options_version", "radiation_algorithm_version",
            "radiation_limiter", "radiation_flux_semantics", "radiation_reduced_c_factor",
            "leaf_count", "xscale")


def has_metadata(reader, path, name):
    handle = reader.open(path)
    try:
        return bool(reader.lib.DBInqVarExists(handle, name.encode()))
    finally:
        reader.lib.DBClose(handle)


def two_step_config(dt):
    # RK3 divides the time remaining to stop_time across a whole refinement
    # batch, even if stop_step ends this invocation after two steps. Keep that
    # cap inactive so hard_dt sets both steps; verify their sum in the checkpoint.
    return dict(common(0, dt, 100 * dt), stop_step=1)


def configuration(model):
    if model == "m1":
        dt = .02
        config = two_step_config(dt)
        config.update(problem="RADIATION_COUPLED_WAVE", radiation="on", hydro="on",
                      periodic="on", rad_implicit="on", omega=0, xscale=1,
                      atomic_mass=1, atomic_number=1, code_to_g=1e30,
                      code_to_cm=CGS_C, code_to_s=1, cfl=.25,
                      rad_test_chi=1, rad_test_amplitude=.01, correct_am_hydro="off",
                      inflow_bc="off", reflect_bc="off")
    else:
        dt = 1.
        template = Path(__file__).resolve().parents[2] / "radiation_results/configs/subcritical_shock.ini"
        config = read_config(template)
        config.update(two_step_config(dt))
        config.update(cfl=.27, reflect_bc="off", fld_boundary="reflecting")
    config.update(radiation_model=model, number_gpus=0, executors_per_gpu=0)
    # Leave the radiation integrator at its model-specific default. Requesting
    # SSP must preserve M1 Euler and FLD backward Euler independently of hydro.
    config.pop("radiation_integrator", None)
    return config, dt


def invoke(args, root, label, base, request=None, restart=None, expected_error=None,
           expected_integrator="RK3", expect_warning=False):
    folder = root / label
    folder.mkdir()
    config = dict(base, datadir=str(folder) + os.sep)
    if restart:
        config.update(restart_filename=str(restart), radiation="off")
    if request == "config":
        config["time_integrator"] = "SSP_MULTISTEP"
    ini = folder / "run.ini"
    ini.write_text("".join(f"{name}={value}\n" for name, value in config.items()))
    command = [str(args.executable), f"--config_file={ini}", "--hpx:threads=1",
               "--hpx:ini=hpx.threadpools.io_pool_size=1",
               "--hpx:ini=hpx.threadpools.timer_pool_size=1",
               "--hpx:ini=hpx.threadpools.parcel_pool_size=1", "--hpx:ini=hpx.parcel.tcp.enable=0"]
    if request == "cli":
        command.append("--time_integrator=SSP_MULTISTEP")
    print(f"Running {label}", flush=True)
    started = time.monotonic()
    with (folder / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=restart.parent if restart else folder,
                                 env=environment(None), stdout=log, stderr=subprocess.STDOUT,
                                 timeout=args.timeout)
    output = (folder / "run.log").read_text(errors="replace")
    record = {"label": label, "directory": str(folder), "command": command, "config": config,
              "returncode": process.returncode, "wall_seconds": time.monotonic() - started}
    if expected_error:
        require(process.returncode != 0 and expected_error in output,
                f"Expected rejection {expected_error!r} in {folder}:\n{output[-6000:]}")
        require("SSP_MULTISTEP stages=" not in output and not re.search(r"TS \d+::", output)
                and not (folder / "final.silo").exists(), "Rejected raw restart advanced the solution")
        require("Unable to open checkpoint data file" not in output,
                "Invalid radiation checkpoint reached raw-node loading")
        record["expected_error"] = expected_error
        return record
    require(process.returncode == 0, f"Application failed in {folder}:\n{output[-7000:]}")
    require(f"time_integrator = {expected_integrator}" in output,
            f"Wrong selected hydro integrator in {folder}:\n{output[-5000:]}")
    require((WARNING in output) == expect_warning, f"Unexpected/missing radiation SSP warning in {folder}")
    require((folder / "final.silo").is_file(), f"Missing final checkpoint in {folder}")
    if expected_integrator == "RK3":
        require("SSP_MULTISTEP stages=" not in output, "Radiation executed an SSP update")
    else:
        require(re.findall(r"SSP_MULTISTEP stages=(\d+)", output) == ["3", "2"],
                "Raw rejection fixture did not execute one RK3 startup and one SSP step")
    record["warning_count"] = output.count(WARNING)
    return record


def snapshot(reader, record, model, expected_time):
    path = Path(record["directory"]) / "final.silo"
    metadata, fields = reader.snapshot(path, fields=FIELDS, metadata_names=METADATA, block_count=1)
    require(math.isclose(metadata["time"], expected_time, rel_tol=2e-12, abs_tol=2e-14),
            f"Wrong final time for {record['label']}: {metadata['time']} != {expected_time}")
    require(metadata["radiation"] == metadata["hydro"] == metadata["n_species"] == 1,
            "Radiation/hydro state was disabled instead of just the SSP integrator")
    require(metadata["radiation_model"] == (0 if model == "m1" else 1) and
            metadata["radiation_integrator"] == (0 if model == "m1" else 3),
            "SSP request changed the radiation model or its integrator")
    require(not has_metadata(reader, path, "restart_version"),
            "Radiation fallback wrote an unsupported exact SSP checkpoint")
    record["checkpoint_metadata"] = metadata
    return metadata, fields


def compare(expected, actual, description):
    left_meta, left_fields = expected
    right_meta, right_fields = actual
    require(left_meta == right_meta, f"{description}: checkpoint physics metadata differs")
    counts = {}
    for field in FIELDS:
        require(left_fields[field].keys() == right_fields[field].keys(),
                f"{description}: changed {field} block inventory")
        count = 0
        for block in left_fields[field]:
            require(left_fields[field][block] == right_fields[field][block],
                    f"{description}: {field} values differ in {block}")
            count += len(left_fields[field][block])
        counts[field] = count
    return {"description": description, "equal": True, "comparison": "exact stored values",
            "field_value_counts": counts}


def raw_radiation_rejection(args, root, reader, report):
    config = two_step_config(.002)
    config.update(problem="SOD", radiation="off", hydro="on", periodic="off", xscale=1,
                  cfl=.4, correct_am_hydro="off", number_gpus=0, executors_per_gpu=0)
    seed = invoke(args, root, "exact-hydro-seed", config, request="cli",
                  expected_integrator="SSP_MULTISTEP")
    report["runs"].append(seed)
    checkpoint = Path(seed["directory"]) / "final.silo"
    require(has_metadata(reader, checkpoint, "restart_version"), "Hydro fixture lacks exact SSP payload metadata")
    modified = reader.altered_checkpoint(checkpoint, "radiation-marker", {"radiation": 1})
    missing_shards = root / "exact-radiation-input"
    missing_shards.mkdir()
    radiative = missing_shards / "final.silo"
    shutil.move(str(modified), radiative)
    # Copy only the top-level metadata. Any attempt to restore a raw node would
    # need final.silo.data/, which intentionally does not exist in this folder.
    require(not (missing_shards / "final.silo.data").exists(), "Raw fixture unexpectedly has node shards")
    for request in (None, "cli"):
        label = "exact-radiation-rejected-" + (request or "automatic")
        rejection = invoke(args, root, label, config, request=request, restart=radiative,
                           expected_error=EXACT_ERROR)
        rejection["raw_node_shards_absent"] = True
        report["runs"].append(rejection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--silo-library")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    args.executable = args.executable.resolve()
    require(args.executable.is_file(), "Missing OctoTiger executable")
    require(args.timeout > 0, "Timeout must be positive")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="octotiger-radiation-ssp-disabled-"))
    root.mkdir(parents=True, exist_ok=True)
    reader = Silo(args.silo_library)
    reader.lib.DBInqVarExists.argtypes = [ct.c_void_p, ct.c_char_p]
    reader.lib.DBInqVarExists.restype = ct.c_int
    report = {"status": "running", "scope": "SSP disabled for fresh/restored radiation and rejected for exact radiative payloads",
              "runtime": fingerprint(args.executable, None), "runs": [], "checks": []}
    report_path = root / "validation.json"
    save_report(report_path, report)
    print(f"Radiation SSP exclusion artifacts: {root}", flush=True)
    try:
        for model in ("m1", "fld"):
            config, dt = configuration(model)
            baseline = invoke(args, root, model + "-default", config)
            report["runs"].append(baseline)
            expected = snapshot(reader, baseline, model, 2 * dt)
            initial_meta, initial = reader.snapshot(Path(baseline["directory"]) / "X.0.silo",
                                                   fields=FIELDS, metadata_names=METADATA, block_count=1)
            require(initial_meta["time"] == 0 and any(initial[name] != expected[1][name] for name in FIELDS),
                    f"{model} exclusion test did not evolve")
            for request in ("cli", "config"):
                requested = invoke(args, root, f"{model}-ssp-{request}", config, request=request, expect_warning=True)
                report["runs"].append(requested)
                actual = snapshot(reader, requested, model, 2 * dt)
                report["checks"].append(compare(expected, actual, f"{model}: fresh SSP {request} versus default"))
            # stop_step is relative to each invocation, including restarts.
            resumed_config = dict(config, stop_step=1)
            checkpoint = Path(baseline["directory"]) / "final.silo"
            resumed = invoke(args, root, model + "-restart-default", resumed_config, restart=checkpoint)
            report["runs"].append(resumed)
            resumed_state = snapshot(reader, resumed, model, 4 * dt)
            require(any(resumed_state[1][name] != expected[1][name] for name in FIELDS),
                    f"{model} restart did not advance")
            requested = invoke(args, root, model + "-restart-ssp", resumed_config,
                               request="cli", restart=checkpoint, expect_warning=True)
            report["runs"].append(requested)
            requested_state = snapshot(reader, requested, model, 4 * dt)
            report["checks"].append(compare(resumed_state, requested_state,
                f"{model}: SSP after radiation restored from checkpoint versus default"))
            save_report(report_path, report)
        raw_radiation_rejection(args, root, reader, report)
        report["status"] = "passed"
        save_report(report_path, report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_report(report_path, report)
        raise
    print(f"PASS: radiation uses RK3; M1/FLD outputs unchanged; exact radiative SSP payloads rejected. {report_path}")


if __name__ == "__main__":
    main()
