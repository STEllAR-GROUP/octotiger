#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Exercise both time integrators on small CPU-only hydro/gravity problems."""

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
import sys
import tempfile

from silo_snapshot import SiloSnapshot, compare_checkpoints


# Field ordering from octotiger/hydro_defs.hpp. These runs use one species.
RHO, EGAS, TAU, POT, SX, SY, SZ, SPECIES = 0, 1, 2, 3, 4, 5, 6, 10
N_FIELDS = 11
N_STEPS = 12
HARD_DT = 0.002


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_table(path, columns):
    require(path.is_file(), f"Missing diagnostic file: {path}")
    rows = [[float(token) for token in line.split()]
            for line in path.read_text().splitlines() if line.strip()]
    require(rows, f"Empty diagnostic file: {path}")
    for row in rows:
        require(len(row) == columns, f"Wrong diagnostic column count in {path}")
        require(all(math.isfinite(value) for value in row),
                f"Non-finite diagnostic values in {path}")
    return rows


def maximum_drift(rows, field):
    column = 1 + 2 * field  # sums.dat includes cumulative boundary outflow.
    initial = rows[0][column]
    return max(abs(row[column] - initial) for row in rows) / max(1.0, abs(initial))


def compare_tables(left, right, description, rel_tol=2.0e-12, abs_tol=2.0e-12):
    require(len(left) == len(right), f"Row counts differ for {description}")
    for row_index, (a_row, b_row) in enumerate(zip(left, right)):
        require(len(a_row) == len(b_row), f"Column counts differ for {description}")
        for column, (a, b) in enumerate(zip(a_row, b_row)):
            require(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol),
                    f"Mismatch in {description}, row {row_index}, "
                    f"column {column}: {a} != {b}")


def run_case(executable, scenario, integrator, directory, timeout, *,
             step_count=N_STEPS, start_time=0.0, start_step=0, stop_time=1.0,
             write_checkpoint=False, restart_from=None, integrator_from_config=False,
             expected_integrator=None, code_units=None, omega=0.0, expected_step_sizes=None,
             expected_failure=None, problem="SOD"):
    directory.mkdir()
    if restart_from is not None:
        # Silo mesh references are relative to the run directory.
        checkpoint = directory / restart_from.name
        shutil.copy2(restart_from, checkpoint)
        shutil.copytree(restart_from.parent / (restart_from.name + ".data"),
                        directory / (restart_from.name + ".data"))
    command = [
        str(executable), f"--hpx:threads={os.environ.get('SSP_TEST_HPX_THREADS', '2')}",
        "--hpx:ini=hpx.threadpools.io_pool_size=1",
        "--hpx:ini=hpx.threadpools.timer_pool_size=1",
        "--hpx:ini=hpx.threadpools.parcel_pool_size=1",
        "--hpx:ini=hpx.parcel.tcp.enable=0",
        f"--problem={problem}", "--eos=IDEAL",
        f"--gravity={'on' if scenario == 'gravity' else 'off'}", "--hydro=on",
        "--min_level=1", "--max_level=1", "--n_species=1", "--cfl=0.4",
        f"--hard_dt={HARD_DT}", f"--stop_step={step_count - 1}", f"--stop_time={stop_time}",
        f"--disable_output={'off' if write_checkpoint else 'on'}", "--disable_analytic=on", "--disable_diagnostics=off",
        "--theta=0.5", "--number_gpus=0", "--executors_per_gpu=0",
        "--hydro_host_kernel_type=LEGACY", "--hydro_device_kernel_type=OFF",
        "--multipole_host_kernel_type=LEGACY", "--multipole_device_kernel_type=OFF",
        "--monopole_host_kernel_type=LEGACY", "--monopole_device_kernel_type=OFF",
    ]
    if omega is not None:
        command.append(f"--omega={omega}")
    if code_units is not None:
        for unit, value in zip(("g", "cm", "s"), code_units):
            command.append(f"--code_to_{unit}={value}")
    if integrator is not None:
        if integrator_from_config:
            configuration = directory / "config.ini"
            configuration.write_text(f"time_integrator = {integrator}\n")
            command.append(f"--config_file={configuration}")
        else:
            command.append(f"--time_integrator={integrator}")
    if restart_from is not None:
        command.append(f"--restart_filename={checkpoint}")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_path = directory / "run.log"
    with log_path.open("w") as log:
        completed = subprocess.run(command, cwd=directory, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    log = log_path.read_text()
    if expected_failure is not None:
        require(completed.returncode != 0,
                f"Invalid checkpoint was unexpectedly accepted; inspect {log_path}")
        require(expected_failure in log,
                f"Expected checkpoint diagnostic missing from {log_path}\n{log[-4000:]}")
        require("SSP_MULTISTEP stages=" not in log and not re.search(r"TS (\d+)::", log),
                f"Rejected checkpoint advanced the solution; inspect {log_path}")
        return log
    require(completed.returncode == 0,
            f"Octo-Tiger exited {completed.returncode}; inspect {log_path}\n{log[-4000:]}")
    selected = expected_integrator or integrator or "RK3"
    require(f"time_integrator = {selected}" in log,
            f"Integrator selection missing from {log_path}")
    completed_steps = re.findall(r"TS (\d+)::", log)
    require(completed_steps and int(completed_steps[-1]) == step_count,
            f"Did not complete {step_count} timesteps; inspect {log_path}")
    expected_batches = (start_step + step_count + 4) // 5 - start_step // 5
    require(len(completed_steps) == expected_batches,
            f"Unexpected number of regrid/batch boundaries; inspect {log_path}")
    stages = re.findall(r"SSP_MULTISTEP stages=(\d+) alpha=([^\s]+)", log)
    if selected == "SSP_MULTISTEP":
        require(len(stages) == step_count,
                f"Expected {step_count} SSP stage reports, found {len(stages)} in {log_path}")
        require(any(stage == "2" for stage, _ in stages),
                f"Multistep path was never used; inspect {log_path}")
        if start_step == 0:
            require(stages[0][0] == "3", f"Initial startup must initialize history: {log_path}")
        else:
            require(stages[0][0] == "2", f"Restart discarded active SSP history: {log_path}")
        require([int(stage) for stage, _ in stages] ==
                [3 if step == 0 else 2
                 for step in range(start_step, start_step + step_count)],
                f"Unexpected startup/regrid stage sequence: {log_path}")
    else:
        require(not stages, f"Legacy integrator unexpectedly entered SSP path: {log_path}")
    sums = read_table(directory / "sums.dat", 1 + 2 * N_FIELDS + 3)
    profile = read_table(directory / "line.final.dat", N_FIELDS + 1)
    initial_profile = read_table(directory / "line.0.dat", N_FIELDS + 1)
    expected_time = start_time
    for offset, step in enumerate(range(start_step, start_step + step_count)):
        if expected_step_sizes is None:
            expected_time += min(HARD_DT, (stop_time - expected_time) / (5 - step % 5))
        else:
            expected_time += expected_step_sizes[offset]
    require(math.isclose(sums[-1][0], expected_time, rel_tol=1.0e-12),
            f"Unexpected final time in {directory / 'sums.dat'}")
    for row in profile:
        density, energy, entropy = row[1 + RHO], row[1 + EGAS], row[1 + TAU]
        require(density > 0.0 and entropy > 0.0, f"Non-positive hydro state in {directory}")
        kinetic = 0.5 * sum(row[1 + field] ** 2 for field in (SX, SY, SZ)) / density
        require(energy - kinetic > 0.0, f"Non-positive internal energy in {directory}")
    require(maximum_drift(sums, RHO) < 1.0e-10,
            f"Mass plus boundary outflow was not conserved in {directory}")
    require(maximum_drift(sums, SPECIES) < 1.0e-10,
            f"Species mass plus boundary outflow was not conserved in {directory}")
    energy_drift = maximum_drift(sums, EGAS)
    if scenario == "hydro":
        require(energy_drift < 1.0e-10,
                f"Hydro total energy plus boundary outflow was not conserved: {energy_drift}")
        require(all(row[1 + POT] == 0.0 for row in profile),
                "Gravity potential is nonzero in hydro-only run")
    else:
        require(any(row[1 + POT] < 0.0 for row in profile),
                "Self-gravity did not produce a negative gravitational potential")
    # A stationary or skipped-update path must not satisfy the smoke test.
    require(sorted(profile) != sorted(initial_profile), f"Hydro state never advanced: {directory}")
    if write_checkpoint:
        require((directory / "final.silo").is_file(),
                f"Missing final Silo checkpoint in {directory}")
    return sums, sorted(profile), energy_drift, stages


def run_restart(executable, scenario, directory, timeout, silo_library):
    reader = SiloSnapshot(silo_library)
    cases = [("inside_batch", 3, 7, 1.0, {}), ("batch_boundary", 5, 5, 1.0, {}),
             ("variable_dt", 11, 2, 0.024, {})]
    if scenario == "gravity":
        cases.append(("rotating_units", 3, 7, 1.0,
                      {"code_units": (1.25, 1.75, 0.875), "omega": 0.137}))
    for label, seed_steps, resumed_steps, stop_time, physics in cases:
        case = directory / label
        case.mkdir()
        total_steps = seed_steps + resumed_steps
        uninterrupted = run_case(executable, scenario, "SSP_MULTISTEP",
                                 case / "uninterrupted", timeout, step_count=total_steps,
                                 stop_time=stop_time, write_checkpoint=True, **physics)
        seed = run_case(executable, scenario, "SSP_MULTISTEP",
                        case / "seed", timeout, step_count=seed_steps,
                        stop_time=stop_time, write_checkpoint=True, **physics)
        # A fresh run still defaults to RK3. At this boundary restart omit the
        # selector entirely, requiring SSP_MULTISTEP to come from the checkpoint.
        automatic_selection = label == "batch_boundary"
        resumed_integrator = None if automatic_selection else "SSP_MULTISTEP"
        resumed = run_case(executable, scenario, resumed_integrator,
                           case / "resumed", timeout, step_count=resumed_steps,
                           start_time=seed[0][-1][0], start_step=seed_steps,
                           stop_time=stop_time, write_checkpoint=True,
                           restart_from=case / "seed" / "final.silo",
                           integrator_from_config=not automatic_selection,
                           expected_integrator="SSP_MULTISTEP", omega=None)
        # The rotating case deliberately supplies no nonunit/rotation overrides
        # on restart, so these also have to be restored from checkpoint metadata.
        require(seed[3] == uninterrupted[3][:seed_steps],
                f"Seed stage/alpha sequence changed in {label}")
        require(resumed[3] == uninterrupted[3][seed_steps:],
                f"Restart stage/alpha sequence differs from uninterrupted suffix in {label}")
        if label == "variable_dt":
            require(seed[0][-1][0] < seed_steps * HARD_DT,
                    "Variable-step checkpoint was taken before the timestep changed")
        # The opaque node archives contain unconverted state, gravity, outflows,
        # clocks and multistep history. Compare every byte for every node,
        # including the internal FMM root; rounded text is only a secondary check.
        nodes, byte_count = compare_checkpoints(reader,
            case / "uninterrupted" / "final.silo", case / "resumed" / "final.silo")
        compare_tables([seed[0][-1]], [resumed[0][0]], "checkpoint conserved sums",
                       rel_tol=0.0, abs_tol=0.0)
        compare_tables([uninterrupted[0][-1]], [resumed[0][-1]],
                       "uninterrupted/restarted conserved sums", rel_tol=0.0, abs_tol=0.0)
        compare_tables(uninterrupted[1], resumed[1], "uninterrupted/restarted profiles",
                       rel_tol=0.0, abs_tol=0.0)
        print(f"PASS restart {scenario}/{label}: {seed_steps}+{resumed_steps} steps; "
              f"{nodes} node archives/{byte_count} bytes and master metadata identical; "
              "SSP history and stage sequence preserved; "
              + ("checkpoint integrator selection verified." if automatic_selection
                 else "config selection verified."))

    # Changing stop_time intentionally changes the trajectory. Use it here to
    # impose abrupt changes inside a batch, where valid history must survive.
    # The exact-continuation comparisons above retain the same stopping policy.
    for label, seed_stop, resumed_stop in (
            ("large_decrease", 1.0, 3 * HARD_DT + 4.0e-5),
            ("growth_cap", 1.0e-4, 1.0)):
        case = directory / label
        case.mkdir()
        seed = run_case(executable, scenario, "SSP_MULTISTEP",
                        case / "seed", timeout, step_count=3,
                        stop_time=seed_stop, write_checkpoint=True)
        if label == "growth_cap":
            previous_dt = seed_stop / 5
            expected_sizes = [previous_dt * 1.9, previous_dt * 1.9 * 1.9]
        else:
            expected_sizes = None
        resumed = run_case(executable, scenario, "SSP_MULTISTEP",
                           case / "resumed", timeout, step_count=2,
                           start_time=seed[0][-1][0], start_step=3,
                           stop_time=resumed_stop,
                           restart_from=case / "seed" / "final.silo",
                           expected_step_sizes=expected_sizes)
        require(all(stage == "2" for stage, _ in resumed[3]),
                f"Timestep change discarded valid history: {label}")
        ratios = [float(alpha) for _, alpha in resumed[3]]
        if label == "large_decrease":
            require(ratios[0] > 99, "Large-decrease case did not exceed the old ratio cap")
        else:
            require(all(math.isclose(alpha, 1 / 1.9, rel_tol=1e-6)
                        for alpha in ratios), "Application did not enforce the 1.9 growth cap")
        print(f"PASS {scenario}/{label}: two-stage continuation; alpha={ratios}")


def rewrite_integer_metadata(reader, path, changes):
    """Change only selected metadata in a synthetic problem-ID migration fixture."""
    variables = reader.read(path, changes)
    silo = reader.silo
    silo.DBWrite.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p,
                            C.POINTER(C.c_int), C.c_int, C.c_int]
    silo.DBWrite.restype = C.c_int
    handle = silo.DBOpenReal(str(path.resolve()).encode(), 5, 2)  # UNKNOWN, APPEND
    require(handle, f"Cannot update fixture: {path}")
    try:
        integer_types = {16: C.c_int, 17: C.c_short, 18: C.c_long, 22: C.c_longlong}
        for name, values in changes.items():
            datatype, dimensions, _ = variables[name]
            require(len(dimensions) == 1 and dimensions[0] == len(values),
                    f"Metadata fixture shape changed: {name}")
            require(datatype in integer_types, f"Expected integer metadata: {name}")
            data = (integer_types[datatype] * len(values))(*values)
            dims = (C.c_int * 1)(len(values))
            require(silo.DBWrite(handle, name.encode(), data, dims, 1, datatype) == 0,
                    f"Cannot rewrite fixture metadata: {name}")
    finally:
        require(silo.DBClose(handle) == 0, f"Cannot close fixture: {path}")
    updated = reader.read(path, changes)
    for name, values in changes.items():
        require(reader.integers(updated[name]) == tuple(values),
                f"Fixture metadata did not change: {name}")


def copy_checkpoint(source, directory):
    directory.mkdir()
    checkpoint = directory / source.name
    shutil.copy2(source, checkpoint)
    shutil.copytree(source.parent / (source.name + ".data"),
                    directory / (source.name + ".data"))
    return checkpoint


def run_problem_id_compatibility(executable, directory, timeout, silo_library):
    """Check metadata-only migration of the old non-radiative pulse ID 12."""
    reader = SiloSnapshot(silo_library)
    for integrator in ("SSP_MULTISTEP", "RK3"):
        case = directory / integrator.lower()
        case.mkdir()
        exact = integrator == "SSP_MULTISTEP"
        physics = {"problem": "AMR_ADVECTION"}
        seed = run_case(executable, "hydro", integrator, case / "seed", timeout,
                        step_count=3, write_checkpoint=True, **physics)
        original = case / "seed" / "final.silo"
        names = ["problem", "radiation"]
        if exact:
            names.append("restart_solver_integers")
        metadata = reader.read(original, names)
        require(reader.integers(metadata["problem"]) == (22,),
                "New pulse checkpoints must use problem ID 22")
        require(reader.integers(metadata["radiation"]) == (0,),
                "Pulse fixture unexpectedly enables radiation")
        changes = {"problem": (12,)}
        if exact:
            solver = list(reader.integers(metadata["restart_solver_integers"]))
            require(solver[4] == 22, "Exact solver metadata has the wrong problem ID")
            solver[4] = 12
            changes["restart_solver_integers"] = solver
        legacy = copy_checkpoint(original, case / "legacy_id12")
        rewrite_integer_metadata(reader, legacy, changes)
        if exact:
            require(reader.checkpoint(original) == reader.checkpoint(legacy),
                    "Problem-ID fixture changed opaque node state")
        resumed = []
        for label, checkpoint in (("canonical", original), ("migrated", legacy)):
            result = run_case(executable, "hydro", None if exact else "RK3",
                              case / label, timeout, step_count=2,
                              start_time=seed[0][-1][0], start_step=3 if exact else 0,
                              write_checkpoint=True, restart_from=checkpoint,
                              expected_integrator=integrator, omega=None, **physics)
            resumed.append(result)
            final = case / label / "final.silo"
            require(reader.integers(reader.read(final, ("problem",))["problem"]) == (22,),
                    "Restart failed to write the canonical pulse ID")
        compare_tables(resumed[0][0], resumed[1][0], "canonical/legacy-ID conserved sums",
                       rel_tol=0.0, abs_tol=0.0)
        compare_tables(resumed[0][1], resumed[1][1], "canonical/legacy-ID profiles",
                       rel_tol=0.0, abs_tol=0.0)
        if exact:
            compare_checkpoints(reader, case / "canonical" / "final.silo",
                                case / "migrated" / "final.silo")
            require(resumed[0][3] == resumed[1][3],
                    "Problem-ID migration changed the SSP stage/ratio sequence")
            # Reject inconsistent copies before migration; otherwise normalization
            # could hide disagreement between the two saved problem identifiers.
            inconsistent = copy_checkpoint(legacy, case / "inconsistent_id")
            solver[4] = 22
            rewrite_integer_metadata(reader, inconsistent, {"restart_solver_integers": solver})
            run_case(executable, "hydro", None, case / "inconsistent_rejected", timeout,
                     step_count=2, restart_from=inconsistent,
                     expected_failure="Exact checkpoint problem metadata is inconsistent", **physics)
            # A radiation marker must never be reinterpreted as a hydro pulse,
            # including when the exact payload automatically selects SSP.
            radiative = copy_checkpoint(legacy, case / "radiation_marker")
            rewrite_integer_metadata(reader, radiative, {"radiation": (1,)})
            run_case(executable, "hydro", None, case / "radiation_rejected", timeout,
                     step_count=2, restart_from=radiative, expected_failure="radiation", **physics)
        print(f"PASS problem-ID migration {integrator}: old non-radiative 12 -> 22; "
              "canonical and migrated continuations match exactly.")


def run_legacy_compatibility(executable, legacy_executable, scenario, directory,
                             timeout, silo_library):
    """Use genuine old-executable fixtures; never relabel current archives."""
    reader = SiloSnapshot(silo_library)
    require(legacy_executable.is_file(), f"Legacy executable not found: {legacy_executable}")
    require(legacy_executable != executable,
            "Compatibility fixtures require a distinct legacy executable")
    provenance = {
        "legacy_command": str(legacy_executable),
        # This can be a launcher script. Label it as the command entrypoint,
        # not the underlying executable, whose filename the script may choose.
        "legacy_command_sha256": hashlib.sha256(legacy_executable.read_bytes()).hexdigest(),
        "current_command": str(executable),
        "scenario": scenario,
        "fixtures": [],
    }
    fixtures = {}
    for label, seed_steps in (("inside_batch", 3), ("batch_boundary", 5)):
        seed_dir = directory / ("legacy_" + label)
        result = run_case(legacy_executable, scenario, "SSP_MULTISTEP", seed_dir,
                          timeout, step_count=seed_steps, write_checkpoint=True)
        checkpoint = seed_dir / "final.silo"
        version = reader.integers(reader.read(checkpoint, ("restart_version",))["restart_version"])
        require(version == (1,),
                f"Legacy command did not generate a genuine format-1 fixture: {checkpoint}")
        payloads = reader.checkpoint(checkpoint)
        build_lines = [line for line in (seed_dir / "run.log").read_text().splitlines()
                       if line.startswith("GIT COMMIT:")]
        provenance["fixtures"].append({
            "label": label,
            "accepted_steps": seed_steps,
            "format_version": version[0],
            "checkpoint": str(checkpoint),
            "checkpoint_master_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "build_git_commit_report": build_lines,
            "node_payload_sha256": {str(node): hashlib.sha256(data).hexdigest()
                                    for node, data in sorted(payloads.items())},
        })
        fixtures[label] = (checkpoint, result[0][-1][0])
        print(f"Legacy fixture {scenario}/{label}: executable={legacy_executable}; "
              f"format=1; steps={seed_steps}; nodes={len(payloads)}; "
              f"master_sha256={provenance['fixtures'][-1]['checkpoint_master_sha256']}")
    provenance_path = directory / "legacy_fixture_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    # Neither run crosses an old-code bootstrap at step five before loading.
    # At the new-code final regrid both runs populate identical parent history,
    # allowing exact archive comparison as well as physical-field checks.
    uninterrupted = run_case(executable, scenario, "SSP_MULTISTEP",
                             directory / "current_uninterrupted", timeout,
                             step_count=5, write_checkpoint=True)
    checkpoint, start_time = fixtures["inside_batch"]
    resumed = run_case(executable, scenario, None, directory / "current_resumed",
                       timeout, step_count=2, start_step=3, start_time=start_time,
                       restart_from=checkpoint, expected_integrator="SSP_MULTISTEP",
                       write_checkpoint=True, omega=None)
    nodes, byte_count = compare_checkpoints(reader,
        directory / "current_uninterrupted" / "final.silo",
        directory / "current_resumed" / "final.silo")
    require(resumed[3] == uninterrupted[3][3:],
            "Format-1 inside-batch restart changed the multistep stage/ratio sequence")
    compare_tables([uninterrupted[0][-1]], [resumed[0][-1]],
                   "format-1 continuation conserved sums", rel_tol=0.0, abs_tol=0.0)
    compare_tables(uninterrupted[1], resumed[1],
                   "format-1 continuation profiles", rel_tol=0.0, abs_tol=0.0)
    print(f"PASS format-1 compatibility {scenario}/inside_batch: "
          f"3+2 steps; {nodes} node archives/{byte_count} bytes identical; "
          "checkpoint integrator selection and uninterrupted two-stage continuation verified.")

    checkpoint, start_time = fixtures["batch_boundary"]
    message = ("Format-1 checkpoint at a regrid boundary may lack AMR history; "
               "restart from an inside-batch checkpoint or use the original executable")
    run_case(executable, scenario, None, directory / "boundary_rejected", timeout,
             step_count=2, start_step=5, start_time=start_time,
             restart_from=checkpoint, omega=None, expected_failure=message)
    print(f"PASS format-1 compatibility {scenario}/batch_boundary: "
          "rejected before advancing with the specific recovery diagnostic.")
    print(f"Legacy fixture provenance: {provenance_path}")


def run_baselines(executable, scenario, directory, timeout):
    results = []
    for label, integrator in (("default", None), ("rk3", "RK3"),
                              ("multistep", "SSP_MULTISTEP")):
        results.append(run_case(executable, scenario, integrator,
                                directory / label, timeout))
    compare_tables(results[0][0], results[1][0], "default/RK3 conserved sums")
    compare_tables(results[0][1], results[1][1], "default/RK3 final hydro profiles")
    if scenario == "gravity":
        # The FMM force and gravitational work update have finite numerical
        # error. Bound the new method against RK3 at the same mesh and dt.
        allowed_drift = max(1.0e-8, 4.0 * results[1][2])
        require(results[2][2] <= allowed_drift,
                f"Gravity energy drift {results[2][2]:.6e} exceeds "
                f"RK3-based bound {allowed_drift:.6e}")
    print(f"PASS {scenario}: default=RK3; {N_STEPS} SSP steps; "
          "startup and multistep stages; finite positive states; conserved mass.")
    print("Normalized total-energy drifts (default, RK3, multistep): "
          + ", ".join(f"{result[2]:.6e}" for result in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("scenario", choices=("hydro", "gravity", "restart", "restart_hydro",
                        "restart_gravity", "compatibility_hydro", "compatibility_gravity", "problem_id"))
    parser.add_argument("--work-directory", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--silo-library", type=Path)
    parser.add_argument("--legacy-executable", type=Path,
                        help="format-1 Octo-Tiger executable/launcher for compatibility scenarios")
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()
    if args.scenario.startswith("compatibility_") and args.legacy_executable is None:
        parser.error("compatibility scenarios require --legacy-executable (a genuine format-1 writer)")
    executable = args.executable.resolve()
    require(executable.is_file(), f"Executable not found: {executable}")
    args.work_directory.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"ssp_multistep_{args.scenario}_",
                                      dir=args.work_directory.resolve()))
    try:
        if args.scenario == "problem_id":
            run_problem_id_compatibility(executable, directory, args.timeout, args.silo_library)
        elif args.scenario.startswith("compatibility_"):
            scenario = args.scenario[len("compatibility_"):]
            run_legacy_compatibility(executable, args.legacy_executable.resolve(), scenario,
                                     directory, args.timeout, args.silo_library)
        elif args.scenario == "restart":
            for scenario in ("hydro", "gravity"):
                child = directory / scenario
                child.mkdir()
                run_restart(executable, scenario, child, args.timeout, args.silo_library)
        elif args.scenario.startswith("restart_"):
            scenario = args.scenario[len("restart_"): ]
            run_restart(executable, scenario, directory, args.timeout, args.silo_library)
        else:
            run_baselines(executable, args.scenario, directory, args.timeout)
    except Exception:
        print(f"Retained runtime test artifacts: {directory}", file=sys.stderr)
        raise
    if args.keep_output:
        print(f"Runtime test artifacts: {directory}")
    else:
        shutil.rmtree(directory)


if __name__ == "__main__":
    main()
