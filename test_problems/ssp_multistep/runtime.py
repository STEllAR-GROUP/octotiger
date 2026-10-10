#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Exercise both time integrators on small CPU-only hydro/gravity problems."""

import argparse
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
             expected_integrator=None, code_units=None, omega=0.0):
    directory.mkdir()
    if restart_from is not None:
        # Silo mesh references are relative to the run directory.
        checkpoint = directory / restart_from.name
        shutil.copy2(restart_from, checkpoint)
        shutil.copytree(restart_from.parent / (restart_from.name + ".data"),
                        directory / (restart_from.name + ".data"))
    command = [
        str(executable), "--hpx:threads=2",
        "--hpx:ini=hpx.threadpools.io_pool_size=1",
        "--hpx:ini=hpx.threadpools.timer_pool_size=1",
        "--hpx:ini=hpx.threadpools.parcel_pool_size=1",
        "--hpx:ini=hpx.parcel.tcp.enable=0",
        "--problem=SOD", "--eos=IDEAL",
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
        if start_step % 5 == 0:
            require(stages[0][0] == "3", f"Batch startup must initialize history: {log_path}")
        else:
            require(stages[0][0] == "2", f"Restart discarded active SSP history: {log_path}")
        require([int(stage) for stage, _ in stages] ==
                [3 if step % 5 == 0 else 2
                 for step in range(start_step, start_step + step_count)],
                f"Unexpected startup/regrid stage sequence: {log_path}")
    else:
        require(not stages, f"Legacy integrator unexpectedly entered SSP path: {log_path}")
    sums = read_table(directory / "sums.dat", 1 + 2 * N_FIELDS + 3)
    profile = read_table(directory / "line.final.dat", N_FIELDS + 1)
    initial_profile = read_table(directory / "line.0.dat", N_FIELDS + 1)
    expected_time = start_time
    for step in range(start_step, start_step + step_count):
        expected_time += min(HARD_DT, (stop_time - expected_time) / (5 - step % 5))
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
    parser.add_argument("scenario", choices=("hydro", "gravity", "restart", "restart_hydro", "restart_gravity"))
    parser.add_argument("--work-directory", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--silo-library", type=Path)
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    require(executable.is_file(), f"Executable not found: {executable}")
    args.work_directory.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"ssp_multistep_{args.scenario}_",
                                      dir=args.work_directory.resolve()))
    try:
        if args.scenario == "restart":
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
