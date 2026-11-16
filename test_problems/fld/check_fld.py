#!/usr/bin/env python3
"""Run bounded FLD application checks and verify numerical diagnostics."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess

try:
    import resource
except ImportError:
    resource = None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def near(actual, expected, tolerance, message):
    require(math.isfinite(actual) and
            abs(actual - expected) <= tolerance * max(1.0, abs(expected)),
            f"{message}: actual={actual:.17g}, expected={expected:.17g}")


def backward_euler_temperature(thermal, radiation, dt, rate):
    """Independent monotone scalar reference for rho=cv=ar=1."""
    total = thermal + radiation
    lower, upper = 0.0, total
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        residual = middle - thermal - dt * rate * (total - middle - middle**4)
        if residual > 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def command_for(args):
    amr = args.case == "amr"
    pulse = args.case in ("pulse", "amr")
    command = [
        str(args.exe.resolve()), f"--hpx:threads={args.threads}", "--hpx:bind=none",
        "--problem=sod", f"--hydro={'on' if args.case == 'rotation' else 'off'}",
        "--radiation=off", "--gravity=off",
        "--fld=on", "--hydro_host_kernel_type=LEGACY", "--hydro_device_kernel_type=OFF",
        "--disable_output=on", "--disable_diagnostics=on", "--disable_analytic=on",
        "--correct_am_hydro=off", "--sod_gamma=1.6666666666666667",
        "--sod_rhol=1", f"--sod_rhor={0.5 if amr else 1.0}",
        "--sod_pl=0.66666666666666663",
        f"--sod_pr={0.33333333333333331 if amr else 0.66666666666666663}",
        "--xscale=1", f"--min_level={1 if amr else args.level}",
        f"--max_level={3 if amr else args.level}",
        f"--unigrid={'off' if amr else 'on'}",
        "--fld_cv=1", "--fld_ar=1", "--fld_initial_energy=0.1",
        f"--fld_initial_pulse={0.5 if pulse else 0.0}",
        f"--fld_absorption={0.0 if pulse else 1000.0}",
        f"--fld_scattering={1.0 if pulse else 0.0}",
        "--fld_density_exponent=0", "--fld_temperature_exponent=0",
        f"--fld_c={1.0 if pulse else 100.0}",
        f"--fld_tolerance={args.tolerance:.17g}",
        f"--fld_max_iterations={1 if args.case == 'nonconvergence' else 100}",
        "--fld_max_cycles=200", "--hard_dt=0.01",
        f"--stop_step={1 if args.case == 'nonconvergence' else 3}",
        "--stop_time=0.03",
    ]
    if args.case == "rotation":
        # A uniform inertial fluid crosses both sides of the rotating box. The
        # default outflow-only condition would remove its incoming mesh-relative
        # velocity and invalidate the spatially uniform reference solution.
        command.extend(("--omega=0.7", "--inflow_bc=on"))
    if args.case == "dual_energy":
        # Force the real application's entropy-variable fallback at low Mach.
        # The production-helper test separately exercises high-Mach cancellation.
        command.append("--dual_energy_sw1=2")
    return command


def parse_steps(output):
    result = []
    required = {"dt", "iterations", "cycles", "residual", "radiation_energy",
                "thermal_energy", "min_energy", "min_temperature", "shared",
                "leaves", "cells", "tau_thermal_energy", "dual_energy_cells", "max_energy"}
    for line in output.splitlines():
        if not line.startswith("FLD_STEP "):
            continue
        row = dict(token.split("=", 1) for token in line.split()[1:])
        require(required.issubset(row), f"missing FLD diagnostics: {required - row.keys()}")
        values = {name: float(value) for name, value in row.items()}
        require(all(math.isfinite(value) for value in values.values()),
                "FLD diagnostics contain a nonfinite value")
        require(values["min_energy"] >= 0.0 and values["min_temperature"] > 0.0,
                "FLD returned inadmissible radiation or gas temperature")
        require(values["shared"] == 1.0, "grid and fld_grid have different radiation buffers")
        require(values["dt"] > 0.0 and values["residual"] >= 0.0,
                "FLD timestep or residual is invalid")
        require(values["leaves"] >= 1.0 and values["cells"] >= values["leaves"],
                "FLD reported an invalid mesh size")
        near(values["tau_thermal_energy"], values["thermal_energy"], 5.0e-11,
             "tau^(5/3) does not match the committed gas thermal energy")
        if "gas_total_energy" in values and "initial_gas_total_energy" in values:
            before = values["initial_gas_total_energy"] + values["initial_radiation_energy"]
            after = (values["gas_total_energy"] + values["radiation_energy"] +
                     values.get("escaped_diffusion_energy", 0.0))
            near(after, before, 5.0e-9,
                 "implicit radiation exchange failed gas-total/radiation energy conservation")
        result.append(values)
    return result


def run_application(command, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    for token in command:
        if token.startswith("--restart_filename="):
            checkpoint = Path(token.split("=", 1)[1])
            # OctoTiger's Silo multimesh stores relative shard paths, resolved
            # against the process directory by its existing restart loader.
            shard_name = checkpoint.name + ".data"
            shard_link = work_dir / shard_name
            if not shard_link.exists():
                shard_link.symlink_to(checkpoint.parent / shard_name, target_is_directory=True)
    (work_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, cwd=work_dir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=600)
    (work_dir / "application.log").write_text(result.stdout)
    return result


def replace_option(command, name, value):
    prefix = f"--{name}="
    return [token for token in command if not token.startswith(prefix)] + [f"{prefix}{value}"]


def accepted_rows(result, tolerance, minimum_steps=2):
    require(result.returncode == 0,
            f"FLD application exited {result.returncode}; see application.log")
    rows = parse_steps(result.stdout)
    require(len(rows) >= minimum_steps, "FLD test did not complete the required implicit steps")
    require(all(row["residual"] <= tolerance for row in rows),
            "FLD accepted a timestep above the nonlinear residual tolerance")
    return rows


def check_restart(args, command):
    # The first two steps create a nontrivial radiation/material state. Compare
    # continued Silo evolution with an uninterrupted four-step run, and inspect
    # the very first restarted input state before any new solve can hide a reset.
    first_command = replace_option(command, "disable_output", "off")
    first_command = replace_option(first_command, "stop_time", "0.02")
    first_command = replace_option(first_command, "stop_step", "4")
    first = accepted_rows(run_application(first_command, args.work_dir / "first"), args.tolerance)
    checkpoint = (args.work_dir / "first" / "final.silo").resolve()
    require(checkpoint.is_file(), "application did not create the final Silo checkpoint")
    continuous_command = replace_option(command, "stop_time", "0.04")
    continuous_command = replace_option(continuous_command, "stop_step", "4")
    continuous = accepted_rows(run_application(continuous_command, args.work_dir / "continuous"),
                               args.tolerance, 4)
    restart_command = continuous_command + [f"--restart_filename={checkpoint}"]
    restarted = accepted_rows(run_application(restart_command, args.work_dir / "restarted"),
                              args.tolerance)
    refined_command = replace_option(restart_command, "min_level", str(args.level + 1))
    refined_command = replace_option(refined_command, "max_level", str(args.level + 1))
    refined = accepted_rows(run_application(refined_command, args.work_dir / "refined"),
                            args.tolerance)
    coarsened = []
    if args.level > 0:
        coarse_command = replace_option(restart_command, "min_level", str(args.level - 1))
        coarse_command = replace_option(coarse_command, "max_level", str(args.level - 1))
        coarsened = accepted_rows(run_application(coarse_command, args.work_dir / "coarsened"),
                                 args.tolerance)
    check_tolerance = max(5.0e-8, 50.0 * args.tolerance)
    sequences = [restarted, refined] + ([coarsened] if coarsened else [])
    for sequence in sequences:
        for field in ("radiation_energy", "thermal_energy"):
            initial_field = "initial_" + field
            require(initial_field in sequence[0], "restart check requires initial-state diagnostics")
            near(sequence[0][initial_field], first[-1][field], check_tolerance,
                 f"Silo restart/regrid did not preserve {field}")
        for field in ("radiation_energy", "thermal_energy", "min_energy", "min_temperature"):
            near(sequence[-1][field], continuous[-1][field], check_tolerance,
                 f"restarted and uninterrupted {field} differ")
    require(all(row["leaves"] == 8**(args.level + 1) for row in refined),
            "restart refinement did not create the requested finer topology")
    require(all(row["leaves"] == 8**args.level for row in restarted),
            "ordinary restart did not preserve the leaf topology")
    require(all(row["leaves"] == 8**(args.level - 1) for row in coarsened),
            "restart coarsening did not create the requested coarser topology")
    check_uniform(first, check_tolerance)
    check_uniform(continuous, check_tolerance)
    metrics = {"first": first, "continuous": continuous, "restarted": restarted,
               "refined_restart": refined, "coarsened_restart": coarsened}
    (args.work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(restarted[-1], sort_keys=True))


def check_uniform(rows, tolerance):
    thermal, radiation = 1.0, 0.1
    for row in rows:
        old_total = thermal + radiation
        thermal = backward_euler_temperature(thermal, radiation, row["dt"], 1.0e5)
        radiation = old_total - thermal
        near(row["thermal_energy"] / 8.0, thermal, tolerance,
             "uniform thermal coupling disagrees with the backward-Euler root")
        near(row["radiation_energy"] / 8.0, radiation, tolerance,
             "uniform radiation coupling disagrees with the backward-Euler root")
        near(row["min_temperature"], thermal, tolerance,
             "uniform thermal state developed spatial variation")
        near(row["min_energy"], radiation, tolerance,
             "uniform radiation state developed spatial variation")
        near((row["thermal_energy"] + row["radiation_energy"]) / 8.0,
             1.1, tolerance, "uniform absorption/emission failed energy conservation")


def check_pulse(rows, amr, tolerance):
    first = rows[0]
    for index, row in enumerate(rows):
        near(row["thermal_energy"], 6.0 if amr else 8.0, tolerance,
             "elastic scattering changed gas thermal energy")
        near(row["radiation_energy"], first["radiation_energy"], tolerance,
             "closed-boundary diffusion failed radiation conservation")
        if "initial_radiation_energy" in row:
            near(row["radiation_energy"], row["initial_radiation_energy"], tolerance,
                 "diffusion lost radiation during this implicit step")
        require(row["min_energy"] >= 0.1 - tolerance,
                "diffusion violated the initial radiation lower bound")
        if index > 0:
            require(row["max_energy"] < rows[index - 1]["max_energy"] - tolerance,
                    "scattering diffusion did not spread the Gaussian radiation peak")
        if amr:
            require(8 < row["leaves"] < 512,
                    "AMR test did not exercise a mixed-resolution leaf hierarchy")
    if not amr:
        count = round(first["cells"] ** (1.0 / 3.0))
        require(count**3 == round(first["cells"]), "uniform mesh cell count is not cubic")
        dx = 2.0 / count
        one_dimensional_sum = sum(math.exp(-((-1.0 + (i + 0.5) * dx) / 0.2)**2)
                                  for i in range(count))
        initial_radiation = 0.8 + 0.5 * (dx * one_dimensional_sum)**3
        near(first["radiation_energy"], initial_radiation, tolerance,
             "first diffusion solve changed the initialized Gaussian integral")
        initial_peak = 0.1 + 0.5 * math.exp(-3.0 * (0.5 * dx / 0.2)**2)
        require(first["max_energy"] < initial_peak - tolerance,
                "first diffusion solve did not reduce the initial Gaussian peak")


def main():
    if resource is not None:
        _, hard_limit = resource.getrlimit(resource.RLIMIT_CORE)
        resource.setrlimit(resource.RLIMIT_CORE, (0, hard_limit))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("uniform", "pulse", "amr", "rotation", "restart",
                                          "dual_energy", "nonconvergence"),
                        required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()
    if not 0 <= args.level <= 2 or args.threads < 1:
        parser.error("require level 0..2 and positive thread count")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(args)
    if args.case == "restart":
        check_restart(args, command)
        return
    result = run_application(command, args.work_dir)
    if args.case == "nonconvergence":
        require(result.returncode != 0, "an unconverged FLD step incorrectly returned success")
        require(re.search(r"FLD[^\n]*(converg|iteration|cycle)", result.stdout,
                          flags=re.IGNORECASE) is not None,
                "application failed for a reason unrelated to FLD convergence")
        print("Expected FLD nonconvergence reported")
        return
    rows = accepted_rows(result, args.tolerance)
    check_tolerance = max(5.0e-8, 50.0 * args.tolerance)
    if args.case in ("uniform", "rotation", "dual_energy"):
        check_uniform(rows, check_tolerance)
        require(all(row["leaves"] == 8**args.level for row in rows),
                "uniform test did not use the requested leaf topology")
        if args.case == "dual_energy":
            require(all(row["dual_energy_cells"] == row["cells"] for row in rows),
                    "forced-fallback fixture did not recover every temperature from tau")
    else:
        check_pulse(rows, args.case == "amr", check_tolerance)
    (args.work_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows[-1], sort_keys=True))


if __name__ == "__main__":
    main()
