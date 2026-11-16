#!/usr/bin/env python3
"""Bounded WD/FLD application checks; the SCF and shock cases are optional smokes."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import time

from check_fld import replace_option, require


GAMMA = 5.0 / 3.0
KB = 1.380658e-16
MH = 1.6733e-24
HPLANCK = 6.6260755e-27
LIGHT_SPEED = 2.99792458e10
SIGMA = 5.67051e-5
ACGS = 6.00228e22
BCGS = 2.0 * 9.81011e5
LENGTH = 1.0e9
MASS = BCGS * LENGTH**3
TIME = LENGTH * math.sqrt(BCGS / ACGS)
C = LIGHT_SPEED * TIME / LENGTH
AR = 4.0 * SIGMA / LIGHT_SPEED / ACGS
TEMPERATURE = 3.0e8
CUTOFF = 1.0e4
CV = KB / MH * (TIME / LENGTH)**2 * 0.75 / (GAMMA - 1.0)


def near(actual, expected, message, relative=2.0e-7, absolute=1.0e-13):
    require(math.isfinite(actual) and
            abs(actual - expected) <= absolute + relative * abs(expected),
            f"{message}: actual={actual:.17g}, expected={expected:.17g}")


def cold_energy(rho):
    """Cold EOS energy for the SOD fixture's unchanged A=B=1 code constants."""
    x = rho**(1.0 / 3.0)
    if x < 0.01:
        return 2.4 * x**5
    pressure = x * (2.0 * x*x - 3.0) * math.sqrt(1.0 + x*x) + 3.0 * math.asinh(x)
    return 8.0 * (math.sqrt(1.0 + x*x) - 1.0) * rho - pressure


def sod_pressure(rho, temperature=TEMPERATURE):
    # SOD initializes Eg=p/(gamma-1). Supply the cold+thermal total explicitly;
    # dual_energy_sw1=0 then recovers the thermal reservoir by subtraction.
    return (GAMMA - 1.0) * (cold_energy(rho) + rho * CV * temperature)


def command_for(args, **changes):
    options = {
        "problem": "sod", "eos": "wd", "hydro": "off", "gravity": "off",
        "radiation": "off", "fld": "on", "n_species": 1,
        "atomic_mass": 4, "atomic_number": 2,
        "hydro_host_kernel_type": "LEGACY", "hydro_device_kernel_type": "OFF",
        "monopole_host_kernel_type": "LEGACY", "multipole_host_kernel_type": "LEGACY",
        "monopole_device_kernel_type": "OFF", "multipole_device_kernel_type": "OFF",
        "disable_output": "on", "disable_diagnostics": "on", "disable_analytic": "on",
        "correct_am_hydro": "off", "sod_gamma": GAMMA,
        "sod_rhol": 1.0, "sod_rhor": 1.0,
        "sod_pl": sod_pressure(1.0), "sod_pr": sod_pressure(1.0),
        "sod_phi": 90, "sod_theta": 0, "dual_energy_sw1": 0,
        "xscale": 1, "min_level": args.level, "max_level": args.level,
        "unigrid": "on", "code_to_g": MASS, "code_to_cm": LENGTH, "code_to_s": TIME,
        "fld_material_model": "fully_ionized", "fld_unit_system": "physical",
        "fld_opacity_model": "free_free_thomson", "fld_opacity_temperature_floor": CUTOFF,
        "fld_gaunt": 1, "fld_initial_energy": 0, "fld_initial_pulse": 0,
        "fld_absorption": 1000, "fld_rosseland": 2,
        "fld_tolerance": args.tolerance, "fld_max_iterations": 100, "fld_max_cycles": 200,
        "hard_dt": 1.0e-4, "stop_step": 3, "stop_time": 3.0e-4,
    }
    options.update(changes)
    command = [str(args.exe.resolve()), f"--hpx:threads={args.threads}", "--hpx:bind=none"]
    for name, value in options.items():
        if isinstance(value, (list, tuple)):
            command.extend([f"--{name}"] + [str(item) for item in value])
        else:
            command.append(f"--{name}={value}")
    return command


def run(args, command, name):
    work = args.work_dir / name
    work.mkdir(parents=True, exist_ok=True)
    for token in command:
        if token.startswith("--restart_filename="):
            checkpoint = Path(token.split("=", 1)[1])
            link = work / (checkpoint.name + ".data")
            if not link.exists():
                link.symlink_to(str(checkpoint) + ".data", target_is_directory=True)
    (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    remaining = args.deadline - time.monotonic()
    require(remaining > 0.0, "WD/FLD application-check wall-time budget exhausted")
    process = subprocess.Popen(command, cwd=work, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = process.communicate(timeout=remaining)
    except subprocess.TimeoutExpired as error:
        process.kill()
        output, _ = process.communicate()
        (work / "application.log").write_text(output)
        raise RuntimeError(f"{name} exceeded the total {args.timeout}s check budget") from error
    except BaseException:
        process.kill()
        output, _ = process.communicate()
        (work / "application.log").write_text(output)
        raise
    (work / "application.log").write_text(output)
    require(process.returncode == 0, f"{name} exited {process.returncode}; see {work}/application.log")
    return output


def parse_steps(output, tolerance, minimum=1):
    rows = []
    required = {"dt", "residual", "radiation_energy", "thermal_energy", "min_energy",
                "min_temperature", "cold_energy", "atmosphere_thermal_energy",
                "atmosphere_radiation_energy", "opacity_regularized_cells", "shared",
                "initial_radiation_energy", "initial_thermal_energy", "gas_total_energy",
                "initial_gas_total_energy", "tau_thermal_energy", "cells", "leaves"}
    for line in output.splitlines():
        if not line.startswith("FLD_STEP "):
            continue
        row = {key: float(value) for key, value in
               (token.split("=", 1) for token in line.split()[1:])}
        require(required.issubset(row), f"missing WD diagnostics: {required - row.keys()}")
        require(all(math.isfinite(value) for value in row.values()), "nonfinite WD diagnostics")
        require(row["min_energy"] >= 0 and row["min_temperature"] >= 0,
                "WD material or radiation energy became negative")
        require(row["shared"] == 1 and row["dt"] > 0 and 0 <= row["residual"] <= tolerance,
                "WD FLD failed its shared-state, timestep, or convergence contract")
        require(0 <= row["opacity_regularized_cells"] <= row["cells"],
                "opacity cutoff count includes nonleaf/ghost cells")
        require(row["cold_energy"] > 0, "fixture did not exercise the WD cold reservoir")
        near(row["tau_thermal_energy"], row["thermal_energy"], "tau/thermal agreement",
             relative=5.0e-10, absolute=1.0e-25)
        near(row["gas_total_energy"] + row["radiation_energy"] +
             row.get("escaped_diffusion_energy", 0.0),
             row["initial_gas_total_energy"] + row["initial_radiation_energy"],
             "gas-total plus radiation conservation in the implicit step", relative=5.0e-10)
        rows.append(row)
    require(len(rows) >= minimum, "WD fixture did not complete the required implicit steps")
    return rows


def planck_absorption(rho, temperature, model, independent_absorption=1000.0):
    if model == "independent":
        return independent_absorption
    number = rho * BCGS / MH
    return (3.692e8 * number**2 * 0.5 * (HPLANCK / KB)**3 *
            max(temperature, CUTOFF)**-3.5 * 15.0 / math.pi**4 * LENGTH)


def backward_euler(thermal, radiation, dt, model, independent_absorption=1000.0):
    total = thermal + radiation
    low, high = 0.0, total / CV
    for _ in range(120):
        temperature = 0.5 * (low + high)
        residual = (CV * temperature - thermal - dt * C *
                    planck_absorption(1.0, temperature, model, independent_absorption) *
                    (total - CV * temperature - AR * temperature**4))
        if residual > 0.0:
            high = temperature
        else:
            low = temperature
    temperature = 0.5 * (low + high)
    return CV * temperature, total - CV * temperature


def check_uniform(rows, model, volume=8.0, independent_absorption=1000.0):
    thermal, radiation = CV * TEMPERATURE, 0.0
    for row in rows:
        thermal, radiation = backward_euler(thermal, radiation, row["dt"], model, independent_absorption)
        near(row["thermal_energy"] / volume, thermal, "WD scalar BE thermal reservoir")
        near(row["radiation_energy"] / volume, radiation, "WD scalar BE emitted radiation",
             relative=2.0e-5, absolute=2.0e-10)
        near(row["min_temperature"], thermal / CV, "WD physical Kelvin temperature")
        near(row["cold_energy"] / volume, cold_energy(1.0), "cold reservoir changed")
        require(row["opacity_regularized_cells"] == 0, "heated fixture unexpectedly hit cutoff")
    require(rows[-1]["radiation_energy"] > 0, "heated WD material did not emit radiation")


def cold_case(args):
    command = command_for(args, problem="star", gravity="on", xscale=2.56,
                          star_rho_out=1.0e-10, rho_floor=1.0e-10, tau_floor=1.0e-30,
                          disable_diagnostics="off",
                          hard_dt=5.0e-7, stop_time=1.0e-6, stop_step=1)
    metrics = {}
    for name, cutoff in (("cold", 1.0e4), ("cold_cutoff_low", 1.0e3), ("cold_cutoff_high", 1.0e5)):
        output = run(args, replace_option(command, "fld_opacity_temperature_floor", cutoff), name)
        rows = parse_steps(output, args.tolerance)
        for row in rows:
            # The tiny entropy floor remains a thermal floor, not cold energy.
            require(row["thermal_energy"] / row["cold_energy"] < 1.0e-12,
                    "cold degeneracy energy was treated as thermal energy")
            require(row["radiation_energy"] <=
                    2.0 * row["initial_thermal_energy"] + 1.0e-100,
                    "opacity cutoff emitted more energy than the cold material reservoir")
            require(row["opacity_regularized_cells"] == row["cells"],
                    "cold-star opacity regularization count is incomplete")
        near(rows[-1]["cold_energy"], rows[0]["cold_energy"], "frozen cold-star reservoir")
        if name != "cold":
            for field in ("cold_energy", "thermal_energy", "radiation_energy"):
                near(rows[-1][field], metrics["cold"][-1][field],
                     f"cold cutoff sensitivity: {field}", absolute=1.0e-50)
        metrics[name] = rows
    model = re.search(r"Equilibrium WD: mass=(\S+) g radius=(\S+) cm rho_center=(\S+) code", output)
    require(model is not None, "cold STAR did not report its initialized WD model")
    metrics.update(mass_g=float(model[1]), radius_cm=float(model[2]), rho_center_code=float(model[3]))
    dynamic = parse_steps(run(args, replace_option(command, "hydro", "on"), "cold_dynamic"), args.tolerance)
    initial = metrics["cold"][0]
    final = dynamic[-1]
    reference_line = shock_line(args.work_dir / "cold" / "line.final.dat")
    dynamic_line = shock_line(args.work_dir / "cold_dynamic" / "line.final.dat")
    require(len(reference_line) == len(dynamic_line), "cold dynamic profile topology changed")
    density_l1 = sum(abs(a[1] - b[1]) for a, b in zip(reference_line, dynamic_line))
    density_sum = sum(row[1] for row in reference_line)
    metrics["cold_dynamic"] = dynamic
    metrics["cold_dynamic_metrics"] = {
        "cold_energy_relative_drift": (final["cold_energy"] - initial["cold_energy"]) / initial["cold_energy"],
        "thermal_fraction": final["thermal_energy"] / initial["cold_energy"],
        "radiation_fraction": final["radiation_energy"] / initial["cold_energy"],
        "density_line_relative_l1": density_l1 / density_sum,
        "density_line_max_initial": max(row[1] for row in reference_line),
        "density_line_max_final": max(row[1] for row in dynamic_line),
        "elapsed_code_time": sum(row["dt"] for row in dynamic),
    }
    metrics["limitations"] = ("Frozen cold cutoff checks plus two tiny coupled hydro/gravity steps. "
                              "Density and energy drift are recorded; no resolved equilibrium or long-time stability claim.")
    return metrics


def independent_case(args):
    rows = parse_steps(run(args, command_for(args, fld_opacity_model="independent"), "independent"), args.tolerance, 2)
    check_uniform(rows, "independent")
    metrics = {"independent": rows}
    for factor in (0.5, 2.0):
        # Hold rho_unit and velocity_unit fixed, preserving A=B=1, CV and ar.
        # Rescale the box, dt and inverse-code-length means so the physical
        # domain, elapsed time and opacities remain exactly the same.
        name = f"unit_scale_{factor:g}"
        command = command_for(args, fld_opacity_model="independent",
                              code_to_g=MASS * factor**3, code_to_cm=LENGTH * factor,
                              code_to_s=TIME * factor, xscale=1.0 / factor,
                              hard_dt=1.0e-4 / factor, stop_time=3.0e-4 / factor,
                              fld_absorption=1000.0 * factor, fld_rosseland=2.0 * factor)
        scaled = parse_steps(run(args, command, name), args.tolerance, 2)
        volume = 8.0 / factor**3
        check_uniform(scaled, "independent", volume, 1000.0 * factor)
        require(len(scaled) == len(rows), "equivalent unit mappings took different step counts")
        for original, alternate in zip(rows, scaled):
            for field in ("radiation_energy", "thermal_energy", "cold_energy"):
                near(alternate[field] / volume, original[field] / 8.0,
                     f"unit-invariant {field}", relative=5.0e-7)
            near(alternate["min_temperature"], original["min_temperature"], "unit-invariant Kelvin temperature")
            near(alternate["dt"] * factor, original["dt"], "unit-invariant physical timestep")
        metrics[name] = scaled
    return metrics


def amr_case(args):
    command = command_for(args, min_level=1, max_level=3, unigrid="off",
                          sod_rhor=0.5, sod_pr=sod_pressure(0.5),
                          fld_opacity_model="independent", fld_absorption=0,
                          fld_rosseland=2, fld_rosseland_density_exponent=1,
                          fld_initial_energy=1.0e-4, fld_initial_pulse=5.0e-4)
    rows = parse_steps(run(args, command, "amr"), args.tolerance, 2)
    for row in rows:
        require(8 < row["leaves"] < 512, "WD transport did not exercise mixed AMR levels")
        near(row["thermal_energy"], row["initial_thermal_energy"], "pure transport heated WD gas")
        near(row["radiation_energy"], row["initial_radiation_energy"],
             "closed AMR transport lost radiation", relative=2.0e-7)
        near(row["cold_energy"], rows[0]["cold_energy"], "AMR transport changed cold energy")
    require(rows[-1]["max_energy"] < rows[0]["max_energy"], "AMR radiation pulse did not spread")
    return {"amr": rows}


def restart_case(args):
    base = command_for(args, fld_opacity_model="independent", stop_time=4.0e-4, stop_step=8)
    first_command = replace_option(replace_option(base, "disable_output", "off"), "stop_time", 2.0e-4)
    first = parse_steps(run(args, first_command, "first"), args.tolerance, 2)
    checkpoint = (args.work_dir / "first" / "final.silo").resolve()
    require(checkpoint.is_file(), "WD checkpoint was not written")
    continuous = parse_steps(run(args, base, "continuous"), args.tolerance, 4)
    metrics = {"first": first, "continuous": continuous}
    for name, level in (("restarted", args.level), ("refined", args.level + 1),
                        ("coarsened", max(0, args.level - 1))):
        command = base + [f"--restart_filename={checkpoint}"]
        command = replace_option(replace_option(command, "min_level", level), "max_level", level)
        rows = parse_steps(run(args, command, name), args.tolerance, 2)
        for field in ("radiation_energy", "thermal_energy"):
            near(rows[0]["initial_" + field], first[-1][field], f"restart input {field}")
        for field in ("radiation_energy", "thermal_energy", "cold_energy", "min_temperature"):
            near(rows[-1][field], continuous[-1][field], f"{name} final {field}", relative=5.0e-6)
        require(all(row["leaves"] == 8**level for row in rows), f"{name} topology mismatch")
        metrics[name] = rows
    check_uniform(continuous, "independent")
    return metrics


def shock_line(path):
    rows = [[float(item) for item in line.split()] for line in path.read_text().splitlines()]
    require(rows and all(len(row) >= 8 for row in rows), "missing shock line diagnostics")
    require(all(math.isfinite(value) for row in rows for value in row), "nonfinite shock profile")
    require(all(row[1] > 0 and row[3] >= 0 for row in rows), "invalid shock density/thermal tracer")
    return rows


def shock_case(args):
    # This exercises hydrodynamic transport and physical emission. The ideal SOD
    # analytic solution is intentionally disabled: this is a WD EOS shock smoke.
    base = command_for(args, hydro="on", disable_diagnostics="off", sod_rhor=0.5,
                       sod_pr=sod_pressure(0.5, 0.5 * TEMPERATURE),
                       stop_time=0.1, stop_step=64, hard_dt=0.005)
    metrics = {}
    lines = {}
    for name, timestep, enabled in (("radiation_off", 0.005, False),
                                    ("radiation_on", 0.005, True),
                                    ("half_dt", 0.0025, True)):
        command = replace_option(replace_option(base, "hard_dt", timestep), "fld", "on" if enabled else "off")
        output = run(args, command, name)
        metrics[name] = parse_steps(output, args.tolerance, 2) if enabled else []
        lines[name] = shock_line(args.work_dir / name / "line.final.dat")
        # The initial fluid is stationary. Require measurable motion so this
        # cannot pass by checking only the local relaxation of the initial jump.
        speed = max(math.sqrt(sum(value * value for value in row[5:8])) / row[1]
                    for row in lines[name])
        require(speed > 0.005, "WD pressure discontinuity did not develop hydrodynamic motion")
        metrics[name + "_maximum_speed"] = speed
    coarse, fine = metrics["radiation_on"][-1], metrics["half_dt"][-1]
    require(coarse["radiation_energy"] > 0 and fine["radiation_energy"] > 0,
            "heated WD shock emitted no radiation")
    near(coarse["radiation_energy"], fine["radiation_energy"],
         "shock radiation changed excessively under dt/2", relative=0.1)
    require(len(lines["radiation_off"]) == len(lines["radiation_on"]), "shock line topology mismatch")
    difference = sum(abs(a[3]**GAMMA - b[3]**GAMMA)
                     for a, b in zip(lines["radiation_off"], lines["radiation_on"]))
    metrics["thermal_line_l1_difference"] = difference / len(lines["radiation_off"])
    require(difference > 0, "radiation on/off shock profiles are indistinguishable")
    metrics["limitations"] = ("Evolved to code time 0.1 with measurable hydrodynamic motion. "
                              "The coarse shock profile and two timesteps are a smoke comparison, "
                              "not a resolved-shock or convergence-order measurement.")
    return metrics


def scf_case(args):
    work = args.work_dir / "scf"
    work.mkdir(parents=True, exist_ok=True)
    # Synthetic q=.7 fixture only. These masses are not a selected production DWD.
    (work / "scf.init").write_text("M1=1.0\nM2=0.7\na=1.0\nfill2=0.99\n"
                                 "fill1=0.99\nequal_struct_eos=1\ncontact_fill=0\n")
    command = command_for(args, problem="dwd", gravity="on", hydro="off",
                          n_species=5, atomic_mass=[4] * 5, atomic_number=[2] * 5,
                          min_level=3, max_level=3, xscale=1, disable_diagnostics="off",
                          disable_output="off",
                          hard_dt=1.0e-8, stop_time=1.0e-8, stop_step=1,
                          tau_floor=1.0e-30, scf_output_frequency=256, scf_max_iterations=16)
    output = run(args, command, "scf")
    return scf_metrics(output, work, args.tolerance)


def scf_metrics(output, work, tolerance):
    rows = parse_steps(output, tolerance)
    records = [[float(value) for value in line.split()]
               for line in (work / "log.txt").read_text().splitlines()]
    require(records and all(len(row) >= 22 for row in records), "missing SCF iteration diagnostics")
    require(len(records) == 16, "SCF did not honor its fixed 16-iteration budget")
    last_iteration = records[-1]
    require(all(math.isfinite(value) for value in last_iteration), "SCF returned nonfinite iteration diagnostics")
    require(last_iteration[3] > 0 and last_iteration[4] > 0, "SCF failed to retain two stellar masses")
    near(last_iteration[4] / last_iteration[3], 0.7,
         "SCF iteration mass normalization missed its target ratio", relative=1.0e-5)
    final_records = [[float(value) for value in line.split()]
                     for line in (work / "binary.dat").read_text().splitlines()]
    require(final_records and all(len(row) >= 34 for row in final_records),
            "missing final binary diagnostics")
    final = final_records[-1]
    require(all(math.isfinite(value) for value in final), "SCF returned nonfinite final diagnostics")
    require(final[4] > 0 and final[18] > 0, "final binary lost a stellar component")
    final_ratio = final[18] / final[4]
    return {"scf": rows, "scf_iterations": int(last_iteration[0]) + 1,
            "last_iteration_mass_ratio": last_iteration[4] / last_iteration[3],
            "last_iteration_virial": last_iteration[8],
            "last_iteration_masses_solar": [last_iteration[3], last_iteration[4]],
            "last_iteration_maximum_densities": [last_iteration[1], last_iteration[2]],
            "last_iteration_donor_fill_volume": last_iteration[21],
            "final_code_time": final[0], "final_separation": final[1], "final_omega": final[2],
            "final_mass_ratio": final_ratio,
            "final_mass_ratio_relative_target_difference": final_ratio / 0.7 - 1.0,
            "final_masses_solar": [final[4], final[18]],
            "final_maximum_densities": [final[32], final[33]],
            "limitations": ("Fixed level 3, 16 SCF iterations and one FLD step; no convergence claim. "
                            "Iteration diagnostics precede the final density update; final diagnostics "
                            "come from binary.dat after evolution.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("cold", "heated", "independent", "amr", "restart", "shock", "scf"), required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()
    if not 0 <= args.level <= 1 or args.threads < 1 or args.timeout <= 0:
        parser.error("require level 0..1, positive threads, and positive total timeout")
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.deadline = time.monotonic() + args.timeout
    cases = {"cold": cold_case, "independent": independent_case, "amr": amr_case, "restart": restart_case,
             "shock": shock_case, "scf": scf_case}
    if args.case in cases:
        metrics = cases[args.case](args)
    else:
        model = "independent" if args.case == "independent" else "free_free_thomson"
        rows = parse_steps(run(args, command_for(args, fld_opacity_model=model), args.case), args.tolerance, 2)
        check_uniform(rows, model)
        metrics = {args.case: rows}
    (args.work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"case": args.case, "passed": True, "metrics": str(args.work_dir / "metrics.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
