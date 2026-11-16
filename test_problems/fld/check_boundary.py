#!/usr/bin/env python3
"""Bounded vacuum FLD checks using accepted physical-face energy ledgers."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from check_fld import (accepted_rows, command_for, near, replace_option,
                       require, run_application)


FACES = [f"escape_{axis}_{side}" for axis in "xyz" for side in ("minus", "plus")]


def check_rows(rows, case):
    for row in rows:
        require(row["boundary_vacuum"] == 1, "vacuum boundary was not selected/restored")
        escape = row["escaped_diffusion_energy"]
        require(all(row[face] >= 0 for face in FACES), "vacuum admitted incoming radiation")
        near(sum(row[face] for face in FACES), escape, 2e-13, "face sum differs from escape")
        near(row["boundary_diffusion_luminosity"] * row["dt"], escape, 2e-13,
             "boundary luminosity has incorrect timestep scaling")
        before = row["initial_radiation_energy"] + row["initial_thermal_energy"]
        after = row["radiation_energy"] + row["thermal_energy"] + escape
        near(after, before, 2e-10, "material/radiation/escape ledger did not close")
        if case == "cold":
            require(escape == 0 and row["radiation_energy"] == 0,
                    "vacuum boundary injected radiation into a zero radiation state")
        else:
            require(escape > 0, "vacuum boundary did not release radiation")
        if case not in ("coupled", "wd"):
            near(row["thermal_energy"], row["initial_thermal_energy"], 2e-9,
                 "elastic scattering plus escape changed the gas thermal energy")
            near(row["radiation_energy"] + escape, row["initial_radiation_energy"], 2e-9,
                 "pure-scattering radiation loss differs from outgoing flux")
        if case in ("uniform", "coupled"):
            for face in FACES:
                near(row[face], escape / 6, 2e-9, "uniform cube lost face symmetry")
        if case == "amr":
            require(8 < row["leaves"] < 512, "boundary test did not exercise mixed AMR")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("uniform", "amr", "restart", "cold", "coupled", "wd"),
                        required=True)
    args = parser.parse_args()
    tolerance = 1e-9
    base = SimpleNamespace(exe=args.exe, threads=1, level=1, tolerance=tolerance,
                           case="amr" if args.case == "amr" else "pulse")
    command = command_for(base)
    for name, value in (("fld_boundary", "vacuum"), ("fld_initial_pulse", 0),
                        ("fld_initial_energy", 0 if args.case == "cold" else 0.1)):
        command = replace_option(command, name, value)
    if args.case == "coupled":
        command = replace_option(command, "fld_absorption", 1000)
        command = replace_option(command, "fld_c", 100)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.case == "wd":
        import check_dwd_fld as wd
        # Preserve density/velocity units (and the cold EOS normalization), but
        # use a small physical box so Thomson escape is measurable in three
        # steps. This is a material/transport fixture, not a stellar model.
        scale = 1e-14
        command = wd.command_for(base, fld_boundary="vacuum",
                                 code_to_cm=wd.LENGTH * scale,
                                 code_to_g=wd.MASS * scale**3,
                                 code_to_s=wd.TIME * scale,
                                 fld_initial_energy=0.01)
        result = run_application(command, args.work_dir)
        require(result.returncode == 0, "physical WD vacuum fixture failed; see application.log")
        rows = wd.parse_steps(result.stdout, tolerance)
        check_rows(rows, "wd")
        for row in rows:
            require(row["escaped_diffusion_energy"] >
                    1e-5 * row["initial_radiation_energy"],
                    "physical WD escape is too small to resolve in this fixture")
            wd.near(row["thermal_energy"] + row["radiation_energy"] +
                    row["escaped_diffusion_energy"],
                    row["initial_thermal_energy"] + row["initial_radiation_energy"],
                    "physical WD vacuum thermal/radiation ledger", relative=2e-9, absolute=1e-25)
            wd.near(row["cold_energy"], rows[0]["cold_energy"],
                    "radiative escape changed the frozen WD cold reservoir", relative=2e-13)
        metrics = rows
    elif args.case != "restart":
        rows = accepted_rows(run_application(command, args.work_dir), tolerance)
        check_rows(rows, args.case)
        metrics = rows
    else:
        first_command = replace_option(command, "disable_output", "off")
        first_command = replace_option(first_command, "stop_step", 2)
        first_command = replace_option(first_command, "stop_time", 0.02)
        first = accepted_rows(run_application(first_command, args.work_dir / "first"), tolerance)
        checkpoint = (args.work_dir / "first" / "final.silo").resolve()
        require(checkpoint.is_file(), "missing boundary restart checkpoint")
        continuous_command = replace_option(command, "stop_step", 4)
        continuous_command = replace_option(continuous_command, "stop_time", 0.04)
        continuous = accepted_rows(run_application(continuous_command,
                                                   args.work_dir / "continuous"), tolerance, 4)
        restarted_command = replace_option(continuous_command, "fld_boundary", "reflecting")
        restarted_command.append(f"--restart_filename={checkpoint}")
        restarted = accepted_rows(run_application(restarted_command,
                                                  args.work_dir / "restarted"), tolerance)
        for rows in (first, continuous, restarted):
            check_rows(rows, "restart")
        for field in ("radiation_energy", "thermal_energy", "min_energy", "max_energy",
                      "escaped_diffusion_energy", *FACES):
            near(restarted[-1][field], continuous[-1][field], 2e-9,
                 f"restart changed {field}")
        near(restarted[0]["initial_radiation_energy"], first[-1]["radiation_energy"],
             2e-9, "restart reset radiation energy")
        metrics = dict(first=first, continuous=continuous, restarted=restarted)
    (args.work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
