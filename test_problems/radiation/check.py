#!/usr/bin/env python3
"""Run isolated radiation cases and check final time and finite error norms.

Numerical bounds are restored from the check.py bytecode tracked in dominic-wip2.
They describe the original 32-cell, c=1, t=0.2 regression configurations.
A nonzero application exit status always fails before comparison.
"""
import argparse
import math
from pathlib import Path
import re
import subprocess
import sys

CASES = ("streaming_wave", "streaming_front", "gaussian_pulse", "equilibrium_sphere")
FIELDS = ("er", "fx", "fy", "fz")


def check(text, case, final_time):
    if case not in CASES or not math.isfinite(final_time) or final_time <= 0:
        raise ValueError("Invalid radiation comparison case/time")
    marker = re.findall(r"^RADIATION_TEST_FINISHED\s+(\w+)\s+t=(\S+)\s*$", text, re.M)
    if len(marker) != 1 or marker[0][0].lower() != "radiation_" + case:
        raise ValueError("Missing or incorrect analytic-comparison completion marker")
    time = float(marker[0][1])
    if not math.isfinite(time) or abs(time - final_time) > 1e-11 * final_time:
        raise ValueError(f"Wrong final time: {time}; expected {final_time}")
    rows = {}
    for field, *norms in re.findall(r"^\s*(er|fx|fy|fz)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", text, re.M):
        if field in rows:
            raise ValueError(f"Duplicate error row: {field}")
        values = tuple(map(float, norms))
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Invalid error norm: {field} {values}")
        rows[field] = values
    if set(rows) != set(FIELDS):
        raise ValueError("Missing radiation error norms")
    gaussian_mean = .01 * (math.sqrt(math.pi) * .2 / 2) ** 3
    if case == "streaming_wave":
        limits = {field: (.065, .1, .23) for field in FIELDS}
    elif case == "streaming_front":
        limits = {field: (.075, .21, .8) if field in ("er", "fx") else (1e-12,) * 3 for field in FIELDS}
    elif case == "gaussian_pulse":
        limits = {"er": (.08 * gaussian_mean, 3e-5, 5e-4)}
        limits.update({field: (.06 * gaussian_mean, 2e-5, 3.5e-4) for field in FIELDS[1:]})
    else:
        limits = {"er": (5e-5, 1.5e-4, .0018)}
        limits.update({field: (2.5e-5, 9e-5, .0012) for field in FIELDS[1:]})
    for field, values in rows.items():
        for label, value, bound in zip(("L1", "L2", "Linf"), values, limits[field]):
            if value > bound:
                raise ValueError(f"{case} {field} {label}={value:.6e} exceeds {bound:.6e}")
        print(f"{field:>3}  " + "  ".join(f"{value:.6e}" for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "compare"))
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--time", type=float, default=.2)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    log = args.directory / "run.log"
    if args.mode == "run":
        if args.exe is None or args.config is None:
            parser.error("run requires --exe and --config")
        for name in ("L1.dat", "L2.dat", "Linf.dat"):
            (args.directory / name).unlink(missing_ok=True)
        command = [str(args.exe.resolve()), "--config_file=" + str(args.config.resolve()),
                   f"--max_level={args.level}", f"--min_level={args.level}",
                   f"--stop_time={args.time}", "--code_to_cm=1", "--code_to_g=1",
                   "--code_to_s=3.3356409519815204e-11", "--hpx:threads=2"]
        if args.reference:
            command.append("--rad_reference=" + str(args.reference.resolve()))
        with log.open("w") as out:
            result = subprocess.run(command, cwd=args.directory, stdout=out, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(errors="replace")[-12000:], file=sys.stderr)
            raise RuntimeError(f"Octo-TIGER failed with exit code {result.returncode}; see {log}")
    check(log.read_text(), args.case, args.time)
    print(f"PASS {args.case}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
