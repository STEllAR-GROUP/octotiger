#!/usr/bin/env python3
"""Run a bounded, static Scorpio-MG/FMM comparison without changing its inputs."""

import argparse
import json
import math
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--amr", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-cycles", type=int, default=200)
    parser.add_argument("--write-fields", action="store_true")
    parser.add_argument("--boundary", choices=("analytic", "multipole"), default="analytic")
    parser.add_argument("--expect-failure", action="store_true")
    args = parser.parse_args()
    if args.level < 0 or args.level > 4:
        parser.error("this bounded check supports levels 0 through 4")
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.exe.resolve()),
        f"--hpx:threads={args.threads}", "--hpx:bind=none",
        "--problem=solid_sphere", "--hydro=off", "--radiation=off",
        "--gravity=on", "--gravity_compare=on",
        "--disable_output=on", "--disable_diagnostics=on",
        "--disable_analytic=on", "--stop_step=1",
        "--correct_am_grav=off", "--theta=0.35",
        "--solid_sphere_mass=1", "--solid_sphere_radius=0.3",
        "--solid_sphere_xcenter=0.17", "--solid_sphere_ycenter=-0.11",
        "--solid_sphere_zcenter=0.07", "--solid_sphere_rho_min=0", "--rho_floor=0",
        f"--min_level={min(1, args.level) if args.amr else args.level}",
        f"--max_level={args.level}",
        f"--unigrid={'off' if args.amr else 'on'}",
        f"--gravity_mg_tolerance={args.tolerance:.17g}",
        f"--gravity_mg_max_cycles={args.max_cycles}",
        f"--gravity_mg_boundary={args.boundary}",
        f"--gravity_compare_output={'on' if args.write_fields else 'off'}",
    ]
    (args.work_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, cwd=args.work_dir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=600)
    (args.work_dir / "application.log").write_text(result.stdout)
    rows = [line for line in result.stdout.splitlines()
            if line.startswith("MG_GRAVITY_RESULT ")]
    if args.expect_failure:
        if result.returncode == 0:
            raise RuntimeError("an unconverged gravity solve incorrectly returned success")
        if "Static gravity multigrid reached its cycle limit without convergence" not in result.stdout:
            raise RuntimeError("failure was unrelated to multigrid convergence")
        print("Expected multigrid nonconvergence reported")
        return
    if result.returncode != 0:
        raise RuntimeError(f"gravity comparison exited {result.returncode}; see application.log")
    if len(rows) != 1:
        raise RuntimeError(f"expected one MG_GRAVITY_RESULT, found {len(rows)}")
    fields = dict(token.split("=", 1) for token in rows[0].split()[1:])
    required = {"cycles", "residual", "relative_residual", "phi_l1", "phi_l2",
                "phi_linf", "force_l2", "fmm_phi_l2", "fmm_force_l2",
                "delta_phi_l2", "delta_force_l2", "leaves", "cells",
                "mg_seconds", "fmm_seconds", "represented_mass"}
    if not required.issubset(fields):
        raise RuntimeError(f"missing comparison metrics: {required - fields.keys()}")
    metrics = {key: float(fields[key]) for key in required}
    if not all(math.isfinite(value) and value >= 0.0 for value in metrics.values()):
        raise RuntimeError("comparison reported nonfinite or negative metrics")
    if metrics["relative_residual"] > args.tolerance * 1.001:
        raise RuntimeError("converged report does not satisfy the requested true residual")
    if metrics["cycles"] > args.max_cycles or metrics["cells"] < metrics["leaves"]:
        raise RuntimeError("invalid solver iteration or mesh counts")
    if not args.amr and metrics["leaves"] != 8 ** args.level:
        raise RuntimeError("uniform comparison did not use the requested leaf topology")
    if args.amr and not 8 ** min(1, args.level) < metrics["leaves"] < 8 ** args.level:
        raise RuntimeError("AMR comparison did not exercise a mixed-resolution leaf mesh")
    if metrics["represented_mass"] <= 0.0:
        raise RuntimeError("comparison has no represented mass")
    expected_boundary = ("multipole_l4_ghost_centers" if args.boundary == "multipole"
                         else "analytic_sphere_ghost_centers")
    if fields.get("boundary") != expected_boundary:
        raise RuntimeError("comparison did not use the requested boundary condition")
    metrics["boundary"] = fields["boundary"]
    (args.work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
