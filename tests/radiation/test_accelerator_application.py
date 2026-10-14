#!/usr/bin/env python3
"""Short coupled application parity checks, including Silo fields and budgets.

These are four-step implementation checks, not completed physical benchmarks.
Runs are serial with two HPX workers; all outputs are retained for inspection.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from test_hydro_application import CASES, snapshot
from test_restart import Silo

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=("cuda", "kokkos"), default=("cuda", "kokkos"))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--silo-library")
    parser.add_argument("--radiation-integrator", choices=("euler",), default="euler")
    args = parser.parse_args()
    folder = (args.work_dir or Path(tempfile.mkdtemp(prefix="radiation-accelerator-"))).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    silo = Silo(args.silo_library)
    report = {"scope": "four-step coupled application parity, not final physical runs", "radiation_integrator": args.radiation_integrator, "comparisons": []}
    for case in args.cases:
        reference = None
        # Exercise both physical light speed and the factors used by the studies.
        factor = {"subcritical_shock": 0.03, "supercritical_shock": 0.1, "radiating_blast": 1.0}[case]
        for backend in ("cpu", *args.backends):
            output = folder / f"{case}-{backend}"
            command = [sys.executable, str(ROOT / "tests/radiation/test_hydro_application.py"),
                       "--executable", str(args.executable.resolve()), "--case", case,
                       "--config", str(ROOT / f"radiation_results/configs/{case}.ini"),
                       "--threads", "2", "--work-dir", str(output), "--radiation-backend", backend,
                       "--radiation-reduced-c-factor", str(factor),
                       "--radiation-integrator", args.radiation_integrator]
            if args.silo_library:
                command += ["--silo-library", args.silo_library]
            subprocess.run(command, check=True, timeout=210)
            metadata, fields = snapshot(silo, output / "final.silo")
            if backend == "cpu":
                reference = (metadata, fields)
                continue
            expected_metadata, expected_fields = reference
            if metadata["time"] != expected_metadata["time"]:
                raise AssertionError(f"{case}/{backend}: physical times differ")
            worst = 0.0
            cells_checked = 0
            for field, blocks in fields.items():
                expected_blocks = expected_fields[field]
                if blocks.keys() != expected_blocks.keys():
                    raise AssertionError(f"{case}/{backend}/{field}: mesh differs")
                for block, values in blocks.items():
                    expected = expected_blocks[block]
                    if len(values) != len(expected):
                        raise AssertionError("Field length mismatch")
                    scale = max(max(map(abs, expected)), max(map(abs, values)), 1e-300)
                    error = max(abs(x - y) for x, y in zip(values, expected)) / scale
                    if not math.isfinite(error) or error > 5e-11:
                        raise AssertionError(f"{case}/{backend}/{field}/{block}: relative error {error}")
                    worst = max(worst, error)
                    cells_checked += len(values)
            report["comparisons"].append({"case": case, "backend": backend,
                "radiation_reduced_c_factor": factor, "field_values_compared": cells_checked,
                "worst_field_scaled_error": worst})
    (folder / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
