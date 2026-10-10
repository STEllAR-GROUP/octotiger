#!/usr/bin/env python3
"""Check all four production regressions on a serial mesh (no HPX)."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--cells", type=int, default=32)
parser.add_argument("--cxx", default="g++")
parser.add_argument("--sanitize", action="store_true")
parser.add_argument("--output", type=Path)
parser.add_argument("--radiation-reduced-c-factor", type=float, default=1)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("radiation_check", ROOT / "test_problems/radiation/check.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
with tempfile.TemporaryDirectory(prefix="radiation-regression-") as temporary:
    output = Path(temporary) / "results"
    command = [sys.executable, str(ROOT / "radiation_results/validate_serial.py"),
               "--root", str(ROOT), "--reference", str(args.reference.resolve()),
               "--cells", str(args.cells), "--radiation-reduced-c-factor", str(args.radiation_reduced_c_factor), "--cxx", args.cxx, "--output", str(output)]
    if args.sanitize:
        command.append("--sanitize")
    subprocess.run(command, check=True)
    logs = []
    for case in comparison.CASES:
        text = (output / case.upper() / "run.log").read_text()
        comparison.check(text, case, .2 / args.radiation_reduced_c_factor)
        logs.append(text)
    if args.output:
        args.output.write_text("".join(logs))
