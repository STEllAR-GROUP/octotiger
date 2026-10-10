#!/usr/bin/env python3
"""Exercise radiation ghost exchange and regridding on a genuinely mixed mesh."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "radiation_results/amr"))
import study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    output = args.work_dir or Path(tempfile.mkdtemp(prefix="octotiger-gradient-radiation-"))
    print("Mixed radiation AMR artifacts:", output.resolve(), flush=True)
    study.run(argparse.Namespace(
        plan=ROOT / "radiation_results/amr/initial-thresholds.json",
        executable=args.executable, generator=None, threads=args.threads,
        steps=6, timeout=150, memory_limit_gib=4, allow_all_fine_through=3,
        keep_going=False, output=output, cases=["equilibrium_sphere"], levels=[4]))
    report = json.loads((output / "equilibrium_sphere-l4/validation.json").read_text())
    if not report["all_observed_ratios_in_target"]:
        raise AssertionError("Sphere regression must retain both active levels with a cell ratio in [0.1,1]")
    print("PASS: mixed L3/L4 radiation evolution, regrid, cell counts, and energy ledger", flush=True)


if __name__ == "__main__":
    main()
