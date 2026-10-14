#!/usr/bin/env python3
"""Exercise radiation ghost exchange and regridding on a genuinely mixed mesh.

The default M1 Euler L3/L4 case retains the original occupancy requirement. The
optional L2/L3 case uses the strongest symmetry group from the threshold plan;
it must remain mixed, but its measured ratio need not fit the L3/L4 target.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "radiation_results/amr"))
import study


def checkpoint_metadata(path, library, integrator, level, threshold, histogram, final_time):
    sys.path.insert(0, str(ROOT / "tests/radiation"))
    from test_restart import Silo

    silo = Silo(library)
    expected = {
        "radiation_options_version": 3,
        "radiation_model": 0, "radiation_integrator": 0,
        "radiation_algorithm_version": 1, "radiation_flux_semantics": 0,
        "radiation_reduced_c_factor": 1,
        "radiation": 1, "rad_implicit": 1, "periodic": 0,
        "gradient_amr_options_version": 1,
        "gradient_refine_field": 2, "gradient_refine_mode": 0,
        "gradient_refine_threshold": threshold,
        "gradient_min_level": level - 1, "gradient_max_level": level,
        "leaf_count": histogram["coarse_leaf_blocks"] + histogram["fine_leaf_blocks"],
    }
    metadata, _ = silo.snapshot(path, fields=(), metadata_names=expected)
    for name, value in expected.items():
        if not math.isfinite(metadata[name]) or not math.isclose(metadata[name], value, rel_tol=3e-13):
            raise AssertionError(f"Checkpoint {name}: got {metadata[name]!r}, expected {value!r}")
    if not math.isclose(metadata["time"], final_time, rel_tol=3e-13):
        raise AssertionError("Checkpoint and final conservation sample times differ")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--radiation-integrator", choices=("euler", "rk1"), default="euler")
    parser.add_argument("--radiation-backend", choices=("cpu", "cuda", "kokkos"), default="cpu")
    parser.add_argument("--finest-level", type=int, choices=(3, 4), default=4,
                        help="Use 3 for a smaller mixed L2/L3 run; default retains L3/L4 occupancy target")
    parser.add_argument("--timeout", type=float, default=150)
    parser.add_argument("--check-checkpoint", action="store_true",
                        help="Write final.silo and verify saved integrator, source, and AMR metadata")
    parser.add_argument("--silo-library", help="Silo shared library; supplying it enables checkpoint checks")
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--threads and --timeout must be positive")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    output = (args.work_dir or Path(tempfile.mkdtemp(prefix="octotiger-gradient-radiation-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Mixed radiation AMR artifacts:", output, flush=True)
    plan = json.loads((ROOT / "radiation_results/amr/initial-thresholds.json").read_text())
    level = args.finest_level
    rows = [row for row in plan["rows"] if row["case"] == "equilibrium_sphere" and row["finest_level"] == level]
    if len(rows) != 1:
        raise AssertionError("Expected exactly one sphere threshold proposal for the requested level")
    row = rows[0]
    proposal = row["selected"] if level == 4 else row["minimum_nonempty"]
    threshold = proposal["threshold"]
    folder = output / f"equilibrium_sphere-l{level}"
    folder.mkdir()  # Preserve earlier results, including failed runs.
    check_checkpoint = args.check_checkpoint or args.silo_library is not None
    config = study.read_config(ROOT / "radiation_results/configs/equilibrium_sphere.ini")
    config.update(
        unigrid="off", min_level=level - 1, max_level=level, stop_step=5,
        disable_analytic="on", disable_output="off" if check_checkpoint else "on",
        disable_diagnostics="on", odt="1e99", datadir=str(folder) + os.sep,
        gradient_refine_field=row["field"], gradient_refine_mode=row["mode"],
        gradient_refine_threshold=threshold, gradient_refine_floor=row["floor"],
        gradient_refine_buffer=row["buffer"], gradient_refine_coarsen_factor=.8,
        radiation_reduced_c_factor=1, radiation_integrator=args.radiation_integrator,
        radiation_backend=args.radiation_backend, silo_num_groups=1,
        code_to_g=1, code_to_cm=study.C, code_to_s=1)
    (folder / "run.ini").write_text("".join(f"{name}={value}\n" for name, value in config.items()))
    command = [str(args.executable.resolve()), "--config_file=" + str(folder / "run.ini"),
               "--hpx:threads=" + str(args.threads)]
    report = dict(case="equilibrium_sphere", finest_level=level, command=command,
                  scope="Bounded mixed-AMR evolution and regrid; not full-time accuracy validation",
                  radiation_integrator=args.radiation_integrator, radiation_backend=args.radiation_backend,
                  requested_steps=6, threshold=threshold, status="running")
    try:
        with (folder / "run.log").open("w") as log:
            report["process_resources"] = study.run_bounded(command, folder, log, args.timeout, 4)
        report["histograms"] = study.read_histogram(folder / "gradient-amr.csv", level)
        report["conservation"] = study.check_ledger(folder / "radiation-conservation.csv")
        report["all_observed_ratios_in_target"] = all(
            item["status"] == "ratio_in_target" for item in report["histograms"])
        if not all(item["coarse_leaf_blocks"] > 0 and item["fine_leaf_blocks"] > 0
                   for item in report["histograms"]):
            raise AssertionError("Sphere regression must retain both active levels through evolution and regrid")
        if level == 4 and not report["all_observed_ratios_in_target"]:
            raise AssertionError("L3/L4 sphere regression requires a measured cell ratio in [0.1,1]")
        # Retry presence is evidence, not a requirement of this smooth fixture.
        log_text = (folder / "run.log").read_text(errors="replace")
        report["retry_messages"] = [line for line in log_text.splitlines() if "Radiation " in line and " retry" in line]
        if check_checkpoint:
            report["checkpoint_metadata"] = checkpoint_metadata(
                folder / "final.silo", args.silo_library, args.radiation_integrator,
                level, threshold, report["histograms"][-1], report["conservation"]["final_time"])
        report["status"] = "application_checks_passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        report["partial_evidence"] = study.partial_evidence(folder)
        raise
    finally:
        study.write_json(folder / "validation.json", report)
        study.write_json(output / "summary.json", {"reports": [report]})
    print(f"PASS: mixed L{level-1}/L{level} {args.radiation_integrator}/{args.radiation_backend} "
          "radiation evolution, regrid, cell counts, and energy ledger", flush=True)


if __name__ == "__main__":
    main()
