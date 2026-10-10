#!/usr/bin/env python3
"""Preserve radiation when gradient AMR refines a time-zero checkpoint.

Write a legacy-control (field=none) level-one checkpoint, restart it with an
enabled radiation gradient, and rewrite at time zero on level two. Compare
every fine radiation cell against piecewise-constant prolongation of its saved
coarse cell. This exercises regridding without mixing in transport error.
The two application invocations run sequentially with one HPX worker each.
"""
import argparse
import ctypes as ct
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent / "radiation"))
from test_restart import CGS_C, FIELDS, Silo


def cells_by_position(blocks, level):
    """Decode public Octo-Tiger octal block IDs and x-fast Silo arrays."""
    cells = {}
    for name, values in blocks.items():
        node = int(name.strip("/").split("/")[0], 8)
        position = [0, 0, 0]
        depth = 0
        while node != 1:
            if node <= 0:
                raise AssertionError(f"Invalid block ID in {name}")
            for axis in (2, 1, 0):
                position[axis] = 2 * position[axis] + (node & 1)
                node >>= 1
            depth += 1
        if depth != level or len(values) != 512:
            raise AssertionError(f"Expected a level-{level} 8-cubed block: {name}")
        for index, value in enumerate(values):
            local = (index % 8, (index // 8) % 8, index // 64)
            cell = tuple(8 * position[axis] + local[axis] for axis in range(3))
            if cell in cells:
                raise AssertionError(f"Overlapping checkpoint cell {cell}")
            cells[cell] = value
    if len(cells) != (8 * 2**level)**3:
        raise AssertionError("Checkpoint does not cover the uniform domain exactly once")
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--silo-library")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    folder = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="gradient-restart-zero-"))
    folder.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    print(f"Time-zero radiation restart artifacts: {folder}", flush=True)
    base = {
        "problem": "RADIATION_STREAMING_WAVE", "radiation": "on", "hydro": "off",
        "gravity": "off", "periodic": "on", "rad_implicit": "off", "omega": 0,
        "unigrid": "off", "min_level": 1, "max_level": 1, "xscale": 1,
        "n_species": 5, "cfl": .25, "hard_dt": .001, "stop_time": .002,
        "stop_step": 1, "odt": 1, "disable_output": "off", "disable_diagnostics": "on",
        "disable_analytic": "on", "correct_am_hydro": "off", "rewrite_silo": "on",
        "hydro_host_kernel_type": "LEGACY", "hydro_device_kernel_type": "OFF",
        "monopole_device_kernel_type": "OFF", "multipole_device_kernel_type": "OFF",
        "silo_num_groups": 1, "code_to_cm": CGS_C, "code_to_g": 1, "code_to_s": 1,
    }

    def invoke(name, restart=None):
        output = folder / name
        output.mkdir()
        config = dict(base, datadir=str(output) + os.sep)
        if restart:
            config.update({
                "restart_filename": str(restart), "max_level": 2,
                "gradient_refine_field": "radiation_energy", "gradient_refine_mode": "relative",
                "gradient_refine_threshold": 1e-8, "gradient_refine_floor": 1e-12,
            })
        else:
            config["gradient_refine_field"] = "none"
        ini = output / "run.ini"
        ini.write_text("".join(f"{name}={value}\n" for name, value in config.items()))
        command = [str(executable), "--config_file=" + str(ini), "--hpx:threads=1"]
        with (output / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=restart.parent if restart else output,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        checkpoint = output / "X.0.silo"
        if result.returncode or not checkpoint.is_file():
            text = (output / "run.log").read_text(errors="replace")
            raise AssertionError(f"Time-zero rewrite failed in {output}:\n{text[-5000:]}")
        return checkpoint

    silo = Silo(args.silo_library)
    coarse_checkpoint = invoke("coarse")
    handle = silo.open(coarse_checkpoint)
    try:
        exists = silo.lib.DBInqVarExists
        exists.argtypes, exists.restype = [ct.c_void_p, ct.c_char_p], ct.c_int
        if exists(handle, b"gradient_amr_options_version"):
            raise AssertionError("Source checkpoint must omit gradient metadata to allow opt-in on restart")
    finally:
        silo.lib.DBClose(handle)
    metadata_names = ("radiation", "hydro", "gravity", "leaf_count")
    coarse_metadata, coarse_fields = silo.snapshot(coarse_checkpoint, metadata_names=metadata_names, block_count=8)
    fine_checkpoint = invoke("refined", restart=coarse_checkpoint)
    fine_metadata, fine_fields = silo.snapshot(fine_checkpoint, metadata_names=metadata_names + (
        "gradient_amr_options_version", "gradient_refine_field", "gradient_min_level", "gradient_max_level"), block_count=64)
    if coarse_metadata["time"] != 0 or fine_metadata["time"] != 0:
        raise AssertionError("Test must regrid without any time evolution")
    if fine_metadata["gradient_amr_options_version"] != 1 or fine_metadata["gradient_refine_field"] != 2:
        raise AssertionError("Restart did not enable radiation gradient refinement")
    if fine_metadata["gradient_min_level"] != 1 or fine_metadata["gradient_max_level"] != 2:
        raise AssertionError("Restart did not retain the requested adjacent levels")
    checks = {}
    for field in FIELDS:
        coarse = cells_by_position(coarse_fields[field], 1)
        fine = cells_by_position(fine_fields[field], 2)
        scale = max(abs(value) for value in coarse.values())
        if not scale or max(coarse.values()) == min(coarse.values()):
            raise AssertionError(f"Source {field} must be nonzero and spatially varying")
        error = max(abs(value - coarse[tuple(index // 2 for index in position)])
                    for position, value in fine.items())
        if error > 3e-13 * scale:
            raise AssertionError(f"Time-zero refinement changed {field}: max error {error}, scale {scale}")
        coarse_mean = math.fsum(coarse.values()) / len(coarse)
        fine_mean = math.fsum(fine.values()) / len(fine)
        if abs(coarse_mean - fine_mean) > 3e-13 * scale:
            raise AssertionError(f"Time-zero refinement changed the volume mean of {field}")
        checks[field] = {"coarse_cells": len(coarse), "fine_cells": len(fine),
                         "maximum_absolute_error": error, "relative_linf": error / scale,
                         "coarse_volume_mean": coarse_mean, "fine_volume_mean": fine_mean}
    report = {"status": "pass", "scope": "Time-zero radiation checkpoint refinement; no transport steps",
              "coarse_checkpoint": str(coarse_checkpoint), "refined_checkpoint": str(fine_checkpoint),
              "coarse_metadata": coarse_metadata, "refined_metadata": fine_metadata, "fields": checks}
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "pass", "report": str(folder / "validation.json"),
                      "maximum_relative_linf": max(value["relative_linf"] for value in checks.values())}, indent=2))


if __name__ == "__main__":
    main()
