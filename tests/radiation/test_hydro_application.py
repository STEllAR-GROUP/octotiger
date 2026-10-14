#!/usr/bin/env python3
"""Check a short coupled radiation-hydrodynamic run in physical CGS units.

This is a smoke/regression test, not a comparison at a published final time.
All cells in the initial and final Silo checkpoints are inspected. The source
ledger must demonstrate matter-radiation exchange, and its radiation energy
balance must close including boundary transport. No Python Silo package is
required; test_restart provides the public Silo C-API reader.
The optional reduced light-speed factor defaults to one. Its value must be
preserved in both checkpoints; a reduced-speed smoke is an implementation
check, not an accuracy validation of the approximation.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from test_restart import CGS_C, Silo


CASES = ("subcritical_shock", "supercritical_shock", "radiating_blast")
METADATA = ("code_to_g", "code_to_cm", "code_to_s", "n_species", "problem",
            "hydro", "radiation", "rad_implicit", "periodic", "gravity", "radiation_reduced_c_factor", "radiation_integrator", "radiation_model", "radiation_algorithm_version")


def read_config(path):
    result = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def snapshot(silo, path):
    handle = silo.open(path)
    try:
        n_species = int(silo.metadata(handle, "n_species"))
    finally:
        silo.lib.DBClose(handle)
    fields = ("egas", "tau", "sx", "sy", "sz", "er", "fx", "fy", "fz")
    fields += tuple(f"rho_{index + 1}" for index in range(n_species))
    metadata, raw = silo.snapshot(path, fields=fields, metadata_names=METADATA, block_count=8)
    # Silo object paths include the field name (/octree-location/er). Match
    # different physical fields by the shared leaf path.
    values = {name: {key.rsplit("/", 1)[0]: value for key, value in blocks.items()}
              for name, blocks in raw.items()}
    return metadata, values


def check_state(metadata, fields):
    for name in ("hydro", "radiation", "rad_implicit"):
        if metadata[name] != 1:
            raise AssertionError(f"Coupled benchmark requires {name}=on")
    if metadata["periodic"] != 0:
        raise AssertionError("Radiation shock/blast benchmarks require physical boundaries")
    density = []
    internal = []
    radiation = []
    flux_ratio = []
    for block in sorted(fields["er"]):
        arrays = {field: values[block] for field, values in fields.items()}
        if any(len(values) != len(arrays["er"]) for values in arrays.values()):
            raise AssertionError(f"Inconsistent checkpoint field lengths in {block}")
        for index, energy in enumerate(arrays["er"]):
            rho = sum(arrays[f"rho_{species + 1}"][index]
                      for species in range(int(metadata["n_species"])))
            if rho <= 0 or energy <= 0 or arrays["tau"][index] <= 0:
                raise AssertionError(f"Nonpositive density, radiation or dual energy in {block}:{index}")
            kinetic = sum(arrays[field][index] ** 2 for field in ("sx", "sy", "sz")) / (2 * rho)
            thermal = arrays["egas"][index] - kinetic
            if not math.isfinite(thermal) or thermal <= 0:
                raise AssertionError(f"Nonpositive gas thermal energy in {block}:{index}")
            flux = math.hypot(*(arrays[field][index] for field in ("fx", "fy", "fz")))
            ratio = flux / (CGS_C * energy)
            if ratio > 1 + 2e-12:
                raise AssertionError(f"Unrealizable physical radiation flux in {block}:{index}: {ratio}")
            density.append(rho)
            internal.append(thermal)
            radiation.append(energy)
            flux_ratio.append(ratio)
    return {"density": density, "gas_internal_energy": internal,
            "radiation_energy": radiation, "reduced_flux": flux_ratio}


def check_light_speed_factor(metadata, expected):
    actual = metadata["radiation_reduced_c_factor"]
    if not math.isfinite(actual) or actual != expected:
        raise AssertionError(f"Checkpoint reduced light-speed factor {actual!r} differs from requested {expected!r}")


def relative_change(before, after):
    scale = max(abs(value) for value in before)
    return max(abs(a - b) for a, b in zip(before, after)) / scale


def check_conservation(path):
    with path.open() as stream:
        rows = [{name: float(value) for name, value in row.items()} for row in csv.DictReader(stream)]
    if len(rows) < 2 or rows[-1]["t"] <= rows[0]["t"]:
        raise AssertionError("Radiation conservation ledger contains no evolved sample")
    if any(not math.isfinite(value) for row in rows for value in row.values()):
        raise AssertionError("Nonfinite radiation conservation ledger")
    initial, final = rows[0], rows[-1]
    source = final["er_source"] - initial["er_source"]
    boundary = final["er_boundary"] - initial["er_boundary"]
    change = final["er"] - initial["er"]
    scale = max(abs(initial["er"]), abs(final["er"]), abs(source), abs(boundary))
    residual = abs(change + boundary - source) / scale
    if residual > 1e-10:
        raise AssertionError(f"Radiation energy ledger does not close: residual={residual}")
    if abs(source) <= 1e-14 * scale:
        raise AssertionError("Smoke run did not exercise thermal matter-radiation exchange")
    return {"relative_energy_residual": residual, "relative_source_exchange": source / scale}


def check_slices(folder, metadata):
    rows = []
    for path in sorted((folder / "radiation-hydro-slices").glob("*.csv")):
        with path.open() as stream:
            rows.extend({name: float(value) for name, value in row.items()} for row in csv.DictReader(stream))
    if len(rows) != 16 * 16 or len({(row["x"], row["y"]) for row in rows}) != len(rows):
        raise AssertionError("Website export must contain one complete, unduplicated 16-by-16 slice")
    energy_unit = metadata["code_to_g"] / (metadata["code_to_cm"] * metadata["code_to_s"] ** 2)
    radiation_constant = 4 * 5.67051e-5 / CGS_C
    for row in rows:
        if any(not math.isfinite(value) for value in row.values()):
            raise AssertionError("Nonfinite website slice value")
        if any(row[name] <= 0 for name in ("rho", "er", "gas_temperature", "radiation_temperature")):
            raise AssertionError("Nonpositive website slice thermodynamic state")
        if not math.isclose(row["t"], metadata["time"], rel_tol=2e-13):
            raise AssertionError("Website slice time does not match the final checkpoint")
        if not math.isclose(row["z"], row["dx"] / 2, rel_tol=2e-13):
            raise AssertionError("Website slice is not the plane nearest positive z=0")
        expected = (row["er"] * energy_unit / radiation_constant) ** .25
        if not math.isclose(row["radiation_temperature"], expected, rel_tol=2e-12):
            raise AssertionError("Website radiation temperature is inconsistent with CGS radiation energy")
    return {"cells": len(rows), "minimum_gas_temperature_K": min(row["gas_temperature"] for row in rows),
            "maximum_gas_temperature_K": max(row["gas_temperature"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--radiation-backend", choices=("cpu", "cuda", "kokkos"), default="cpu")
    parser.add_argument("--radiation-integrator", choices=("euler", "rk1"), default="euler")
    parser.add_argument("--silo-library")
    parser.add_argument("--threads", type=int, default=2, choices=(1, 2))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--radiation-reduced-c-factor", type=float, default=1,
                        help="Constant radiation propagation speed / physical c, in (0,1]; default 1")
    args = parser.parse_args()
    if not math.isfinite(args.radiation_reduced_c_factor) or not 0 < args.radiation_reduced_c_factor <= 1:
        parser.error("--radiation-reduced-c-factor must be finite and in (0,1]")
    executable = args.executable.resolve()
    config = read_config(args.config.resolve())
    folder = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix=f"radiation-{args.case}-"))
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "final.silo").exists():
        parser.error("--work-dir already contains final.silo; choose an empty output directory")
    config.update({
        "max_level": 1, "min_level": 1, "unigrid": "on", "stop_step": 3,
        "disable_output": "off", "disable_analytic": "on", "disable_diagnostics": "on",
        "hydro_host_kernel_type": "LEGACY", "hydro_device_kernel_type": "OFF",
        "monopole_device_kernel_type": "OFF", "multipole_device_kernel_type": "OFF",
        "silo_num_groups": 1, "odt": "1e99", "datadir": str(folder) + os.sep,
        "radiation_reduced_c_factor": args.radiation_reduced_c_factor,
        "radiation_backend": args.radiation_backend,
        "radiation_integrator": args.radiation_integrator,
    })
    config_path = folder / "run.ini"
    config_path.write_text("".join(f"{name}={value}\n" for name, value in config.items()))
    (folder / "radiation-hydro-slices").mkdir(exist_ok=True)
    print(f"Radiation {args.case} smoke artifacts: {folder}", flush=True)
    command = [str(executable), "--config_file=" + str(config_path), f"--hpx:threads={args.threads}"]
    with (folder / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
    if process.returncode:
        tail = (folder / "run.log").read_text(errors="replace")[-6000:]
        raise AssertionError(f"Application failed ({process.returncode}):\n{tail}")
    silo = Silo(args.silo_library)
    initial_metadata, initial_fields = snapshot(silo, folder / "X.0.silo")
    final_metadata, final_fields = snapshot(silo, folder / "final.silo")
    check_light_speed_factor(initial_metadata, args.radiation_reduced_c_factor)
    check_light_speed_factor(final_metadata, args.radiation_reduced_c_factor)
    if initial_metadata["time"] != 0 or final_metadata["time"] <= 0:
        raise AssertionError("Smoke run must advance from t=0")
    for metadata in (initial_metadata, final_metadata):
        if metadata["radiation_integrator"] != 0:
            raise AssertionError("Checkpoint integrator differs from the requested method")
        if metadata["radiation_model"] != 0 or metadata["radiation_algorithm_version"] != 1:
            raise AssertionError("Checkpoint does not identify M1 radiation algorithm version 1")
    initial = check_state(initial_metadata, initial_fields)
    final = check_state(final_metadata, final_fields)
    changes = {name: relative_change(initial[name], final[name])
               for name in ("density", "gas_internal_energy", "radiation_energy")}
    if changes["gas_internal_energy"] <= 1e-12 or changes["radiation_energy"] <= 1e-12:
        raise AssertionError(f"Gas and radiation must both evolve: {changes}")
    if args.case.endswith("shock") and max(final["density"]) <= max(initial["density"]) * (1 + 1e-10):
        raise AssertionError("Reflecting-wall shock did not compress the gas")
    report = {
        "case": args.case, "validation": "four-step coupled smoke; not a published-time comparison",
        "radiation_backend": args.radiation_backend,
        "radiation_integrator": args.radiation_integrator,
        "radiation_reduced_c_factor": final_metadata["radiation_reduced_c_factor"],
        "physical_c": CGS_C, "transport_c": CGS_C * final_metadata["radiation_reduced_c_factor"],
        "cells": len(final["density"]), "physical_time_s": final_metadata["time"] * final_metadata["code_to_s"],
        "relative_changes": changes, "maximum_reduced_flux": max(final["reduced_flux"]),
        "conservation": check_conservation(folder / "radiation-conservation.csv"),
        "website_slice": check_slices(folder, final_metadata),
    }
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
