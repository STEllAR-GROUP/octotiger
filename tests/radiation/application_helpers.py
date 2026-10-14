"""Shared production radiation regression helpers."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from test_restart import CGS_C

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def save_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def environment(library_dir):
    result = dict(os.environ, OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if library_dir:
        previous = result.get("LD_LIBRARY_PATH", "")
        result["LD_LIBRARY_PATH"] = str(library_dir) + (os.pathsep + previous if previous else "")
    return result

def fingerprint(executable, library_dir):
    libraries = {}
    output = subprocess.check_output(["ldd", str(executable)], text=True,
                                     env=environment(library_dir), timeout=30)
    for line in output.splitlines():
        parts = line.split()
        if "=>" in parts:
            index = parts.index("=>") + 1
            if index < len(parts) and parts[index].startswith("/"):
                path = Path(parts[index]).resolve()
                libraries[str(path)] = digest(path)
    if library_dir:
        expected = str((library_dir / "libhpx_octolib.so").resolve())
        require(expected in libraries, f"Runtime did not select saved radiation library {expected}")
    return {"executable": str(executable), "executable_sha256": digest(executable),
            "library_dir": str(library_dir) if library_dir else None,
            "shared_library_sha256": libraries}

def conservation(initial, final, metadata):
    epsilon = metadata["radiation_reduced_c_factor"]
    light_speed = CGS_C * metadata["code_to_s"] / metadata["code_to_cm"]
    def totals(state):
        energy = math.fsum(gas + radiation / epsilon for gas, radiation in zip(state["egas"], state["er"]))
        momentum = [math.fsum(gas + flux / (epsilon * light_speed)
                             for gas, flux in zip(state[mom], state[rad]))
                    for mom, rad in zip(("sx", "sy", "sz"), ("fx", "fy", "fz"))]
        return math.fsum(state["rho"]), energy, momentum
    mass0, energy0, momentum0 = totals(initial)
    mass1, energy1, momentum1 = totals(final)
    momentum_scale = math.fsum(
        math.hypot(*(state[name][index] for name in ("sx", "sy", "sz"))) +
        math.hypot(*(state[name][index] for name in ("fx", "fy", "fz"))) / (epsilon * light_speed)
        for state in (initial, final) for index in range(len(state["rho"])))
    result = {"relative_mass_residual": abs(mass1 - mass0) / mass0,
              "relative_total_RSLA_energy_residual": abs(energy1 - energy0) / abs(energy0),
              "relative_total_RSLA_momentum_residual": math.dist(momentum0, momentum1) / max(momentum_scale, 1e-300),
              "momentum_normalization": "sum of magnitudes of material and RSLA radiation momenta in both snapshots",
              "total_energy_definition": "sum(Egas + Er/epsilon)",
              "total_momentum_definition": "sum(rho*v + (F/c)/(epsilon*c))",
              "initial_total_energy": energy0, "final_total_energy": energy1,
              "initial_total_momentum": momentum0, "final_total_momentum": momentum1}
    for name in ("relative_mass_residual", "relative_total_RSLA_energy_residual", "relative_total_RSLA_momentum_residual"):
        require(result[name] < 1e-9, f"Periodic conservation failed: {name}={result[name]}")
    return result
