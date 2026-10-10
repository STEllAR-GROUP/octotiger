#!/usr/bin/env python3
"""Compare an actual level-one radiation checkpoint/resume with an uninterrupted run.

Uses the installed Silo shared library through its public C API; no Python Silo,
NumPy, or HDF5 package is required. All four full-volume radiation fields must
agree, and deliberately conflicting restart options must be replaced by the
checkpoint's physical parameters. Each application process runs sequentially.
"""

import argparse
import ctypes as ct
import ctypes.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


FIELDS = ("er", "fx", "fy", "fz")
CGS_C = 2.99792458e10
# Public Silo DBdatatype values from silo.h.
SILO_TYPES = {16: ct.c_int, 17: ct.c_short, 18: ct.c_long,
              19: ct.c_float, 20: ct.c_double, 21: ct.c_char, 22: ct.c_longlong}
EXPECTED = {
    "radiation_options_version": 2,
    "radiation_reduced_c_factor": 1.0,
    "radiation": 1,
    "periodic": 1,
    "rad_implicit": 0,
    "rad_test_chi": 3.5,
    "rad_test_width": 0.23,
    "rad_test_background": 1.7,
    "rad_test_amplitude": 0.031,
    "rad_test_luminosity": 0.044,
    "rad_reference": "restart-reference-unused.bin",
    "code_to_g": 2.75,
    "code_to_cm": CGS_C,
    "code_to_s": 1.0,
}


class Multivar(ct.Structure):
    # Prefix of the stable public DBmultivar struct; later members are unused.
    _fields_ = [("id", ct.c_int), ("nvars", ct.c_int), ("ngroups", ct.c_int),
                ("varnames", ct.POINTER(ct.c_char_p))]


class Quadvar(ct.Structure):
    # Prefix of the stable public DBquadvar struct, through the array dimensions.
    _fields_ = [("id", ct.c_int), ("name", ct.c_char_p), ("units", ct.c_char_p),
                ("label", ct.c_char_p), ("cycle", ct.c_int), ("meshid", ct.c_int),
                ("vals", ct.POINTER(ct.c_void_p)), ("datatype", ct.c_int),
                ("nels", ct.c_int), ("nvals", ct.c_int), ("ndims", ct.c_int),
                ("dims", ct.c_int * 3)]


class Silo:
    def __init__(self, library):
        library = library or ctypes.util.find_library("siloh5") or ctypes.util.find_library("silo")
        if not library:
            raise RuntimeError("Cannot locate the Silo shared library; use --silo-library")
        # Some distributions build Silo's HDF5 driver with C++ symbols but omit
        # libstdc++ from its shared-library dependencies. The application already
        # provides it; provide the same runtime for this standalone reader.
        cxx_runtime = ctypes.util.find_library("stdc++")
        if cxx_runtime:
            ct.CDLL(cxx_runtime, mode=ct.RTLD_GLOBAL)
        self.lib = ct.CDLL(library)
        signatures = {
            "DBOpenReal": ([ct.c_char_p, ct.c_int, ct.c_int], ct.c_void_p),
            "DBClose": ([ct.c_void_p], ct.c_int),
            "DBGetVarType": ([ct.c_void_p, ct.c_char_p], ct.c_int),
            "DBGetVarLength": ([ct.c_void_p, ct.c_char_p], ct.c_int),
            "DBReadVar": ([ct.c_void_p, ct.c_char_p, ct.c_void_p], ct.c_int),
            "DBGetMultivar": ([ct.c_void_p, ct.c_char_p], ct.POINTER(Multivar)),
            "DBFreeMultivar": ([ct.POINTER(Multivar)], None),
            "DBGetQuadvar": ([ct.c_void_p, ct.c_char_p], ct.POINTER(Quadvar)),
            "DBFreeQuadvar": ([ct.POINTER(Quadvar)], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = result

    def open(self, path):
        handle = self.lib.DBOpenReal(os.fsencode(path), 5, 1)  # DB_UNKNOWN, DB_READ
        if not handle:
            raise AssertionError(f"Cannot open Silo checkpoint {path}")
        return handle

    def metadata(self, handle, name):
        encoded = name.encode()
        datatype = self.lib.DBGetVarType(handle, encoded)
        length = self.lib.DBGetVarLength(handle, encoded)
        if datatype not in SILO_TYPES or length < 1:
            raise AssertionError(f"Missing/invalid Silo metadata {name}")
        buffer = (SILO_TYPES[datatype] * length)()
        if self.lib.DBReadVar(handle, encoded, buffer) != 0:
            raise AssertionError(f"Cannot read Silo metadata {name}")
        if datatype == 21:
            return bytes(buffer).rstrip(b"\0").decode()
        if length != 1:
            raise AssertionError(f"Expected scalar Silo metadata {name}")
        return buffer[0]

    def snapshot(self, path, fields=FIELDS, metadata_names=EXPECTED, block_count=8):
        handle = self.open(path)
        try:
            metadata = {name: self.metadata(handle, name) for name in metadata_names}
            metadata["time"] = self.metadata(handle, "cgs_time") / self.metadata(handle, "code_to_s")
            result = {}
            for field in fields:
                multivar = self.lib.DBGetMultivar(handle, field.encode())
                if not multivar:
                    raise AssertionError(f"Checkpoint does not contain {field}")
                try:
                    count = multivar.contents.nvars
                    if count != block_count:
                        raise AssertionError(f"Expected {block_count} blocks for {field}, got {count}")
                    names = [multivar.contents.varnames[index].decode() for index in range(count)]
                finally:
                    self.lib.DBFreeMultivar(multivar)
                blocks = {}
                for name in names:
                    filename, object_name = name.split(":", 1)
                    block_file = self.open(path.parent / filename)
                    try:
                        variable = self.lib.DBGetQuadvar(block_file, object_name.encode())
                        if not variable:
                            raise AssertionError(f"Cannot read {field} block {name}")
                        try:
                            value = variable.contents
                            if value.ndims != 3 or value.nvals != 1 or value.nels != math.prod(value.dims):
                                raise AssertionError(f"Invalid radiation array shape for {name}")
                            if value.datatype not in (19, 20) or value.nels <= 0:
                                raise AssertionError(f"Invalid radiation array datatype/length for {name}")
                            data = ct.cast(value.vals[0], ct.POINTER(SILO_TYPES[value.datatype]))
                            values = [float(data[index]) for index in range(value.nels)]
                            if not all(math.isfinite(number) for number in values):
                                raise AssertionError(f"Nonfinite radiation field in {name}")
                            # Object paths encode octree locations; filenames differ between runs.
                            blocks[object_name] = values
                        finally:
                            self.lib.DBFreeQuadvar(variable)
                    finally:
                        self.lib.DBClose(block_file)
                if len(blocks) != block_count:
                    raise AssertionError(f"Duplicate/missing radiation block paths for {field}")
                result[field] = blocks
            return metadata, result
        finally:
            self.lib.DBClose(handle)


def close(actual, expected, name, tolerance=3e-13):
    if not math.isfinite(actual) or abs(actual - expected) > tolerance * max(abs(expected), 1e-300):
        raise AssertionError(f"{name}: got {actual!r}, expected {expected!r}")


def check_metadata(metadata, time):
    for name, expected in EXPECTED.items():
        if isinstance(expected, str):
            if metadata[name] != expected:
                raise AssertionError(f"{name}: got {metadata[name]!r}, expected {expected!r}")
        else:
            close(metadata[name], expected, name)
    close(metadata["time"], time, "checkpoint time")


def run(executable, folder, final_time, threads, timeout, restart=None):
    folder.mkdir(parents=True)
    config = {
        "problem": "RADIATION_STREAMING_WAVE", "radiation": "on", "hydro": "off",
        "gravity": "off", "omega": 0, "periodic": "on", "rad_implicit": "off",
        "unigrid": "on", "min_level": 1, "max_level": 1, "xscale": 1,
        "n_species": 5, "cfl": 0.25, "hard_dt": 0.001, "stop_time": final_time,
        "stop_step": 100000, "odt": 1, "disable_output": "off",
        "disable_diagnostics": "on", "disable_analytic": "on",
        "hydro_host_kernel_type": "LEGACY", "hydro_device_kernel_type": "OFF",
        "monopole_device_kernel_type": "OFF", "multipole_device_kernel_type": "OFF",
        "silo_num_groups": 1, "datadir": str(folder) + os.sep,
    }
    config.update({name: value for name, value in EXPECTED.items()
                   if name.startswith("rad_test_") or name in ("rad_reference", "radiation_reduced_c_factor", "code_to_g", "code_to_cm", "code_to_s")})
    if restart:
        config.update({
            "restart_filename": str(restart), "periodic": "off", "rad_implicit": "on",
            "rad_test_chi": 0.75, "rad_test_width": 0.19, "rad_test_background": 5,
            "rad_test_amplitude": 0.2, "rad_test_luminosity": 0.9,
            "rad_reference": "deliberately-wrong-reference.bin",
            "radiation_reduced_c_factor": 0.9 if EXPECTED["radiation_reduced_c_factor"] != 0.9 else 1.0,
            "code_to_g": 1, "code_to_cm": CGS_C / 2, "code_to_s": 2,
        })
    config_path = folder / "run.ini"
    config_path.write_text("".join(f"{name}={value}\n" for name, value in config.items()))
    command = [str(executable), "--config_file=" + str(config_path), "--hpx:threads=" + str(threads)]
    with (folder / "run.log").open("w") as log:
        # Legacy Silo restarts resolve relative shard names from the working
        # directory. Keep that established invocation contract; datadir still
        # places every resumed checkpoint in its separate output directory.
        run_directory = restart.parent if restart else folder
        process = subprocess.run(command, cwd=run_directory, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if process.returncode:
        tail = (folder / "run.log").read_text(errors="replace")[-6000:]
        raise AssertionError(f"Application failed ({process.returncode}) in {folder}:\n{tail}")
    checkpoint = folder / "final.silo"
    if not checkpoint.is_file():
        raise AssertionError(f"Application did not write {checkpoint}; see {folder / 'run.log'}")
    return checkpoint


def compare_fields(direct, resumed):
    errors = {}
    for field in FIELDS:
        if direct[field].keys() != resumed[field].keys():
            raise AssertionError(f"Restart changed mesh inventory for {field}")
        reference = []
        actual = []
        for block in sorted(direct[field]):
            if len(direct[field][block]) != len(resumed[field][block]):
                raise AssertionError(f"Restart changed array length for {field} {block}")
            reference.extend(direct[field][block])
            actual.extend(resumed[field][block])
        scale = max(abs(value) for value in reference)
        if scale == 0 or max(reference) == min(reference):
            raise AssertionError(f"Streaming-wave {field} is unexpectedly zero/constant")
        differences = [abs(a - b) / scale for a, b in zip(actual, reference)]
        errors[field] = {"relative_l1": sum(differences) / len(differences),
                         "relative_linf": max(differences), "cells": len(reference)}
        if errors[field]["relative_linf"] > 5e-11:
            raise AssertionError(f"Restart changed {field}: {errors[field]}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--silo-library")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--work-dir", type=Path, help="Optional empty output directory; kept for inspection")
    parser.add_argument("--factor", type=float, default=1.0, help="Radiation reduced-c factor to preserve across restart")
    args = parser.parse_args()
    if not 1 <= args.threads <= 2:
        parser.error("--threads must be 1 or 2; runs share the local resource budget")
    if not math.isfinite(args.factor) or not 0 < args.factor <= 1:
        parser.error("--factor must be finite and satisfy 0 < factor <= 1")
    EXPECTED["radiation_reduced_c_factor"] = args.factor
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"Executable does not exist: {executable}")
    # Retain failed runs, and print the location so application diagnostics are inspectable.
    folder = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="octotiger-radiation-restart-"))
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Radiation restart artifacts: {folder}", flush=True)
    silo = Silo(args.silo_library)
    uninterrupted = run(executable, folder / "direct", 0.016, args.threads, args.timeout)
    checkpoint = run(executable, folder / "checkpoint", 0.008, args.threads, args.timeout)
    metadata, initial_fields = silo.snapshot(checkpoint)
    check_metadata(metadata, 0.008)
    resumed_path = run(executable, folder / "resumed", 0.016, args.threads, args.timeout, checkpoint)
    direct_metadata, direct_fields = silo.snapshot(uninterrupted)
    resumed_metadata, resumed_fields = silo.snapshot(resumed_path)
    check_metadata(direct_metadata, 0.016)
    check_metadata(resumed_metadata, 0.016)
    # Ensure the compared continuation actually evolved after loading its checkpoint.
    change = max(abs(direct_fields["er"][block][index] - value)
                 for block, values in initial_fields["er"].items() for index, value in enumerate(values))
    if change <= 1e-8 * max(abs(value) for values in initial_fields["er"].values() for value in values):
        raise AssertionError("Restart comparison did not evolve the checkpoint")
    errors = compare_fields(direct_fields, resumed_fields)
    (folder / "comparison.json").write_text(json.dumps(errors, indent=2) + "\n")
    print(json.dumps(errors, indent=2))
    print("Radiation restart field and physical-parameter checks passed")


if __name__ == "__main__":
    main()
