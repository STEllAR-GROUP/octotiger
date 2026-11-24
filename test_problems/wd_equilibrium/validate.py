# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Check a short, nonrotating one-solar-mass WD run using its CGS Silo fields.

The physical checks complement the reference-file comparison. They establish a
resolved initial star and limited drift over this short run; they do not claim
exact discrete hydrostatic balance or stability over a stellar dynamical time.
Only the saved leaf meshes are integrated. Thermodynamics use the WD EOS and
the solver's dual-energy switch. Negative conserved gas energy can occur with
gas-plus-potential energy conservation; it is recorded diagnostically, while
the auxiliary thermal state supplies physical pressure. NumPy and a shared
Silo library are required, but Python HDF5 bindings are not used.
"""

import argparse
import ctypes as C
import ctypes.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


class DBtoc(C.Structure):
    # Public Silo DBtoc prefix, through directory names; later fields unused.
    _fields_ = [field for name in (
        "curve", "multimesh", "multimeshadj", "multivar", "multimat",
        "multimatspecies", "csgmesh", "csgvar", "defvars", "qmesh", "qvar",
        "ucdmesh", "ucdvar", "ptmesh", "ptvar", "mat", "matspecies", "var",
        "obj", "dir") for field in
        ((name + "_names", C.POINTER(C.c_char_p)), ("n" + name, C.c_int))]


class DBquadmesh(C.Structure):
    # Public Silo DBquadmesh prefix, through coordinate dimensions.
    _fields_ = [
        ("id", C.c_int), ("block_no", C.c_int), ("group_no", C.c_int),
        ("name", C.c_char_p), ("cycle", C.c_int), ("coord_sys", C.c_int),
        ("major_order", C.c_int), ("stride", C.c_int * 3),
        ("coordtype", C.c_int), ("facetype", C.c_int), ("planar", C.c_int),
        ("coords", C.c_void_p * 3), ("datatype", C.c_int), ("time", C.c_float),
        ("dtime", C.c_double), ("min_extents", C.c_float * 6),
        ("max_extents", C.c_float * 6), ("labels", C.c_char_p * 3),
        ("units", C.c_char_p * 3), ("ndims", C.c_int), ("nspace", C.c_int),
        ("nnodes", C.c_int), ("dims", C.c_int * 3),
    ]


class DBquadvar(C.Structure):
    # Public Silo DBquadvar prefix, through zone dimensions.
    _fields_ = [
        ("id", C.c_int), ("name", C.c_char_p), ("units", C.c_char_p),
        ("label", C.c_char_p), ("cycle", C.c_int), ("meshid", C.c_int),
        ("vals", C.POINTER(C.c_void_p)), ("datatype", C.c_int),
        ("nels", C.c_int), ("nvals", C.c_int), ("ndims", C.c_int),
        ("dims", C.c_int * 3),
    ]


class SiloReader:
    def __init__(self, library=None):
        # Some Linux Silo installations leave their C++ runtime symbols unresolved.
        cxx = ctypes.util.find_library("stdc++")
        if cxx:
            self.cxx = C.CDLL(cxx, mode=C.RTLD_GLOBAL)
        library = library or ctypes.util.find_library("siloh5") or ctypes.util.find_library("silo")
        if not library:
            raise RuntimeError("Shared Silo library not found; pass --silo-library")
        self.library = C.CDLL(str(library))
        signatures = {
            "DBOpenReal": ([C.c_char_p, C.c_int, C.c_int], C.c_void_p),
            "DBClose": ([C.c_void_p], C.c_int),
            "DBSetDir": ([C.c_void_p, C.c_char_p], C.c_int),
            "DBGetToc": ([C.c_void_p], C.POINTER(DBtoc)),
            "DBGetQuadmesh": ([C.c_void_p, C.c_char_p], C.POINTER(DBquadmesh)),
            "DBFreeQuadmesh": ([C.POINTER(DBquadmesh)], None),
            "DBGetQuadvar": ([C.c_void_p, C.c_char_p], C.POINTER(DBquadvar)),
            "DBFreeQuadvar": ([C.POINTER(DBquadvar)], None),
            "DBInqVarExists": ([C.c_void_p, C.c_char_p], C.c_int),
            "DBGetVarByteLength": ([C.c_void_p, C.c_char_p], C.c_int),
            "DBGetVarType": ([C.c_void_p, C.c_char_p], C.c_int),
            "DBReadVar": ([C.c_void_p, C.c_char_p, C.c_void_p], C.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = result

    def open(self, path):
        handle = self.library.DBOpenReal(str(Path(path).resolve()).encode(), 5, 1)
        if not handle:
            raise RuntimeError(f"Cannot open Silo file: {path}")
        return handle

    @staticmethod
    def array(pointer, datatype, count):
        types = {19: C.c_float, 20: C.c_double}
        if datatype not in types or not pointer or count <= 0:
            raise RuntimeError("Expected nonempty Silo floating-point data")
        return np.ctypeslib.as_array(C.cast(pointer, C.POINTER(types[datatype])),
                                    shape=(count,)).astype(np.float64, copy=True)

    def scalar(self, path, name):
        handle = self.open(path)
        try:
            encoded = name.encode()
            if self.library.DBInqVarExists(handle, encoded) != 1:
                raise RuntimeError(f"Missing Silo metadata {path}:{name}")
            datatype = self.library.DBGetVarType(handle, encoded)
            length = self.library.DBGetVarByteLength(handle, encoded)
            if datatype != 20 or length != C.sizeof(C.c_double):
                raise RuntimeError(f"Expected double Silo scalar {path}:{name}")
            value = C.c_double()
            if self.library.DBReadVar(handle, encoded, C.byref(value)) != 0:
                raise RuntimeError(f"Cannot read Silo metadata {path}:{name}")
            return value.value
        finally:
            self.library.DBClose(handle)

    def snapshot(self, path):
        handle = self.open(path)
        blocks = {}
        try:
            def visit(directory):
                if self.library.DBSetDir(handle, directory.encode()) != 0:
                    raise RuntimeError(f"Cannot enter Silo directory: {directory}")
                toc_pointer = self.library.DBGetToc(handle)
                if not toc_pointer:
                    raise RuntimeError(f"Cannot list Silo directory: {directory}")
                toc = toc_pointer.contents
                # Copy names: Silo invalidates DBtoc pointers when changing directory.
                directories = [toc.dir_names[i].decode() for i in range(toc.ndir)]
                mesh_names = [toc.qmesh_names[i].decode() for i in range(toc.nqmesh)]
                variable_names = [toc.qvar_names[i].decode() for i in range(toc.nqvar)]
                if mesh_names:
                    if mesh_names != ["quadmesh"]:
                        raise RuntimeError(f"Unexpected leaf meshes: {directory}:{mesh_names}")
                    mesh_pointer = self.library.DBGetQuadmesh(handle, b"quadmesh")
                    if not mesh_pointer:
                        raise RuntimeError(f"Cannot read leaf mesh: {directory}")
                    try:
                        mesh = mesh_pointer.contents
                        if mesh.ndims != 3 or mesh.coordtype != 130:
                            raise RuntimeError("Expected three-dimensional collinear Silo mesh")
                        axes = tuple(self.array(mesh.coords[d], mesh.datatype, mesh.dims[d])
                                     for d in range(3))
                        time_s = float(mesh.dtime)
                    finally:
                        self.library.DBFreeQuadmesh(mesh_pointer)
                    species = sorted(name for name in variable_names
                                     if re.fullmatch(r"rho_[1-9][0-9]*", name))
                    if not species:
                        raise RuntimeError(f"No species density fields in {directory}")
                    fields = {}
                    for name in species + ["sx", "sy", "sz", "egas", "tau"]:
                        if name not in variable_names:
                            raise RuntimeError(f"Missing leaf field {directory}:{name}")
                        pointer = self.library.DBGetQuadvar(handle, name.encode())
                        if not pointer:
                            raise RuntimeError(f"Cannot read leaf field {directory}:{name}")
                        try:
                            var = pointer.contents
                            expected = tuple(len(axis) - 1 for axis in axes)
                            if (var.nvals != 1 or var.ndims != 3
                                    or tuple(var.dims) != expected
                                    or var.nels != math.prod(expected)):
                                raise RuntimeError(f"Unexpected field dimensions {directory}:{name}")
                            fields[name] = self.array(var.vals[0], var.datatype, var.nels)
                        finally:
                            self.library.DBFreeQuadvar(pointer)
                    fields["rho"] = sum(fields[name] for name in species)
                    blocks[directory] = {"axes": axes, "time_s": time_s, "fields": fields}
                for name in directories:
                    visit(directory.rstrip("/") + "/" + name)

            visit("/")
        finally:
            self.library.DBClose(handle)
        if not blocks:
            raise RuntimeError(f"No saved leaf meshes in {path}")
        return blocks


def snapshot_paths(path):
    """Accept an Octo-Tiger master file or its single-locality data/0.silo file."""
    path = Path(path).resolve()
    if path.parent.name.endswith(".silo.data"):
        return path.parent.with_name(path.parent.name[:-5]), path
    return path, Path(str(path) + ".data") / "0.silo"


def block_geometry(block):
    axes = block["axes"]
    widths = [np.diff(axis) for axis in axes]
    if any(np.any(width <= 0) for width in widths):
        raise RuntimeError("Mesh coordinates must increase along every axis")
    if any(not np.allclose(width, width[0], rtol=1e-12, atol=0) for width in widths):
        raise RuntimeError("Expected uniform cell spacing within each leaf block")
    dx = float(widths[0][0])
    if not all(np.isclose(width[0], dx, rtol=1e-12, atol=0) for width in widths):
        raise RuntimeError("Expected cubic leaf cells")
    centers = [(axis[1:] + axis[:-1]) * 0.5 for axis in axes]
    # Octo-Tiger stores x fastest, followed by y and z.
    z, y, x = np.meshgrid(centers[2], centers[1], centers[0], indexing="ij")
    xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return dx, dx ** 3, xyz


def degenerate_thermodynamics(density):
    """Independent, cancellation-resistant cold WD pressure and energy in CGS."""
    # These physical constants match the mu_e = 2 model and src/physcon.cpp.
    a = np.longdouble(6.00228e22)
    b = np.longdouble(2.0 * 9.81011e5)
    x = np.cbrt(density.astype(np.longdouble) / b)
    x2 = x * x
    small = x < 0.05
    pressure = np.empty_like(x)
    energy = np.empty_like(x)
    # The series follow the independent integrals for a zero-temperature
    # electron gas. Direct formulas subtract nearly equal terms at small x.
    y = x2[small]
    x5 = x[small] * y * y
    pressure[small] = a * x5 * (np.longdouble(8) / 5 + y *
        (-np.longdouble(4) / 7 + y * (np.longdouble(1) / 3 + y *
            (-np.longdouble(5) / 22 + y * (np.longdouble(35) / 208 - y *
                np.longdouble(21) / 160)))))
    energy[small] = a * x5 * (np.longdouble(12) / 5 + y *
        (-np.longdouble(3) / 7 + y * (np.longdouble(1) / 6 + y *
            (-np.longdouble(15) / 176 + y * (np.longdouble(21) / 416 - y *
                np.longdouble(21) / 640)))))
    z, z2 = x[~small], x2[~small]
    pressure[~small] = a * (z * (2 * z2 - 3) * np.sqrt(1 + z2) +
                           3 * np.arcsinh(z))
    enthalpy = 8 * a / b * z2 / (np.sqrt(1 + z2) + 1)
    energy[~small] = density[~small] * enthalpy - pressure[~small]
    return pressure.astype(np.float64), energy.astype(np.float64)


def effective_thermodynamics(fields):
    """Recover pressure with physics::to_prim's default dual-energy switch."""
    density = fields["rho"]
    pressure_cold, energy_cold = degenerate_thermodynamics(density)
    momentum = np.column_stack([fields[name] for name in ("sx", "sy", "sz")])
    kinetic = 0.5 * np.sum(momentum * momentum, axis=1) / density
    thermal = fields["egas"] - kinetic - energy_cold
    fallback = thermal <= 0.001 * fields["egas"]
    # Silo stores tau with units (energy density)^(1/gamma), so the same
    # homogeneous switch and power law apply directly to its CGS fields.
    thermal[fallback] = fields["tau"][fallback] ** (5.0 / 3.0)
    pressure = pressure_cold + (2.0 / 3.0) * thermal
    if (not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(thermal))
            or np.any(pressure <= 0) or np.any(thermal < 0)):
        raise RuntimeError("Nonfinite or nonphysical pressure after WD dual-energy recovery")
    return pressure, thermal, fallback


def analyze(initial, final, radius, expected_mass, args):
    if initial.keys() != final.keys():
        raise RuntimeError("Initial/final leaf inventories differ; density drift requires identical meshes")
    accum = {name: 0.0 for name in ("mass_initial", "mass_final", "density_l1",
                                  "initial_speed2", "final_speed2")}
    first_moment_initial = np.zeros(3)
    first_moment_final = np.zeros(3)
    momentum_initial = np.zeros(3)
    momentum_final = np.zeros(3)
    finest = math.inf
    coarsest = 0.0
    widths = []
    initial_time = None
    final_time = None
    thermodynamics = {
        phase: {"negative_egas_cells": 0, "minimum_egas_erg_cm3": math.inf,
                "negative_egas_mass_g": 0.0, "negative_egas_integral_erg": 0.0,
                "minimum_effective_pressure_erg_cm3": math.inf,
                "minimum_effective_thermal_energy_erg_cm3": math.inf,
                "dual_energy_fallback_cells": 0}
        for phase in ("initial", "final")}
    for name in sorted(initial):
        a, b = initial[name], final[name]
        if any(not np.array_equal(x, y) for x, y in zip(a["axes"], b["axes"])):
            raise RuntimeError(f"Initial/final leaf coordinates differ: {name}")
        for block in (a, b):
            if any(not np.all(np.isfinite(value)) for value in block["fields"].values()):
                raise RuntimeError(f"Nonfinite hydrodynamic field in leaf {name}")
            if np.any(block["fields"]["rho"] <= 0) or np.any(block["fields"]["tau"] < 0):
                raise RuntimeError(f"Nonpositive density or negative thermal tracer in leaf {name}")
        dx, volume, xyz = block_geometry(a)
        for phase, block in (("initial", a), ("final", b)):
            fields = block["fields"]
            pressure, thermal, fallback = effective_thermodynamics(fields)
            negative = fields["egas"] < 0
            summary = thermodynamics[phase]
            summary["negative_egas_cells"] += int(np.count_nonzero(negative))
            summary["minimum_egas_erg_cm3"] = min(summary["minimum_egas_erg_cm3"],
                                                    float(np.min(fields["egas"])))
            summary["negative_egas_mass_g"] += float(np.sum(fields["rho"][negative])) * volume
            summary["negative_egas_integral_erg"] += float(np.sum(-fields["egas"][negative])) * volume
            summary["minimum_effective_pressure_erg_cm3"] = min(
                summary["minimum_effective_pressure_erg_cm3"], float(np.min(pressure)))
            summary["minimum_effective_thermal_energy_erg_cm3"] = min(
                summary["minimum_effective_thermal_energy_erg_cm3"], float(np.min(thermal)))
            summary["dual_energy_fallback_cells"] += int(np.count_nonzero(fallback))
        finest, coarsest = min(finest, dx), max(coarsest, dx)
        widths.append(dx)
        rho_a, rho_b = a["fields"]["rho"], b["fields"]["rho"]
        p_a = np.column_stack([a["fields"][field] for field in ("sx", "sy", "sz")])
        p_b = np.column_stack([b["fields"][field] for field in ("sx", "sy", "sz")])
        accum["mass_initial"] += float(np.sum(rho_a)) * volume
        accum["mass_final"] += float(np.sum(rho_b)) * volume
        accum["density_l1"] += float(np.sum(np.abs(rho_b - rho_a))) * volume
        accum["initial_speed2"] += float(np.sum(p_a * p_a / rho_a[:, None])) * volume
        accum["final_speed2"] += float(np.sum(p_b * p_b / rho_b[:, None])) * volume
        first_moment_initial += np.sum(rho_a[:, None] * xyz, axis=0) * volume
        first_moment_final += np.sum(rho_b[:, None] * xyz, axis=0) * volume
        momentum_initial += np.sum(p_a, axis=0) * volume
        momentum_final += np.sum(p_b, axis=0) * volume
        if initial_time is None:
            initial_time, final_time = a["time_s"], b["time_s"]
        if a["time_s"] != initial_time or b["time_s"] != final_time:
            raise RuntimeError("Leaf output times disagree within a snapshot")
    m0, m1 = accum["mass_initial"], accum["mass_final"]
    if radius <= 0 or not np.isfinite(radius) or min(m0, m1) <= 0:
        raise RuntimeError("Radius and integrated masses must be finite and positive")
    velocity_scale = math.sqrt(6.67259e-8 * expected_mass / radius)
    binding_scale = 6.67259e-8 * expected_mass * expected_mass / radius
    for phase, mass in (("initial", m0), ("final", m1)):
        summary = thermodynamics[phase]
        summary["negative_egas_mass_fraction"] = summary.pop("negative_egas_mass_g") / mass
        summary["negative_egas_over_binding_scale"] = (
            summary["negative_egas_integral_erg"] / binding_scale)
    com0, com1 = first_moment_initial / m0, first_moment_final / m1
    ratios = [int(round(math.log2(width / finest))) for width in widths]
    if any(not math.isclose(width / finest, 2 ** ratio, rel_tol=1e-11)
           for width, ratio in zip(widths, ratios)):
        raise RuntimeError("Leaf cell widths are not related by octree refinement")
    metrics = {
        "initial_time_s": initial_time,
        "final_time_s": final_time,
        "radius_cm": radius,
        "expected_mass_g": expected_mass,
        "mass_initial_g": m0,
        "mass_final_g": m1,
        "initial_mass_relative_error": abs(m0 / expected_mass - 1),
        "mass_relative_drift": abs(m1 / m0 - 1),
        "leaf_blocks": len(initial),
        "leaf_blocks_by_levels_below_finest": {str(level): ratios.count(level)
                                               for level in sorted(set(ratios))},
        "finest_dx_cm": finest,
        "coarsest_dx_cm": coarsest,
        "diameter_in_finest_cells": 2 * radius / finest,
        "initial_com_cm": com0.tolist(),
        "final_com_cm": com1.tolist(),
        "com_relative_drift": float(np.linalg.norm(com1 - com0)) / radius,
        "initial_com_speed_cm_s": float(np.linalg.norm(momentum_initial)) / m0,
        "final_com_speed_cm_s": float(np.linalg.norm(momentum_final)) / m1,
        "dynamical_velocity_cm_s": velocity_scale,
        "initial_rms_velocity_over_dynamical": math.sqrt(accum["initial_speed2"] / m0) / velocity_scale,
        "final_rms_velocity_over_dynamical": math.sqrt(accum["final_speed2"] / m1) / velocity_scale,
        "density_relative_l1_drift": accum["density_l1"] / m0,
    }
    checks = {
        "evolved_from_zero_time": initial_time == 0 and final_time > initial_time,
        "one_solar_mass": metrics["initial_mass_relative_error"] < args.mass_tolerance,
        "fifty_finest_cells_across_diameter": math.isclose(2 * radius / finest, 50, rel_tol=1e-10),
        "density_amr_has_coarser_leaves": len(set(ratios)) > 1,
        "initial_momenta_zero": metrics["initial_rms_velocity_over_dynamical"] < 1e-14,
        "mass_conserved": metrics["mass_relative_drift"] < args.mass_drift_tolerance,
        "com_preserved": metrics["com_relative_drift"] < args.com_tolerance,
        "limited_velocity_growth": metrics["final_rms_velocity_over_dynamical"] < args.velocity_tolerance,
        "limited_density_drift": metrics["density_relative_l1_drift"] < args.density_tolerance,
        "positive_effective_thermodynamics": True,
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics,
            "thermodynamics": thermodynamics,
            "scope": "Three-step physical sanity checks with dual-energy pressure recovery. "
                     "Negative conserved gas energy is diagnostic, not a thermodynamic failure. "
                     "No exact discrete equilibrium or long-run stability claim.",
            "tolerances": {"mass_relative_error": args.mass_tolerance,
                           "mass_relative_drift": args.mass_drift_tolerance,
                           "com_drift_over_radius": args.com_tolerance,
                           "rms_velocity_over_dynamical": args.velocity_tolerance,
                           "density_relative_l1_drift": args.density_tolerance,
                           "dual_energy_switch": 0.001}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    parser.add_argument("--radius-cm", type=float,
                        help="Physical stellar radius; default: initial code_to_cm times radius-code")
    parser.add_argument("--radius-code", type=float, default=1.0)
    parser.add_argument("--mass-g", type=float, default=1.989e33)
    parser.add_argument("--silo-library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mass-tolerance", type=float, default=0.003)
    parser.add_argument("--mass-drift-tolerance", type=float, default=1e-7)
    parser.add_argument("--com-tolerance", type=float, default=1e-6)
    parser.add_argument("--velocity-tolerance", type=float, default=0.02)
    parser.add_argument("--density-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    try:
        reader = SiloReader(args.silo_library)
        master_initial, data_initial = snapshot_paths(args.initial)
        _, data_final = snapshot_paths(args.final)
        radius = (args.radius_cm if args.radius_cm is not None
                  else reader.scalar(master_initial, "code_to_cm") * args.radius_code)
        result = analyze(reader.snapshot(data_initial), reader.snapshot(data_final),
                         radius, args.mass_g, args)
        status = 0 if result["passed"] else 1
    except (OSError, RuntimeError, ValueError) as error:
        result, status = {"passed": False, "error": str(error)}, 2
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return status


if __name__ == "__main__":
    sys.exit(main())
