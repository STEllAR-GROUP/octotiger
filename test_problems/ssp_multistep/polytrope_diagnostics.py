#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Analyze the translating n=1.5 polytrope in Octo-Tiger Silo snapshots.

Reads an existing results.json without modifying it and writes a neighboring
equilibrium-analysis.json. All reported physical quantities use code units.
The continuum reference is sampled at cell centers; its error includes initial
sampling, numerical Lane-Emden initialization, AMR, atmosphere and hydro/gravity
errors. It is not an exact discrete equilibrium or an isolated temporal error.
"""

import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = {"n": 1.5, "rho_center": 1.0, "G": 1.0,
              "radius": 1.0 / 3.0, "alpha": 0.09123047097730642,
              "initializer_dr": 1.0 / (3.0 * 1024.0)}


def lane_emden(step):
    """Regular-origin fourth-order RK solution, through the first zero.

    The final step is bisected to locate the zero. Clamping theta in the source
    only extends the RHS beyond the first zero while bracketing that surface.
    """
    n = PARAMETERS["n"]
    h = step
    theta = 1 - h*h/6 + n*h**4/120 - n*(8*n-5)*h**6/15120
    phi = -h/3 + n*h**3/30 - n*(8*n-5)*h**5/2520
    xi = h
    xis, thetas, phis = [0.0, xi], [1.0, theta], [0.0, phi]

    def derivative(x, y, z):
        return z, -(max(y, 0.0)**n + 2*z/x)

    def advance(x, y, z, interval):
        k1y, k1z = derivative(x, y, z)
        k2y, k2z = derivative(x+interval/2, y+interval*k1y/2, z+interval*k1z/2)
        k3y, k3z = derivative(x+interval/2, y+interval*k2y/2, z+interval*k2z/2)
        k4y, k4z = derivative(x+interval, y+interval*k3y, z+interval*k3z)
        return (y+interval*(k1y+2*k2y+2*k3y+k4y)/6,
                z+interval*(k1z+2*k2z+2*k3z+k4z)/6)

    while xi < 10:
        new_theta, new_phi = advance(xi, theta, phi, h)
        if new_theta <= 0:
            lo, hi = 0.0, h
            for _ in range(45):
                middle = (lo+hi)/2
                if advance(xi, theta, phi, middle)[0] > 0:
                    lo = middle
                else:
                    hi = middle
            surface_step = (lo+hi)/2
            _, surface_phi = advance(xi, theta, phi, surface_step)
            xis.append(xi+surface_step)
            thetas.append(0.0)
            phis.append(surface_phi)
            break
        xi += h
        theta, phi = new_theta, new_phi
        xis.append(xi)
        thetas.append(theta)
        phis.append(phi)
    else:
        raise RuntimeError("Lane-Emden first zero was not found")

    xis = np.asarray(xis)
    thetas = np.asarray(thetas)
    phis = np.asarray(phis)
    dimensionless_mass = -xis[-1]**2 * phis[-1]
    # Fourth radial moment, with a sufficiently fine trapezoid quadrature. This
    # integral is used only for a reference shape diagnostic, not normalization.
    integrand = xis**4 * thetas**n
    fourth_radial_moment = float(np.sum(np.diff(xis)*(integrand[:-1]+integrand[1:])/2))
    return {"xi": xis, "theta": thetas, "phi": phis,
            "xi_surface": float(xis[-1]), "surface_derivative": float(phis[-1]),
            "dimensionless_mass": float(dimensionless_mass),
            "fourth_radial_moment": fourth_radial_moment, "step": h}


def normalization(table, coarse):
    p = PARAMETERS
    mass = 4*math.pi*p["alpha"]**3*p["rho_center"]*table["dimensionless_mass"]
    k = 4*math.pi*p["G"]*p["alpha"]**2*p["rho_center"]**((p["n"]-1)/p["n"])/(p["n"]+1)
    relative_mass_difference = abs(table["dimensionless_mass"]/coarse["dimensionless_mass"]-1)
    absolute_surface_difference = abs(table["xi_surface"]-coarse["xi_surface"])
    if relative_mass_difference > 1e-8 or absolute_surface_difference > 1e-8:
        raise RuntimeError("Independent Lane-Emden step-halving check did not converge")
    return {**p, "ideal_star_mass": mass,
            "polytropic_K": k, "polytropic_relation": "P = K * rho**(1 + 1/n)",
            "dynamical_time": math.sqrt(p["radius"]**3/(p["G"]*mass)),
            "dynamical_time_definition": "sqrt(R**3 / (G*M_ideal))",
            "central_density_time": 1/math.sqrt(p["G"]*p["rho_center"]),
            "ideal_rms_radius": p["alpha"]*math.sqrt(table["fourth_radial_moment"]/table["dimensionless_mass"]),
            "computed_surface_radius": p["alpha"]*table["xi_surface"],
            "lane_emden": {key: table[key] for key in ("xi_surface", "surface_derivative", "dimensionless_mass", "step")},
            "lane_emden_step_halving": {"coarse_step": coarse["step"],
                                       "fine_step": table["step"],
                                       "relative_mass_difference": relative_mass_difference,
                                       "absolute_surface_difference": absolute_surface_difference}}


def density_reference(radius, table, atmosphere):
    theta = np.interp(radius/PARAMETERS["alpha"], table["xi"], table["theta"], left=1.0, right=0.0)
    return np.where(radius <= PARAMETERS["radius"],
                    np.maximum(PARAMETERS["rho_center"]*theta**PARAMETERS["n"], atmosphere), atmosphere)


def analyze_snapshot(reader, snapshot, table, scales, center, velocity, atmosphere, dual_energy_switch):
    path = snapshot["path"]
    time = snapshot["time"]
    expected_center = center + time*velocity
    mass = volume = rho_max = radius_second = radius_second_centers = velocity_second = l1 = 0.0
    position_first = np.zeros(3)
    velocity_first = np.zeros(3)
    cells = 0
    total_mass = pressure_integral = potential_energy = gas_energy = 0.0
    total_velocity_second = 0.0
    total_velocity_first = np.zeros(3)
    total_momentum = np.zeros(3)
    for _, xyz, fields in reader.meshes(path, field_names=("rho_1", "rho_2", "sx", "sy", "sz", "egas", "tau", "pot")):
        x, y, z = xyz
        dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
        dv = dz[:, None, None]*dy[None, :, None]*dx[None, None, :]
        positions = (0.5*(x[1:]+x[:-1])[None, None, :],
                     0.5*(y[1:]+y[:-1])[None, :, None],
                     0.5*(z[1:]+z[:-1])[:, None, None])
        rho = fields["rho_1"]+fields["rho_2"]
        if not all(np.all(np.isfinite(value)) for value in fields.values()) or np.any(rho <= 0) or np.any(fields["rho_1"] < 0):
            raise RuntimeError(f"Invalid density or momentum in {path}")
        star_weight = dv*fields["rho_1"]
        thermal = fields["egas"] - 0.5*sum(fields[field]**2 for field in ("sx", "sy", "sz"))/rho
        pressure = (2.0/3.0)*np.where(thermal <= dual_energy_switch*fields["egas"], fields["tau"]**(5.0/3.0), thermal)
        if np.any(pressure <= 0) or not np.all(np.isfinite(pressure)):
            raise RuntimeError(f"Invalid physical pressure in {path}")
        total_mass += float(np.sum(dv*rho))
        pressure_integral += float(np.sum(dv*pressure))
        potential_energy += float(np.sum(dv*fields["pot"]))*0.5
        gas_energy += float(np.sum(dv*fields["egas"]))
        relative_position = [positions[axis]-expected_center[axis] for axis in range(3)]
        relative_velocity = [fields[field]/rho-velocity[axis] for axis, field in enumerate(("sx", "sy", "sz"))]
        radius_square = sum(component**2 for component in relative_position)
        cell_radius_square = (dx[None, None, :]**2+dy[None, :, None]**2+dz[:, None, None]**2)/12
        mass += float(np.sum(star_weight))
        volume += float(np.sum(dv))
        rho_max = max(rho_max, float(np.max(rho)))
        radius_second_centers += float(np.sum(star_weight*radius_square))
        radius_second += float(np.sum(star_weight*(radius_square+cell_radius_square)))
        velocity_second += float(np.sum(star_weight*sum(component**2 for component in relative_velocity)))
        total_velocity_second += float(np.sum(dv*rho*sum(component**2 for component in relative_velocity)))
        for axis in range(3):
            position_first[axis] += np.sum(star_weight*relative_position[axis])
            velocity_first[axis] += np.sum(star_weight*relative_velocity[axis])
            total_velocity_first[axis] += np.sum(dv*rho*relative_velocity[axis])
            total_momentum[axis] += np.sum(dv*fields[("sx", "sy", "sz")[axis]])
        reference = density_reference(np.sqrt(radius_square), table, atmosphere)
        l1 += float(np.sum(dv*np.abs(rho-reference)))
        cells += rho.size
    if mass <= 0 or not math.isclose(volume, 8.0, rel_tol=1e-12):
        raise RuntimeError(f"Invalid star mass or domain volume in {path}")
    relative_centroid = position_first/mass
    relative_bulk = velocity_first/mass
    radius_variance = radius_second/mass - float(relative_centroid @ relative_centroid)
    radius_variance_centers = radius_second_centers/mass - float(relative_centroid @ relative_centroid)
    velocity_variance = velocity_second/mass - float(relative_bulk @ relative_bulk)
    if min(radius_variance, radius_variance_centers, velocity_variance) < -1e-12:
        raise RuntimeError(f"Negative central moment in {path}")
    bulk_velocity = velocity+relative_bulk
    peculiar_kinetic_energy = 0.5*(total_velocity_second - 2*float(relative_bulk@total_velocity_first) + total_mass*float(relative_bulk@relative_bulk))
    total_energy = gas_energy + potential_energy
    bulk_frame_total_energy = total_energy-float(bulk_velocity@total_momentum)+0.5*total_mass*float(bulk_velocity@bulk_velocity)
    if potential_energy >= 0:
        raise RuntimeError(f"Nonnegative gravitational binding energy in {path}")
    virial_residual = (2*peculiar_kinetic_energy+3*pressure_integral+potential_energy)/abs(potential_energy)
    return {"path": path, "step": snapshot["step"], "time": time,
            "time_over_dynamical_time": time/scales["dynamical_time"],
            "cells": cells, "leaf_count": snapshot["leaf_count"],
            "rho_max": rho_max, "star_species_mass": mass,
            "star_centroid": (expected_center+relative_centroid).tolist(),
            "expected_translated_center": expected_center.tolist(),
            "centroid_error": relative_centroid.tolist(),
            "centroid_error_norm": float(np.linalg.norm(relative_centroid)),
            "star_rms_radius": math.sqrt(max(radius_variance, 0.0)),
            "star_rms_radius_cell_centers": math.sqrt(max(radius_variance_centers, 0.0)),
            "star_bulk_velocity": (velocity+relative_bulk).tolist(),
            "bulk_velocity_error": relative_bulk.tolist(),
            "bulk_velocity_error_norm": float(np.linalg.norm(relative_bulk)),
            "star_rms_peculiar_velocity": math.sqrt(max(velocity_variance, 0.0)),
            "total_mass": total_mass,
            "peculiar_kinetic_energy_all_fluid": peculiar_kinetic_energy,
            "peculiar_kinetic_energy_stellar_material": 0.5*mass*max(velocity_variance, 0.0),
            "pressure_volume_integral": pressure_integral,
            "gravitational_potential_energy": potential_energy,
            "total_energy": total_energy,
            "bulk_frame_total_energy": bulk_frame_total_energy,
            "bulk_frame_total_energy_over_gravitational_binding": bulk_frame_total_energy/abs(potential_energy),
            "virial_residual": virial_residual,
            "density_l1_volume_integral": l1,
            "density_l1_per_ideal_star_mass": l1/scales["ideal_star_mass"]}


def run_summary(series):
    first, last = series[0], series[-1]
    for row in series:
        row["rho_max_relative_change_from_initial"] = row["rho_max"]/first["rho_max"]-1
        row["star_rms_radius_change_from_initial"] = row["star_rms_radius"]-first["star_rms_radius"]
        row["star_rms_radius_relative_change_from_initial"] = row["star_rms_radius"]/first["star_rms_radius"]-1
        row["star_mass_relative_change_from_initial"] = row["star_species_mass"]/first["star_species_mass"]-1
        row["centroid_drift_from_initial_offset"] = (np.asarray(row["centroid_error"])-first["centroid_error"]).tolist()
        row["centroid_drift_from_initial_offset_norm"] = float(np.linalg.norm(row["centroid_drift_from_initial_offset"]))
    return {"initial": first, "final": last,
            "maximum_absolute_virial_residual": max(abs(row["virial_residual"]) for row in series),
            "maximum_density_l1_per_ideal_star_mass": max(row["density_l1_per_ideal_star_mass"] for row in series),
            "maximum_absolute_rho_max_relative_change": max(abs(row["rho_max_relative_change_from_initial"]) for row in series),
            "maximum_absolute_rms_radius_relative_change": max(abs(row["star_rms_radius_relative_change_from_initial"]) for row in series),
            "maximum_rms_peculiar_velocity": max(row["star_rms_peculiar_velocity"] for row in series),
            "maximum_centroid_drift_from_initial_offset": max(row["centroid_drift_from_initial_offset_norm"] for row in series)}


def safe_ratio(numerator, denominator):
    return numerator/denominator if denominator != 0 else None


def analyze_results(results_path, source_root=DEFAULT_ROOT, silo_library=None):
    """Return diagnostics and save equilibrium-analysis.json alongside input."""
    results_path, source_root = Path(results_path), Path(source_root)
    sys.path.insert(0, str(source_root/"test_problems/ssp_multistep"))
    from amr_gravity import GravityMeshReader
    raw = results_path.read_bytes()
    data = json.loads(raw)
    table = lane_emden(0.0001)
    scales = normalization(table, lane_emden(0.0002))
    settings = data["settings"]
    center = np.asarray(settings["center"], dtype=float)
    velocity = np.asarray(settings["velocity"], dtype=float)
    atmosphere = float(settings["star_rho_out"])
    reader = GravityMeshReader(silo_library)
    result = {"source_results": str(results_path.resolve()),
              "source_results_sha256": hashlib.sha256(raw).hexdigest(),
              "units": "code units; Silo fields and time converted using each snapshot's stored metadata",
              "normalization": scales, "settings": settings,
              "definitions": {
                  "rho_max": "maximum total density rho_1 + rho_2 over all leaf cells",
                  "star_weight": "rho_1 * cell volume; rho_1 is the stellar material tracer",
                  "star_rms_radius": "sqrt(integral rho_1 * |x-centroid|^2 dV / star mass), using the piecewise-constant cell reconstruction, including each cell's internal spatial variance",
                  "star_rms_radius_cell_centers": "same mass-weighted moment using only cell centers, provided to expose the spatial sampling convention",
                  "star_bulk_velocity": "integral rho_1 * v dV / star mass, where v = momentum / (rho_1 + rho_2)",
                  "star_rms_peculiar_velocity": "sqrt(integral rho_1 * |v-bulk velocity|^2 dV / star mass)",
                  "density_l1_volume_integral": "sum cell volume * abs(total density - point-sampled translated continuum Lane-Emden density), with the initializer's atmosphere floor and radius cutoff",
                  "density_l1_per_ideal_star_mass": "density_l1_volume_integral / ideal zero-atmosphere Lane-Emden mass",
                  "virial_residual": "(2*T_pec + 3*integral(P)dV + W)/abs(W), W=0.5*integral(pot)dV, T_pec uses all fluid relative to the stellar tracer bulk velocity; pressure follows the IDEAL dual-energy switch",
                  "bulk_frame_total_energy_over_gravitational_binding": "(gas plus gravitational energy, transformed to the stellar bulk frame)/abs(W); negative values indicate negative total energy in this frame",
                  "ratios": "SSP transfer divided by ordinary RK3; a missing ratio means the RK3 denominator was zero"},
              "caveats": [
                  "The translated continuum polytrope is a diagnostic reference, not an exact discrete hydro/gravity equilibrium.",
                  "Cell-center reference sampling and the finite-step initializer produce a nonzero initial density error; initial and final values are retained separately.",
                  "The stationary finite atmosphere, truncation, gravity approximation, spatial discretization and AMR can cause departures from the ideal reference.",
                  "A single resolution and pair of runs do not establish convergence or an isolated temporal error.",
                  "The virial diagnostic integrates all fluid in the finite box and omits surface-pressure and boundary-flux terms; it is not an exact finite-domain virial identity.",
                  "No accuracy pass/fail tolerance is imposed by this helper."],
              "runs": [], "comparisons": {}}
    for run in data["runs"]:
        rows = run["snapshot_diagnostics"] if "snapshot_diagnostics" in run else run["snapshots"]
        series = []
        # Different named files can contain the same final state. Keep one entry
        # per saved state, preferring final.silo for the terminal snapshot.
        unique = {}
        for row in sorted(rows, key=lambda item: (item["step"], item["path"])):
            key = (row["step"], row["time"])
            if key not in unique or Path(row["path"]).name == "final.silo":
                unique[key] = row
        for row in sorted(unique.values(), key=lambda item: (item["step"], item["time"])):
            series.append(analyze_snapshot(reader, row, table, scales, center, velocity, atmosphere, settings["dual_energy_switch"]))
        if not series:
            raise RuntimeError(f"No snapshots for {run['mode']}")
        analyzed = {"mode": run["mode"], "directory": run["directory"],
                    "snapshots": series, "summary": run_summary(series)}
        result["runs"].append(analyzed)
        print(f"Analyzed {run['mode']}: {len(series)} states, final density L1/M = {series[-1]['density_l1_per_ideal_star_mass']:.8g}", flush=True)
    by_mode = {run["mode"]: run for run in result["runs"]}
    if "transfer" in by_mode and "rk3" in by_mode:
        transfer, rk3 = (by_mode[mode]["summary"] for mode in ("transfer", "rk3"))
        if not math.isclose(transfer["final"]["time"], rk3["final"]["time"], rel_tol=1e-11, abs_tol=1e-12):
            raise RuntimeError("Cannot compare final errors at different physical times")
        result["comparisons"] = {
            "final_ssp_over_rk3": {key: safe_ratio(transfer["final"][key], rk3["final"][key]) for key in
                ("density_l1_per_ideal_star_mass", "star_rms_peculiar_velocity", "centroid_error_norm", "centroid_drift_from_initial_offset_norm")},
            "maximum_ssp_over_rk3": {key: safe_ratio(transfer[key], rk3[key]) for key in
                ("maximum_absolute_virial_residual", "maximum_density_l1_per_ideal_star_mass", "maximum_absolute_rho_max_relative_change", "maximum_absolute_rms_radius_relative_change", "maximum_rms_peculiar_velocity", "maximum_centroid_drift_from_initial_offset")},
            "final_rms_radius_relative_change_ssp_minus_rk3": transfer["final"]["star_rms_radius_relative_change_from_initial"]-rk3["final"]["star_rms_radius_relative_change_from_initial"],
            "final_rho_max_relative_change_ssp_minus_rk3": transfer["final"]["rho_max_relative_change_from_initial"]-rk3["final"]["rho_max_relative_change_from_initial"]}
    destination = results_path.parent/"equilibrium-analysis.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(destination, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--silo-library", type=Path)
    args = parser.parse_args()
    analyze_results(args.results, args.source_root, args.silo_library)


if __name__ == "__main__":
    main()
