#!/usr/bin/env python3
"""Check the two z-axis blackbody photosphere light curves in the application."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess


# The application's cgs physical constants, not rounded textbook substitutes.
KB = 1.380658e-16
MH = 1.6733e-24
SIGMA = 5.67051e-5
LIGHT_SPEED = 2.99792458e10
GAMMA = 5.0 / 3.0
MU = 0.5  # One fully ionized hydrogen species.
FIELDS = ("time", "step", "liso_plus", "liso_minus", "area_plus", "area_minus",
          "unresolved_area_plus", "unresolved_area_minus", "sample")
HEADER = ("time_s", "step", "liso_plus_erg_s", "liso_minus_erg_s", "area_plus_cm2",
          "area_minus_cm2", "unresolved_area_plus_cm2", "unresolved_area_minus_cm2", "sample")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def near(actual, expected, message, relative=2.0e-10, absolute=1.0e-13):
    require(math.isfinite(actual) and
            abs(actual - expected) <= absolute + relative * abs(expected),
            f"{message}: actual={actual:.17g}, expected={expected:.17g}")


def ideal_pressure(density, temperature, length_unit=1.0, time_unit=1.0):
    return density * KB / (MU * MH) * temperature * (time_unit / length_unit)**2


def command_for(args, **overrides):
    options = {
        "problem": "sod", "eos": "IDEAL", "hydro": "off", "radiation": "off",
        "gravity": "off", "fld": "off", "hydro_host_kernel_type": "LEGACY",
        "hydro_device_kernel_type": "OFF", "disable_output": "on",
        "disable_diagnostics": "on", "disable_analytic": "on", "cdisc_detect": "off",
        "correct_am_hydro": "off", "sod_gamma": GAMMA, "sod_rhol": 2.0,
        "sod_rhor": 2.0, "sod_pl": ideal_pressure(2.0, 1.0e4),
        "sod_pr": ideal_pressure(2.0, 1.0e4), "sod_phi": 90, "sod_theta": 0,
        "n_species": 1, "atomic_mass": 1, "atomic_number": 1,
        "xscale": 1, "min_level": 1, "max_level": 1, "unigrid": "on",
        "stop_step": 0, "stop_time": 0.03, "light_curve": "on",
        "light_curve_interval": 1,
        "light_curve_output": "light_curve.dat", "light_curve_absorption": 2.0,
        "light_curve_scattering": 0.0, "light_curve_density_exponent": 0.0,
        "light_curve_temperature_exponent": 0.0,
        "code_to_g": 1.0, "code_to_cm": 1.0, "code_to_s": 1.0,
    }
    options.update(overrides)
    return [str(args.exe.resolve()), f"--hpx:threads={args.threads}", "--hpx:bind=none"] + [
        f"--{name}={value}" for name, value in options.items()]


def parse_rows(output, prefix):
    rows = []
    for line in output.splitlines():
        if line.startswith(prefix + " "):
            rows.append({name: float(value) for name, value in
                         (token.split("=", 1) for token in line.split()[1:])})
    return rows


def check_disjoint(rectangles):
    """Sweep patch edges; touching edges are allowed, overlapping areas are not."""
    events = []
    for index, (xmin, xmax, ymin, ymax) in enumerate(rectangles):
        events.extend(((xmin, 1, index, ymin, ymax), (xmax, 0, index, ymin, ymax)))
    active = {}
    for _, entering, index, ymin, ymax in sorted(events):
        if not entering:
            del active[index]
            continue
        for other_min, other_max in active.values():
            overlap = min(ymax, other_max) - max(ymin, other_min)
            tolerance = 1.0e-12 * max(ymax - ymin, other_max - other_min)
            require(overlap <= tolerance, "saved photosphere patches overlap within a view")
        active[index] = (ymin, ymax)


def check_mesh(curve_file, row):
    sample_dir = Path(str(curve_file) + ".mesh") / f"sample_{int(row['sample'])}"
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    for field in FIELDS:
        near(float(manifest[field]), row[field], f"mesh manifest differs from {field}")
    half_width = float(manifest["domain_half_width"])
    require(math.isfinite(half_width) and half_width > 0, "invalid saved camera size")
    require(set(manifest["opacity"]) >= {"absorption", "scattering", "density_exponent",
                                         "temperature_exponent"},
            "mesh manifest omitted opacity settings")
    require(isinstance(manifest["units"], dict) and manifest["units"],
            "mesh manifest omitted physical units")
    require(manifest["files"] and len(set(manifest["files"])) == len(manifest["files"]),
            "mesh manifest has no locality files or duplicates")
    integrals = {side: {"area": [], "temperature4_area": [], "patches": [], "cells": 0}
                 for side in ("plus", "minus")}
    cells_seen, localities_seen = set(), set()
    for filename in manifest["files"]:
        require(Path(filename).name == filename, "locality mesh path is not a local filename")
        mesh = json.loads((sample_dir / filename).read_text())
        locality = mesh["locality"]
        require(isinstance(locality, int) and locality >= 0 and locality not in localities_seen,
                "duplicate or invalid locality mesh")
        require(filename == f"locality_{locality}.json", "locality mesh filename mismatch")
        localities_seen.add(locality)
        for cell in mesh["cells"]:
            require(cell["view"] in ("plus_z", "minus_z"), "invalid saved observer direction")
            side = cell["view"].removesuffix("_z")
            center = [float(cell[axis]) for axis in ("x", "y", "z")]
            width, temperature = float(cell["width"]), float(cell["temperature"])
            require(all(math.isfinite(value) for value in center + [width, temperature]) and
                    width > 0 and temperature >= 0, "invalid photosphere cell geometry or temperature")
            key = (side, *center, width)
            require(key not in cells_seen, "one source cell was saved more than once for a view")
            cells_seen.add(key)
            tolerance = 1.0e-11 * half_width
            require(all(abs(value) + 0.5 * width <= half_width + tolerance for value in center),
                    "saved source cell lies outside the domain")
            require(cell["patches"], "saved source cell has no visible patches")
            areas = []
            for patch in cell["patches"]:
                require(len(patch) == 4 and all(math.isfinite(value) for value in patch),
                        "invalid saved visible patch")
                xmin, xmax, ymin, ymax = patch
                require(xmax > xmin and ymax > ymin, "saved patch has nonpositive area")
                require(xmin >= center[0] - 0.5 * width - tolerance and
                        xmax <= center[0] + 0.5 * width + tolerance and
                        ymin >= center[1] - 0.5 * width - tolerance and
                        ymax <= center[1] + 0.5 * width + tolerance,
                        "visible patch is outside its actual source cell")
                areas.append((xmax - xmin) * (ymax - ymin))
                integrals[side]["patches"].append(patch)
            area = math.fsum(areas)
            near(float(cell["visible_area"]), area, "saved cell visible area differs from its patches")
            integrals[side]["area"].append(area)
            integrals[side]["temperature4_area"].append(temperature**4 * area)
            integrals[side]["cells"] += 1
    summary = {}
    for side, integral in integrals.items():
        check_disjoint(integral["patches"])
        area = math.fsum(integral["area"])
        luminosity = 4.0 * SIGMA * math.fsum(integral["temperature4_area"])
        near(area, row[f"area_{side}"], f"saved {side} mesh does not reproduce photosphere area")
        near(luminosity, row[f"liso_{side}"], f"saved {side} mesh does not reproduce luminosity")
        near(area + row[f"unresolved_area_{side}"], 4.0 * half_width**2,
             f"{side} resolved plus unresolved area does not cover the camera")
        summary[side] = {"cells": integral["cells"], "patches": len(integral["patches"]),
                         "area": area, "luminosity": luminosity}
    return summary


def run(args, name, **options):
    work_dir = args.work_dir / name
    work_dir.mkdir(parents=True, exist_ok=True)
    curve_file = work_dir / "light_curve.dat"
    # Each fixture owns this output; avoid mixing appended rows from a prior run.
    curve_file.unlink(missing_ok=True)
    command = command_for(args, **options)
    (work_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, cwd=work_dir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=180)
    (work_dir / "application.log").write_text(result.stdout)
    require(result.returncode == 0,
            f"{name} exited {result.returncode}; see {work_dir / 'application.log'}")
    rows = parse_rows(result.stdout, "LIGHT_CURVE")
    require(rows, f"{name} did not write an initial light-curve sample")
    require(curve_file.is_file(), f"{name} did not create light_curve.dat")
    lines = curve_file.read_text().splitlines()
    headers = [line[1:].split() for line in lines if line.startswith("#")]
    require(headers == [list(HEADER)], f"unexpected light-curve file header: {headers}")
    file_rows = [list(map(float, line.split())) for line in lines
                 if line.strip() and not line.startswith("#")]
    require(len(file_rows) == len(rows), "file and stdout light-curve sample counts differ")
    meshes = []
    for index, (row, file_row) in enumerate(zip(rows, file_rows)):
        require(set(FIELDS).issubset(row), "LIGHT_CURVE is missing a required field")
        require(len(file_row) == len(FIELDS), "light-curve file has the wrong column count")
        require(all(math.isfinite(row[field]) and row[field] >= 0.0 for field in FIELDS),
                "light-curve output contains an invalid value")
        require(row["step"].is_integer() and row["sample"].is_integer(),
                "light-curve step or sample is not an integer")
        for field, value in zip(FIELDS, file_row):
            near(value, row[field], f"file/stdout mismatch in {field}")
        if index > 0:
            require(row["step"] > rows[index - 1]["step"] and
                    row["time"] > rows[index - 1]["time"],
                    "light-curve samples did not advance in time and step")
            require(row["sample"] > rows[index - 1]["sample"], "sample IDs did not advance")
        meshes.append(check_mesh(curve_file, row))
    near(rows[0]["time"], 0.0, "initial sample time")
    near(rows[0]["step"], 0.0, "initial sample step")
    return {"light_curve": rows, "meshes": meshes, "fld": parse_rows(result.stdout, "FLD_STEP"),
            "leaves": [int(value) for value in re.findall(
                r"rebalancing \d+ nodes with (\d+) leaves", result.stdout)]}


def check_photosphere(row, temperature_plus, temperature_minus=None, length_unit=1.0):
    if temperature_minus is None:
        temperature_minus = temperature_plus
    area = 4.0 * length_unit**2
    for side, temperature in (("plus", temperature_plus), ("minus", temperature_minus)):
        near(row[f"area_{side}"], area, f"{side} projected photosphere area")
        near(row[f"unresolved_area_{side}"], 0.0, f"{side} unresolved area")
        near(row[f"liso_{side}"], 4.0 * SIGMA * temperature**4 * area,
             f"{side} isotropic-equivalent luminosity")


def check_thin(row):
    for side in ("plus", "minus"):
        near(row[f"area_{side}"], 0.0, f"{side} thin-ray photosphere area")
        near(row[f"unresolved_area_{side}"], 4.0, f"{side} thin-ray unresolved area")
        near(row[f"liso_{side}"], 0.0, f"{side} thin-ray photosphere luminosity")


def static_row(metrics):
    require(len(metrics["light_curve"]) == 1, "static fixture emitted extra samples")
    require(not metrics["fld"], "hydro-only observer unexpectedly ran FLD")
    return metrics["light_curve"][0]


def check_uniform(args):
    result = {}
    for name, options in (("static", {}), ("rotating", {"omega": 0.7}),
                          ("units", {"code_to_g": 2, "code_to_cm": 3, "code_to_s": 5,
                                     "sod_pl": ideal_pressure(2, 1.0e4, 3, 5),
                                     "sod_pr": ideal_pressure(2, 1.0e4, 3, 5)})):
        result[name] = run(args, name, **options)
        check_photosphere(static_row(result[name]), 1.0e4,
                          length_unit=3 if name == "units" else 1)
    # The near half-slab is already optically thick: each observer must select
    # its own near material, rather than sharing a ray or reversing its order.
    result["layers"] = run(args, "layers", sod_phi=0,
                           sod_pl=ideal_pressure(2, 1.0e4),
                           sod_pr=ideal_pressure(2, 2.0e4))
    check_photosphere(static_row(result["layers"]), 2.0e4, 1.0e4)
    # Existing stop_step is inclusive: 2 requests three steps in one outer
    # batch. The diagnostic must sample the second completed individual step.
    result["evolving"] = run(args, "evolving", hydro="on", cfl=0.4, hard_dt=1.0e-10,
                             stop_time=3.0e-10, stop_step=2, light_curve_interval=2)
    evolving = result["evolving"]
    require(not evolving["fld"], "hydro-only cadence fixture unexpectedly ran FLD")
    require([row["step"] for row in evolving["light_curve"]] == [0.0, 2.0],
            "hydro-only light curve skipped the second step within an evolution batch")
    for row in evolving["light_curve"]:
        check_photosphere(row, 1.0e4)
    return result


def check_opacity(args):
    result = {}
    # alpha is already an inverse length: at rho=2, alpha=.3 gives tau=.6,
    # not 1.2. Scattering, in contrast, is a mass opacity and requires rho.
    result["absorption_thin"] = run(args, "absorption_thin", light_curve_absorption=0.3)
    check_thin(static_row(result["absorption_thin"]))
    common = {"sod_rhol": 0.5, "sod_rhor": 0.5,
              "sod_pl": ideal_pressure(0.5, 1.0e4),
              "sod_pr": ideal_pressure(0.5, 1.0e4), "light_curve_absorption": 0}
    for name, scattering in (("scattering_thin", 0.75), ("scattering_thick", 1.25)):
        result[name] = run(args, name, light_curve_scattering=scattering, **common)
        row = static_row(result[name])
        if name.endswith("thin"):
            check_thin(row)
        else:
            check_photosphere(row, 1.0e4)
    # rho^1 T^-1 turns this coefficient into alpha=.6 per unit length.
    result["power_law"] = run(args, "power_law", light_curve_absorption=3000,
                              light_curve_density_exponent=1,
                              light_curve_temperature_exponent=-1)
    check_photosphere(static_row(result["power_law"]), 1.0e4)
    return result


def check_amr(args):
    common = {"min_level": 1, "max_level": 3, "unigrid": "off",
              "sod_rhol": 1.0, "sod_rhor": 0.5,
              "sod_pl": ideal_pressure(1.0, 1.0e4),
              "sod_pr": ideal_pressure(0.5, 1.0e4)}
    result = {}
    for name, absorption in (("thick", 0.6), ("thin", 0.3), ("z_layers", 0.6)):
        result[name] = run(args, name, light_curve_absorption=absorption,
                           sod_phi=0 if name == "z_layers" else 90, **common)
        require(result[name]["leaves"] and 8 < result[name]["leaves"][-1] < 512,
                "AMR fixture did not create a mixed-resolution hierarchy")
        row = static_row(result[name])
        if name != "thin":
            check_photosphere(row, 1.0e4)
        else:
            # This also detects counting covered parent cells along a ray.
            check_thin(row)
        if name == "z_layers":
            # The photosphere lies behind the refined density interface. Fine
            # beams reaching a coarse source cell must retain their footprints
            # and be grouped under that one actual cell in the saved mesh.
            mesh = result[name]["meshes"][0]
            require(any(mesh[side]["patches"] > mesh[side]["cells"]
                        for side in ("plus", "minus")),
                    "z-layer AMR rays did not exercise partial patches on coarse hit cells")
    return result


def check_fld(args):
    result = run(args, "coupled", fld="on", sod_rhol=1, sod_rhor=1,
                 sod_pl=GAMMA - 1, sod_pr=GAMMA - 1, fld_cv=1, fld_ar=1, fld_c=100,
                 fld_absorption=1000, fld_scattering=0, fld_initial_energy=0.1,
                 fld_tolerance=1.0e-10, fld_max_iterations=100, fld_max_cycles=200,
                 hard_dt=0.01, stop_step=3, stop_time=0.03, light_curve_interval=2,
                 code_to_s=2)
    rows, thermal = result["light_curve"], result["fld"]
    require(len(thermal) == 3, "FLD fixture did not complete three implicit steps")
    require([row["step"] for row in rows] == [0.0, 2.0],
            "light-curve interval did not select every second completed step")
    check_photosphere(rows[0], 1.0)
    check_photosphere(rows[1], thermal[1]["min_temperature"])
    near(rows[1]["time"], 2.0 * sum(row["dt"] for row in thermal[:2]),
         "light-curve time is not in physical seconds")
    require(rows[1]["liso_plus"] < rows[0]["liso_plus"],
            "light curve did not respond to implicit gas cooling")
    return {"coupled": result}


def check_ipr(args):
    temperature = 1.0e6
    energy = temperature**GAMMA
    radiation = 4.0 * SIGMA / LIGHT_SPEED * temperature**4
    gas_cv = KB / (MU * MH * (GAMMA - 1.0))
    density = (energy - radiation) / (gas_cv * temperature)
    require(density > 0 and radiation > 0.5 * energy, "invalid IPR reference state")
    # Sod sets tau=e^(1/gamma). This choice makes that exactly T while also
    # satisfying e=rho*Cv*T+a*T^4, so the static IPR state is consistent.
    result = run(args, "stored_temperature", eos="IPR", sod_rhol=density,
                 sod_rhor=density, sod_pl=(GAMMA - 1) * energy,
                 sod_pr=(GAMMA - 1) * energy, ipr_nr_tol=1.0e-12)
    check_photosphere(static_row(result), temperature)
    return {"stored_temperature": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("uniform", "opacity", "amr", "fld", "ipr"),
                        required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    result = {"uniform": check_uniform, "opacity": check_opacity, "amr": check_amr,
              "fld": check_fld, "ipr": check_ipr}[args.case](args)
    (args.work_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case": args.case, "runs": len(result), "status": "passed"}))


if __name__ == "__main__":
    main()
