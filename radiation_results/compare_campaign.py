#!/usr/bin/env python3
"""Compare saved M1 RK1 and FLD backward-Euler output CSVs.

No solver, reference generator, network service or deployment is invoked. Each
level is kept separate. Slice discrepancies use a common Cartesian partition
of the saved leaf-cell averages; they are not full-volume accuracy norms.
"""

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
import numpy as np

CASES = ("streaming_wave", "streaming_front", "gaussian_pulse", "equilibrium_sphere",
         "subcritical_shock", "supercritical_shock", "radiating_blast", "marshak")
COUPLED_WITHOUT_REFERENCE = ("subcritical_shock", "supercritical_shock", "radiating_blast")
METHODS = ("m1", "fld")
PAIRED_CASES = frozenset(CASES)
INTEGRATORS = {"m1": "euler", "fld": "backward_euler"}
FLUX_SEMANTICS = {"m1": "evolved_m1_flux", "fld": "derived_fld_flux"}
LABELS = {"m1": "M1 RK1", "fld": "FLD backward Euler"}
FLD_REFERENCE_MODELS = {"gaussian_pulse": "periodic-constant-diffusion", "marshak": "su-olson-linear-halfspace",
                        "equilibrium_sphere": "steady-gaussian-emitter"}
FLD_REFERENCE = {
    "streaming_wave": ("no supplied FLD spatial analytic reference",
        "The exact M1 periodic-advection profile does not solve nonlinear FLD. The periodic radiation-energy "
        "integral is an exact invariant, but it does not specify the evolving spatial profile. Cross-model "
        "differences measure closure and discretization differences, not FLD error against that M1 solution."),
    "streaming_front": ("no supplied FLD spatial analytic reference",
        "The exact M1 translated-front profile does not solve nonlinear FLD. The periodic radiation-energy "
        "integral is an exact invariant, but it does not specify the evolving spatial profile. Cross-model "
        "differences measure closure and discretization differences, not FLD error against that M1 solution."),
    "gaussian_pulse": ("periodic constant-diffusion analytic reference",
        "Each retained Fourier mode decays as exp[-D |k|² t], with D = c/(3χ) and reference flux -D grad(E). "
        "It uses the same filtered initial energy as the M1 run, while M1 has its separate telegraph reference. "
        "This is an analytic diffusion-limit reference, not an exact solution of nonlinear flux-limited diffusion."),
    "equilibrium_sphere": ("steady Gaussian-emitter diffusion-limit analytic reference",
        "The steady Gaussian-emitter solution assumes constant linear diffusion. Saved reference values are "
        "cell averages evaluated with four Gauss points per axis. Nonlinear flux limiting, finite-domain "
        "boundary treatment, quadrature, and discretization can contribute to departures from this reference; "
        "it is not an exact stationary solution of nonlinear FLD."),
    "marshak": ("Su–Olson linear diffusion analytic reference",
        "The semi-infinite Su–Olson solution describes constant diffusion and equal linear radiation/material "
        "heat capacities. It is a diffusion-limit reference for FLD. Flux limiting, the finite-domain far "
        "boundary, cell-center sampling, and discretization can all contribute to its measured discrepancy.")}
FLD_REFERENCE.update({case: ("no supplied independent reference profile",
    "Moving gas and radiation evolve together with mixed-frame FLD. The published benchmark curves are not "
    "embedded; comparisons with M1 measure closure and discretization differences, not exact-solution error.")
    for case in COUPLED_WITHOUT_REFERENCE})
RAD_FIELDS = ("er", "fx", "fy", "fz")
HYDRO_FIELDS = ("rho", "vx", "vy", "vz", "gas_temperature", "radiation_temperature", *RAD_FIELDS)
FIELD_LABELS = {"er": "Radiation energy density", "fx": "Radiation flux x", "fy": "Radiation flux y",
                "fz": "Radiation flux z", "rho": "Gas density", "vx": "Gas velocity x",
                "vy": "Gas velocity y", "vz": "Gas velocity z", "gas_temperature": "Gas temperature",
                "radiation_temperature": "Radiation temperature", "material_egas": "Material energy density"}
REFERENCE = {
    "marshak": ("Su–Olson diffusion reference", "The supplied Su–Olson radiation/material reference solves "
                "the linear thermal-capacity diffusion problem in normalized units. The saved parameter values "
                "define its match to the source model; diffusion and finite-speed M1 transport can differ. "
                "Reference values are evaluated at cell centers, without an additional cell-volume average."),
    "streaming_wave": ("free-streaming analytic profile", "Periodic-advection cell averages are exact at zero opacity. Positive-opacity runs use this as a free-streaming-limit reference."),
    "streaming_front": ("free-streaming analytic profile", "Translated-front cell averages are exact at zero opacity and are a limit reference at positive opacity. "
                        "A discontinuity need not converge at second order."),
    "gaussian_pulse": ("linearized analytic reference", "Fourier telegraph-equation reference for isotropic, "
                       "linearized radiation moments. Nonlinear M1 can retain a closure-model error at finite amplitude."),
    "equilibrium_sphere": ("diffusion-limit analytic reference", "Steady diffusion solution for a Gaussian emitter. "
                           "Finite-flux nonlinear M1 and quadrature errors can produce a reference-error floor."),
    "subcritical_shock": ("no supplied analytic or digitized numerical profile", "Ensman (1994), Figure 9, "
                          "is the existing published numerical comparison target. Its curves are not embedded; "
                          "the local Cartesian adaptation differs from that original setup."),
    "supercritical_shock": ("no supplied analytic or digitized numerical profile", "Ensman (1994), Figure 13, "
                            "is the existing published numerical comparison target. Its curves are not embedded; "
                            "the local Cartesian adaptation differs from that original setup."),
    "radiating_blast": ("no supplied analytic or digitized numerical profile", "Ensman (1994), Figures 17–22, "
                        "provide published numerical comparisons. Their curves are not embedded, and the original "
                        "shell/deposition differs from this filled-core Cartesian adaptation. The Sedov–Taylor "
                        "solution is not an analytic solution of this radiating setup."),
}
IGNORED_SIGNATURE = {"radiation_model", "radiation_integrator", "radiation_flux_semantics",
                     "radiation_limiter", "disable_analytic", "radiation_algorithm_version",
                     "radiation_frame", "radiation_hydro_scheme",
                     "radiation_backend", "hydro_host_kernel_type", "hydro_device_kernel_type",
                     "hard_dt", "stop_step", "fld_boundary", "fld_nonlinear_tolerance",
                     "fld_linear_tolerance", "fld_max_iterations", "fld_max_linear_iterations"}
STYLE = """body{margin:0;background:#eef3f4;color:#172b35;font:16px/1.55 system-ui,sans-serif}
main{max-width:1180px;margin:28px auto;padding:28px;background:white;border-radius:12px}
h1,h2,h3{line-height:1.2}a{color:#076e76}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px;text-align:left;border-bottom:1px solid #d7e1e5}.scroll{overflow-x:auto}
.note{padding:14px;border-left:4px solid #b17b26;background:#fff6e6}img{width:100%;height:auto}
small{color:#536772}section{border-top:2px solid #d7e1e5;margin-top:28px;padding-top:20px}
@media(max-width:700px){main{margin:0;padding:16px}}"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def escaped(value):
    return html.escape(str(value), quote=True)


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def json_write(path, content):
    atomic_text(path, json.dumps(content, indent=2, allow_nan=False) + "\n")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def norms(values, weights=None):
    values = np.asarray(values, dtype=float)
    require(values.size > 0 and np.isfinite(values).all(), "Cannot measure nonfinite/empty differences")
    return {"L1": float(np.average(np.abs(values), weights=weights)),
            "L2": float(np.sqrt(np.average(values * values, weights=weights))),
            "Linf": float(np.max(np.abs(values)))}


def reference_contract(metadata):
    model, case = metadata["radiation_model"], metadata["case"]
    kind = metadata.get("reference_kind", REFERENCE[case][0])
    sampling = "cell-center point values" if case == "marshak" else "cell averages"
    if kind == "unavailable":
        if model == "fld":
            require(case in {"streaming_wave", "streaming_front", *COUPLED_WITHOUT_REFERENCE} and metadata.get("reference_model") == "none" and
                    metadata.get("reference_exact_for_solver") is False and
                    metadata.get("reference_sampling") == "none",
                    "Unavailable FLD references must explicitly declare no reference model or sampling")
        label, note = FLD_REFERENCE[case] if model == "fld" else ("no supplied analytic reference", "No applicable saved analytic reference.")
        return {"available": False, "kind": kind, "model": None, "exact_for_solver": False,
                "sampling": "none", "label": label, "note": note}
    if kind == "numerical-only" or case in COUPLED_WITHOUT_REFERENCE:
        return {"available": False, "kind": kind, "model": None, "exact_for_solver": False,
                "sampling": sampling, "label": "numerical-only", "note": "No applicable saved analytic reference."}
    if model == "fld":
        require(case in FLD_REFERENCE_MODELS and kind == "linear-diffusion-limit" and
                metadata.get("reference_model") == FLD_REFERENCE_MODELS.get(case) and
                metadata.get("reference_exact_for_solver") is False and
                metadata.get("reference_sampling", sampling) == sampling,
                "FLD references require an explicit applicable diffusion-limit model and non-exact qualification")
        label, note = FLD_REFERENCE[case]
        return {"available": True, "kind": kind, "model": metadata["reference_model"],
                "exact_for_solver": False, "sampling": sampling, "label": label, "note": note}
    label, note = REFERENCE[case]
    return {"available": True, "kind": kind, "model": metadata.get("reference_model", "m1-reference"),
            "exact_for_solver": case in {"streaming_wave", "streaming_front"} and
                metadata.get("streaming_optical_depth", 0) == 0 and kind != "free-streaming-limit",
            "sampling": sampling, "label": label, "note": note}


def method_identity(model, case):
    moving = model == "fld" and case in COUPLED_WITHOUT_REFERENCE
    return (2, "derived_fld_lab_flux") if moving else (1, FLUX_SEMANTICS[model])


def load_run(root, run, manifest=None):
    case, level = run["case"], run["level"]
    model = run["method"]
    folder = root / run["directory"]
    require(folder.resolve().is_relative_to(root), "Run directory escapes the campaign")
    metadata_path = folder / case / f"l{level}" / "run.json"
    require(metadata_path.resolve().is_relative_to(root), "Run metadata escapes the campaign")
    if run.get("run_metadata_sha256"):
        require(digest(metadata_path) == run["run_metadata_sha256"], "Completed run metadata hash changed")
    metadata = json.loads(metadata_path.read_text())
    remote_receipt = None
    if 'remote_import' in run:
        require(manifest is not None, 'Remote output requires its campaign runtime and import receipt')
        from run_campaign import validate_remote_import
        remote_receipt = validate_remote_import(root, manifest, run)
        require(metadata.get('executable_sha256') == remote_receipt['executable_sha256'],
                'Remote output does not match its validated executable')
    elif manifest and manifest.get('executable_sha256'):
        require(metadata.get('executable_sha256') == manifest['executable_sha256'],
                "Model output does not match the campaign's frozen executable")
    algorithm_version, flux_semantics = method_identity(model, case)
    batch = json.loads((folder / "batch.json").read_text())
    require(bool(batch.get("source_sha256")), "Missing saved source provenance")
    require(metadata.get("status") == "complete" and metadata.get("case") == case and
            metadata.get("level") == level and metadata.get("radiation_model") == model and
            metadata.get("radiation_integrator") == INTEGRATORS[model] and
            metadata.get("radiation_algorithm_version") == algorithm_version and
            metadata.get("radiation_flux_semantics") == flux_semantics,
            "Saved run identity/completion does not match the campaign")
    require(not metadata.get("radiation_ader_version") and
            metadata.get("runtime_profile", "baseline") == "baseline",
            "Historical ADER/runtime recovery metadata cannot be relabeled as M1/FLD")
    require(bool(metadata.get("comparison_signature")), "Missing saved physical comparison signature")
    if model == "fld" and case in COUPLED_WITHOUT_REFERENCE:
        require(metadata.get("radiation_frame") == "mixed" and
                metadata.get("radiation_hydro_scheme") == "first_order_conservative",
                "Moving FLD output requires explicit mixed-frame hydro metadata")
    if model == "fld":
        require("reference_kind" in metadata, "FLD output must explicitly declare its reference contract")
    reference = reference_contract(metadata)
    coupled = case in COUPLED_WITHOUT_REFERENCE
    stem = "hydro-slice" if coupled else "slice"
    path = folder / "plots" / case / f"l{level}" / f"{stem}.csv"
    require(path.resolve().is_relative_to(root), "Slice CSV escapes the campaign")
    with path.open() as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = [[float(row[name]) for name in columns] for row in reader]
    require(bool(rows), "Saved slice CSV is empty")
    array = np.asarray(rows)
    require(np.isfinite(array).all(), "Saved slice contains nonfinite data")
    values = {name: array[:, index] for index, name in enumerate(columns)}
    fields = HYDRO_FIELDS if coupled else RAD_FIELDS
    if case == "marshak":
        fields = (*fields, "material_egas")
    required = ("t", "x", "y", "z", "dx", *fields)
    require(all(name in values for name in required), "Saved slice is missing required columns")
    require(np.all(values["dx"] > 0), "Invalid leaf-cell width")
    time = float(metadata["time"])
    require(time > 0 and math.isfinite(time), "Invalid physical time")
    require(np.allclose(values["t"], time, rtol=2e-12, atol=0), "CSV and metadata times differ")
    require(np.allclose(values["z"], values["dx"] / 2, rtol=2e-10, atol=0),
            "Comparison requires the saved slice adjacent to z=0 on its positive side")
    if metadata.get("adaptive"):
        # The saved plot CSV is a constant projection onto finest-level pixels.
        # Its x/y centers are display pixels, while dx and z retain the native
        # leaf-cell width. Use the display width for area weights and raster
        # footprints; otherwise coarse leaves overlap in the common partition.
        pixel_dx = float(metadata["dx"])
        require(math.isfinite(pixel_dx) and pixel_dx > 0, "Invalid adaptive display pixel width")
        ratio = values["dx"] / pixel_dx
        max_ratio = 2 ** (int(metadata["level"]) - int(metadata["min_level"]))
        require(np.all(ratio >= 1) and np.all(ratio <= max_ratio) and
                np.allclose(np.log2(ratio), np.rint(np.log2(ratio)), rtol=0, atol=1e-10),
                "Adaptive display source widths are not dyadic")
        values["native_dx"] = values["dx"]
        values["dx"] = np.full_like(values["dx"], pixel_dx)
    units = json.loads(path.with_name(f"{stem}.units.json").read_text())
    reference_errors = {}
    has_reference = reference["available"]
    if has_reference:
        require(all(field + "_ref" in values for field in fields), "Missing saved analytic/reference columns")
        for field in fields:
            reference_errors[field] = norms(values[field] - values[field + "_ref"], values["dx"] ** 2)
    else:
        # Numerical-only records may originate from a generic CSV writer. Do
        # not render any stale/reference-shaped columns as an FLD solution.
        values = {name: value for name, value in values.items() if not name.endswith(("_ref", "_error"))}
    require(all(name in units and isinstance(units[name], str) for name in ("t", "x", "y", "z", "dx", *fields)),
            "Saved physical-field units are missing")
    return {"run": run, "metadata": metadata, "values": values, "fields": fields, "units": units,
            "path": path, "reference_errors": reference_errors,
            "has_reference": has_reference,
            "reference": reference,
            "metadata_path": metadata_path, "csv_sha256": digest(path), "metadata_sha256": digest(metadata_path),
            "source_sha256": batch["source_sha256"], "remote_receipt": remote_receipt}


def bounds(data):
    values = data["values"]
    half = values["dx"] / 2
    return (float(np.min(values["x"] - half)), float(np.max(values["x"] + half)),
            float(np.min(values["y"] - half)), float(np.max(values["y"] + half)))


def signature(metadata):
    result = {}
    for key, value in metadata.get("comparison_signature", {}).items():
        if key in IGNORED_SIGNATURE:
            continue
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            result[key] = value
    return result


def compatible(left, right, root=None, manifest=None):
    for key in ("time", "radiation_reduced_c_factor", "length"):
        a, b = float(left["metadata"][key]), float(right["metadata"][key])
        require(math.isclose(a, b, rel_tol=2e-12, abs_tol=0), f"The models have different {key}")
    require(left["fields"] == right["fields"], "The models have different physical fields")
    unit_fields = ("t", "x", "y", "z", "dx", *left["fields"])
    require(all(left["units"][field] == right["units"][field] for field in unit_fields),
            "The models have different saved physical-field units")
    require(signature(left["metadata"]) == signature(right["metadata"]),
            "The models have different saved physical/reconstruction settings")
    require(left["metadata"].get("executable_sha256") == right["metadata"].get("executable_sha256") and
            bool(left["metadata"].get("executable_sha256")), "The models have different/missing executable provenance")
    require(left["source_sha256"] == right["source_sha256"], "The models have different source provenance")
    require(left["metadata"].get("solver_library_sha256") == right["metadata"].get("solver_library_sha256"),
            "The models have different solver library provenance")
    remote = left.get('remote_receipt')
    require(bool(remote) == bool(right.get('remote_receipt')),
            'The models have different local/remote execution provenance')
    if manifest and manifest.get("executable_sha256"):
        if remote:
            from run_campaign import validate_remote_receipt
            for data in (left, right):
                validate_remote_receipt(manifest, data['run'], data['remote_receipt'])
            expected = manifest['remote_runtime']['executable_sha256']
        else:
            expected = manifest['executable_sha256']
        require(left["metadata"]["executable_sha256"] == expected,
                "Model output does not match the campaign's " +
                ("validated remote executable" if remote else "frozen executable"))
    if manifest and manifest.get("snapshot_sha256") and not remote:
        frozen_sources = manifest["snapshot_sha256"]
        require(all(frozen_sources.get(".campaign/source/" + path) == value
                    for path, value in left["source_sha256"].items()),
                "Model output does not match the campaign's frozen source")
    left_bounds, right_bounds = bounds(left), bounds(right)
    length = left_bounds[1] - left_bounds[0]
    require(np.allclose(left_bounds, right_bounds, rtol=0, atol=length * 2e-12),
            "The models cover different slice domains")


def raster(data, extent, spacing):
    x0, x1, y0, y1 = extent
    nx, ny = round((x1 - x0) / spacing), round((y1 - y0) / spacing)
    require(0 < nx <= 4096 and 0 < ny <= 4096, "Invalid or excessive comparison grid")
    selected = [*data["fields"], *(field + "_ref" for field in data["fields"]
                                  if field + "_ref" in data["values"])]
    result = {field: np.zeros((ny, nx)) for field in selected}
    coverage = np.zeros((ny, nx), dtype=np.uint8)
    values = data["values"]
    for row in range(len(values["dx"])):
        width = values["dx"][row]
        continuous = ((values["x"][row] - width / 2 - x0) / spacing,
                      (values["y"][row] - width / 2 - y0) / spacing, width / spacing)
        ix, iy, size = (round(value) for value in continuous)
        require(all(abs(value - round(value)) < 1e-7 for value in continuous),
                "Leaf-cell edges do not align on the common Cartesian partition")
        require(size > 0 and 0 <= ix < ix + size <= nx and 0 <= iy < iy + size <= ny,
                "Leaf cell lies outside the comparison grid")
        patch = np.s_[iy:iy + size, ix:ix + size]
        require(not np.any(coverage[patch]), "Overlapping leaf cells in saved slice")
        coverage[patch] = 1
        for field in selected:
            result[field][patch] = values[field][row]
    require(bool(np.all(coverage)), "Saved leaf slice leaves holes in the comparison domain")
    return result


def record_run(root, data):
    metadata = data["metadata"]
    return {"time": metadata["time"], "csv": str(data["path"].relative_to(root)),
            "csv_sha256": data["csv_sha256"], "metadata_sha256": data["metadata_sha256"],
            "leaf_slice_cells": metadata.get("mesh", {}).get("native_slice_cells", len(data["values"]["x"])),
            "display_slice_pixels": len(data["values"]["x"]),
            "reference_slice_errors": data["reference_errors"],
            "solver_full_volume_reference_errors": metadata.get("norms", {})
                if data["has_reference"] and metadata["radiation_model"] == "m1" else {},
            "reference_error_scope": "area-weighted saved positive-side slice; not full-volume norms",
            "reference_model": data["reference"]["model"],
            "reference_exact_for_solver": data["reference"]["exact_for_solver"],
            "reference_sampling": data["reference"]["sampling"],
            "reference_note": data["reference"]["note"],
            "reference_sha256": metadata.get("reference_sha256"),
            "reference_generator_sha256": metadata.get("reference_generator_sha256"),
            "reference_diffusion_coefficient": metadata.get("reference_diffusion_coefficient"),
            "analytic_mean_energy": metadata.get("analytic_mean_energy"),
            "analytic_invariants": metadata.get("fld_diagnostics", {}).get("analytic_invariants", {}),
            "slice_physical_bounds": metadata.get("fld_diagnostics", {}).get("slice_physical_bounds", {}),
            "reference_kind": metadata.get("reference_kind", REFERENCE[data["run"]["case"]][0]),
            "radiation_model": metadata["radiation_model"],
            "radiation_integrator": metadata["radiation_integrator"],
            "radiation_algorithm_version": metadata["radiation_algorithm_version"],
            "radiation_flux_semantics": metadata["radiation_flux_semantics"],
            "radiation_reduced_c_factor": metadata["radiation_reduced_c_factor"],
            "units": data["units"],
            "executable_sha256": metadata.get("executable_sha256"),
            "solver_library_sha256": metadata.get("solver_library_sha256"),
            "source_sha256": data["source_sha256"],
            "execution_provenance": ({"host": "QueenBee", "owner": data['remote_receipt']['owner'],
                "job_id": data['remote_receipt']['job_id'], "node": data['remote_receipt']['node'],
                "nodes": 1, "tasks": 1, "source_commit": data['remote_receipt']['source_commit'],
                "source_tree_sha256": data['remote_receipt']['source_tree_sha256'],
                "solver_library_sha256": data['remote_receipt']['solver_library_sha256'],
                "import_receipt": data['run']['remote_import']} if data.get('remote_receipt') else {"host": "local"}),
            "checkpoint_resume": metadata.get("checkpoint_resume")}


def make_plot(folder, case, level, loaded, projected, extent, spacing, pyplot):
    fields = next(iter(loaded.values()))["fields"]
    sample = next(iter(projected.values()))[fields[0]]
    x = extent[0] + (np.arange(sample.shape[1]) + .5) * spacing
    # The central positive row samples the same x-y positions for every method.
    row = max(0, min(sample.shape[0] - 1, round(-extent[2] / spacing)))
    figure, axes = pyplot.subplots(math.ceil(len(fields) / 2), 2, figsize=(11, 2.8 * math.ceil(len(fields) / 2)),
                                  squeeze=False, constrained_layout=True)
    colors = {"m1": "#c56a17", "fld": "#076e76"}
    csv_columns = {"x": x, "y": np.full(x.size, extent[2] + (row + .5) * spacing)}
    for index, field in enumerate(fields):
        axis = axes.flat[index]
        for method, grid in projected.items():
            profile = grid[field][row]
            axis.step(x, profile, where="mid", color=colors[method], label=LABELS[method], linewidth=1.6,
                      linestyle="-" if method == "m1" else "--",
                      marker="o" if method == "m1" else None, markerfacecolor="none",
                      markersize=3, markevery=max(1, len(x) // 20))
            csv_columns[f"{method}_{field}"] = profile
            if field + "_ref" in grid:
                reference = grid[field + "_ref"][row]
                csv_columns[f"{method}_{field}_ref"] = reference
                label = ("FLD diffusion-limit reference" if method == "fld" else
                         "M1 telegraph reference" if case == "gaussian_pulse" else "M1 reference")
                axis.step(x, reference, where="mid", color=colors[method],
                          linestyle=":" if method == "fld" else "-.", label=label, linewidth=1.1)
        axis.set_title(FIELD_LABELS[field], fontsize=10)
        units = next(iter(loaded.values()))["units"]
        axis.set_ylabel(units[field], fontsize=9)
        axis.set_xlabel(f'x ({units["x"]})', fontsize=9)
        axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useOffset=False)
        axis.grid(alpha=.18)
        axis.legend(fontsize=7)
    for axis in axes.flat[len(fields):]:
        axis.set_visible(False)
    time = next(iter(loaded.values()))["metadata"]["time"]
    time_unit = next(iter(loaded.values()))["units"]["t"]
    figure.suptitle(f'{case.replace("_", " ").title()} · level {level} · t = {time:g} {time_unit}\n'
                   'Shared x coordinates; positive central row of the saved z = 0⁺ slice', fontsize=12)
    temporary = folder / "profiles.tmp.png"
    figure.savefig(temporary, dpi=145)
    pyplot.close(figure)
    temporary.replace(folder / "profiles.png")
    temporary_csv = folder / "comparison.csv.tmp"
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_columns)
        writer.writerows(zip(*csv_columns.values()))
    temporary_csv.replace(folder / "comparison.csv")


def metric_table(metrics, units):
    if not metrics:
        return "<p>No reference-error norms are available for this problem.</p>"
    rows = "".join(f'<tr><th>{escaped(FIELD_LABELS.get(field, field))}</th><td>{escaped(units.get(field, ""))}</td>'
                   + "".join(f'<td>{value[norm]:.6e}</td>' for norm in ("L1", "L2", "Linf")) + "</tr>"
                   for field, value in metrics.items())
    return '<div class="scroll"><table><thead><tr><th>Field</th><th>Units</th><th>L₁</th><th>L₂</th>' \
           '<th>L∞</th></tr></thead><tbody>' + rows + "</tbody></table></div>"


def html_page(title, body):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" ' \
           f'content="width=device-width,initial-scale=1"><title>{escaped(title)}</title><style>{STYLE}</style>' \
           f'</head><body><main>{body}</main></body></html>'


def validate_manifest(manifest):
    require(manifest.get("schema") == 2 and manifest.get("workflow") == "m1-fld-radiation",
            "M1/FLD comparisons require a fresh schema-2 m1-fld-radiation campaign")
    models = manifest.get("methods")
    require(isinstance(models, list) and bool(models) and
            models == [model for model in METHODS if model in models],
            "Campaign models must be a nonempty canonical subset of m1 and fld")
    require(not manifest.get("runtime_profiles") and not manifest.get("runtime_comparisons"),
            "Historical runtime recovery profiles are not supported in the M1/FLD workflow")
    runs = manifest.get("runs", [])
    require(isinstance(runs, list), "Campaign runs must be a list")
    identities = []
    for run in runs:
        model = run.get("method")
        require(run.get("case") in CASES and model in models and
                run.get("model") == model and run.get("integrator") == INTEGRATORS[model] and
                type(run.get("level")) is int and run["level"] >= 0,
                "Invalid campaign case/model/integrator/level identity")
        require(model != "fld" or run["case"] in PAIRED_CASES,
                "FLD campaign case is unsupported")
        require(run.get("runtime_profile", "baseline") == "baseline",
                "Historical runtime recovery profiles cannot be used for M1/FLD comparisons")
        identities.append((run["case"], run["level"], model))
    require(len(identities) == len(set(identities)), "Duplicate comparison identity")
    unsupported = manifest.get("unsupported", [])
    require(isinstance(unsupported, list), "Unsupported combinations must be a list")
    for item in unsupported:
        require(isinstance(item, dict) and item.get("case") in CASES and item.get("model") in models and
                type(item.get("level")) is int and item["level"] >= 0 and bool(item.get("reason")),
                "Invalid unsupported model/problem entry")
        require((item['case'], item['level'], item['model']) not in identities,
                "A model/problem cannot be both scheduled and unsupported")
    selected = manifest.get("cases", [case for case in CASES if any(run["case"] == case for run in runs)])
    require(isinstance(selected, list) and all(case in CASES for case in selected) and
            len(selected) == len(set(selected)) and all(run["case"] in selected for run in runs),
            "Invalid campaign case selection")
    return [case for case in selected if any(run["case"] == case for run in runs)]


def refresh(root, manifest):
    selected = validate_manifest(manifest)
    root = Path(root).resolve()
    output = root / "comparisons"
    output.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    runs = manifest.get("runs", [])
    indexed = {(run["case"], run["level"], run["method"]): run for run in runs}
    unavailable = {(item['case'], item['level'], item['model']): item['reason']
                   for item in manifest.get('unsupported', [])}
    models = manifest["methods"]
    summary = {"schema": 2, "workflow": "m1-fld-radiation", "methods": models,
               "scope": "M1/FLD model discrepancies and separate model-specific reference errors",
               "dependencies": {"numpy": np.__version__, "matplotlib": matplotlib.__version__}, "cases": {}}
    scope_note = ('<p class="note">M1 and FLD solve different transport closures. Both updates are first order '
                  'in time. A small model discrepancy does not establish accuracy against an exact solution. '
                  'M1 evolves radiation flux; FLD reconstructs diagnostic flux from its energy gradient. '
                  'Flux differences are not errors in a shared radiation-momentum conservation law. '
                  'FLD uses physical light speed on the CPU. Moving cases include gas transport and radiation force/work; '
                  'their derived lab-frame flux includes radiation enthalpy advection.</p>')
    cards = []
    for case in selected:
        title = case.replace("_", " ").title()
        reference_kind, reference_note = REFERENCE[case]
        planned_models = [model for model in models if model == "m1" or case in PAIRED_CASES]
        planned_models = [model for model in planned_models if
            any(run['case'] == case and run['method'] == model for run in runs) or
            not any(identity[0] == case and identity[2] == model for identity in unavailable)]
        paired = len(planned_models) == 2
        body = (f'<p><a href="index.html">All problem comparisons</a> · <a href="../index.html">Campaign</a></p>'
                f'<h1>{escaped(title)}: ' + ' and '.join(LABELS[model] for model in planned_models) + '</h1>')
        if "m1" in planned_models:
            body += f'<p><strong>M1 reference:</strong> {escaped(reference_kind)}. {escaped(reference_note)}</p>'
        if "fld" in planned_models:
            label, note = FLD_REFERENCE[case]
            body += f'<p><strong>FLD reference:</strong> {escaped(label)}. {escaped(note)}</p>'
        reasons = sorted({reason for identity, reason in unavailable.items() if identity[0] == case})
        if reasons:
            body += ('<p class="note">Unavailable model/problem combinations: ' +
                     escaped('; '.join(reasons)) + '. No excluded matching partner is assumed.</p>')
        elif case not in PAIRED_CASES:
            body += '<p class="note">This case is retained for M1 only. No unsupported FLD run or matching partner is planned.</p>'
        elif not paired:
            body += '<p class="note">Only the selected model is planned; no cross-model discrepancy is computed.</p>'
        body += scope_note
        case_report = []
        levels = sorted({run["level"] for run in runs if run["case"] == case})
        for level in levels:
            level_models = [model for model in planned_models if (case, level, model) not in unavailable]
            paired = len(level_models) == 2
            record = {"case": case, "level": level, "reference_kind": "model-specific", "methods": {},
                      "status": "pending", "direct_slice_difference": {}, "difference_kind": "model discrepancy",
                      "difference_direction": "fld minus m1",
                      "flux_comparison": "derived FLD diagnostic flux minus evolved M1 flux; not momentum error"}
            record["reference_sampling"] = "cell-center point values" if case == "marshak" else "cell averages"
            loaded = {}
            section = f'<section id="l{level}"><h2>Level {level}</h2>'
            for model in level_models:
                run = indexed.get((case, level, model))
                status = run.get("status", "pending") if run else "pending"
                if status == "completed":
                    try:
                        loaded[model] = load_run(root, run, manifest)
                        record["methods"][model] = record_run(root, loaded[model])
                        provenance = record['methods'][model]['execution_provenance']
                        if provenance['host'] == 'QueenBee':
                            section += (f'<p>{escaped(LABELS[model])} ran on one QueenBee node, '
                                f'{escaped(provenance["node"])}, job {escaped(provenance["job_id"])}. '
                                'Its executable, source, libraries, configuration, and imported outputs '
                                'are bound to the verified completion receipt.</p>')
                    except (OSError, ValueError, KeyError) as error:
                        loaded.pop(model, None)
                        record["methods"][model] = {"error": str(error)}
                    page = root / run["directory"] / f"{case}.html"
                    if page.is_file():
                        section += (f'<p><a href="../{escaped(page.relative_to(root))}">{escaped(LABELS[model])}: '
                                    'complete results, parameters and reference discussion</a></p>')
                else:
                    section += f'<p>{escaped(LABELS[model])}: {escaped(status)}.</p>'
            folder = output / case / f"l{level}"
            folder.mkdir(parents=True, exist_ok=True)
            try:
                if len(loaded) == 2:
                    compatible(loaded["m1"], loaded["fld"], root, manifest)
                    record["status"] = "matched"
                elif loaded:
                    record["status"] = "awaiting_matching_method" if paired else next(iter(loaded)) + "_only"
                if loaded:
                    extent = bounds(next(iter(loaded.values())))
                    spacing = min(float(np.min(data["values"]["dx"])) for data in loaded.values())
                    projected = {model: raster(data, extent, spacing) for model, data in loaded.items()}
                    record["common_partition"] = {"extent": extent, "spacing": spacing,
                                                  "scope": "piecewise-constant leaf averages on common finest display pixels"}
                    if len(loaded) == 2:
                        record["direct_slice_difference"] = {
                            field: norms(projected["fld"][field] - projected["m1"][field])
                            for field in loaded["m1"]["fields"]}
                    make_plot(folder, case, level, loaded, projected, extent, spacing, pyplot)
                    record["plot"] = f"{case}/l{level}/profiles.png"
                    section += (f'<p><strong>{escaped(record["status"].replace("_", " "))}</strong>. '
                                'No other refinement level or physical time is substituted for a missing partner.</p>'
                                f'<a href="{case}/l{level}/profiles.png"><img src="{case}/l{level}/profiles.png" '
                                f'alt="Level {level} model profiles and separately applicable references"></a>'
                                f'<p><a href="{case}/l{level}/comparison.csv">Matched profile coordinates and values (CSV)</a></p>')
            except (ValueError, KeyError) as error:
                record.update(status="incompatible", reason=str(error), direct_slice_difference={})
                section += (f'<p class="note">Direct comparison withheld: {escaped(error)}. '
                            'Individual model-specific reference errors remain available below.</p>')
            if record["direct_slice_difference"]:
                section += '<h3>FLD minus M1: model discrepancy on the saved slice</h3>' + metric_table(
                    record["direct_slice_difference"], loaded["m1"]["units"])
            for model, data in loaded.items():
                section += f'<h3>{escaped(LABELS[model])}: model-specific reference diagnostics</h3>'
                if data["has_reference"]:
                    section += (f'<p><strong>{escaped(data["reference"]["label"])}</strong>. '
                        f'{escaped(data["reference"]["note"])}</p>'
                        '<p>Area-weighted reference errors on this model’s original leaf slice.</p>') + metric_table(
                        data["reference_errors"], data["units"])
                    volume = record["methods"][model]["solver_full_volume_reference_errors"]
                    if volume:
                        section += ('<details><summary>Solver-reported full-volume reference norms</summary>' +
                                    metric_table(volume, data["units"]) + '</details>')
                else:
                    section += '<p>Numerical-only output: no applicable embedded reference curve or analytic error norm.</p>'
                    invariants = record["methods"][model]["analytic_invariants"]
                    if invariants:
                        section += ('<h4>Exact integral invariant</h4><p>Saved full-volume energy-history diagnostics. '
                                    'These check the radiation-energy integral, not an analytic spatial profile.</p>'
                                    f'<pre>{escaped(json.dumps(invariants, indent=2, allow_nan=False))}</pre>')
                section += (f'<p><a href="../{escaped(data["path"].relative_to(root))}">Source slice CSV</a> · '
                            f'<a href="../{escaped(data["metadata_path"].relative_to(root))}">Run metadata</a></p>')
            if case in COUPLED_WITHOUT_REFERENCE and loaded:
                from momentum_diagnostics import diagnose_run, write_pair
                diagnostics = []
                missing = []
                for model, data in loaded.items():
                    try:
                        diagnostics.append(diagnose_run(data["metadata_path"].parent))
                    except (OSError, ValueError, KeyError) as error:
                        missing.append(f"{LABELS[model]}: {error}")
                if diagnostics:
                    momentum_folder = folder / "momentum"
                    record["momentum_diagnostics"] = write_pair(diagnostics, momentum_folder)
                    section += (f'<h4>Total gas and radiation momentum</h4><p><a href="{case}/l{level}/momentum/index.html">'
                                'Momentum components, magnitude, and available boundary balances</a></p>')
                if missing:
                    record["momentum_unavailable"] = missing
                    section += '<p>Momentum diagnostics unavailable for ' + escaped('; '.join(missing)) + '.</p>'
            elif case not in COUPLED_WITHOUT_REFERENCE:
                section += ('<p>Momentum plots are omitted for this fixed-material problem. The prescribed medium '
                            'acts as an external momentum reservoir; radiation can exchange momentum with it '
                            'without evolving a compensating gas momentum.</p>')
            for model, result in record["methods"].items():
                if "error" in result:
                    record["status"] = "invalid_output"
                    section += f'<p class="note">{escaped(LABELS[model])}: {escaped(result["error"])}</p>'
            section += f'<p><a href="{case}/l{level}/metrics.json">Model discrepancy and input hashes (JSON)</a></p></section>'
            json_write(folder / "metrics.json", record)
            case_report.append(record)
            body += section
        body += ('<section><h2>How these comparisons are measured</h2><p>All field values retain their saved units. '
                 'Central profiles share x coordinates and the positive row adjacent to y = z = 0. Adaptive '
                 'leaf-cell averages are prolonged as constants onto the finest common Cartesian x-y partition, '
                 'without extrapolation or time interpolation. Different leaf sizes may average different z extents '
                 'adjacent to the plane. These are saved finite-volume slices.</p>'
                 '<p>Slice L₁ is mean absolute discrepancy, L₂ is root-mean-square discrepancy, and L∞ is maximum '
                 'absolute discrepancy, with area weighting. These are two-dimensional diagnostics; separately '
                 'labeled solver reference norms cover the full volume. Each model is measured against its own '
                 'saved reference. A diffusion-limit error includes departures of the nonlinear closure from that limit.</p>')
        if case == "marshak":
            body += ('<p>The saved Su–Olson reference uses cell-center point values without a cell-volume average. '
                     'Reference differences for both models include sampling, closure, finite-domain, and '
                     'discretization effects; Su–Olson is not an exact nonlinear M1 or flux-limited FLD solution.</p>')
        body += '</section>'
        atomic_text(output / f"{case}.html", html_page(title + " model comparison", body))
        summary["cases"][case] = case_report
        statuses = "; ".join(f'L{item["level"]}: {item["status"].replace("_", " ")}' for item in case_report)
        card_reference = "; ".join(reference_kind + " (M1)" if model == "m1" else
                                  FLD_REFERENCE[case][0] + (" (FLD limit)" if case in FLD_REFERENCE_MODELS else " (FLD)")
                                  for model in planned_models)
        cards.append(f'<tr><th><a href="{case}.html">{escaped(title)}</a></th><td>{escaped(card_reference)}</td>'
                     f'<td>{escaped(statuses)}</td></tr>')
    intro = ('<p><a href="../index.html">Campaign dashboard</a></p><h1>M1 RK1 and FLD: model comparisons</h1>'
             '<p>All eight problems compare both closures at the same refinement level and physical time. '
             'Partial pairs remain visible while their partner is pending. '
             'References are attached to the model they describe.</p>' + scope_note +
             '<div class="scroll"><table><thead><tr><th>Problem</th><th>Available reference</th>'
             '<th>Comparison status</th></tr></thead><tbody>' + "".join(cards) +
             '</tbody></table></div><p><a href="summary.json">Comparison manifest (JSON)</a></p>')
    atomic_text(output / "index.html", html_page("M1 RK1 and FLD comparisons", intro))
    json_write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.campaign / "campaign.json").read_text())
    summary = refresh(args.campaign, manifest)
    print(json.dumps({"index": str(args.campaign.resolve() / "comparisons/index.html"),
                      "pairs": sum(item["status"] == "matched"
                                   for records in summary["cases"].values() for item in records)}, indent=2))


if __name__ == "__main__":
    main()
