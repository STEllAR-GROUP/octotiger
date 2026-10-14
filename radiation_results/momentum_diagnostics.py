#!/usr/bin/env python3
"""Physical gas+radiation momentum histories, with explicit budget availability.

No simulation, VisIt invocation, movie rendering, or publication is performed.
Scientific plots use Matplotlib with a noninteractive backend.
The new runtime ledger is preferred to full-volume Silo-query integrals. Old
FLD flux budget placeholders are never interpreted as measured zero traction.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
import html
import hashlib
import json
import math
import os
import io
import tempfile
from pathlib import Path

AXES = ("x", "y", "z")
MODELS = {"m1": "M1 RK1", "fld": "FLD"}
FIXED = {"streaming_wave", "streaming_front", "gaussian_pulse", "equilibrium_sphere", "marshak"}
COUPLED = {"subcritical_shock", "supercritical_shock", "radiating_blast"}
FAMILIES = ("raw_delta", "boundary_corrected_delta", "accounted_balance")


def decimal(value, name="number"):
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise ValueError(f"Invalid {name}") from error
    if not result.is_finite():
        raise ValueError(f"Nonfinite {name}")
    return result


def vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must have three components")
    return [decimal(v, name) for v in value]


def flag(value):
    if value in (True, 1, "1", "on", "true"):
        return True
    if value in (False, 0, "0", "off", "false"):
        return False
    raise ValueError("Invalid explicit availability/configuration flag")


def read_runtime(path: Path):
    records = []
    units = None
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            units = units or row["units"]
            if row["units"] != units:
                raise ValueError("Momentum ledger changes units")
            record = {key: row[key] for key in ("time", "physical_c", "volume", "gas_energy", "radiation_energy")}
            record["gas_momentum"] = [row[f"gas_p{d}"] for d in AXES]
            record["radiation_momentum"] = [row[f"radiation_p{d}"] for d in AXES]
            record["gas_boundary_momentum"] = [row[f"gas_p{d}_boundary"] for d in AXES]
            for label, available, columns in (
                ("radiation_boundary_momentum", "radiation_boundary_available", [f"radiation_p{d}_boundary" for d in AXES]),
                ("external_impulse", "external_impulse_available", [f"external_p{d}" for d in AXES]),
            ):
                if flag(row[available]):
                    record[label] = [row[column] for column in columns]
                elif any(row[column].strip() for column in columns):
                    raise ValueError(f"Unavailable {label} must be blank, not a fabricated value")
            records.append(record)
    return {"schema": 1, "units": units, "integration": "full-volume leaf-cell",
            "precision": "runtime long-double reduction", "records": records,
            "source": str(Path(path).resolve())}


def add_legacy_budgets(document, metadata, path):
    """Match exact saved-state times; never interpolate cumulative impulses."""
    model = metadata.get("radiation_model")
    periodic = flag(metadata.get("config", {}).get("periodic", False))
    fixed = metadata.get("case") in FIXED
    records = [dict(record) for record in document["records"]]
    rows = []
    if path is not None and Path(path).is_file() and model == "m1":
        with Path(path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    config = metadata.get("config", {})
    normalized = metadata.get("case") == "marshak"
    mass = decimal(1 if normalized else config.get("code_to_g", 1))
    length = decimal(1 if normalized else config.get("code_to_cm", 1))
    duration = decimal(1 if normalized else config.get("code_to_s", 1))
    momentum_unit = mass * length / duration
    for record in records:
        t = decimal(record["time"])
        c_physical = decimal(record["physical_c"])
        c_code = c_physical * duration / length
        if periodic:
            record.setdefault("gas_boundary_momentum", [0, 0, 0])
            record.setdefault("radiation_boundary_momentum", [0, 0, 0])
        if fixed:
            # Gas transport is explicitly disabled in these saved benchmarks.
            record.setdefault("gas_boundary_momentum", [0, 0, 0])
        else:
            record.setdefault("external_impulse", [0, 0, 0])
        matched = [row for row in rows if abs(decimal(row["t"]) * duration - t) <=
                   Decimal("1e-12") * max(Decimal(1), abs(t))]
        if len(matched) > 1:
            raise ValueError("Ambiguous legacy momentum budget time")
        if matched:
            row = matched[0]
            scale = momentum_unit / (c_code * c_code)
            record["radiation_boundary_momentum"] = [str(decimal(row[f"f{d}_boundary"]) * scale) for d in AXES]
            if fixed:
                record["external_impulse"] = [str(decimal(row[f"f{d}_source"]) * scale) for d in AXES]
    return {**document, "records": records}


def build_diagnostic(volume_document, metadata, radiation_csv_path=None, total_momentum_csv_path=None):
    """Return dimensional and normalized vector diagnostics; no files are changed."""
    with localcontext() as context:
        context.prec = 50
        if total_momentum_csv_path is not None:
            document = read_runtime(Path(total_momentum_csv_path))
        else:
            document = add_legacy_budgets(volume_document, metadata, radiation_csv_path)
        if document.get("schema") != 1 or document.get("integration") != "full-volume leaf-cell":
            raise ValueError("Momentum requires documented full-volume leaf-cell integrals")
        if document.get("units") not in {"cgs", "code", "normalized"}:
            raise ValueError("Momentum integral units are missing")
        model, case = metadata.get("radiation_model"), metadata.get("case")
        if model not in MODELS or not isinstance(case, str):
            raise ValueError("Momentum model/case metadata is missing")
        records = document.get("records")
        if not isinstance(records, list) or len(records) < 2:
            raise ValueError("Momentum history needs at least two physical states")
        values = []
        previous_time = None
        c = None
        volume = None
        for record in records:
            t = decimal(record["time"], "time")
            speed = decimal(record["physical_c"], "physical c")
            vol = decimal(record["volume"], "volume")
            if speed <= 0 or vol <= 0 or t < 0 or (previous_time is not None and t <= previous_time):
                raise ValueError("Momentum times must increase; c and volume must be positive")
            c = c or speed
            volume = volume or vol
            if abs(speed - c) > Decimal("1e-12") * c or abs(vol - volume) > Decimal("1e-10") * volume:
                raise ValueError("Momentum history changes light speed or domain volume")
            previous_time = t
            pg = vector(record["gas_momentum"], "gas momentum")
            pr = vector(record["radiation_momentum"], "radiation momentum")
            eg = decimal(record["gas_energy"], "gas energy")
            er = decimal(record["radiation_energy"], "radiation energy")
            if eg < 0 or er < 0:
                raise ValueError("Momentum normalization cannot use negative energy")
            values.append({"time": t, "total": [a + b for a, b in zip(pg, pr)],
                           "gas": pg, "radiation": pr, "energy": eg + er,
                           **{key: vector(record[key], key) if key in record else None for key in
                              ("gas_boundary_momentum", "radiation_boundary_momentum", "external_impulse")}})
        first = values[0]
        scale = first["energy"] / c if first["time"] == 0 else Decimal(0)
        basis = "initial total gas+radiation energy / physical c"
        if scale == 0 and case == "marshak" and first["time"] == 0:
            # The normalized benchmark has prescribed incident flux=1 through
            # the x-minus area for the full configured interval. Use the same
            # fixed input scale for both models, not their resulting energies.
            length = decimal(metadata["length"], "Marshak domain length")
            end = decimal(metadata["time"], "Marshak configured final time")
            incident = decimal(metadata.get("incident_flux", 1))
            if length <= 0 or end <= 0 or incident <= 0:
                raise ValueError("Invalid prescribed Marshak incident-energy scale")
            scale = incident * length * length * end / c
            basis = "prescribed incident flux × x-minus area × configured duration / physical c"
        if scale <= 0:
            scale = None
            basis = "unavailable: missing t=0 or zero initial energy without a prescribed input scale"
        fixed = case in FIXED
        epsilon = float(metadata.get("radiation_reduced_c_factor", 1))
        if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError("Invalid reduced-light-speed factor")
        initial_momentum = sum(component * component for component in first["total"]).sqrt()
        amplification = None if scale is None else initial_momentum / scale
        notes = ["P = integral(rho v + F_lab/c²) dV; gas energy includes internal and kinetic energy, not rest-mass energy.",
                 "Raw change includes physical boundary impulses and external fixed-matter forces."]
        if amplification is not None:
            notes.append("Normalization by E0/c can amplify floating-point roundoff relative to initial bulk momentum. "
                         f"Here |P0|/(E0/c) = {float(amplification):.6g}; this ratio describes the change of scale, not an error bound.")
        if model == "fld":
            notes.append("FLD flux is derived. Physical total momentum is a diagnostic, not an exactly conserved evolved FLD invariant.")
        elif epsilon != 1:
            notes.append("Reduced-c M1 conserves a weighted momentum, not this physical total; physical changes are not a solver-error test.")
        else:
            notes.append("Physical-c M1 should balance recorded boundary and external impulses to numerical accuracy.")
        if fixed:
            notes.append("Matter is externally held fixed. M1 radiation source momentum is an external impulse, not cancelled gas recoil.")
        if not str(document.get("precision", "")).startswith("runtime"):
            notes.append("Silo/VisIt floating-point integrals have extraction roundoff; these are not long-double solver residuals.")
        output_rows = []
        for value in values:
            delta = [x - initial for x, initial in zip(value["total"], first["total"])]
            boundary = None
            if all(value[key] is not None and first[key] is not None for key in
                   ("gas_boundary_momentum", "radiation_boundary_momentum")):
                boundary = [sum(value[key][d] - first[key][d] for key in
                                ("gas_boundary_momentum", "radiation_boundary_momentum")) for d in range(3)]
            external = None
            if value["external_impulse"] is not None and first["external_impulse"] is not None:
                external = [x - initial for x, initial in zip(value["external_impulse"], first["external_impulse"])]
            corrected = [x + b for x, b in zip(delta, boundary)] if boundary is not None else None
            accounted = [x - s for x, s in zip(corrected, external)] if corrected is not None and external is not None else None
            row = {"time": float(value["time"]), "gas_momentum": list(map(float, value["gas"])),
                   "radiation_momentum": list(map(float, value["radiation"])),
                   "outward_boundary_impulse": None if boundary is None else list(map(float, boundary)),
                   "external_impulse": None if external is None else list(map(float, external))}
            for label, quantity in zip(FAMILIES, (delta, corrected, accounted)):
                magnitude = sum(x * x for x in quantity).sqrt() if quantity is not None else None
                row[label] = None if quantity is None else list(map(float, quantity))
                row[label + "_magnitude"] = None if magnitude is None else float(magnitude)
                row[label + "_normalized"] = None if quantity is None or scale is None else [float(x / scale) for x in quantity]
                row[label + "_magnitude_normalized"] = None if magnitude is None or scale is None else float(magnitude / scale)
            output_rows.append(row)
        available = {family: all(row[family] is not None for row in output_rows) for family in FAMILIES}
        missing = {}
        if not available["boundary_corrected_delta"]:
            missing["boundary_corrected_delta"] = "Full gas and radiation boundary impulse is not recorded; zero placeholders were not used."
        if not available["accounted_balance"]:
            missing["accounted_balance"] = "Complete boundary and external fixed-matter impulse records are unavailable."
        return {"schema": 1, "case": case, "level": metadata.get("level"), "model": model,
                "model_label": MODELS[model], "units": document["units"], "physical_c": float(c),
                "precision": document.get("precision", "saved-state floating-point integration"),
                "source": document.get("source"), "fixed_material": fixed, "notes": notes,
                "normalization": {"value": None if scale is None else float(scale), "basis": basis,
                                  "initial_total_energy": float(first["energy"]), "initial_time": float(first["time"]),
                                  "initial_momentum_magnitude": float(initial_momentum),
                                  "momentum_to_normalization_ratio": None if amplification is None else float(amplification)},
                "available": available, "unavailable_reasons": missing, "records": output_rows}


def diagnose_run(folder: Path, volume_integrals=None):
    folder = Path(folder)
    metadata = json.loads((folder / "run.json").read_text())
    ledger = folder / "momentum-diagnostics.csv"
    if ledger.is_file():
        return build_diagnostic(None, metadata, total_momentum_csv_path=ledger)
    if volume_integrals is None:
        receipt = folder / "movies/radiation-energy-flux/extraction.json"
        if not receipt.is_file():
            raise ValueError("No runtime momentum ledger or verified full-volume extraction receipt")
        entry = json.loads(receipt.read_text()).get("volume_integrals")
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError("Extraction receipt has no full-volume integral path")
        volume_integrals = Path(entry["path"])
        if not volume_integrals.is_absolute():
            volume_integrals = folder / volume_integrals
        if not entry.get("sha256") or hashlib.sha256(volume_integrals.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Full-volume extraction hash does not match its receipt")
    if isinstance(volume_integrals, (str, Path)):
        document = json.loads(Path(volume_integrals).read_text())
    else:
        document = volume_integrals
    return build_diagnostic(document, metadata, radiation_csv_path=folder / "radiation-conservation.csv")


def plot_svg(diagnostics, family, png_path=None):
    """Matplotlib scientific SVG, optionally saving the identical figure as PNG."""
    # The reporting process shares a strict CPU budget with solver validation.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    if "MPLCONFIGDIR" not in os.environ:
        cache = Path(tempfile.gettempdir()) / f"octotiger-momentum-matplotlib-{os.getuid()}"
        cache.mkdir(mode=0o700, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    normalized = all(d["normalization"]["value"] is not None for d in diagnostics)
    suffix = "_normalized" if normalized else ""
    title = {"raw_delta": r"Raw total momentum change $\Delta\mathbf{P}$",
             "boundary_corrected_delta": r"Boundary-corrected momentum $\Delta\mathbf{P}+\mathbf{B}_{\rm out}$",
             "accounted_balance": r"Accounted balance $\Delta\mathbf{P}+\mathbf{B}_{\rm out}-\mathbf{J}_{\rm ext}$"}[family]
    subtitle = ("Normalized by each run’s initial total gas+radiation energy / physical c"
                if normalized else "Dimensional momentum; initial-energy normalization unavailable")
    colors = {"m1": "#2563a8", "fld": "#c55c19"}
    styles = {"m1": "-", "fld": "--"}
    unit = "dimensionless" if normalized else ("g cm s⁻¹" if diagnostics[0]["units"] == "cgs" else "code momentum")
    time_unit = "Time (s)" if diagnostics[0]["units"] == "cgs" else "Time (code units)"
    settings = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                "axes.labelsize": 10, "svg.fonttype": "none", "svg.hashsalt": "octotiger-momentum",
                "axes.formatter.useoffset": False, "axes.spines.top": False, "axes.spines.right": False}
    with matplotlib.rc_context(settings):
        figure = Figure(figsize=(12.5, 8.2), facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 2).flat
        figure.subplots_adjust(left=.09, right=.97, top=.80, bottom=.17, hspace=.43, wspace=.30)
        figure.suptitle(title, y=.98, fontsize=16)
        figure.text(.5, .94, subtitle, ha="center", fontsize=11)
        handles = []
        for diagnostic in diagnostics:
            missing = "" if diagnostic["available"][family] else " (unavailable: missing ledger)"
            handles.append(Line2D([], [], color=colors[diagnostic["model"]],
                                  linestyle=styles[diagnostic["model"]], linewidth=2,
                                  label=diagnostic["model_label"] + missing))
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .915),
                      ncol=2, frameon=False)
        floor = None
        for panel, axis in enumerate(axes):
            axis.set_title(("x component", "y component", "z component", "Magnitude (logarithmic scale)")[panel])
            axis.set_xlabel(time_unit)
            axis.set_ylabel(unit)
            curves = []
            for diagnostic in diagnostics:
                key = family + ("_magnitude" if panel == 3 else "") + suffix
                points = [(row["time"], row[key] if panel == 3 else
                           (None if row[key] is None else row[key][panel])) for row in diagnostic["records"]]
                if any(value is None for _, value in points):
                    continue
                curves.append((diagnostic["model"], points))
            if not curves:
                axis.text(.5, .5, "Unavailable: required impulse ledger missing",
                          ha="center", va="center", transform=axis.transAxes, wrap=True)
                axis.set_xticks([])
                axis.set_yticks([])
                continue
            all_points = [point for _, points in curves for point in points]
            if panel == 3:
                floor = 1.e-18 if normalized else max(max(abs(value) for _, value in all_points) * 1.e-16, 1.e-300)
                axis.set_yscale("log")
            for model, points in curves:
                times = [time for time, _ in points]
                values = [max(value, floor) if panel == 3 else value for _, value in points]
                axis.plot(times, values, color=colors[model], linestyle=styles[model], linewidth=1.9,
                          marker="o" if len(times) <= 12 else None, markersize=3.5)
            if panel != 3:
                axis.axhline(0, color="#6b7280", linewidth=.7, alpha=.65)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
                if all(value == 0 for _, value in all_points):
                    axis.set_ylim(-1.e-16 if normalized else -1.e-30, 1.e-16 if normalized else 1.e-30)
            axis.grid(True, which="major", color="#dce2e7", linewidth=.7)
            axis.margins(x=.025)
        note = ("Normalization by E₀/c can magnify floating-point roundoff relative to |P₀| when bulk momentum dominates."
                if normalized else "No initial-energy scale was substituted for an unavailable normalization.")
        figure.text(.09, .075, note, fontsize=9, color="#4b5563")
        if floor is not None:
            figure.text(.09, .047, f"Magnitude values below {floor:.1e} share the plot floor; exported CSV retains the original values.",
                        fontsize=9, color="#4b5563")
        if any(not diagnostic["available"][family] for diagnostic in diagnostics):
            figure.text(.09, .021, "Unavailable budget curves are omitted; missing boundary traction is never replaced by zero.",
                        fontsize=9, color="#4b5563")
        stream = io.StringIO()
        figure.savefig(stream, format="svg", metadata={"Date": None, "Creator": "OctoTiger momentum diagnostics / Matplotlib"})
        if png_path is not None:
            figure.savefig(Path(png_path), format="png", dpi=140)
        figure.clear()
        return stream.getvalue()


def write_pair(diagnostics, output: Path):
    diagnostics = sorted(diagnostics, key=lambda value: list(MODELS).index(value["model"]))
    if not diagnostics or len({d["model"] for d in diagnostics}) != len(diagnostics):
        raise ValueError("Momentum pair needs unique M1/FLD diagnostics")
    if len({(d["case"], d["level"], d["units"]) for d in diagnostics}) != 1:
        raise ValueError("Momentum comparisons require matching case, level, and units")
    if diagnostics[0]["case"] in FIXED:
        return {"schema": 1, "case": diagnostics[0]["case"], "level": diagnostics[0]["level"],
                "omitted": True, "plots": [], "reason": "Momentum plots are omitted because matter is externally held fixed; the external reservoir prevents a closed gas+radiation momentum comparison."}
    if diagnostics[0]["case"] not in COUPLED:
        raise ValueError("Momentum plots are restricted to the three moving-material benchmarks")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for diagnostic in diagnostics:
        (output / f'{diagnostic["model"]}-momentum.json').write_text(json.dumps(diagnostic, indent=2, allow_nan=False) + "\n")
        columns = ["time", "normalization"] + [family + suffix for family in FAMILIES for suffix in
            ("_x", "_y", "_z", "_magnitude", "_x_normalized", "_y_normalized", "_z_normalized", "_magnitude_normalized")]
        with (output / f'{diagnostic["model"]}-momentum.csv').open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in diagnostic["records"]:
                row = {"time": record["time"], "normalization": diagnostic["normalization"]["value"]}
                for family in FAMILIES:
                    for suffix in ("", "_normalized"):
                        quantity = record[family + suffix]
                        for d, axis in enumerate(AXES):
                            row[family + "_" + axis + suffix] = None if quantity is None else quantity[d]
                        row[family + "_magnitude" + suffix] = record[family + "_magnitude" + suffix]
                writer.writerow(row)
    plots = []
    for family in FAMILIES:
        name = family.replace("_", "-") + ".svg"
        (output / name).write_text(plot_svg(diagnostics, family))
        plots.append(name)
    notes = "".join(f'<h2>{html.escape(d["model_label"])}</h2><p>Normalization: {html.escape(d["normalization"]["basis"])} = {d["normalization"]["value"]}.</p>' +
                    "".join(f'<p>{html.escape(note)}</p>' for note in d["notes"]) +
                    "".join(f'<p>{html.escape(reason)}</p>' for reason in d["unavailable_reasons"].values()) for d in diagnostics)
    images = "".join(f'<img src="{name}" alt="{html.escape(name)}" style="max-width:100%;height:auto">' for name in plots)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Total momentum comparison</title>'
        '<main style="max-width:1160px;margin:2rem auto;font:16px/1.5 sans-serif"><h1>Total gas+radiation momentum</h1>'
        '<p>B_out is the cumulative outward momentum flux; J_external is the impulse supplied by externally fixed matter. '
        'Gas/radiation internal exchange is not subtracted from their total.</p>' + images + notes + '</main>')
    return {"schema": 1, "case": diagnostics[0]["case"], "level": diagnostics[0]["level"],
            "page": str((output / "index.html").resolve()), "plots": plots,
            "models": [d["model"] for d in diagnostics]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-folder", type=Path)
    parser.add_argument("--fld-folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = [diagnose_run(folder) for folder in (args.m1_folder, args.fld_folder) if folder is not None]
    print(json.dumps(write_pair(diagnostics, args.output), indent=2))


if __name__ == "__main__":
    main()
