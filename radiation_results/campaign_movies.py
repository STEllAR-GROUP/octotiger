#!/usr/bin/env python3
"""Render completed campaign Silo states with serial VisIt 3.4.2 and FFmpeg.

This is a reporting step: invoke at a simulation boundary, never alongside a
12-worker solver. It changes neither campaign.json nor the frozen solver input.
The VisIt API names follow the installed 3.4.2 Python attributes manual; in
particular, VectorAttributes supplies native nVectors/useStride/stride thinning.
"""

from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from campaign_profile_plots import PROFILE_SCRIPT


COUPLED = {"subcritical_shock", "supercritical_shock", "radiating_blast"}
SHOCKS = {"subcritical_shock", "supercritical_shock"}
STYLE_VERSION = 3
MOVIE_SUBDIR = Path("movies/radiation-energy-flux")
MODEL_LABELS = {"m1": "M1 RK1", "fld": "FLD backward Euler"}
MODEL_INTEGRATORS = {"m1": "euler", "fld": "backward_euler"}
MODEL_FLUX = {"m1": "evolved_m1_flux", "fld": "derived_fld_flux"}
FLD_REFERENCE_MODELS = {"gaussian_pulse": "periodic-constant-diffusion", "marshak": "su-olson-linear-halfspace",
                        "equilibrium_sphere": "steady-gaussian-emitter"}
FLD_CASES = {"streaming_wave", "streaming_front", "gaussian_pulse", "equilibrium_sphere", "marshak"}
CASES = {"streaming_wave", "streaming_front", "gaussian_pulse", "equilibrium_sphere",
         "subcritical_shock", "supercritical_shock", "radiating_blast", "marshak"}


def model_contract(metadata: dict, run: dict | None = None) -> dict:
    """Validate model identity before generating plots or reusing a cache."""
    model = metadata.get("radiation_model")
    if (model not in MODEL_LABELS or metadata.get("radiation_integrator") != MODEL_INTEGRATORS[model]
            or metadata.get("radiation_algorithm_version") != 1
            or metadata.get("radiation_flux_semantics") != MODEL_FLUX[model]
            or metadata.get("radiation_ader_version")
            or metadata.get("runtime_profile", "baseline") != "baseline"):
        raise ValueError("Movie input requires explicit current M1/FLD model, integrator, version, and flux semantics")
    if run is not None and (run.get("method") != model or run.get("model") != model
                            or run.get("integrator") != MODEL_INTEGRATORS[model]):
        raise ValueError("Movie run model/integrator does not match its saved metadata")
    if model == "fld" and metadata.get("case") not in FLD_CASES:
        raise ValueError("FLD campaign movies support stationary radiation problems only")
    reference = {"reference_kind": metadata.get("reference_kind", "not recorded"),
                 "reference_model": metadata.get("reference_model"),
                 "reference_exact_for_solver": metadata.get("reference_exact_for_solver"),
                 "reference_description": "Movies show saved numerical states; model-specific analytic comparisons are in the final-time profiles."}
    if reference["reference_kind"] == "not recorded":
        reference["reference_description"] = "Movies show saved numerical states; see the final-time run report for analytic-reference applicability."
    if reference["reference_kind"] in {"numerical-only", "unavailable"}:
        reference["reference_description"] = "Movies show saved numerical states; no applicable analytic reference is recorded."
    if model == "fld" and reference["reference_kind"] == "unavailable":
        if (metadata["case"] not in {"streaming_wave", "streaming_front"} or
                reference["reference_model"] != "none" or reference["reference_exact_for_solver"] is not False or
                metadata.get("reference_sampling") != "none"):
            raise ValueError("Unavailable FLD references must explicitly declare no reference model or sampling")
        reference["reference_description"] = ("Movies show saved numerical states. No nonlinear FLD spatial analytic "
            "reference is supplied for this benchmark; the M1 streaming profile is not an FLD reference. "
            "The final-time report includes exact invariant checks.")
    if model == "fld" and reference["reference_kind"] not in {"numerical-only", "not recorded", "unavailable"}:
        if (metadata["case"] not in FLD_REFERENCE_MODELS or reference["reference_kind"] != "linear-diffusion-limit" or
                reference["reference_model"] != FLD_REFERENCE_MODELS[metadata["case"]] or
                reference["reference_exact_for_solver"] is not False):
            raise ValueError("FLD movie references require an applicable, explicitly qualified diffusion-limit model")
        reference["reference_description"] = ("Final-time profiles compare FLD with a linear diffusion-limit analytic reference, "
            "not an exact solution of nonlinear flux-limited diffusion. The movie shows numerical states only.")
    return {"radiation_model": model, "radiation_integrator": MODEL_INTEGRATORS[model],
            "radiation_algorithm_version": 1, "radiation_flux_semantics": MODEL_FLUX[model],
            "model_label": MODEL_LABELS[model], **reference,
            "flux_label": "Derived FLD flux" if model == "fld" else "Evolved M1 flux",
            "flux_description": ("Arrows show derived FLD diagnostic flux -D grad(E); radiation momentum is not evolved."
                                 if model == "fld" else "Arrows show the independently evolved M1 radiation flux.")}


def validate_movie_manifest(manifest: dict) -> None:
    models = manifest.get("methods")
    if (manifest.get("schema") != 2 or manifest.get("workflow") != "m1-fld-radiation"
            or not isinstance(models, list) or not models
            or models != [model for model in MODEL_LABELS if model in models]
            or manifest.get("runtime_profiles") or manifest.get("runtime_comparisons")):
        raise ValueError("Movies require a fresh schema-2 m1-fld-radiation campaign without recovery profiles")
    if not isinstance(manifest.get("runs"), list):
        raise ValueError("Movie campaign runs must be a list")
    identities = set()
    for run in manifest.get("runs", []):
        model = run.get("method")
        identity = (run.get("case"), run.get("level"), model)
        if (model not in models or run.get("case") not in CASES or run.get("model") != model
                or run.get("integrator") != MODEL_INTEGRATORS[model]
                or type(run.get("level")) is not int or run["level"] < 0
                or run.get("runtime_profile", "baseline") != "baseline" or identity in identities
                or (model == "fld" and run.get("case") not in FLD_CASES)):
            raise ValueError("Invalid movie campaign model/case identity")
        identities.add(identity)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numerical_silos(folder: Path) -> list[Path]:
    numbered = []
    for path in folder.iterdir():
        match = re.fullmatch(r"X\.(\d+)\.silo", path.name)
        if match:
            numbered.append((int(match[1]), path.resolve()))
    numbered.sort()
    files = [path for _, path in numbered] + [folder / "final.silo"]
    if not numbered or any(not path.is_file() or not path.stat().st_size for path in files):
        raise ValueError(f"Need nonempty X.*.silo and final.silo in {folder}")
    return files


def playback_schedule(times: list[float], seconds: float, fps: int, hold: float,
                      movie_timing: dict | None = None) -> tuple[list[int], dict]:
    """Sample a continuous (optionally piecewise) simulation clock at video FPS.

    Each picture is an existing state held until the next recorded state. If
    video sampling would omit even one stored state, reject the schedule rather
    than silently discard it or fabricate an interpolated picture.
    """
    if len(times) < 2 or any(not math.isfinite(t) for t in times):
        raise ValueError("Need at least two finite snapshot times")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Snapshot times must strictly increase")
    if not isinstance(fps, int) or fps < 1 or not math.isfinite(hold) or hold < 0:
        raise ValueError("Invalid FPS or endpoint hold")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Movie duration must be positive and finite")
    pause = round(hold * fps)
    if movie_timing is None:
        active = round(seconds * fps) - 2 * pause
        requested = [{"end_time": times[-1], "seconds": active / fps}]
        reason = "Uniform actual-simulation-time playback"
    else:
        if not isinstance(movie_timing, dict) or not isinstance(movie_timing.get("segments"), list) or not movie_timing["segments"]:
            raise ValueError("movie_timing requires a nonempty segments list")
        requested = movie_timing["segments"]
        reason = movie_timing.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError("movie_timing reason must be text")
    segments = []
    start_time = times[0]
    first_frame = pause
    for index, segment in enumerate(requested):
        if not isinstance(segment, dict):
            raise ValueError("Every movie timing segment must be an object")
        try:
            end_time, duration = float(segment["end_time"]), float(segment["seconds"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Each movie timing segment needs numeric end_time and seconds") from error
        if not math.isfinite(end_time) or end_time <= 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Timing segment end_time and seconds must be positive and finite")
        if index == len(requested) - 1:
            if not math.isclose(end_time, times[-1], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Final movie timing segment must match the saved final time")
            end_time = times[-1]
        if end_time <= start_time or end_time > times[-1]:
            raise ValueError("Movie timing segment ends must strictly increase within the saved time interval")
        count = round(duration * fps)
        if count < 2:
            raise ValueError("Every timing segment must have at least two video frames")
        segments.append({"start_time": start_time, "end_time": end_time,
                         "requested_seconds": duration, "seconds": count / fps,
                         "video_frames": count, "first_video_frame": first_frame,
                         "end_video_frame_exclusive": first_frame + count})
        start_time = end_time
        first_frame += count
    count = first_frame + pause
    if count > 1_000_000:
        raise ValueError("Invalid movie duration or endpoint hold")
    sequence = [0] * pause
    for segment in segments:
        for i in range(segment["video_frames"]):
            physical_time = segment["start_time"] + (segment["end_time"] - segment["start_time"]) * i / segment["video_frames"]
            sequence.append(max(0, bisect.bisect_right(times, physical_time) - 1))
    if pause:
        sequence.extend([len(times) - 1] * pause)
    else:
        # With no endpoint hold, show the endpoint on the last output frame.
        sequence[-1] = len(times) - 1
    missing = sorted(set(range(len(times))) - set(sequence))
    if missing:
        missing_times = ", ".join(f"{times[index]:.9g}" for index in missing[:8])
        raise ValueError(f"Playback would skip {len(missing)} recorded states (times {missing_times}); increase segment seconds or FPS")
    return sequence, {"mode": "piecewise_actual_time" if movie_timing else "uniform_actual_time",
                      "reason": reason, "segments": segments, "fps": fps,
                      "endpoint_hold_seconds": pause / fps, "active_seconds": sum(s["seconds"] for s in segments),
                      "seconds": len(sequence) / fps, "video_frames": len(sequence),
                      "all_recorded_states_displayed": True,
                      "interpolation": "none; zero-order hold of stored states",
                      "frame_sampling": "Each segment samples its starting time and excludes its ending time; final state is shown during the endpoint hold or last frame"}


def frame_schedule(times: list[float], seconds: float, fps: int, hold: float,
                   movie_timing: dict | None = None) -> list[int]:
    return playback_schedule(times, seconds, fps, hold, movie_timing)[0]


def movie_view(metadata: dict, requested: dict | None) -> tuple[list[float], dict | None]:
    half = float(metadata["length"]) / 2
    if not math.isfinite(half) or half <= 0:
        raise ValueError("Movie domain length must be finite and positive")
    if requested is None:
        return [-half, half, -half, half], None
    if not isinstance(requested, dict) or not isinstance(requested.get("reason", ""), str):
        raise ValueError("movie_view must be an object with a text reason")
    bounds = []
    for axis in ("x", "y"):
        values = requested.get(axis)
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError("movie_view needs two bounds for each of x and y")
        try:
            low, high = map(float, values)
        except (TypeError, ValueError) as error:
            raise ValueError("movie_view bounds must be numeric") from error
        if not math.isfinite(low) or not math.isfinite(high) or low >= high or low < -half or high > half:
            raise ValueError("movie_view bounds must be finite, ordered and inside the full domain")
        bounds.extend([low, high])
    return bounds, {"x": bounds[:2], "y": bounds[2:], "reason": requested.get("reason", "")}


def effective_movie_view(args: argparse.Namespace, requested: dict | None) -> dict | None:
    """An explicit full-domain policy takes precedence over saved display crops."""
    return None if getattr(args, "full_domain", False) else requested


def effective_physical_scope(args: argparse.Namespace, run: dict | None) -> str | None:
    """Keep scientific scope while updating the former shock display wording."""
    run = run or {}
    scope = run.get("physical_scope")
    if getattr(args, "full_domain", False) and run.get("case") in SHOCKS and isinstance(scope, str):
        scope = scope.replace("cropped central maps", "full-domain central maps")
    return scope


def source_signature(folder: Path, files: list[Path], args: argparse.Namespace,
                     run: dict | None = None) -> dict:
    inputs = []
    for silo in files:
        associated = [silo]
        data = Path(str(silo) + ".data")
        if data.is_dir():
            associated += sorted(path for path in data.rglob("*") if path.is_file())
        for path in associated:
            stat = path.stat()
            inputs.append({"path": str(path.relative_to(folder)), "size": stat.st_size,
                           "mtime_ns": stat.st_mtime_ns})
    return {"style_version": STYLE_VERSION, "renderer_sha256": sha256(Path(__file__)),
            "profile_renderer_sha256": sha256(Path(__file__).with_name("campaign_profile_plots.py")),
            "run_metadata_sha256": sha256(folder / "run.json"), "inputs": inputs,
            "width": args.width, "height": args.height, "vectors": args.vectors,
            "arrow_stride": args.arrow_stride, "fps": args.fps, "seconds": args.seconds,
            "hold": args.hold, "visit": str(Path(args.visit).resolve()),
            "movie_timing": (run or {}).get("movie_timing"),
            "movie_view": effective_movie_view(args, (run or {}).get("movie_view")),
            "full_domain": bool(getattr(args, "full_domain", False)),
            "physical_scope": effective_physical_scope(args, run)}


def prepare_private_encoding(frames_json: Path, output_directory: Path,
                             args: argparse.Namespace, movie_timing: dict | None = None) -> dict:
    """Reuse saved PNGs for private encoding QA; write a plan, launch nothing.

    The new directory contains links to the existing images and an FFmpeg argv
    list. It never writes a production campaign-movie.json or changes originals.
    The caller must execute the returned command and inspect the result before
    claiming video validation. This is intentionally separate from publication.
    """
    frames_json = frames_json.resolve()
    rendered = json.loads(frames_json.read_text())
    if rendered.get("status") != "complete":
        raise ValueError("Private encoding needs a complete saved frame manifest")
    frames = rendered["frames"]
    panels = rendered["panels"]
    if panels not in (["radiation"], ["radiation", "matter"], ["radiation", "matter", "temperature"]):
        raise ValueError("Unsupported saved movie panel layout")
    sequence, playback = playback_schedule([float(frame["time"]) for frame in frames],
                                           args.seconds, args.fps, args.hold, movie_timing)
    sources = {}
    for panel in panels:
        for index, frame in enumerate(frames):
            path = (frames_json.parent / frame[panel + "_image"]).resolve()
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing cached image: {path}")
            sources[panel, index] = path
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    for panel in panels:
        for index, state in enumerate(sequence):
            (output_directory / f"{panel}_{index:06d}.png").symlink_to(sources[panel, state])
    command = [args.ffmpeg, "-hide_banner", "-nostdin", "-v", "warning", "-y", "-xerror",
               "-filter_threads", "1", "-filter_complex_threads", "1"]
    for panel in panels:
        command += ["-threads", "1", "-framerate", str(args.fps), "-i", str(output_directory / (panel + "_%06d.png"))]
    if len(panels) == 3:
        command += ["-filter_complex", "[0:v][1:v]hstack=inputs=2[top];[top][2:v]vstack=inputs=2[v]", "-map", "[v]"]
    elif len(panels) == 2:
        command += ["-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]"]
    else:
        command += ["-map", "0:v"]
    output = output_directory / "private-qa.mp4"
    command += ["-frames:v", str(len(sequence)), "-an", "-c:v", "libx264", "-threads", "1", "-crf", "18",
                "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    plan = {"encoding_only_qa": True, "publication_allowed": False, "source_frames_json": str(frames_json),
            "playback": playback, "frame_state_indices": sequence, "ffmpeg_command": command,
            "output": str(output)}
    write_json(output_directory / "encoding-plan.json", plan)
    return plan


# Executed by VisIt's embedded Python, with its plotting API in the global scope.
# Scalar queries operate on the sliced AMR data, not a uniform-grid CSV export.
VISIT_SCRIPT = r'''
import json, math, os, sys, traceback
with open("request.json") as stream:
    request = json.load(stream)
metadata = request["metadata"]
database = request["database"]
half = float(metadata["length"]) / 2
# A tiny positive offset picks the z>=0 zone averages at the mesh interface.
slice_z = float(metadata["dx"]) * 1.0e-6

def check_stop():
    for path in request.get("stop_paths", []):
        if os.path.exists(path):
            raise RuntimeError("STOP marker present: " + path)

def checked(value, action):
    if value == 0:
        raise RuntimeError(action + " failed: " + GetLastError())
    return value

def slice_plot():
    if request.get("movie_view"):
        # Crop native cells before glyph sampling so a small viewport still
        # receives the requested sparse arrows instead of a few whole-box ones.
        checked(AddOperator("Box", 0), "Add view-region Box")
        box = BoxAttributes()
        box.minx, box.maxx, box.miny, box.maxy = request["view_bounds"]
        box.minz, box.maxz = -half, half
        checked(SetOperatorOptions(box), "Set view-region Box")
    checked(AddOperator("Slice", 0), "Add Slice")
    atts = SliceAttributes()
    atts.originType = atts.Intercept
    atts.originIntercept = slice_z
    atts.axisType = atts.ZAxis
    atts.normal = (0, 0, 1)
    atts.upAxis = (0, 1, 0)
    atts.project2d = 1
    atts.interactive = 0
    checked(SetOperatorOptions(atts), "Set Slice")

def scalar_plot(name):
    DeleteAllPlots()
    checked(AddPlot("Pseudocolor", name, 1, 0), "Add " + name)
    atts = PseudocolorAttributes()
    atts.centering = atts.Zonal
    atts.lightingFlag = 0
    checked(SetPlotOptions(atts), "Set pseudocolor")
    slice_plot()
    checked(DrawPlots(), "Draw " + name)

def query_range():
    checked(Query("MinMax", use_actual_data=1), "MinMax")
    result = GetQueryOutputValue()
    if len(result) != 2 or not all(math.isfinite(float(v)) for v in result):
        raise RuntimeError("Invalid slice extrema: " + repr(result))
    return [float(result[0]), float(result[1])]

def scalar_style(low, high, gaussian=False):
    observed = [low, high]
    if gaussian:
        if high <= 0:
            raise RuntimeError("Gaussian energy excess has no positive values")
        return {"scale": "log", "limits": [high * 1.0e-6, high],
                "floor": high * 1.0e-6, "unclipped_range": [low, high]}
    if high > 0 and low >= 0 and (low == 0 or high / low >= 1000):
        floor = max(low, high * 1.0e-6)
        return {"scale": "log", "limits": [floor, high], "floor": floor,
                "unclipped_range": [low, high]}
    if low == high:
        pad = max(abs(low) * 1.0e-8, 1.0e-30)
        low, high = low - pad, high + pad
    return {"scale": "linear", "limits": [low, high], "floor": None,
            "unclipped_range": observed}

def text_label(text, x, y, size):
    obj = CreateAnnotationObject("Text2D")
    obj.text = text
    obj.position = (x, y)
    obj.height = size
    obj.fontFamily = obj.Arial
    obj.fontBold = 0
    return obj

def configure_panel(panel, style, density_contours):
    DeleteAllPlots()
    if panel == "radiation":
        raw = "campaign_energy_excess" if request["gaussian"] else "er"
        field = "campaign_energy_display"
        label = "Radiation energy density"
        if request["gaussian"]:
            label += " - background"
        units = request["energy_units"]
        vector = "campaign_flux"
        arrows = request["flux_label"] + " direction in xy; equal-length arrows"
        legend_units = "dimensionless" if metadata.get("units", {}).get("system") == "dimensionless" else units
        legend_title = ("E - E_bg" if request["gaussian"] else "E") + "\n(" + legend_units + ")"
    else:
        raw, field, label = "rho", "campaign_density_display", "Mass density"
        units, vector = request["density_units"], "campaign_velocity"
        arrows = "Velocity direction in xy; arrows have equal length"
        legend_title = "rho\n(" + units + ")"
    expression = raw if style["floor"] is None else "max(%s,%.17g)" % (raw, style["floor"])
    checked(DefineScalarExpression(field, expression), "Define display scalar")
    checked(AddPlot("Pseudocolor", field, 1, 0), "Add panel")
    atts = PseudocolorAttributes()
    atts.scaling = atts.Log if style["scale"] == "log" else atts.Linear
    atts.minFlag, atts.maxFlag = 1, 1
    atts.min, atts.max = style["limits"]
    atts.centering, atts.lightingFlag = atts.Zonal, 0
    atts.colorTableName = "campaign_viridis"
    style["color_table"] = atts.colorTableName
    checked(SetPlotOptions(atts), "Set scalar style")
    slice_plot()
    checked(AddPlot("Mesh", "quadmesh", 1, 0), "Add central mesh")
    ma = MeshAttributes()
    ma.legendFlag = 0
    ma.lineWidth = 0
    ma.meshColorSource = ma.MeshCustom
    ma.meshColor = (70, 70, 70, 255)
    ma.opaqueMode = ma.Off
    ma.showInternal = 1
    ma.opacity = 0.16
    checked(SetPlotOptions(ma), "Set faint mesh")
    slice_plot()
    if panel == "radiation" and density_contours:
        checked(AddPlot("Contour", "rho", 1, 0), "Add density contours")
        ca = ContourAttributes()
        ca.contourMethod = ca.Value
        ca.contourValue = tuple(density_contours)
        ca.colorType = ca.ColorBySingleColor
        ca.singleColor = (55, 55, 55, 255)
        ca.lineWidth = 0
        ca.legendFlag = 0
        checked(SetPlotOptions(ca), "Set density contours")
        slice_plot()
    checked(AddPlot("Vector", vector, 1, 0), "Add arrows")
    va = VectorAttributes()
    va.glyphLocation = va.AdaptsToMeshResolution
    va.origOnly = 1
    va.useStride = 1 if request["arrow_stride"] else 0
    va.stride = request["arrow_stride"] or 1
    va.nVectors = request["vectors"]
    va.colorByMagnitude = 0
    va.vectorColor = (255, 255, 255, 255)
    va.useLegend = 0
    va.scaleByMagnitude = 0
    va.autoScale = 0
    va.scale = min(request["view_bounds"][1] - request["view_bounds"][0],
                   request["view_bounds"][3] - request["view_bounds"][2]) * 0.035
    va.headOn, va.headSize, va.lineWidth = 1, 0.3, 1
    va.vectorOrigin = va.Middle
    checked(SetPlotOptions(va), "Set native vector thinning")
    slice_plot()
    checked(DrawPlots(), "Draw panel")
    legend = GetAnnotationObject(GetPlotList().GetPlots(0).plotName)
    legend.useCustomTitle = 1
    legend.customTitle = legend_title
    legend.fontHeight = 0.017
    aa = AnnotationAttributes()
    aa.userInfoFlag, aa.databaseInfoFlag, aa.timeInfoFlag = 0, 0, 0
    aa.backgroundColor, aa.foregroundColor = (255,255,255,255), (0,0,0,255)
    aa.axes2D.visible = 1
    for axis, title in ((aa.axes2D.xAxis, "x"), (aa.axes2D.yAxis, "y")):
        axis.title.userTitle, axis.title.userUnits = 1, 1
        axis.title.title, axis.title.units = title, request["length_units"]
        axis.title.font.bold, axis.title.font.italic = 0, 0
    SetAnnotationAttributes(aa)
    view = View2DAttributes()
    view.windowCoords = tuple(request["view_bounds"])
    view.viewportCoords = (0.22, 0.95, 0.20, 0.82)
    view.fullFrameActivationMode = view.Off
    SetView2D(view)
    labels = [text_label(label + " (" + units + "); " + style["scale"] + " scale", 0.06, 0.965, 0.024),
              text_label(request["title"], 0.06, 0.93, 0.020),
              text_label("", 0.23, 0.865, 0.021),
              text_label(arrows, 0.06, 0.075, 0.018)]
    sample = ("every %dth arrow" % request["arrow_stride"] if request["arrow_stride"]
              else "approximately %d arrows" % request["vectors"])
    labels.append(text_label("Central xy slice (z=0+" + (", cropped" if request["movie_view"] else ", full domain")
                             + "); faint cell mesh; " + sample, 0.06, 0.046, 0.016))
    if panel == "radiation" and density_contours:
        labels.append(text_label("Gray density contours: " + ", ".join("%.3g" % x for x in density_contours)
                                 + " " + request["density_units"], 0.06, 0.105, 0.015))
    if style["floor"] is not None:
        labels.append(text_label("Values <= %.3g %s use the floor color; fixed limits for all frames" % (style["floor"], units),
                                 0.06, 0.02, 0.015))
    else:
        labels.append(text_label("Fixed color limits for all frames; recorded states, no interpolation",
                                 0.06, 0.02, 0.015))
    return labels

def save_png(filename):
    sa = SaveWindowAttributes()
    sa.outputToCurrentDirectory = 0
    sa.outputDirectory = os.getcwd()
    sa.fileName = filename
    sa.family = 0
    sa.format = sa.PNG
    sa.width, sa.height = request["width"], request["height"]
    sa.resConstraint = sa.NoConstraint
    sa.screenCapture = 0
    SetSaveWindowAttributes(sa)
    checked(SaveWindow(), "Save " + filename)

# PROFILE_HELPERS

try:
    check_stop()
    color_table = ColorControlPointList()
    color_table.ClearControlPoints()
    for position, color in request["color_points"]:
        point = ColorControlPoint()
        point.position = position
        point.colors = tuple(color)
        color_table.AddControlPoints(point)
    checked(AddColorTable("campaign_viridis", color_table), "Register fixed viridis color table")
    checked(OpenDatabase(database), "Open Silo series")
    checked(DefineVectorExpression("campaign_flux", "{fx,fy,fz}"), "Define flux")
    if request["coupled"]:
        # Octo-Tiger stores rho=sum(rho_i), vx/vy/vz as Silo expressions.
        checked(DefineVectorExpression("campaign_velocity", "{vx,vy,vz}"), "Define velocity")
    background = float(metadata.get("background", 0)) if request["gaussian"] else 0
    checked(DefineScalarExpression("campaign_energy_excess", "er-(%.17g)" % background), "Define excess")
    scalar_plot("campaign_energy_excess" if request["gaussian"] else "er")
    frames, energy_ranges = [], []
    for state, silo in enumerate(request["silos"]):
        check_stop()
        checked(SetTimeSliderState(state), "Select Silo state")
        checked(Query("Time"), "Read Silo time")
        time = float(GetQueryOutputValue())
        extrema = query_range()
        if not math.isfinite(time) or time < 0:
            raise RuntimeError("Invalid Silo time")
        if frames and time <= frames[-1]["time"]:
            if os.path.basename(silo) == "final.silo" and math.isclose(time, frames[-1]["time"], rel_tol=1e-12, abs_tol=1e-14):
                frames.pop()
                energy_ranges.pop()
            else:
                raise RuntimeError("Silo times fail strict increasing order")
        frames.append({"state": state, "time": time, "silo": silo})
        energy_ranges.append(extrema)
    if len(frames) < 2 or not math.isclose(frames[0]["time"], 0, abs_tol=max(float(metadata["time"])*1e-10,1e-14)):
        raise RuntimeError("Missing initial state")
    if not math.isclose(frames[-1]["time"], float(metadata["time"]), rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError("Snapshot final time does not match completed run")
    styles = {"radiation": scalar_style(min(x[0] for x in energy_ranges), max(x[1] for x in energy_ranges), request["gaussian"])}
    density_contours = []
    if request["coupled"]:
        scalar_plot("rho")
        ranges = []
        for frame in frames:
            check_stop()
            checked(SetTimeSliderState(frame["state"]), "Select density time")
            ranges.append(query_range())
        styles["matter"] = scalar_style(min(x[0] for x in ranges), max(x[1] for x in ranges))
        low, high = min(x[0] for x in ranges), max(x[1] for x in ranges)
        if high > low * (1 + 1e-10):
            density_contours = ([math.exp(math.log(low) + (math.log(high)-math.log(low))*i/5) for i in range(1,5)]
                                if low > 0 and high/low >= 1000 else [low+(high-low)*i/5 for i in range(1,5)])
    profiles = scan_temperature_profiles(frames) if request["temperature_profile"] else None
    for panel, style in styles.items():
        labels = configure_panel(panel, style, density_contours)
        for i, frame in enumerate(frames):
            check_stop()
            checked(SetTimeSliderState(frame["state"]), "Select movie time")
            labels[2].text = "t = %.7g %s | stored state %d/%d" % (frame["time"], request["time_units"], i+1, len(frames))
            filename = "%s_%06d.png" % (panel, i)
            save_png(filename)
            frame[panel + "_image"] = filename
        for label in labels:
            label.Delete()
    if profiles is not None:
        render_temperature_profiles(frames, profiles)
    check_stop()
    result = {"status": "complete", "frames": frames, "styles": styles, "density_contours": density_contours,
              "panels": list(styles) + (["temperature"] if profiles is not None else []),
              "temperature_profile": profiles,
              "slice_z": slice_z, "background_subtracted": background,
              "geometry": "Positive-side cell averages adjacent to z=0; slice offset dx_finest*1e-6",
              "mesh": {"variable": "quadmesh", "central_slice": True, "opacity": 0.16, "lineWidth": 0},
              "vector_projection": "Full stored 3D flux/velocity vectors projected by VisIt Slice into xy; equal-length direction glyphs",
              "vector_attributes": {"useStride": bool(request["arrow_stride"]), "stride": request["arrow_stride"] or 1,
                                    "nVectors": request["vectors"], "scaleByMagnitude": False, "autoScale": False}}
    with open("frames.json", "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print("CAMPAIGN_MOVIE_RENDER_COMPLETE", flush=True)
except Exception:
    with open("failure.txt", "w") as stream:
        traceback.print_exc(file=stream)
    traceback.print_exc()
    sys.exit(2)
sys.exit(0)
'''
VISIT_SCRIPT = VISIT_SCRIPT.replace("# PROFILE_HELPERS", PROFILE_SCRIPT)


def run_logged(command: list[str], cwd: Path, log: Path, *, allow_shutdown_250=False) -> int:
    stop_paths = []
    if (cwd / "request.json").exists():
        stop_paths = json.loads((cwd / "request.json").read_text()).get("stop_paths", [])
    with log.open("w") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 1800
        try:
            while process.poll() is None:
                if any(Path(path).exists() for path in stop_paths):
                    raise RuntimeError("Movie command cancelled by STOP marker")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Movie command exceeded 30 minutes; see {log}")
                # VisIt's CLI can remain alive after its viewer aborts. Detect
                # that concrete failure instead of waiting forever at the gate.
                if "VisIt's viewer exited abnormally" in log.read_text(errors="replace"):
                    raise RuntimeError(f"VisIt viewer crashed; see {log}")
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if process.returncode and not (allow_shutdown_250 and process.returncode == 250):
        raise RuntimeError(f"Command exited {process.returncode}; see {log}")
    return process.returncode


def stopped(campaign: Path) -> None:
    if (campaign / "STOP").exists():
        raise RuntimeError(f"Campaign STOP marker present: {campaign / 'STOP'}")


def write_visit_job(metadata: dict, files: list[Path], diagnostics: Path,
                    args: argparse.Namespace, title: str, *, coupled: bool,
                    stop_paths: list[Path] | None = None, view: dict | None = None) -> dict:
    """Write a VisIt job without starting any process, also useful for private QA.

    This helper does not publish assets or mark production runs complete. A caller
    may use short retained coupled diagnostics to exercise plotting privately.
    Production publication always goes through render_one's completed-run checks.
    """
    contract = model_contract(metadata)
    if contract["radiation_model"] == "fld" and coupled:
        raise ValueError("Stationary FLD movies cannot display a coupled gas-velocity panel")
    diagnostics.mkdir(parents=True, exist_ok=True)
    units = metadata.get("units", {})
    view_bounds, normalized_view = movie_view(metadata, effective_movie_view(args, view))
    installation = Path(args.visit).resolve().parents[1]
    tables = sorted(installation.glob("*/linux-x86_64/resources/colortables/viridis.ct"))
    if not tables:
        raise ValueError("Installed VisIt viridis color table is missing")
    tree = ET.parse(tables[-1])
    color_points = []
    for node in tree.findall(".//Object[@name='ColorControlPoint']"):
        position = float(node.find("Field[@name='position']").text)
        color = [int(v) for v in node.find("Field[@name='colors']").text.split()]
        color_points.append([position, color])
    if len(color_points) < 2:
        raise ValueError("Invalid installed viridis color table")
    request = {**contract, "metadata": metadata, "database": str(diagnostics / "series.visit"),
               "color_points": color_points, "color_table_sha256": sha256(tables[-1]),
               "silos": [str(path) for path in files], "coupled": coupled,
               "temperature_profile": metadata["case"] in SHOCKS, "profile_height": 400,
               "view_bounds": view_bounds, "movie_view": normalized_view,
               "profile_full_domain": True,
               "view_policy": "full_domain" if normalized_view is None else "configured_crop",
               "full_domain_forced": bool(getattr(args, "full_domain", False)),
               "gaussian": metadata["case"] == "gaussian_pulse", "width": args.width,
               "height": args.height, "vectors": args.vectors, "arrow_stride": args.arrow_stride,
               "energy_units": units.get("energy_density", "Silo units"),
               "density_units": "g/cm^3" if units.get("system") == "CGS" else "Silo units",
               "time_units": units.get("time", "Silo time"), "length_units": units.get("length", "Silo units"),
               "title": title, "stop_paths": [str(path) for path in (stop_paths or [])]}
    write_json(diagnostics / "request.json", request)
    for path in files:
        if "\n" in str(path) or "\r" in str(path):
            raise ValueError("Silo paths cannot contain newlines")
    (diagnostics / "series.visit").write_text("\n".join(str(path) for path in files) + "\n")
    (diagnostics / "render.py").write_text(VISIT_SCRIPT)
    return request


def render_one(campaign: Path, run: dict, args: argparse.Namespace) -> dict:
    stopped(campaign)
    folder = (campaign / run["directory"] / run["case"] / f"l{run['level']}").resolve()
    folder.relative_to(campaign)
    metadata_path = folder / "run.json"
    metadata = json.loads(metadata_path.read_text())
    contract = model_contract(metadata, run)
    if metadata.get("status") != "complete" or metadata["case"] != run["case"] or metadata["level"] != run["level"]:
        raise ValueError(f"Completed run identity mismatch: {metadata_path}")
    if sha256(metadata_path) != run.get("run_metadata_sha256"):
        raise ValueError(f"Completed metadata hash mismatch: {metadata_path}")
    files = numerical_silos(folder)
    # Validate requested playback before starting a renderer. Actual Silo times
    # receive the same validation after rendering, including no skipped states.
    playback_schedule([0., float(metadata["time"])], args.seconds, args.fps, args.hold, run.get("movie_timing"))
    movie_view(metadata, effective_movie_view(args, run.get("movie_view")))
    if not isinstance(run.get("physical_scope", ""), str):
        raise ValueError("physical_scope must be text")
    signature = source_signature(folder, files, args, run)
    destination = folder / MOVIE_SUBDIR
    manifest_path = destination / "campaign-movie.json"
    if manifest_path.exists() and not args.force:
        cached = json.loads(manifest_path.read_text())
        if cached.get("status") == "complete" and cached.get("signature") == signature:
            if all((destination / key).is_file() and sha256(destination / key) == value
                   for key, value in cached.get("sha256", {}).items()) and cached.get("sha256"):
                print(f"Reusing movie: {run['id']}", flush=True)
                return cached
    diagnostics = destination / "renders" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    diagnostics.mkdir(parents=True)
    request = write_visit_job(metadata, files, diagnostics, args,
                              f"{run['case'].replace('_', ' ')} | {contract['model_label']} | level {run['level']} | "
                              + ("AMR" if metadata.get("adaptive") else "uniform mesh"),
                              coupled=run["case"] in COUPLED,
                              stop_paths=[campaign / "STOP"] + args.stop_file, view=run.get("movie_view"))
    print(f"Rendering {run['id']} from {len(files)} stored Silo files", flush=True)
    code = run_logged([args.visit, "-cli", "-nowin", "-noconfig", "-s", str(diagnostics / "render.py")],
                      diagnostics, diagnostics / "visit.log", allow_shutdown_250=True)
    messages = (diagnostics / "visit.log").read_text(errors="replace")
    errors = ["VisIt: Error", "VisIt could not", "Traceback (most recent call last)",
              "There was an error when trying to", "SetTimeSliderState was called when there was no time slider"]
    if any(marker in messages for marker in errors) or "CAMPAIGN_MOVIE_RENDER_COMPLETE" not in messages:
        raise RuntimeError(f"VisIt render incomplete or reported errors; see {diagnostics / 'visit.log'}")
    rendered = json.loads((diagnostics / "frames.json").read_text())
    if rendered["status"] != "complete":
        raise RuntimeError("Incomplete VisIt frame manifest")
    frames = rendered["frames"]
    panels = rendered["panels"]
    for panel in panels:
        for frame in frames:
            image = diagnostics / frame[panel + "_image"]
            if not image.is_file() or image.stat().st_size == 0:
                raise RuntimeError(f"Missing frame: {image}")
        run_logged([args.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
                    "-filter_threads", "1", "-i", str(diagnostics / (panel + "_%06d.png")),
                    "-frames:v", str(len(frames)), "-threads", "1", "-f", "null", "-"],
                   diagnostics, diagnostics / (panel + "-verify.log"))
    times = [frame["time"] for frame in frames]
    sequence, playback = playback_schedule(times, args.seconds, args.fps, args.hold, run.get("movie_timing"))
    encode_dir = diagnostics / "encode"
    encode_dir.mkdir()
    for panel in panels:
        for i, state in enumerate(sequence):
            os.link(diagnostics / frames[state][panel + "_image"], encode_dir / f"{panel}_{i:06d}.png")
    prefix = [args.ffmpeg, "-hide_banner", "-nostdin", "-v", "warning", "-y", "-xerror",
              "-filter_threads", "1", "-filter_complex_threads", "1"]
    inputs = []
    for panel in panels:
        inputs += ["-threads", "1", "-framerate", str(args.fps), "-i", str(encode_dir / (panel + "_%06d.png"))]
    if len(panels) == 3:
        composition = "[0:v][1:v]hstack=inputs=2[top];[top][2:v]vstack=inputs=2[v]"
    elif len(panels) == 2:
        composition = "[0:v][1:v]hstack=inputs=2[v]"
    else:
        composition = None
    filtering = ["-filter_complex", composition, "-map", "[v]"] if composition else ["-map", "0:v"]
    encoded = diagnostics / "movie.mp4"
    run_logged(prefix + inputs + filtering + ["-frames:v", str(len(sequence)), "-an", "-c:v", "libx264",
               "-threads", "1", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)],
               diagnostics, diagnostics / "encode.log")
    probe = subprocess.run([args.ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(encoded)],
                           capture_output=True, text=True, check=True)
    media = json.loads(probe.stdout)
    write_json(diagnostics / "ffprobe.json", media)
    stream = media["streams"][0]
    if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p" or int(stream["nb_frames"]) != len(sequence):
        raise RuntimeError("Encoded video failed format/frame-count checks")
    run_logged([args.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(encoded),
                "-threads", "1", "-f", "null", "-"], diagnostics, diagnostics / "verify-video.log")
    # Poster shows a middle stored state to avoid an uninformative initial state.
    poster_state = len(frames) // 2
    poster = diagnostics / "poster.png"
    if composition:
        poster_inputs = []
        for panel in panels:
            poster_inputs += ["-threads", "1", "-i", str(diagnostics / frames[poster_state][panel + "_image"])]
        run_logged(prefix + poster_inputs + filtering + ["-frames:v", "1", "-threads", "1", str(poster)],
                   diagnostics, diagnostics / "poster.log")
    else:
        shutil.copy2(diagnostics / frames[poster_state]["radiation_image"], poster)
    stopped(campaign)
    assets = {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
              for name, path in (("movie.mp4", encoded), ("poster.png", poster))}
    description = ("Central z=0+ radiation energy slice with faint cell mesh and equal-length flux direction arrows. "
                   + contract["flux_description"] + " ")
    if request["coupled"]:
        description += "Gray density contours and a matching density/velocity panel. "
    if request["temperature_profile"]:
        description += "Synchronized gas and radiation temperature profiles along the central x line appear below the maps. "
    if request["gaussian"]:
        description += "Gaussian energy excess above the recorded background uses a labeled logarithmic floor. "
    description += "Fixed color limits; actual stored Silo times held without solution interpolation."
    description += " " + contract["reference_description"]
    if run.get("movie_timing"):
        description += " Playback segments: " + "; ".join(
            f"{segment['start_time']:g}–{segment['end_time']:g} {request['time_units']} over {segment['seconds']:g} playback seconds"
            for segment in playback["segments"]) + "."
        if playback["reason"]:
            description += " " + playback["reason"]
    if request["movie_view"]:
        bounds = request["view_bounds"]
        description += (f" Cropped view: x=[{bounds[0]:g}, {bounds[1]:g}], "
                        f"y=[{bounds[2]:g}, {bounds[3]:g}] {request['length_units']}.")
        if request["movie_view"]["reason"]:
            description += " " + request["movie_view"]["reason"]
        if request["temperature_profile"]:
            description += " The temperature profile retains full-domain x coverage below the cropped maps."
    else:
        description += " The maps show the full simulation domain without cropping."
        if request["temperature_profile"]:
            description += " The temperature profile also spans the full domain."
    physical_scope = effective_physical_scope(args, run)
    if physical_scope:
        description += " " + physical_scope
    result = {"schema": 2, **contract, "status": "complete", "run_id": run["id"], "case": run["case"],
              "run_metadata_sha256": sha256(metadata_path), "assets": assets,
              "title": request["title"], "description": description, "frames": len(frames),
              "scale": rendered["styles"]["radiation"]["scale"],
              "level": run["level"], "method": run["method"], "renderer": "VisIt 3.4.2",
              "visit_exit_code": code, "path": str(destination / "movie.mp4"),
              "poster": str(destination / "poster.png"), "signature": signature,
              "created_utc": datetime.now(timezone.utc).isoformat(), "panels": panels,
              "color_scale": rendered["styles"]["radiation"]["scale"], "styles": rendered["styles"],
              "color_limits": rendered["styles"]["radiation"]["limits"],
              "color_table_sha256": request["color_table_sha256"], "density_contours": rendered["density_contours"],
              "background_subtracted": rendered["background_subtracted"], "geometry": rendered["geometry"],
              "slice_z": rendered["slice_z"], "vector_projection": rendered["vector_projection"],
              "mesh": rendered["mesh"],
              "temperature_profile": rendered["temperature_profile"],
              "vector_attributes": rendered["vector_attributes"], "source_snapshots": len(frames),
              "displayed_snapshots": len(set(sequence)), "snapshot_times": times,
              "seconds": len(sequence)/args.fps, "fps": args.fps, "video_frames": len(sequence),
              "endpoint_hold_seconds": round(args.hold*args.fps)/args.fps,
              "timing": "Actual Silo time gaps; stored states held without solution interpolation; endpoint holds included",
              "movie_timing": run.get("movie_timing"), "playback": playback,
              "movie_view": request["movie_view"], "view_bounds": request["view_bounds"],
              "view_policy": request["view_policy"], "profile_full_domain": request["profile_full_domain"],
              "full_domain_forced": request["full_domain_forced"],
              "physical_scope": physical_scope,
              "poster_time": times[poster_state], "time_units": request["time_units"],
              "field_units": request["energy_units"], "frames_json": str(diagnostics / "frames.json"),
              "units": metadata.get("units", {}),
              "flux_fields": {"components": ["fx", "fy", "fz"], "storage": "physical flux density F; not F/c",
                              "semantics": contract["radiation_flux_semantics"],
                              "description": contract["flux_description"],
                              "units": metadata.get("units", {}).get("flux", "Silo units"),
                              "glyph_lengths": "equal-length directions; no magnitude information"},
              "velocity_fields": ({"components": ["vx", "vy", "vz"], "source": "Silo expressions; momentum/rho with recorded rotating-frame correction",
                                   "units": "cm/s" if metadata.get("units", {}).get("system") == "CGS" else "Silo units",
                                   "glyph_lengths": "equal-length directions; no magnitude information"}
                                  if request["coupled"] else None),
              "diagnostics": str(diagnostics), "sha256": {name: details["sha256"] for name, details in assets.items()}}
    for source, name in ((encoded, "movie.mp4"), (poster, "poster.png")):
        temporary = destination / (name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination / name)
    write_json(manifest_path, result)
    # Hard links only; original rendered PNGs and all command logs remain intact.
    shutil.rmtree(encode_dir)
    print(f"Completed movie: {run['id']} ({len(frames)} stored states)", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-id")
    selection.add_argument("--all-completed", action="store_true")
    parser.add_argument("--plan", action="store_true", help="List jobs without starting VisIt/FFmpeg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--full-domain", action="store_true",
                        help="Show the complete simulation domain in maps and profiles, overriding every saved movie_view crop")
    parser.add_argument("--stop-file", type=Path, action="append", default=[], help="Additional cancellation marker to check between frames")
    parser.add_argument("--visit", default=os.environ.get("VISIT") or shutil.which("visit") or str(Path.home()/"visit3_4_2.linux-x86_64/bin/visit"))
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    parser.add_argument("--width", type=int, default=900, help="Width per panel, even number")
    parser.add_argument("--height", type=int, default=800, help="Height per panel, even number")
    parser.add_argument("--vectors", type=int, default=144)
    parser.add_argument("--arrow-stride", type=int, default=0, help="Use every Nth native vector instead of a target count")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--hold", type=float, default=1)
    args = parser.parse_args()
    if min(args.width, args.height) < 256 or args.width % 2 or args.height % 2:
        parser.error("Movie dimensions must be even and >=256")
    if args.vectors < 1 or args.arrow_stride < 0 or args.fps < 1 or args.hold < 0 or not math.isfinite(args.seconds):
        parser.error("Invalid vector or playback settings")
    frame_schedule([0., 1.], args.seconds, args.fps, args.hold)
    campaign = args.campaign.resolve()
    manifest = json.loads((campaign / "campaign.json").read_text())
    validate_movie_manifest(manifest)
    runs = [run for run in manifest["runs"] if run["status"] == "completed" and (args.run_id is None or args.run_id == run["id"])]
    if args.run_id and not runs:
        parser.error("The requested run is absent or not completed")
    if args.plan:
        print(json.dumps([{"run_id": run["id"], "panels": 3 if run["case"] in SHOCKS else 2 if run["case"] in COUPLED else 1,
                           "model": run["model"], "integrator": run["integrator"],
                           "model_label": MODEL_LABELS[run["model"]],
                           "radiation_flux_semantics": MODEL_FLUX[run["model"]],
                           "movie_timing": run.get("movie_timing"),
                           "movie_view": effective_movie_view(args, run.get("movie_view")),
                           "full_domain_forced": args.full_domain,
                           "physical_scope": effective_physical_scope(args, run),
                           "directory": str(campaign/run["directory"]/run["case"]/f"l{run['level']}"/MOVIE_SUBDIR)} for run in runs], indent=2))
        return 0
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "VTK_SMP_MAX_THREADS", "LP_NUM_THREADS"):
        os.environ[name] = "1"
    # Affinity is inherited by VisIt, its engine, and FFmpeg. Even helper threads
    # share this one CPU; the surrounding publisher calls us between simulations.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for run in runs:
        render_one(campaign, run, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Movie rendering failed: {error}", file=sys.stderr)
        raise SystemExit(1)
