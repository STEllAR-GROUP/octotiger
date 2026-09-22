"""Plots, Silo renderings, movies, and HTML for configured CTest scenarios."""
from __future__ import annotations

import html
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import textwrap


fieldNames = ["rho", "egas", "tau", "pot", "sx", "sy", "sz",
               "zx", "zy", "zz", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5"]


def readTables(folder: Path, name: str):
    """Read every finite numeric table with this basename from retained raw data."""
    import numpy as np

    answer = []
    for path in sorted((folder / "raw").rglob(name)):
        try:
            data = np.loadtxt(path, ndmin=2)
        except (OSError, ValueError):
            continue
        if data.size and data.ndim == 2 and data.shape[1] >= 2 and np.all(np.isfinite(data)):
            answer.append((path, data))
    return answer


def savePlot(path: Path, figure):
    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=125)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def diagnosticPlots(folder: Path):
    """Create self-describing plots from the legacy numeric diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    products = []
    profiles = readTables(folder, "line.final.dat")
    if not profiles:
        numbered = sorted((folder / "raw").rglob("line.*.dat"))
        profiles = readTables(folder, numbered[-1].name) if numbered else []
    for number, (source, data) in enumerate(profiles):
        count = min(data.shape[1] - 1, 6)
        if count < 1:
            continue
        fig, axes = plt.subplots(count, 1, figsize=(9, max(3.2, 2.1 * count)), sharex=True)
        axes = np.atleast_1d(axes)
        for column, axis in enumerate(axes, 1):
            label = fieldNames[column - 1] if column - 1 < len(fieldNames) else f"field {column}"
            axis.plot(data[:, 0], data[:, column], linewidth=1.25)
            axis.set_ylabel(label)
            axis.grid(alpha=.22)
        axes[-1].set_xlabel("x")
        fig.suptitle("Final center-line profile")
        target = folder / "plots" / ("line-profile.png" if number == 0 else f"line-profile-{number}.png")
        savePlot(target, fig)
        plt.close(fig)
        products.append({"kind": "plot", "title": "Final center-line profile",
                         "path": target.relative_to(folder).as_posix(),
                         "source": source.relative_to(folder).as_posix()})

    norms = {name: readTables(folder, name) for name in ("L1.dat", "L2.dat", "Linf.dat")}
    if any(norms.values()):
        labels, series = [], []
        for name, tables in norms.items():
            if not tables:
                continue
            data = tables[0][1][-1]
            values = np.abs(data[2:])
            if values.size:
                labels.append(name.removesuffix(".dat"))
                series.append(values)
        if series:
            width = .8 / len(series)
            count = max(len(row) for row in series)
            x = np.arange(count)
            fig, axis = plt.subplots(figsize=(max(8, count * .7), 4.8))
            for offset, (label, values) in enumerate(zip(labels, series)):
                axis.bar(x[:len(values)] + (offset - (len(series) - 1) / 2) * width,
                         np.maximum(values, np.finfo(float).tiny), width, label=label)
            names = [fieldNames[index] if index < len(fieldNames) else f"field {index + 1}"
                     for index in range(count)]
            axis.set_xticks(x, names, rotation=35, ha="right")
            axis.set_yscale("log")
            axis.set_ylabel("error norm")
            axis.set_title("Final analytic error norms")
            axis.legend()
            axis.grid(axis="y", alpha=.22)
            target = folder / "plots" / "error-norms.png"
            savePlot(target, fig)
            plt.close(fig)
            products.append({"kind": "plot", "title": "Final analytic error norms",
                             "path": target.relative_to(folder).as_posix()})

    for filename, title, columns in (
            ("temp.dat", "Matter/radiation temperature history", [(1, "gas"), (2, "radiation")]),
            ("step.dat", "Timestep and AMR history", [(2, "dt"), (14, "grids"), (15, "leaves"), (16, "AMR boundaries")]),
            ("sums.dat", "Conserved-sum history", [(1, "sum 1"), (3, "sum 2"), (5, "sum 3")])):
        for number, (source, data) in enumerate(readTables(folder, filename)):
            available = [(index, label) for index, label in columns if index < data.shape[1]]
            if not available or data.shape[0] < 2:
                continue
            fig, axes = plt.subplots(len(available), 1,
                                     figsize=(9, max(3.2, 2.15 * len(available))), sharex=True)
            axes = np.atleast_1d(axes)
            x = data[:, 1] if filename == "step.dat" and data.shape[1] > 1 else data[:, 0]
            for axis, (column, label) in zip(axes, available):
                axis.plot(x, data[:, column], linewidth=1.2)
                axis.set_ylabel(label)
                axis.grid(alpha=.22)
            axes[-1].set_xlabel("time")
            fig.suptitle(title)
            stem = filename.removesuffix(".dat") + ("" if number == 0 else f"-{number}")
            target = folder / "plots" / (stem + ".png")
            savePlot(target, fig)
            plt.close(fig)
            products.append({"kind": "plot", "title": title,
                             "path": target.relative_to(folder).as_posix(),
                             "source": source.relative_to(folder).as_posix()})
    return products


def siloKey(path: Path):
    match = re.fullmatch(r"X\.(\d+)\.silo", path.name)
    if match:
        return (0, int(match.group(1)))
    if path.name == "final.silo":
        return (1, 0)
    return (2, path.name)


def silos(folder: Path):
    values = [path for path in (folder / "raw").rglob("*.silo")
              if ".silo.data" not in path.as_posix() and path.name != "analytic.silo"]
    # A legacy variant should have one root file for each state.  Rejecting
    # duplicate basenames avoids silently splicing unrelated time sequences.
    byName = {}
    for path in values:
        if path.name in byName:
            raise ValueError(f"Ambiguous retained Silo state {path.name}")
        byName[path.name] = path
    return sorted(byName.values(), key=siloKey)


def visitExecutable(value: Path):
    path = value.expanduser().resolve()
    if path.is_dir():
        for candidate in (path / "bin" / "visit", path / "visit", path / "bin" / "frontendlauncher"):
            if candidate.is_file():
                return candidate
    if path.name == "frontendlauncher" and (path.parent / "visit").is_file():
        return path.parent / "visit"
    return path


def visitScript(databases, requestedFields, output: Path):
    """Return a VisIt CLI program which discovers actual scalar names safely."""
    config = {"databases": [str(path) for path in databases],
              "fields": requestedFields, "output": str(output)}
    return textwrap.dedent("""
        import json, os
        config = %s
        products = []

        def scalar_names(metadata):
            answer = []
            for index in range(metadata.GetNumScalars()):
                answer.append(metadata.GetScalars(index).name)
            return answer

        def select(names, requested):
            for candidate in requested:
                for name in names:
                    if name == candidate or name.endswith('/' + candidate):
                        return name
            return None

        for frame, database in enumerate(config['databases']):
            if not OpenDatabase(database):
                raise RuntimeError('VisIt could not open ' + database)
            metadata = GetMetaData(database)
            available = scalar_names(metadata)
            for field_index, requested in enumerate(config['fields']):
                variable = select(available, [requested])
                if variable is None:
                    continue
                DeleteAllPlots()
                if AddPlot('Pseudocolor', variable) == 0:
                    raise RuntimeError('VisIt could not plot ' + variable)
                AddOperator('Slice')
                attributes = SliceAttributes()
                attributes.axisType = attributes.ZAxis
                attributes.originType = attributes.Intercept
                attributes.originIntercept = 0
                attributes.project2d = 1
                SetOperatorOptions(attributes)
                DrawPlots()
                target_dir = os.path.join(config['output'], requested)
                if not os.path.isdir(target_dir):
                    os.makedirs(target_dir)
                target = os.path.join(target_dir, 'frame-%%04d' %% frame)
                save = SaveWindowAttributes()
                save.outputToCurrentDirectory = 0
                save.outputDirectory = target_dir
                save.fileName = os.path.basename(target)
                save.family = 0
                save.format = save.PNG
                save.width = 960
                save.height = 720
                SetSaveWindowAttributes(save)
                saved = SaveWindow()
                products.append({'requested': requested, 'variable': variable,
                                 'frame': frame, 'database': database,
                                 'path': target + '.png'})
            DeleteAllPlots()
            CloseDatabase(database)
        with open(os.path.join(config['output'], 'visit-products.json'), 'w') as stream:
            json.dump(products, stream, indent=2)
        exit()
        """) % repr(config)


def siloProducts(folder: Path, descriptor, visit: Path, ffmpeg: str):
    databases = silos(folder)
    if not databases:
        raise ValueError("No retained root Silo states are available for rendering")
    family = descriptor["family"]
    requested = ["pot", "rho"] if family == "gravity" else ["rho", "egas"]
    render = folder / "rendered"
    render.mkdir()
    script = render / "render.py"
    script.write_text(visitScript(databases, requested, render))
    executable = visitExecutable(visit)
    if not executable.is_file():
        raise ValueError(f"VisIt executable unavailable: {executable}")
    command = [str(executable), "-cli", "-nowin", "-s", str(script)]
    completed = subprocess.run(command, cwd=folder, text=True, capture_output=True)
    (render / "visit.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"VisIt rendering failed with exit code {completed.returncode}; see rendered/visit.log")
    manifestPath = render / "visit-products.json"
    if not manifestPath.is_file():
        raise RuntimeError("VisIt did not publish rendered/visit-products.json")
    rendered = json.loads(manifestPath.read_text())
    products = []
    byRequested = {field: [] for field in requested}
    for item in rendered:
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"VisIt did not create {path}")
        byRequested[item["requested"]].append(path)
    for field, frames in byRequested.items():
        if not frames:
            continue
        frames.sort()
        still = folder / "plots" / f"field-{field}.png"
        still.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frames[-1], still)
        products.append({"kind": "image", "title": f"Final {field} field",
                         "path": still.relative_to(folder).as_posix(),
                         "frames": len(frames)})

    if "movie" in descriptor.get("visualization_products", []):
        primary = byRequested.get("rho", [])
        if len(primary) < 2:
            raise ValueError("A movie was requested but fewer than two rendered Silo states exist")
        movie = folder / "plots" / "field-rho.mp4"
        pattern = render / "rho" / "frame-%04d.png"
        process = subprocess.run([ffmpeg, "-y", "-framerate", "4", "-i", str(pattern),
                                  "-c:v", "libx264", "-pix_fmt", "yuv420p", str(movie)],
                                 text=True, capture_output=True)
        (render / "ffmpeg.log").write_text(process.stdout + process.stderr)
        if process.returncode or not movie.is_file() or movie.stat().st_size == 0:
            raise RuntimeError("Movie encoding failed; see rendered/ffmpeg.log")
        # Decode the entire stream, rather than accepting a nonempty container.
        check = subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(movie),
                                "-map", "0:v:0", "-f", "null", "-"],
                               text=True, capture_output=True)
        if check.returncode:
            raise RuntimeError("Movie validation failed: " + check.stderr.strip())
        products.append({"kind": "video", "title": "Density evolution",
                         "path": movie.relative_to(folder).as_posix(),
                         "frames": len(primary)})
    return products


def visualize(folder: Path, descriptor, visit: Path | None, ffmpeg: str):
    record = {"status": "passed", "products": [], "requested": descriptor.get("visualization_products", [])}
    try:
        record["products"].extend(diagnosticPlots(folder))
    except Exception as error:  # A plot failure is retained and visible, never hidden.
        record.update(status="failed", reason="Diagnostic plotting failed: " + str(error))
        return record
    wantsFields = any(name in record["requested"] for name in
                       ("field_stills", "field_maps", "radial_profiles", "movie"))
    if not wantsFields:
        return record
    if visit is None:
        record.update(status="not_requested",
                      reason="Silo rendering was not requested; pass --visit to render fields and applicable movies.")
        return record
    try:
        record["products"].extend(siloProducts(folder, descriptor, visit, ffmpeg))
    except Exception as error:
        record.update(status="failed", reason=str(error))
    return record


def resolveStatus(values):
    values = set(values)
    if "failed" in values:
        return "failed"
    if values & {"running", "planned"}:
        return "running"
    if "conditional" in values:
        return "conditional"
    return "passed" if values else "running"


def writeReport(output: Path, manifest):
    """Publish the machine manifest and a live, human-readable family report."""
    (output / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    status = resolveStatus(item.get("status", "planned") for item in manifest["tests"])
    refresh = '<meta http-equiv="refresh" content="5">' if status == "running" else ""
    esc = html.escape
    css = """
:root{color-scheme:light dark;--bg:#0b1020;--panel:#141c31;--line:#2b3857;--text:#edf2ff;--muted:#aebbd6;--pass:#3bc982;--fail:#ff6b72;--cond:#f0b95b;--run:#75b8ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px 22px 70px}article,.group{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0}.group{background:#10172a}h1{font-size:2.2rem}h2{margin-bottom:.25rem}.muted,.reason{color:var(--muted)}.badge{font-weight:800;text-transform:uppercase;font-size:.78rem;letter-spacing:.04em}.passed{color:var(--pass)}.failed{color:var(--fail)}.conditional,.not_requested{color:var(--cond)}.running,.planned{color:var(--run)}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.product{border:1px solid var(--line);border-radius:9px;padding:10px}.product img,.product video{width:100%;max-height:560px;object-fit:contain;background:#050812}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid var(--line);padding:6px}a{color:#8fc5ff}code{overflow-wrap:anywhere}details{margin-top:12px}
"""
    parts = ["<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
             refresh, "<title>Configured Octo-TIGER verification</title><style>", css,
             "</style></head><body><main><h1>Configured Hydro/Gravity verification</h1>",
             '<p class="muted">Live CTest results, solver diagnostics, field renderings, movies where physically applicable, logs, and retained raw evidence.</p>',
             f'<p>Overall: <span class="badge {status}">{status}</span> · application HPX threads: {manifest["execution"]["application_threads"]}</p>']
    quarantined = manifest.get("preexisting_build_output", [])
    if quarantined:
        parts.append(f'<p class="reason">Preserved {len(quarantined)} products from an earlier interrupted run under '
                     '<a href="preexisting-build-output/manifest.json">preexisting-build-output</a> before starting CTest.</p>')
    for item in manifest["tests"]:
        itemStatus = item.get("status", "planned")
        descriptor = item["descriptor"]
        parts += [f'<article id="{esc(item["identifier"], quote=True)}"><span class="badge {esc(itemStatus)}">{esc(itemStatus)}</span>',
                  f'<h2>{esc(item["identifier"])}</h2><p>{esc(descriptor["regime"])}</p>']
        if item.get("reason"):
            parts.append(f'<p class="reason">{esc(item["reason"])}</p>')
        requested = descriptor.get("visualization_products", [])
        parts.append('<p class="muted">Requested products: ' + esc(", ".join(requested) or "none") + "</p>")
        for group in item.get("groups", []):
            groupStatus = group.get("status", "planned")
            parts += [f'<section class="group"><span class="badge {esc(groupStatus)}">{esc(groupStatus)}</span>',
                      f'<h3>{esc(group["name"])}</h3>']
            if group.get("reason"):
                parts.append(f'<p class="reason">{esc(group["reason"])}</p>')
            execution = group.get("execution", {})
            checks = execution.get("checks", [])
            if checks:
                parts.append("<table><thead><tr><th>CTest check</th><th>Status</th><th>Seconds</th></tr></thead><tbody>")
                for check in checks:
                    parts.append(f'<tr><td>{esc(check["name"])}</td><td class="{esc(check["status"])}">{esc(check["status"])}</td><td>{esc(str(check.get("seconds") or ""))}</td></tr>')
                parts.append("</tbody></table>")
            visual = group.get("visualization")
            if visual:
                parts.append(f'<p>Visual products: <span class="badge {esc(visual["status"])}">{esc(visual["status"])}</span></p>')
                if visual.get("reason"):
                    parts.append(f'<p class="reason">{esc(visual["reason"])}</p>')
                parts.append('<div class="gallery">')
                base = Path(item["identifier"]) / group["name"]
                for product in visual.get("products", []):
                    url = (base / product["path"]).as_posix()
                    parts.append(f'<div class="product"><h4>{esc(product["title"])}</h4>')
                    if product["kind"] == "video":
                        parts.append(f'<video controls preload="metadata" src="{esc(url, quote=True)}"></video>')
                    else:
                        parts.append(f'<a href="{esc(url, quote=True)}"><img loading="lazy" src="{esc(url, quote=True)}" alt="{esc(product["title"], quote=True)}"></a>')
                    parts.append("</div>")
                parts.append("</div>")
            location = Path(item["identifier"]) / group["name"]
            links = []
            for filename in ("checks.log", "checks-LastTest.log", "cleanup.log", "run.json"):
                if (output / location / filename).is_file():
                    links.append(f'<a href="{esc((location / filename).as_posix(), quote=True)}">{esc(filename)}</a>')
            if links:
                parts.append("<p>Logs and metadata: " + " · ".join(links) + "</p>")
            artifacts = group.get("artifacts", [])
            if artifacts:
                parts.append(f'<details><summary>{len(artifacts)} retained raw artifacts</summary><ul>')
                for artifact in artifacts:
                    url = (location / artifact["path"]).as_posix()
                    parts.append(f'<li><a href="{esc(url, quote=True)}">{esc(artifact["path"])}</a> ({artifact["bytes"]} bytes)</li>')
                parts.append("</ul></details>")
            parts.append("</section>")
        parts.append("</article>")
    parts.append('<p><a href="verification.json">Machine-readable verification manifest</a></p></main></body></html>')
    document = "".join(parts)
    for name in ("index.html", "report.html"):
        (output / name).write_text(document)
