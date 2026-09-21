"""Build the descriptor-driven Octo-TIGER verification website.

The numerical adapters retain ownership of their detailed reports.  This
module owns the site root: it inventories every registered descriptor, merges
the family result manifests, and links the detailed reports without changing
their evidence or status.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path
from typing import Any


APPLICATION_CASES = {
    "streaming_wave": "radiation.skinner_ostriker.streaming_wave",
    "streaming_front": "radiation.skinner_ostriker.streaming_front",
    "gaussian_pulse": "radiation.skinner_ostriker.gaussian_pulse",
    "equilibrium_sphere": "radiation.skinner_ostriker.equilibrium_sphere",
}
KNOWN_STATUSES = {"passed", "failed", "conditional", "running", "complete", "not_run"}
DISPLAY_STATUS = {"complete": "completed", "not_run": "not run"}
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _status(value: Any) -> str:
    return value if value in KNOWN_STATUSES else "conditional"


def _result(status: str, link: str | None = None, reason: str | None = None,
            source: str | None = None) -> dict[str, Any]:
    return {"status": _status(status), "link": link, "reason": reason, "source": source}


def _native_results(root: Path) -> dict[str, dict[str, Any]]:
    candidates = [root / "radiation" / "summary.json", root / "summary.json"]
    for path in candidates:
        value = _read_json(path)
        if not isinstance(value, list):
            continue
        prefix = "radiation/" if path.parent.name == "radiation" else ""
        answer: dict[str, dict[str, Any]] = {}
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            answer[item["id"]] = _result(
                item.get("status", "conditional"), prefix + "report.html",
                item.get("reason"), "native radiation suite")
        return answer
    return {}


def _scenario_results(root: Path, family: str) -> dict[str, dict[str, Any]]:
    value = _read_json(root / family / "verification.json")
    prefix = family + "/"
    if value is None:
        value = _read_json(root / "verification.json")
        prefix = ""
    if not isinstance(value, dict):
        return {}
    answer: dict[str, dict[str, Any]] = {}
    for item in value.get("tests", []):
        if not isinstance(item, dict) or item.get("family", family) != family:
            continue
        identifier = item.get("identifier")
        if isinstance(identifier, str) and identifier.startswith(family + "."):
            answer[identifier] = _result(
                item.get("status", "conditional"), prefix + "report.html",
                item.get("reason"), value.get("harness", {}).get("adapter", "scenario"))
    return answer


def _preserve_application_report(root: Path, requested: bool) -> bool:
    """Move the old four-case landing page aside once, leaving assets in place."""
    index = root / "index.html"
    saved = root / "radiation-application.html"
    case_pages = any((root / (name + ".html")).is_file() for name in APPLICATION_CASES)
    if not index.is_file():
        return saved.is_file()
    text = index.read_text(encoding="utf-8", errors="replace")
    if 'name="octotiger-unified-site"' in text:
        return saved.is_file()
    if not (requested or case_pages):
        return saved.is_file()
    # A resumed application run regenerates index.html. Refresh the preserved
    # subreport from that new page before restoring the unified landing page.
    saved.write_text(text, encoding="utf-8")
    return True


def _application_results(root: Path, available: bool) -> dict[str, dict[str, Any]]:
    if not available:
        return {}
    answer = {}
    locations = [(root / "radiation" / "application", "radiation/application/"),
                 (root, "")]
    for location, prefix in locations:
        for case, identifier in APPLICATION_CASES.items():
            page = location / (case + ".html")
            if page.is_file():
                answer[identifier] = _result(
                    "complete", prefix + page.name,
                    "Application run completed; use its detailed report for numerical diagnostics.",
                    "full application radiation report")
    return answer


def collect(root: Path, descriptors: dict[str, tuple[Path, dict[str, Any]]],
            legacy_application: bool = False) -> dict[str, Any]:
    root = root.resolve()
    application = (_preserve_application_report(root, legacy_application) or
                   (root / "radiation" / "application" / "index.html").is_file())
    observed: dict[str, dict[str, Any]] = {}
    observed.update(_scenario_results(root, "hydro"))
    observed.update(_scenario_results(root, "gravity"))
    observed.update(_native_results(root))
    observed.update(_application_results(root, application))

    tests = []
    for identifier, (path, descriptor) in sorted(descriptors.items()):
        result = observed.get(identifier, _result("not_run", reason="No result in this published batch."))
        tests.append({
            "identifier": identifier,
            "family": descriptor["family"],
            "suite": descriptor["suite"],
            "name": descriptor["name"],
            "regime": descriptor["regime"],
            "status": result["status"],
            "result_link": result.get("link"),
            "reason": result.get("reason"),
            "result_source": result.get("source"),
            "descriptor": path.relative_to(SOURCE_ROOT).as_posix()
                if path.is_relative_to(SOURCE_ROOT) else path.as_posix(),
            "reference_kind": descriptor.get("reference_data", {}).get("kind"),
            "reference": descriptor.get("reference_data", {}).get("description"),
            "build_type": descriptor.get("build_type"),
            "resolution_levels": descriptor.get("resolution_levels", []),
        })

    counts = {status: sum(test["status"] == status for test in tests)
              for status in ("passed", "failed", "conditional", "complete", "running", "not_run")}
    summary = _read_json(root / "summary.json")
    family_results: dict[str, dict[str, Any]] = {}
    if isinstance(summary, dict):
        for item in summary.get("families", []):
            if isinstance(item, dict) and item.get("family") in {"hydro", "gravity", "radiation"}:
                family = item["family"]
                family_results[family] = {
                    "status": _status(item.get("status", "conditional")),
                    "result_link": family + "/report.html",
                    "reason": item.get("reason"),
                }

    family_statuses = {item["status"] for item in family_results.values()}
    summary_status = _status(summary.get("status")) if isinstance(summary, dict) else None
    if counts["failed"] or "failed" in family_statuses:
        overall = "failed"
    elif counts["running"] or "running" in family_statuses or summary_status == "running":
        overall = "running"
    elif counts["conditional"] or counts["not_run"] or counts["complete"]:
        overall = "incomplete"
    else:
        overall = "passed"

    source_commit = summary.get("source_commit") if isinstance(summary, dict) else None
    source = _read_json(root / "source.json")
    if source_commit is None and isinstance(source, dict):
        source_commit = source.get("commit")
    verification = _read_json(root / "verification.json")
    if source_commit is None and isinstance(verification, dict):
        source_commit = verification.get("source", {}).get("commit")

    return {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": source_commit or "unknown",
        "status": overall,
        "counts": counts,
        "test_count": len(tests),
        "application_radiation_report": (
            "radiation/application/index.html"
            if (root / "radiation" / "application" / "index.html").is_file()
            else "radiation-application.html" if application else None),
        "families": family_results,
        "tests": tests,
    }


def _family_status(tests: list[dict[str, Any]]) -> str:
    statuses = {test["status"] for test in tests}
    if "failed" in statuses:
        return "failed"
    if "running" in statuses:
        return "running"
    if statuses == {"passed"}:
        return "passed"
    return "incomplete"


def _render(catalog: dict[str, Any]) -> str:
    esc = html.escape
    refresh = '<meta http-equiv="refresh" content="5">' if catalog["status"] == "running" else ""
    css = """
:root{color-scheme:light dark;--bg:#0b1020;--panel:#141c31;--line:#2b3857;--text:#edf2ff;--muted:#aebbd6;--pass:#3bc982;--fail:#ff6b72;--cond:#f0b95b;--run:#75b8ff;--none:#8490a8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:36px 24px 70px}h1{font-size:2.3rem;margin:.2rem 0}h2{margin-top:2.5rem}h3{margin:.15rem 0}.lede,.meta,.reason{color:var(--muted)}.overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}.stat,.family,.test{background:var(--panel);border:1px solid var(--line);border-radius:12px}.stat{padding:16px}.stat strong{display:block;font-size:1.7rem}.families{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}.family{padding:18px}.suite{margin-top:22px}.tests{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}.test{padding:16px}.badge{display:inline-block;padding:3px 9px;border-radius:99px;font-weight:700;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}.passed{color:var(--pass)}.failed{color:var(--fail)}.conditional,.complete,.incomplete{color:var(--cond)}.running{color:var(--run)}.not_run{color:var(--none)}a{color:#8fc5ff}.test a{display:inline-block;margin-top:8px}.identifier{font:12px ui-monospace,monospace;color:var(--muted);overflow-wrap:anywhere}.reference{font-size:.88rem;color:var(--muted)}footer{margin-top:38px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted)}
"""
    parts = ["<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">", refresh,
             '<meta name="octotiger-unified-site" content="1">',
             "<title>Octo-TIGER verification</title><style>", css, "</style></head><body><main>",
             "<header><div class=\"meta\">UNIFIED PHYSICS VERIFICATION</div><h1>Octo-TIGER verification</h1>",
             "<p class=\"lede\">One catalog for hydro, gravity, and grey radiation. A missing or conditional result remains visible; it is never silently omitted.</p>",
             f'<p>Overall: <span class="badge {esc(catalog["status"])}">{esc(catalog["status"])}</span> &nbsp; Source <code>{esc(catalog["source_commit"])}</code></p></header>',
             '<section class="overview">']
    for key, label in (("test_count", "registered"),):
        parts.append(f'<div class="stat"><strong>{catalog[key]}</strong>{label}</div>')
    for key, label in (("passed", "passed"), ("failed", "failed"), ("conditional", "conditional"),
                       ("complete", "application-complete"), ("not_run", "not run")):
        parts.append(f'<div class="stat"><strong class="{key}">{catalog["counts"][key]}</strong>{label}</div>')
    parts.append('</section><section><h2>Families</h2><div class="families">')
    for family in ("hydro", "gravity", "radiation"):
        tests = [test for test in catalog["tests"] if test["family"] == family]
        recorded = catalog.get("families", {}).get(family, {})
        status = recorded.get("status", _family_status(tests))
        links = sorted({test["result_link"] for test in tests if test.get("result_link")})
        if recorded.get("result_link"):
            links.append(recorded["result_link"])
            links = sorted(set(links))
        if family == "radiation" and catalog.get("application_radiation_report"):
            links.append(catalog["application_radiation_report"])
            links = sorted(set(links))
        parts.append(f'<div class="family"><h3>{esc(family.title())}</h3><span class="badge {status}">{status}</span><p>{len(tests)} registered tests</p>')
        for link in links:
            label = "Application radiation report" if link in {"radiation-application.html", "radiation/application/index.html"} else "Detailed report"
            parts.append(f'<a href="{esc(link, quote=True)}">{label}</a><br>')
        parts.append('</div>')
    parts.append('</div></section>')

    for family in ("hydro", "gravity", "radiation"):
        family_tests = [test for test in catalog["tests"] if test["family"] == family]
        parts.append(f'<section><h2>{esc(family.title())}</h2>')
        for suite in sorted({test["suite"] for test in family_tests}):
            parts.append(f'<div class="suite"><h3>{esc(suite.replace("_", " ").title())}</h3><div class="tests">')
            for test in (test for test in family_tests if test["suite"] == suite):
                status = test["status"]
                parts.append('<article class="test">')
                parts.append(f'<span class="badge {status}">{esc(DISPLAY_STATUS.get(status, status))}</span>')
                parts.append(f'<h3>{esc(test["name"].replace("_", " ").title())}</h3>')
                parts.append(f'<div class="identifier">{esc(test["identifier"])}</div><p>{esc(test["regime"])}</p>')
                if test.get("reason"):
                    parts.append(f'<p class="reason">{esc(test["reason"])}</p>')
                parts.append(f'<p class="reference">Reference: {esc(test.get("reference_kind") or "unspecified")} — {esc(test.get("reference") or "not recorded")}</p>')
                if test.get("result_link"):
                    parts.append(f'<a href="{esc(test["result_link"], quote=True)}">Open result report</a>')
                parts.append('</article>')
            parts.append('</div></div>')
        parts.append('</section>')
    parts.append(f'<footer>Generated {esc(catalog["generated_utc"])}. Machine-readable catalog: <a href="site.json">site.json</a>.</footer></main></body></html>')
    return "".join(parts)


def write(root: Path, descriptors: dict[str, tuple[Path, dict[str, Any]]],
          legacy_application: bool = False) -> dict[str, Any]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = collect(root, descriptors, legacy_application)
    payload = json.dumps(catalog, indent=2, sort_keys=True) + "\n"
    document = _render(catalog)
    (root / "site.json").write_text(payload, encoding="utf-8")
    (root / "index.html").write_text(document, encoding="utf-8")
    (root / "report.html").write_text(document, encoding="utf-8")
    return catalog
