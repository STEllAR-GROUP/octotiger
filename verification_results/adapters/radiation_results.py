"""Compatibility adapter for the existing radiation_results C++ runner."""

from __future__ import annotations

import os
from pathlib import Path


CASE_NAMES = {
    "streaming_wave": "streaming_wave",
    "streaming_front": "streaming_front",
    "gaussian_pulse": "gaussian_pulse",
    "equilibrium_sphere": "equilibrium_sphere",
}


def project_root(source_root: Path) -> Path:
    """Mirror legacy run.sh's PROJECT/src/octotiger layout, with flat-checkout support."""
    source_root = source_root.resolve()
    if source_root.name == "octotiger" and source_root.parent.name == "src":
        return source_root.parent.parent
    return source_root


def legacy_case(name: str) -> str:
    """Return the exact case spelling understood by radiation-results."""
    try:
        return CASE_NAMES[name]
    except KeyError as error:
        raise ValueError(f"radiation_results has no adapter for {name!r}") from error


def command(source_root: Path, mode: str, case: str, arguments: list[str]) -> list[str]:
    if mode not in {"run", "live"}:
        raise ValueError(f"unsupported radiation_results mode: {mode}")
    launcher = Path(
        os.environ.get(
            "OCTOTIGER_VERIFICATION_RADIATION_RUNNER",
            source_root / "verification_results" / "radiation" / "results.sh",
        )
    )
    has_root = any(value == "--root" or value.startswith("--root=") for value in arguments)
    root_arguments = [] if has_root else ["--root", str(project_root(source_root))]
    return [str(launcher), mode, case, *root_arguments, *arguments]
