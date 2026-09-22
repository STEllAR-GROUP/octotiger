"""Compatibility adapter for the existing radiation_results C++ runner."""

from __future__ import annotations

import os
from pathlib import Path


caseNames = {
    "streaming_wave": "streaming_wave",
    "streaming_front": "streaming_front",
    "gaussian_pulse": "gaussian_pulse",
    "equilibrium_sphere": "equilibrium_sphere",
}


def projectRoot(sourceRoot: Path) -> Path:
    """Mirror legacy run.sh's PROJECT/src/octotiger layout, with flat-checkout support."""
    sourceRoot = sourceRoot.resolve()
    if sourceRoot.name == "octotiger" and sourceRoot.parent.name == "src":
        return sourceRoot.parent.parent
    return sourceRoot


def legacyCase(name: str) -> str:
    """Return the exact case spelling understood by radiation-results."""
    try:
        return caseNames[name]
    except KeyError as error:
        raise ValueError(f"radiation_results has no adapter for {name!r}") from error


def command(sourceRoot: Path, mode: str, case: str, arguments: list[str]) -> list[str]:
    if mode not in {"run", "live"}:
        raise ValueError(f"unsupported radiation_results mode: {mode}")
    launcher = Path(
        os.environ.get(
            "OCTOTIGER_VERIFICATION_RADIATION_RUNNER",
            sourceRoot / "verification_results" / "radiation" / "results.sh",
        )
    )
    hasRoot = any(value == "--root" or value.startswith("--root=") for value in arguments)
    rootArguments = [] if hasRoot else ["--root", str(projectRoot(sourceRoot))]
    return [str(launcher), mode, case, *rootArguments, *arguments]
