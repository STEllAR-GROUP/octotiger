#!/usr/bin/env python3
"""Thin common runner over verification-family adapters.

Step 02 deliberately delegates numerical work to radiation_results.  It adds a
stable descriptor/metadata boundary without changing the legacy entry points.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verification_results.adapters import radiation_results
from verification_results.adapters import scenario


SCHEMA_VERSION = 1
SOURCE_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = SOURCE_ROOT / "verification_results"
TEST_ROOT = HARNESS_ROOT / "tests"
RESULT_ROOT = HARNESS_ROOT / "results"
SELECTOR_PREFIX = "radiation.skinner_ostriker."


class HarnessError(RuntimeError):
    pass


def descriptors() -> dict[str, tuple[Path, dict[str, Any]]]:
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(TEST_ROOT.glob("**/test.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_descriptor(value, path)
        identifier = f"{value['family']}.{value['suite']}.{value['name']}"
        if identifier in found:
            raise HarnessError(f"duplicate test identifier {identifier}")
        found[identifier] = (path, value)
    return found


def validate_descriptor(value: dict[str, Any], path: Path | str = "descriptor") -> None:
    required = {
        "schema_version", "family", "suite", "name", "regime", "dimensionality",
        "resolution_levels", "build_type", "required_executable", "parameters",
        "expected_diagnostics", "reference_data", "tolerance_policy",
        "visualization_products", "adapter",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise HarnessError(f"{path}: missing descriptor fields: {', '.join(missing)}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise HarnessError(f"{path}: unsupported schema_version")
    if value["family"] not in {"hydro", "gravity", "radiation"}:
        raise HarnessError(f"{path}: unknown family")
    if value["build_type"] not in ("Debug", "Release", "RelWithDebInfo"):
        raise HarnessError(f"{path}: build_type must be one canonical build mode")
    for field in ("reference_data", "tolerance_policy"):
        if not {"owner", "description"} <= value[field].keys():
            raise HarnessError(f"{path}: {field} requires owner and description")
    if not isinstance(value["dimensionality"], int) or value["dimensionality"] not in {1, 2, 3}:
        raise HarnessError(f"{path}: dimensionality must be 1, 2, or 3")
    levels = value["resolution_levels"]
    if not isinstance(levels, list) or not levels or any(not isinstance(v, int) or v < 0 for v in levels):
        raise HarnessError(f"{path}: resolution_levels must be nonnegative integers")
    if value["adapter"].get("name") == "native_suite":
        p = value["parameters"]
        supported={"streaming_wave_1d", "streaming_wave_pc", "streaming_front_1d", "thin_gaussian", "damped_wave",
                   "static_diffusion", "diffusion_thicker", "thermal_relaxation", "moving_scattering", "boundary_subcycles"}
        if value["name"] not in supported or value["adapter"].get("case") != value["name"]:
            raise HarnessError(f"{path}: unsupported native test case")
        fields = {"units", "length_cm", "domain_cm", "final_time_s", "c_cm_s", "chi_cm_inverse", "amplitude",
                  "width_cm", "velocity_cm_s", "reduced_light_speed_ratio", "rho_g_cm3", "gas_internal_erg_cm3",
                  "radiation_energy_erg_cm3", "frames", "boundary", "reconstruction", "opacity", "output_cadence_s", "hard_dt"}
        if fields - p.keys():
            raise HarnessError(f"{path}: missing physical descriptor fields {sorted(fields-p.keys())}")
        if p["units"] != "CGS" or p["c_cm_s"] != 2.99792458e10 or p["boundary"] != "periodic":
            raise HarnessError(f"{path}: native suite requires physical c in CGS and periodic boundaries")
        length = p["length_cm"]
        if p["domain_cm"] != [[-length/2, length/2]]*3 or length <= 0 or p["final_time_s"] <= 0:
            raise HarnessError(f"{path}: unsupported domain or final time")
        if p["frames"] < 2 or abs(p["output_cadence_s"]*(p["frames"]-1)-p["final_time_s"]) > 1e-12*p["final_time_s"]:
            raise HarnessError(f"{path}: inconsistent output cadence")
        for key in ("length_cm", "final_time_s", "chi_cm_inverse", "amplitude", "width_cm", "velocity_cm_s", "rho_g_cm3", "gas_internal_erg_cm3", "radiation_energy_erg_cm3"):
            if not isinstance(p[key], (int,float)) or not math.isfinite(p[key]):
                raise HarnessError(f"{path}: nonfinite physical parameter {key}")
        if p["chi_cm_inverse"] < 0 or p["rho_g_cm3"] <= 0 or not 0 < p["reduced_light_speed_ratio"] <= 1:
            raise HarnessError(f"{path}: invalid opacity, density, or reduced light speed")
        chi=p["chi_cm_inverse"];name=value["name"]
        absorption=chi if name in {"thermal_relaxation", "damped_wave"} else 0
        scattering=chi if name in {"static_diffusion", "diffusion_thicker", "moving_scattering", "boundary_subcycles"} else 0
        if p["opacity"] != {"model":"grey", "units":"1/cm", "absorption":absorption, "scattering":scattering, "transport_absorption":-1}:
            raise HarnessError(f"{path}: opacity declaration differs from implemented source roles")
        if p["reconstruction"] not in {"production_PLM", "piecewise_constant"}:
            raise HarnessError(f"{path}: unknown reconstruction diagnostic")
    if value["reference_data"].get("kind") not in {
        "analytic", "published", "regression", "qualitative"
    }:
        raise HarnessError(f"{path}: invalid reference kind")
    if value["tolerance_policy"].get("mode") not in {
        "absolute_relative", "convergence", "regression", "qualitative"
    }:
        raise HarnessError(f"{path}: invalid tolerance mode")


def resolve(selector: str) -> tuple[str, Path | None, dict[str, Any] | None]:
    all_descriptors = descriptors()
    if selector in {"all", "radiation", "radiation.skinner_ostriker"}:
        return "all", None, None
    aliases = {
        "wave": "streaming_wave", "front": "streaming_front", "streaming": "streaming_front",
        "diffusion": "gaussian_pulse", "gaussian": "gaussian_pulse", "sphere": "equilibrium_sphere",
    }
    selector = aliases.get(selector, selector)
    if "." not in selector:
        selector = SELECTOR_PREFIX + selector
    try:
        path, descriptor = all_descriptors[selector]
    except KeyError as error:
        choices = ", ".join(sorted(all_descriptors))
        raise HarnessError(f"unknown test or family {selector!r}; choose one of: {choices}") from error
    if descriptor["adapter"]["name"] != "radiation_results":
        raise HarnessError(f"adapter {descriptor['adapter']['name']!r} is not implemented in Step 02")
    return radiation_results.legacy_case(descriptor["adapter"]["case"]), path, descriptor


def option_value(arguments: list[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 == len(arguments):
                raise HarnessError(f"missing value for {name}")
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def has_option(arguments: list[str], name: str) -> bool:
    return any(argument == name or argument.startswith(name + "=") for argument in arguments)


def default_output(mode: str) -> Path:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    prefix = "live-" if mode == "live" else "run-"
    return RESULT_ROOT / f"{prefix}{now}"


def safe_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    # Results may live below SOURCE_ROOT only in the explicitly ignored results directory.
    allowed = RESULT_ROOT.resolve()
    if resolved == allowed or allowed in resolved.parents:
        return resolved
    source = SOURCE_ROOT.resolve()
    if resolved == source or source in resolved.parents:
        raise HarnessError(f"generated output may not be written under the source tree: {source}")
    return resolved


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=SOURCE_ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    version = HARNESS_ROOT / "SOURCE_VERSION"
    if arguments == ("rev-parse", "HEAD") and version.exists():
        exported = version.read_text().strip()
        if re.fullmatch(r"[0-9a-f]{40}", exported):
            return exported
    return "unknown"


def positional_settings(
    arguments: list[str], descriptor: dict[str, Any] | None, mode: str
) -> tuple[list[int], str]:
    values: list[str] = []
    skip = False
    options_with_values = {
        "--root", "--build", "--exe", "--generator", "--output", "--resume", "--reference",
        "--threads", "--jobs", "--snapshots", "--fps", "--width", "--height", "--cells", "--time",
        "--seconds", "--hold", "--position", "--odt", "--hard-dt", "--minimum", "--maximum",
        "--gnuplot", "--ffmpeg", "--cxx", "--field", "--view", "--axis", "--color",
        "--rad-reconstruction", "--visit", "--session-dir",
    }
    for argument in arguments:
        if skip:
            skip = False
            continue
        if argument in options_with_values:
            skip = True
        elif not argument.startswith("--"):
            values.append(argument)
    levels = [int(value) for value in values if re.fullmatch(r"[0-9]+", value)]
    build = option_value(arguments, "--build")
    nonnumeric = [value for value in values if not re.fullmatch(r"[0-9]+", value)]
    if build is None and nonnumeric:
        build = nonnumeric[-1]
    if not levels:
        levels = [2, 3, 4] if mode == "live" else [2, 3]
    if build is None:
        build = "Debug" if mode == "live" else "Release"
    return levels, build


def compiler_metadata(build: str, project_root: Path) -> dict[str, str]:
    directory = Path(build)
    if not directory.is_absolute():
        directory = project_root / "build" / "octotiger" / build.lower()
    cache = directory / "CMakeCache.txt"
    result = {"build_type": build if build.lower() in {"debug", "release", "relwithdebinfo"} else "unknown", "build_directory": str(directory), "compiler": "unknown"}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
                result["compiler"] = line.split("=", 1)[1]
            elif line.startswith("CMAKE_CXX_COMPILER_VERSION:STRING="):
                result["compiler_version"] = line.split("=", 1)[1]
            elif line.startswith("CMAKE_BUILD_TYPE:STRING="):
                result["build_type"] = line.split("=", 1)[1] or "unknown"
    return result


def metadata(
    mode: str, selector: str, path: Path | None, descriptor: dict[str, Any] | None,
    arguments: list[str], output: Path, legacy_command: list[str],
) -> dict[str, Any]:
    levels, build = positional_settings(arguments, descriptor, mode)
    commit = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))
    descriptor_hash = None
    if path:
        descriptor_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    root_value = option_value(legacy_command, "--root")
    project_root = Path(root_value).resolve() if root_value else SOURCE_ROOT
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"commit": commit, "dirty": dirty},
        "harness": {"name": "verification_results", "adapter": "radiation_results"},
        "test": {
            "selector": selector,
            "family": descriptor["family"] if descriptor else "radiation",
            "suite": descriptor["suite"] if descriptor else "skinner_ostriker",
            "name": descriptor["name"] if descriptor else "all",
            "regime": descriptor["regime"] if descriptor else "mixed",
            "dimensionality": descriptor["dimensionality"] if descriptor else 3,
            "descriptor": str(path.relative_to(SOURCE_ROOT)) if path else None,
            "descriptor_sha256": descriptor_hash,
            "parameters": descriptor["parameters"] if descriptor else {},
        },
        "build": compiler_metadata(build, project_root),
        "execution": {
            "mode": mode,
            "required_executable": descriptor["required_executable"] if descriptor else "octotiger",
            "resolution_levels": levels,
            "thread_count": int(option_value(arguments, "--threads") or 12),
            "timestep_controls": {
                "time": option_value(arguments, "--time") or "adapter-default",
                "odt": option_value(arguments, "--odt") or "adapter-default",
                "hard_dt": option_value(arguments, "--hard-dt") or "adapter-default",
                "snapshots": option_value(arguments, "--snapshots") or "adapter-default",
            },
            "arguments": arguments,
            "legacy_command": legacy_command,
        },
        "policy": {
            "expected_diagnostics": descriptor["expected_diagnostics"] if descriptor else [],
            "reference_data": descriptor["reference_data"] if descriptor else {"kind": "mixed"},
            "tolerance": descriptor["tolerance_policy"] if descriptor else {"mode": "mixed"},
            "visualization_products": descriptor["visualization_products"] if descriptor else [],
        },
        "artifacts": {
            "root": str(output), "numerical_data": ".", "plots": "plots", "movies": "movies",
            "web": ".", "logs": ".",
        },
        "legacy": {"batch_metadata": "batch.json", "run_metadata": "<case>/l<level>/run.json"},
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Unified Octo-TIGER verification harness")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list available test descriptors")
    site = commands.add_parser("site", help="regenerate a unified website from an existing result directory")
    site.add_argument("results", type=Path)
    for name in ("plan", "run", "live", "suite"):
        item = commands.add_parser(name, help=f"{name} a test or family through its adapter")
        item.add_argument("selector")
        item.add_argument("arguments", nargs=argparse.REMAINDER, help="adapter options and resolution levels")
    return result


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    available = descriptors()
    if options.command == "list":
        for identifier, (_, descriptor) in available.items():
            print(f"{identifier}\t{descriptor['regime']}")
        return 0
    if options.command == "site":
        from verification_results.web import site
        output = safe_output(options.results)
        if not output.is_dir():
            raise HarnessError(f"result directory does not exist: {output}")
        catalog = site.write(output, available)
        print(f"{output / 'index.html'} ({catalog['status']})")
        # Publishing a failed result is successful site generation. The suite
        # command, not this renderer, owns the physics exit status.
        return 0
    # Suite is additive: historical all/radiation run commands retain their behavior.
    exact = available.get(options.selector)
    if options.command == "suite" or (options.command == "plan" and options.selector == "all") or options.selector in {"hydro", "gravity", "radiation.diagnostics", "radiation.ensman"} or (exact and exact[1]["adapter"]["name"] in {"native_suite", "conditional", "octotiger_scenario"}):
        selected = [(key, value) for key, (_, value) in available.items()
                    if options.selector in {"all", value["family"], value["family"]+"."+value["suite"], key}]
        if not selected:
            raise HarnessError("No suite descriptors match " + options.selector)
        scenario_selected = [(available[key][0], value) for key, value in selected if value["adapter"]["name"] == "octotiger_scenario"]
        native_selected = [(key, value) for key, value in selected if value["adapter"]["name"] != "octotiger_scenario"]
        if scenario_selected and native_selected:
            from verification_results.adapters import unified
            return unified.execute({key: available[key] for key, _ in selected}, options.arguments, plan=options.command == "plan")
        if scenario_selected:
            rc = scenario.execute(scenario_selected, options.arguments, plan=options.command == "plan")
            if options.selector != "all" or not native_selected or options.command == "plan":
                return rc
        if native_selected:
            from verification_results.adapters import native_suite
            return native_suite.execute(native_selected, options.arguments, plan=options.command == "plan")
        return 0
    legacy_case, path, descriptor = resolve(options.selector)
    mode = "live" if options.command == "live" else "run"
    adapter_arguments = list(options.arguments)
    supplied_output = option_value(adapter_arguments, "--output")
    resume = option_value(adapter_arguments, "--resume")
    output = safe_output(Path(supplied_output or resume) if supplied_output or resume else default_output(mode))
    if not supplied_output and not resume:
        adapter_arguments.extend(["--output", str(output)])
    legacy = radiation_results.command(SOURCE_ROOT, mode, legacy_case, adapter_arguments)
    manifest = metadata(mode, options.selector, path, descriptor, adapter_arguments, output, legacy)
    if options.command == "plan":
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    result = subprocess.run(legacy, cwd=SOURCE_ROOT, check=False)
    if result.returncode == 0 and not has_option(adapter_arguments, "--dry-run"):
        output.mkdir(parents=True, exist_ok=True)
        manifest["status"] = "complete"
        (output / "verification.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        from verification_results.web import site
        site.write(output, available, legacy_application=True)
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HarnessError, ValueError, json.JSONDecodeError) as error:
        print(f"verification-results: {error}", file=sys.stderr)
        raise SystemExit(2)
