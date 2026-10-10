#!/usr/bin/env python3
"""Exercise real CLI/config reduced-c bounds, defaults, and checkpoint metadata."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile

from test_restart import CGS_C, Silo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--silo-library")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    folder = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="radiation-reduced-c-options-"))
    folder.mkdir(parents=True, exist_ok=True)
    # Deliberately rejected options may reach the application's fatal handler.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    print(f"Reduced-c option artifacts: {folder}", flush=True)
    base = {
        "problem": "RADIATION_STREAMING_WAVE", "radiation": "on", "hydro": "off",
        "gravity": "off", "periodic": "on", "rad_implicit": "off", "omega": 0,
        "unigrid": "on", "min_level": 1, "max_level": 1, "xscale": 1,
        "n_species": 5, "cfl": 0.25, "hard_dt": 0.001, "stop_time": 0.002,
        "stop_step": 10, "odt": 1, "disable_output": "off", "disable_diagnostics": "on",
        "disable_analytic": "on", "hydro_host_kernel_type": "LEGACY",
        "hydro_device_kernel_type": "OFF", "monopole_device_kernel_type": "OFF",
        "multipole_device_kernel_type": "OFF", "silo_num_groups": 1,
        "code_to_cm": CGS_C, "code_to_g": 1, "code_to_s": 1,
    }

    def invoke(name, factor=None, config_value=False):
        output = folder / name
        output.mkdir()
        config = dict(base, datadir=str(output) + os.sep)
        if config_value and factor is not None:
            config["radiation_reduced_c_factor"] = factor
        ini = output / "run.ini"
        ini.write_text("".join(f"{key}={value}\n" for key, value in config.items()))
        command = [str(executable), "--config_file=" + str(ini), "--hpx:threads=1"]
        if factor is not None and not config_value:
            command.append("--radiation_reduced_c_factor=" + str(factor))
        with (output / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        return output, result.returncode

    help_output = subprocess.run([str(executable), "--help", "--hpx:threads=1"],
                                 text=True, capture_output=True, timeout=60)
    if "radiation_reduced_c_factor" not in help_output.stdout + help_output.stderr:
        raise AssertionError("The reduced-c option is missing from --help")
    failures = []
    for source in ("cli", "config"):
        for index, value in enumerate(("0", "-0.1", "1.01", "nan", "inf", "-inf")):
            output, status = invoke(f"invalid-{source}-{index}", value, source == "config")
            log = (output / "run.log").read_text(errors="replace")
            if status == 0 or "radiation_reduced_c_factor" not in log or (output / "final.silo").exists():
                raise AssertionError(f"Invalid {source} value {value} was not rejected before evolution: {log[-3000:]}")
            failures.append({"source": source, "value": value, "returncode": status})

    silo = Silo(args.silo_library)
    outputs = {}
    metadata_names = ("radiation_options_version", "radiation_reduced_c_factor", "code_to_s", "code_to_cm")
    for name, factor, config_value in (("default", None, False), ("full", "1", False), ("reduced", "0.1", True)):
        output, status = invoke(name, factor, config_value)
        if status:
            raise AssertionError(f"Valid option failed in {output}: {(output / 'run.log').read_text()[-4000:]}")
        metadata, fields = silo.snapshot(output / "final.silo", metadata_names=metadata_names)
        expected = 1.0 if factor is None else float(factor)
        if metadata["radiation_options_version"] != 2 or metadata["radiation_reduced_c_factor"] != expected:
            raise AssertionError(f"Checkpoint did not preserve the factor: {metadata}")
        outputs[name] = (metadata, fields)
    if outputs["default"] != outputs["full"]:
        raise AssertionError("Default behavior differs from explicit factor=1")
    if outputs["full"][1] == outputs["reduced"][1]:
        raise AssertionError("The opt-in reduced factor did not change transport")
    report = {"rejected_values": failures, "default_equals_explicit_full_c": True,
              "accepted_reduced_factor": outputs["reduced"][0]["radiation_reduced_c_factor"],
              "checkpoint_metadata_version": 2}
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
