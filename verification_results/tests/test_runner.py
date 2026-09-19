from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from verification_results import runner
from verification_results.adapters import radiation_results


class DescriptorTests(unittest.TestCase):
    def test_all_radiation_descriptors_validate(self):
        found = runner.descriptors()
        self.assertEqual(
            set(found),
            {
                "radiation.skinner_ostriker.streaming_wave",
                "radiation.skinner_ostriker.streaming_front",
                "radiation.skinner_ostriker.gaussian_pulse",
                "radiation.skinner_ostriker.equilibrium_sphere",
            },
        )
        for _, descriptor in found.values():
            self.assertEqual(descriptor["family"], "radiation")

    def test_adapter_preserves_legacy_case_and_arguments(self):
        command = radiation_results.command(
            runner.SOURCE_ROOT, "live", "streaming_wave", ["2", "3", "Release", "--threads", "7"]
        )
        self.assertEqual(command[1:3], ["live", "streaming_wave"])
        self.assertEqual(command[3:5], ["--root", str(runner.SOURCE_ROOT)])
        self.assertEqual(command[-3:], ["Release", "--threads", "7"])
        self.assertTrue(command[0].endswith("verification_results/radiation/results.sh"))

    def test_adapter_project_root_matches_legacy_layouts(self):
        flat = Path("/work/checkout")
        nested = Path("/work/project/src/octotiger")
        flat_command = radiation_results.command(flat, "run", "streaming_wave", [])
        nested_command = radiation_results.command(nested, "run", "streaming_wave", [])
        self.assertEqual(flat_command[3:5], ["--root", str(flat)])
        self.assertEqual(nested_command[3:5], ["--root", "/work/project"])
        self.assertEqual(radiation_results.project_root(flat), flat)
        self.assertEqual(radiation_results.project_root(nested), Path("/work/project"))

    def test_metadata_build_directory_uses_adapter_project_root(self):
        case, path, descriptor = runner.resolve("wave")
        command = radiation_results.command(
            Path("/work/project/src/octotiger"), "run", case, ["Release"]
        )
        value = runner.metadata(
            "run", "wave", path, descriptor, ["Release"], Path("/tmp/out"), command
        )
        self.assertEqual(
            value["build"]["build_directory"], "/work/project/build/octotiger/release"
        )

    def test_metadata_contract_and_stable_paths(self):
        case, path, descriptor = runner.resolve("wave")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            arguments = ["2", "Release", "--threads", "6", "--time", "1.5", "--output", str(output)]
            command = radiation_results.command(runner.SOURCE_ROOT, "run", case, arguments)
            value = runner.metadata("run", "wave", path, descriptor, arguments, output, command)
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["test"]["name"], "streaming_wave")
        self.assertEqual(value["execution"]["resolution_levels"], [2])
        self.assertEqual(value["execution"]["thread_count"], 6)
        self.assertEqual(value["execution"]["timestep_controls"]["time"], "1.5")
        self.assertIn("commit", value["source"])
        self.assertIn("compiler", value["build"])
        self.assertEqual(value["artifacts"]["plots"], "plots")

    def test_source_input_output_is_rejected(self):
        with self.assertRaises(runner.HarnessError):
            runner.safe_output(runner.TEST_ROOT / "accidental-output")
        with self.assertRaises(runner.HarnessError):
            runner.safe_output(runner.HARNESS_ROOT / "references" / "accidental-output")
        with self.assertRaises(runner.HarnessError):
            runner.safe_output(runner.SOURCE_ROOT / "src" / "accidental-output")

    def test_adapter_defaults_match_legacy_modes(self):
        case, path, descriptor = runner.resolve("wave")
        output = Path("/tmp/verification-default-contract")
        command = radiation_results.command(runner.SOURCE_ROOT, "run", case, [])
        run_meta = runner.metadata("run", "wave", path, descriptor, [], output, command)
        live_meta = runner.metadata("live", "wave", path, descriptor, [], output, command)
        self.assertEqual(run_meta["execution"]["resolution_levels"], [2, 3])
        self.assertEqual(run_meta["build"]["build_type"], "Release")
        self.assertEqual(live_meta["execution"]["resolution_levels"], [2, 3, 4])
        self.assertEqual(live_meta["build"]["build_type"], "Debug")

    def test_value_options_do_not_become_levels_or_build(self):
        case, path, descriptor = runner.resolve("wave")
        arguments = [
            "--fps", "30", "--field", "er", "--width", "1280", "--seconds", "20",
            "--session-dir", "/tmp/sessions",
        ]
        command = radiation_results.command(runner.SOURCE_ROOT, "live", case, arguments)
        value = runner.metadata(
            "live", "wave", path, descriptor, arguments, Path("/tmp/out"), command
        )
        self.assertEqual(value["execution"]["resolution_levels"], [2, 3, 4])
        self.assertEqual(value["build"]["build_type"], "Debug")


class LauncherTests(unittest.TestCase):
    def test_launcher_works_outside_checkout(self):
        launcher = runner.HARNESS_ROOT / "run.sh"
        with tempfile.TemporaryDirectory() as directory:
            listed = subprocess.run(
                [str(launcher), "list"], cwd=directory, text=True, capture_output=True, check=True
            )
            planned = subprocess.run(
                [str(launcher), "plan", "wave", "2", "Release", "--threads", "3"],
                cwd=directory, text=True, capture_output=True, check=True,
            )
        self.assertIn("radiation.skinner_ostriker.streaming_wave", listed.stdout)
        manifest = json.loads(planned.stdout)
        self.assertEqual(manifest["execution"]["thread_count"], 3)
        self.assertEqual(manifest["execution"]["resolution_levels"], [2])
        self.assertEqual(manifest["execution"]["legacy_command"][1:3], ["run", "streaming_wave"])

    def test_common_runner_executes_radiation_adapter_dry_run(self):
        """Exercise process delegation without building or running physics."""
        launcher = runner.HARNESS_ROOT / "run.sh"
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            legacy = fixture / "legacy-runner"
            legacy.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
            legacy.chmod(0o755)
            environment = dict(os.environ)
            environment["OCTOTIGER_VERIFICATION_RADIATION_RUNNER"] = str(legacy)
            completed = subprocess.run(
                [
                    str(launcher), "run", "wave", "0", "--build", "Release", "--dry-run",
                    "--output", str(fixture / "dry-run-output"),
                ],
                cwd=fixture, env=environment, text=True, capture_output=True, check=True,
            )
        self.assertIn("streaming_wave", completed.stdout)
        self.assertIn("--dry-run", completed.stdout)
        self.assertFalse((fixture / "dry-run-output").exists())


if __name__ == "__main__":
    unittest.main()
