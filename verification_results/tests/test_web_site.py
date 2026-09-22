from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verification_results import runner
from verification_results.web import site


class UnifiedWebsiteTests(unittest.TestCase):
    def test_catalog_always_lists_every_registered_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = site.write(Path(tmp), runner.descriptors())
            document = (Path(tmp) / "index.html").read_text()
            self.assertEqual(catalog["test_count"], 22)
            self.assertEqual(catalog["counts"]["not_run"], 22)
            for identifier in runner.descriptors():
                self.assertIn(identifier, document)
            self.assertIn("Hydro", document)
            self.assertIn("Gravity", document)
            self.assertIn("Radiation", document)

    def test_family_manifests_merge_without_hiding_missing_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in ("hydro", "gravity", "radiation"):
                (root / family).mkdir()
                (root / family / "report.html").write_text(family)
            (root / "hydro" / "verification.json").write_text(json.dumps({
                "harness": {"adapter": "configured_ctest"},
                "tests": [{"identifier": "hydro.sod.sod", "family": "hydro", "status": "passed"}],
            }))
            (root / "gravity" / "verification.json").write_text(json.dumps({
                "harness": {"adapter": "configured_ctest"},
                "tests": [{"identifier": "gravity.sphere.self_gravitating_sphere",
                           "family": "gravity", "status": "conditional", "reason": "fixture"}],
            }))
            (root / "radiation" / "summary.json").write_text(json.dumps([
                {"id": "radiation.diagnostics.streaming_wave_1d", "status": "failed", "reason": "gate"},
            ]))
            catalog = site.write(root, runner.descriptors())
            byId = {item["identifier"]: item for item in catalog["tests"]}
            self.assertEqual(byId["hydro.sod.sod"]["status"], "passed")
            self.assertEqual(byId["gravity.sphere.self_gravitating_sphere"]["status"], "conditional")
            self.assertEqual(byId["radiation.diagnostics.streaming_wave_1d"]["status"], "failed")
            self.assertEqual(byId["radiation.ensman.radiative_shock"]["status"], "not_run")
            self.assertEqual(catalog["status"], "failed")
            self.assertIn('href="radiation/report.html"', (root / "index.html").read_text())

    def test_application_report_becomes_subreport_and_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = "<!doctype html><title>Four application tests</title>"
            (root / "index.html").write_text(old)
            for case in site.applicationCases:
                (root / (case + ".html")).write_text(case)
            first = site.write(root, runner.descriptors(), legacyApplication=True)
            self.assertEqual((root / "radiation-application.html").read_text(), old)
            self.assertIn("octotiger-unified-site", (root / "index.html").read_text())
            byId = {item["identifier"]: item for item in first["tests"]}
            self.assertTrue(all(byId[identifier]["status"] == "complete"
                                for identifier in site.applicationCases.values()))
            site.write(root, runner.descriptors(), legacyApplication=True)
            self.assertEqual((root / "radiation-application.html").read_text(), old)
            resumed = "<!doctype html><title>Updated application tests</title>"
            (root / "index.html").write_text(resumed)
            site.write(root, runner.descriptors(), legacyApplication=True)
            self.assertEqual((root / "radiation-application.html").read_text(), resumed)

    def test_nested_application_results_merge_into_unified_live_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            application = root / "radiation" / "application"
            application.mkdir(parents=True)
            application.joinpath("index.html").write_text("application")
            for case in site.applicationCases:
                application.joinpath(case + ".html").write_text(case)
            root.joinpath("summary.json").write_text(json.dumps({
                "status": "running", "families": [{"family": "radiation", "status": "running"}]
            }))
            catalog = site.write(root, runner.descriptors())
            byId = {item["identifier"]: item for item in catalog["tests"]}
            self.assertTrue(all(byId[identifier]["status"] == "complete"
                                for identifier in site.applicationCases.values()))
            document = root.joinpath("index.html").read_text()
            self.assertIn('http-equiv="refresh"', document)
            self.assertIn("radiation/application/index.html", document)

    def test_legacy_launcher_publishes_unified_landing_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "batch"
            fake = root / "fake-radiation-runner"
            fake.write_text("""#!/usr/bin/env python3
import pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
out.mkdir(parents=True)
out.joinpath('index.html').write_text('<title>application report</title>')
for name in ('streaming_wave', 'streaming_front', 'gaussian_pulse', 'equilibrium_sphere'):
    out.joinpath(name + '.html').write_text(name)
""")
            fake.chmod(0o755)
            with patch.dict(os.environ, {"OCTOTIGER_VERIFICATION_RADIATION_RUNNER": str(fake)}):
                self.assertEqual(runner.main(["run", "all", "0", "Release",
                                              "--output", str(output)]), 0)
            self.assertTrue((output / "radiation-application.html").is_file())
            document = (output / "index.html").read_text()
            self.assertIn("octotiger-unified-site", document)
            self.assertIn("hydro.sod.sod", document)
            self.assertIn("radiation.ensman.radiative_shock", document)


if __name__ == "__main__":
    unittest.main()
