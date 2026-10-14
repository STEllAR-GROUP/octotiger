#!/usr/bin/env python3
"""Synthetic saved-output checks: no solver, renderer, or remote operations."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import compare_campaign as comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "index.html").write_text("campaign")

    def manifest(self, runs, **updates):
        result = {"schema": 2, "workflow": "m1-fld-radiation", "methods": ["m1", "fld"], "runs": runs}
        result.update(updates)
        return result

    def add_run(self, model, case="gaussian_pulse", level=0, time=.1, offset=0., completed=True,
                fld_reference=False, reference_offset=0.):
        run = {"case": case, "level": level, "method": model, "model": model,
               "integrator": comparison.INTEGRATORS[model],
               "directory": f"runs/l{level}/{case}/{model}",
               "status": "completed" if completed else "pending"}
        if not completed:
            return run
        folder = self.root / run["directory"]
        data = folder / case / f"l{level}"
        data.mkdir(parents=True)
        (folder / "batch.json").write_text(json.dumps({"source_sha256": {"source.cpp": "same-source"}}))
        (folder / f"{case}.html").write_text("<!doctype html><html><body>saved run</body></html>")
        numerical_only = (model == "fld" and not fld_reference) or case in comparison.COUPLED_WITHOUT_REFERENCE
        metadata = {"status": "complete", "case": case, "level": level, "time": time,
                    "radiation_model": model, "radiation_integrator": run["integrator"],
                    "radiation_algorithm_version": 1, "radiation_flux_semantics": comparison.FLUX_SEMANTICS[model],
                    "radiation_reduced_c_factor": 1., "length": 2.,
                    "reference_kind": "numerical-only" if numerical_only else comparison.REFERENCE[case][0],
                    "executable_sha256": "same-executable", "solver_library_sha256": "same-library",
                    "comparison_signature": {"radiation_model": model,
                        "radiation_integrator": run["integrator"], "opacity": "1", "hard_dt": ".03125"}}
        if model == "fld" and fld_reference:
            metadata.update(reference_kind="linear-diffusion-limit",
                reference_model=comparison.FLD_REFERENCE_MODELS[case], reference_exact_for_solver=False)
        (data / "run.json").write_text(json.dumps(metadata))
        coupled = case in comparison.COUPLED_WITHOUT_REFERENCE
        fields = comparison.HYDRO_FIELDS if coupled else comparison.RAD_FIELDS
        if case == "marshak":
            fields = (*fields, "material_egas")
        columns = ["t", "dx", "x", "y", "z", *fields]
        if not numerical_only:
            columns += [field + "_ref" for field in fields]
        plot = folder / "plots" / case / f"l{level}"
        plot.mkdir(parents=True)
        stem = "hydro-slice" if coupled else "slice"
        with (plot / f"{stem}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for y in (-.5, .5):
                for x in (-.5, .5):
                    row = {"t": time, "dx": 1., "x": x, "y": y, "z": .5}
                    row.update({field: 1. + offset for field in fields})
                    if not numerical_only:
                        row.update({field + "_ref": 1. + reference_offset for field in fields})
                    writer.writerow(row)
        units = {name: "normalized" if case == "marshak" else "test unit" for name in columns}
        (plot / f"{stem}.units.json").write_text(json.dumps(units))
        return run

    def metadata_path(self, run):
        return self.root / run["directory"] / run["case"] / f'l{run["level"]}' / "run.json"

    def change_metadata(self, run, **updates):
        path = self.metadata_path(run)
        value = json.loads(path.read_text())
        value.update(updates)
        path.write_text(json.dumps(value))

    def fake_plot(self, folder, *args):
        (folder / "profiles.png").write_bytes(b"test image")
        (folder / "comparison.csv").write_text("x,m1,fld\n")

    def refresh(self, runs):
        with patch.object(comparison, "make_plot", side_effect=self.fake_plot):
            return comparison.refresh(self.root, self.manifest(runs))

    def test_model_pair_retains_only_applicable_reference_errors(self):
        runs = [self.add_run("m1", offset=.25), self.add_run("fld", offset=.125)]
        originals = {self.metadata_path(run): self.metadata_path(run).read_bytes() for run in runs}
        row = self.refresh(runs)["cases"]["gaussian_pulse"][0]
        self.assertEqual(row["status"], "matched")
        self.assertEqual(row["difference_kind"], "model discrepancy")
        self.assertAlmostEqual(row["direct_slice_difference"]["er"]["L1"], .125)
        self.assertAlmostEqual(row["methods"]["m1"]["reference_slice_errors"]["er"]["L2"], .25)
        self.assertFalse(row["methods"]["fld"]["reference_slice_errors"])
        self.assertEqual(row["methods"]["fld"]["radiation_integrator"], "backward_euler")
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_marshak_compares_both_thermal_energies_without_fld_reference(self):
        runs = [self.add_run("m1", "marshak", offset=.25), self.add_run("fld", "marshak", offset=.125)]
        row = self.refresh(runs)["cases"]["marshak"][0]
        self.assertEqual(row["status"], "matched")
        self.assertAlmostEqual(row["direct_slice_difference"]["material_egas"]["L1"], .125)
        self.assertEqual(row["methods"]["fld"]["units"]["t"], "normalized")
        self.assertAlmostEqual(row["methods"]["m1"]["reference_slice_errors"]["material_egas"]["L1"], .25)
        self.assertFalse(row["methods"]["fld"]["reference_slice_errors"])

    def test_fld_analytic_errors_are_separate_from_m1_reference_and_model_discrepancy(self):
        for case in ("gaussian_pulse", "marshak", "equilibrium_sphere"):
            runs = [self.add_run("m1", case, offset=.25),
                    self.add_run("fld", case, offset=.125, fld_reference=True, reference_offset=.0625)]
            self.change_metadata(runs[1], norms={"er": {"L1": 999, "L2": 999, "Linf": 999}},
                fld_diagnostics={"slice_reference_norms": {"er": {"L1": .0625}}})
            row = self.refresh(runs)["cases"][case][0]
            self.assertEqual(row["status"], "matched")
            self.assertAlmostEqual(row["direct_slice_difference"]["er"]["L1"], .125)
            self.assertAlmostEqual(row["methods"]["m1"]["reference_slice_errors"]["er"]["L1"], .25)
            fld = row["methods"]["fld"]
            self.assertAlmostEqual(fld["reference_slice_errors"]["er"]["L1"], .0625)
            self.assertEqual(fld["reference_model"], comparison.FLD_REFERENCE_MODELS[case])
            self.assertFalse(fld["reference_exact_for_solver"])
            self.assertFalse(fld["solver_full_volume_reference_errors"])
            self.assertIn("not full-volume", fld["reference_error_scope"])
            if case == "marshak":
                self.assertAlmostEqual(fld["reference_slice_errors"]["material_egas"]["L1"], .0625)
            page = (self.root / "comparisons" / f"{case}.html").read_text()
            self.assertIn("diffusion-limit", page)
            self.assertNotIn("999", page)

    def test_fld_references_require_the_right_model_and_explicit_limit_qualification(self):
        run = self.add_run("fld", fld_reference=True)
        original = self.metadata_path(run).read_text()
        for change in ({"reference_kind": "exact"}, {"reference_model": "m1-telegraph"},
                       {"reference_model": "su-olson-linear-halfspace"}, {"reference_exact_for_solver": True},
                       {"reference_exact_for_solver": None}, {"reference_sampling": "point samples"}):
            self.metadata_path(run).write_text(original)
            self.change_metadata(run, **change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "diffusion-limit"):
                comparison.load_run(self.root, run)

    def test_fld_declared_reference_missing_columns_is_rejected(self):
        run = self.add_run("fld")
        self.change_metadata(run, reference_kind="linear-diffusion-limit",
            reference_model="periodic-constant-diffusion", reference_exact_for_solver=False)
        with self.assertRaisesRegex(ValueError, "Missing saved"):
            comparison.load_run(self.root, run)

    def test_stale_fld_reference_columns_and_volume_norms_are_never_presented(self):
        run = self.add_run("fld")
        path = self.root / run["directory"] / "plots/gaussian_pulse/l0/slice.csv"
        with path.open() as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*columns, "er_ref"])
            writer.writeheader()
            writer.writerows(dict(row, er_ref="100") for row in rows)
        self.change_metadata(run, norms={"er": {"L1": 99}})
        loaded = comparison.load_run(self.root, run)
        self.assertNotIn("er_ref", loaded["values"])
        self.assertFalse(comparison.record_run(self.root, loaded)["solver_full_volume_reference_errors"])

    def test_partial_pair_does_not_substitute_another_level(self):
        runs = [self.add_run("m1", level=0), self.add_run("fld", level=0, completed=False),
                self.add_run("fld", level=1)]
        rows = self.refresh(runs)["cases"]["gaussian_pulse"]
        self.assertTrue(all(row["status"] == "awaiting_matching_method" for row in rows))
        self.assertTrue(all(not row["direct_slice_difference"] for row in rows))

    def test_m1_only_hydro_case_has_no_unsupported_partner(self):
        row = self.refresh([self.add_run("m1", "subcritical_shock")])["cases"]["subcritical_shock"][0]
        self.assertEqual(row["status"], "m1_only")
        self.assertEqual(set(row["methods"]), {"m1"})
        self.assertFalse(row["direct_slice_difference"])
        self.assertFalse(row["methods"]["m1"]["reference_slice_errors"])

    def test_full_suite_explicitly_unavailable_fld_cases_do_not_await_a_partner(self):
        runs = [self.add_run('m1', case) for case in comparison.CASES]
        runs += [self.add_run('fld', case, fld_reference=True)
                 for case in ('gaussian_pulse', 'equilibrium_sphere', 'marshak')]
        blocked = [dict(case=case, level=0, model='fld', reason='Known vacuum or moving-gas limitation')
                   for case in comparison.CASES if case not in comparison.FLD_REFERENCE_MODELS]
        manifest = self.manifest(runs, unsupported=blocked)
        with patch.object(comparison, 'make_plot', side_effect=self.fake_plot):
            summary = comparison.refresh(self.root, manifest)
        for case, rows in summary['cases'].items():
            self.assertEqual(rows[0]['status'], 'matched' if case in comparison.FLD_REFERENCE_MODELS else 'm1_only')
            if case not in comparison.FLD_REFERENCE_MODELS:
                self.assertEqual(set(rows[0]['methods']), {'m1'})
                self.assertFalse(rows[0]['direct_slice_difference'])
        page = (self.root / 'comparisons/streaming_wave.html').read_text()
        self.assertIn('Known vacuum or moving-gas limitation', page)
        self.assertIn('No excluded matching partner is assumed', page)
        self.assertNotIn('awaiting matching method', page)

    def test_scheduled_run_cannot_also_be_marked_unsupported(self):
        run = self.add_run('fld', completed=False)
        manifest = self.manifest([run], unsupported=[dict(case=run['case'], level=0, model='fld', reason='blocked')])
        with self.assertRaisesRegex(ValueError, 'both scheduled and unsupported'):
            comparison.validate_manifest(manifest)

    def test_legacy_campaign_and_runtime_profiles_are_rejected_before_writes(self):
        for manifest in ({"schema": 1, "workflow": "level-major-radiation", "runs": []},
                         self.manifest([], runtime_profiles={"baseline": {}}),
                         self.manifest([], runtime_comparisons=[{}]),
                         self.manifest([], methods=["euler", "ader2"])):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                comparison.refresh(self.root, manifest)
        self.assertFalse((self.root / "comparisons").exists())

    def test_unsupported_fld_case_or_inconsistent_model_identity_is_rejected(self):
        for case in comparison.COUPLED_WITHOUT_REFERENCE:
            run = self.add_run("fld", case, completed=False)
            with self.assertRaisesRegex(ValueError, "stationary"):
                comparison.validate_manifest(self.manifest([run]))
        run = self.add_run("m1", completed=False)
        run["integrator"] = "backward_euler"
        with self.assertRaisesRegex(ValueError, "identity"):
            comparison.validate_manifest(self.manifest([run]))

    def test_streaming_fld_has_only_integral_constraints_and_cross_model_discrepancies(self):
        for case in ("streaming_wave", "streaming_front"):
            runs = [self.add_run("m1", case, offset=.25), self.add_run("fld", case, offset=.125)]
            mean = 1. if case == "streaming_wave" else .50000000005
            invariants = {"scope": "full-volume-energy-history", "expected_mean_radiation_energy": mean,
                          "maximum_relative_integrated_energy_error": 1.e-12}
            self.change_metadata(runs[1], reference_kind="unavailable", reference_model="none",
                reference_exact_for_solver=False, reference_sampling="none", analytic_mean_energy=mean,
                fld_diagnostics={"analytic_invariants": invariants})
            row = self.refresh(runs)["cases"][case][0]
            self.assertEqual(row["status"], "matched")
            self.assertAlmostEqual(row["direct_slice_difference"]["er"]["L1"], .125)
            self.assertTrue(row["methods"]["m1"]["reference_exact_for_solver"])
            fld = row["methods"]["fld"]
            self.assertFalse(fld["reference_slice_errors"])
            self.assertFalse(fld["solver_full_volume_reference_errors"])
            self.assertFalse(fld["reference_exact_for_solver"])
            self.assertEqual(fld["reference_sampling"], "none")
            self.assertEqual(fld["analytic_invariants"], invariants)
            page = (self.root / "comparisons" / (case + ".html")).read_text()
            self.assertIn('no supplied FLD spatial analytic reference', page)
            self.assertIn('Exact integral invariant', page)
            self.assertNotIn('retained for M1 only', page)

    def test_streaming_fld_cannot_claim_diffusion_or_m1_spatial_reference(self):
        run = self.add_run("fld", "streaming_wave")
        for updates in ({"reference_kind": "linear-diffusion-limit", "reference_exact_for_solver": False},
                {"reference_kind": "unavailable", "reference_model": "m1-advection", "reference_exact_for_solver": False,
                 "reference_sampling": "none"},
                {"reference_kind": "unavailable", "reference_model": "none", "reference_exact_for_solver": True,
                 "reference_sampling": "none"}):
            self.change_metadata(run, **updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                comparison.load_run(self.root, run)

    def test_duplicate_identity_is_rejected(self):
        run = self.add_run("m1", completed=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            comparison.validate_manifest(self.manifest([run, run]))

    def test_scientific_scope_and_current_labels_on_pages(self):
        self.refresh([self.add_run("m1"), self.add_run("fld")])
        for name in ("index", "gaussian_pulse"):
            page = (self.root / "comparisons" / f"{name}.html").read_text()
            self.assertIn("model discrepanc", page)
            self.assertIn("first order", page)
            self.assertIn("not errors in a shared radiation-momentum", page)
            self.assertNotIn("ADER", page)

    def test_mismatched_physical_time_withholds_overlay(self):
        row = self.refresh([self.add_run("m1", time=.1), self.add_run("fld", time=.2)])["cases"]["gaussian_pulse"][0]
        self.assertEqual(row["status"], "incompatible")
        self.assertFalse(row["direct_slice_difference"])
        self.assertNotIn("plot", row)
        self.assertIn("time", row["reason"])

    def test_mismatched_source_executable_library_physics_or_units_is_rejected(self):
        runs = [self.add_run("m1"), self.add_run("fld")]
        left, right = [comparison.load_run(self.root, run) for run in runs]
        changes = (("executable_sha256", "different"), ("solver_library_sha256", "different"),
                   ("comparison_signature", {"opacity": 2}), ("radiation_reduced_c_factor", .1))
        for key, value in changes:
            original = right["metadata"].get(key)
            right["metadata"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                comparison.compatible(left, right)
            right["metadata"][key] = original
        right["source_sha256"] = {"source.cpp": "different"}
        with self.assertRaisesRegex(ValueError, "source provenance"):
            comparison.compatible(left, right)
        right["source_sha256"] = left["source_sha256"]
        right["units"]["er"] = "wrong units"
        with self.assertRaisesRegex(ValueError, "units"):
            comparison.compatible(left, right)

    def test_model_and_solver_controls_can_differ_but_are_recorded(self):
        runs = [self.add_run("m1"), self.add_run("fld")]
        left, right = [comparison.load_run(self.root, run) for run in runs]
        right["metadata"]["comparison_signature"].update(fld_linear_tolerance=1e-9, hard_dt=.1,
            disable_analytic=True, radiation_limiter="levermore_pomraning")
        comparison.compatible(left, right)
        self.assertEqual(comparison.record_run(self.root, right)["radiation_flux_semantics"], "derived_fld_flux")
        right["metadata"]["comparison_signature"]["eos"] = "unmatched equation of state"
        with self.assertRaisesRegex(ValueError, "physical/reconstruction"):
            comparison.compatible(left, right)

    def test_selected_single_model_has_no_assumed_matching_run(self):
        for model in comparison.METHODS:
            run = self.add_run(model)
            with patch.object(comparison, "make_plot", side_effect=self.fake_plot):
                summary = comparison.refresh(self.root, self.manifest([run], methods=[model]))
            self.assertEqual(summary["methods"], [model])
            row = summary["cases"]["gaussian_pulse"][0]
            self.assertEqual(row["status"], model + "_only")
            self.assertEqual(list(row["methods"]), [model])
            self.assertFalse(row["direct_slice_difference"])

    def test_unselected_model_is_rejected(self):
        with self.assertRaises(ValueError):
            comparison.validate_manifest(self.manifest([self.add_run("fld", completed=False)], methods=["m1"]))

    def test_unsupported_selected_cases_do_not_fabricate_results(self):
        run = self.add_run("fld", completed=False)
        selected = comparison.validate_manifest(self.manifest([run], methods=["fld"],
            cases=["gaussian_pulse", "radiating_blast"]))
        self.assertEqual(selected, ["gaussian_pulse"])

    def test_matching_pair_must_use_frozen_campaign_runtime_and_source(self):
        runs = [self.add_run("m1"), self.add_run("fld")]
        left, right = [comparison.load_run(self.root, run) for run in runs]
        manifest = self.manifest(runs, executable_sha256="same-executable",
            snapshot_sha256={".campaign/source/source.cpp": "same-source"})
        comparison.compatible(left, right, self.root, manifest)
        for change in ({"executable_sha256": "foreign"},
                       {"snapshot_sha256": {".campaign/source/source.cpp": "foreign"}}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "frozen"):
                comparison.compatible(left, right, self.root, dict(manifest, **change))

    def test_saved_metadata_hash_change_is_rejected(self):
        run = self.add_run("m1")
        run["run_metadata_sha256"] = comparison.digest(self.metadata_path(run))
        comparison.load_run(self.root, run)
        self.change_metadata(run, time=.2)
        with self.assertRaisesRegex(ValueError, "metadata hash"):
            comparison.load_run(self.root, run)

    def test_inconsistent_saved_model_flux_semantics_or_reference_is_invalid(self):
        run = self.add_run("fld")
        original = self.metadata_path(run).read_text()
        for change in ({"radiation_model": "m1"}, {"radiation_integrator": "euler"},
                       {"radiation_algorithm_version": 2}, {"radiation_flux_semantics": "evolved_m1_flux"},
                       {"reference_kind": "exact M1"}, {"radiation_ader_version": 6}):
            self.metadata_path(run).write_text(original)
            self.change_metadata(run, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                comparison.load_run(self.root, run)

    def test_adaptive_partition_is_area_weighted_without_holes(self):
        coarse = {"fields": ("er",), "values": {"x": np.array([0.]), "y": np.array([0.]),
                  "dx": np.array([2.]), "er": np.array([2.])}}
        fine = {"fields": ("er",), "values": {"x": np.array([-.5, .5, -.5, .5]),
                "y": np.array([-.5, -.5, .5, .5]), "dx": np.ones(4), "er": np.array([1., 3., 1., 3.])}}
        difference = comparison.raster(fine, (-1., 1., -1., 1.), 1.)["er"] - comparison.raster(coarse, (-1., 1., -1., 1.), 1.)["er"]
        self.assertEqual(comparison.norms(difference), {"L1": 1., "L2": 1., "Linf": 1.})
        fine["values"] = {name: values[:-1] for name, values in fine["values"].items()}
        with self.assertRaisesRegex(ValueError, "holes"):
            comparison.raster(fine, (-1., 1., -1., 1.), 1.)

    def test_adaptive_display_pixels_retain_native_width_without_overlap(self):
        runs = [self.add_run("m1", level=1, offset=.25), self.add_run("fld", level=1, offset=.125)]
        for run in runs:
            self.change_metadata(run, adaptive=True, min_level=0, dx=1., cells=4, length=4.,
                                 mesh={"native_slice_cells": 4})
            path = self.root / run["directory"] / "plots" / run["case"] / "l1/slice.csv"
            columns = ["t", "dx", "x", "y", "z", *comparison.RAD_FIELDS]
            if run["method"] == "m1":
                columns += [field + "_ref" for field in comparison.RAD_FIELDS]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for y in (-1.5, -.5, .5, 1.5):
                    for x in (-1.5, -.5, .5, 1.5):
                        row = {"t": .1, "dx": 2., "x": x, "y": y, "z": 1.}
                        row.update({field: 1. + (.25 if run["method"] == "m1" else .125)
                                    for field in comparison.RAD_FIELDS})
                        if run["method"] == "m1":
                            row.update({field + "_ref": 1. for field in comparison.RAD_FIELDS})
                        writer.writerow(row)
        row = self.refresh(runs)["cases"]["gaussian_pulse"][0]
        self.assertEqual(row["status"], "matched")
        self.assertAlmostEqual(row["direct_slice_difference"]["er"]["L1"], .125)
        self.assertEqual(row["methods"]["m1"]["leaf_slice_cells"], 4)
        self.assertEqual(row["methods"]["m1"]["display_slice_pixels"], 16)

    def test_real_plot_has_separate_model_specific_reference_columns(self):
        runs = [self.add_run("m1", "marshak"),
                self.add_run("fld", "marshak", offset=.125, fld_reference=True, reference_offset=.0625)]
        comparison.refresh(self.root, self.manifest(runs))
        path = self.root / "comparisons/marshak/l0/comparison.csv"
        with path.open() as stream:
            columns = csv.DictReader(stream).fieldnames
        self.assertIn("m1_er_ref", columns)
        self.assertIn("fld_material_egas", columns)
        self.assertIn("fld_er_ref", columns)
        self.assertIn("fld_material_egas_ref", columns)


if __name__ == "__main__":
    unittest.main()
