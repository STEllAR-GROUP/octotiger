"""Small reporting-contract tests; these do not launch VisIt or FFmpeg."""

import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest

import campaign_movies as movies


class MovieTimingTests(unittest.TestCase):
    def test_extraction_manifest_round_trips_into_verified_retention(self):
        import campaign_retention as retention
        from test_campaign_retention import prepare_outputs
        with tempfile.TemporaryDirectory(prefix="movie-extract-test-") as temporary:
            root = Path(temporary)
            run = {"id": "l0-example-m1", "case": "example", "level": 0,
                   "directory": "runs/m1", "stop_time": 1.5, "run_metadata_sha256": "placeholder",
                   "attempts": [{"pid": 100, "process_start_ticks": "123"}]}
            original = prepare_outputs(root, run)
            folder = retention.leaf(root, run)
            (folder / "run.json").write_text("{}")
            run["run_metadata_sha256"] = movies.sha256(folder / "run.json")
            destination = folder / movies.MOVIE_SUBDIR
            movie = json.loads((destination / "campaign-movie.json").read_text())
            movie.update(run_metadata_sha256=run["run_metadata_sha256"], panels=["radiation"])
            movies.write_json(destination / "campaign-movie.json", movie)
            frames = [{"time": time, "profile_data": f"profile-{i}.json", "radiation_image": f"frame-{i}.png"}
                      for i, time in enumerate(movie["snapshot_times"])]
            movies.write_json(destination / "frames.json", {"frames": frames})
            extraction = movies.write_extraction_manifest(folder, destination, movie, original["settings"])
            self.assertEqual(len(extraction["profiles"]), 3)
            receipt = retention.make_receipt(root, {"snapshots": 3}, run,
                {"directory": run["directory"], "claimed_utc": "2026-10-14T12:00:00Z",
                 "solver_identity": {"pid": 100, "process_start_ticks": "123"}})
            path = folder / "retention.json"
            movies.write_json(path, receipt)
            run["retention"] = {"path": str(path.relative_to(root)), "sha256": retention.digest(path)}
            retention.prune(root, run)
            self.assertFalse((folder / "X.1.silo").exists())
            self.assertEqual(retention.validate_retained_cache(root, run, original["settings"])["source_snapshots"], 3)
            (destination / "profile-1.json").write_text("{}")
            with self.assertRaises(ValueError):
                retention.validate_retained_cache(root, run, original["settings"])

    def test_embedded_extraction_uses_full_volume_and_native_profiles(self):
        compile(movies.VISIT_SCRIPT, "embedded_visit_script", "exec")
        self.assertIn('Query("Weighted Variable Sum", use_actual_data=1)', movies.VISIT_SCRIPT)
        self.assertIn('use_sampling=0', movies.VISIT_SCRIPT)
        self.assertIn('"integration": "full-volume leaf-cell"', movies.VISIT_SCRIPT)

    def test_irregular_recorded_times_use_hold_not_interpolation(self):
        times = [0.0, 0.1, 0.9, 1.0]
        sequence = movies.frame_schedule(times, seconds=7, fps=2, hold=1)
        self.assertEqual(sequence, [0, 0] + [0, 1, 1, 1, 1, 1, 1, 1, 1, 2] + [3, 3])
        # Each video frame refers to a stored state, never a weighted blend.
        self.assertTrue(all(isinstance(state, int) for state in sequence))

    def test_endpoint_holds_and_every_stored_state_for_typical_cadence(self):
        sequence = movies.frame_schedule([0., .5, 1.], seconds=5, fps=10, hold=1)
        self.assertEqual(len(sequence), 50)
        self.assertEqual(sequence[:10], [0] * 10)
        self.assertEqual(sequence[-10:], [2] * 10)
        self.assertEqual(set(sequence), {0, 1, 2})

    def test_invalid_times_are_rejected(self):
        for times in ([0.], [0., 0.], [1., 0.], [0., float("nan")]):
            with self.assertRaises(ValueError):
                movies.frame_schedule(times, 12, 24, 1)

    def test_piecewise_blast_has_ten_seconds_per_phase_and_two_endpoint_holds(self):
        times = [0., 37., 100., 395., 400., 575., 1800., 3950., 4000.]
        timing = {"segments": [{"end_time": 400., "seconds": 10.},
                               {"end_time": 4000., "seconds": 10.}],
                  "reason": "Resolve early radiation and later material expansion"}
        sequence, playback = movies.playback_schedule(times, 12, 24, 1, timing)
        self.assertEqual(len(sequence), 528)
        self.assertEqual(playback["seconds"], 22.)
        self.assertEqual(playback["active_seconds"], 20.)
        self.assertEqual(playback["endpoint_hold_seconds"], 1.)
        self.assertEqual([s["video_frames"] for s in playback["segments"]], [240, 240])
        self.assertEqual([(s["first_video_frame"], s["end_video_frame_exclusive"])
                          for s in playback["segments"]], [(24, 264), (264, 504)])
        self.assertEqual(sequence[:24], [0] * 24)
        self.assertEqual(sequence[24], 0)
        self.assertEqual(sequence[263], 3)
        self.assertEqual(sequence[264], 4)
        self.assertEqual(sequence[-24:], [8] * 24)
        self.assertEqual(set(sequence), set(range(len(times))))
        # At five seconds into each phase, sample physical t=200 and t=2200.
        self.assertEqual(sequence[24 + 120], 2)
        self.assertEqual(sequence[264 + 120], 6)

    def test_default_twelve_seconds_retains_ten_active_seconds(self):
        sequence, plan = movies.playback_schedule([0., .11, .5, .8, 1.], 12, 24, 1)
        self.assertEqual(len(sequence), 288)
        self.assertEqual(plan["active_seconds"], 10.)
        self.assertEqual(plan["mode"], "uniform_actual_time")
        self.assertTrue(plan["all_recorded_states_displayed"])

    def test_unrepresentable_recorded_state_fails_instead_of_silently_skipping(self):
        with self.assertRaisesRegex(ValueError, "skip .* recorded states"):
            movies.frame_schedule([0., .1, .9, 1.], 5, 2, 0)
        with self.assertRaisesRegex(ValueError, "skip .* recorded states"):
            movies.frame_schedule([0., 399., 400., 4000.], 12, 24, 1,
                                  {"segments": [{"end_time": 400., "seconds": 10.},
                                                {"end_time": 4000., "seconds": 10.}]})

    def test_segment_validation_and_solver_final_time_tolerance(self):
        invalid = [[], [{"end_time": 1., "seconds": 0.}],
                   [{"end_time": float("nan"), "seconds": 10.}],
                   [{"end_time": 1., "seconds": float("inf")}],
                   [{"end_time": .8, "seconds": 5.}, {"end_time": .5, "seconds": 5.}, {"end_time": 1., "seconds": 5.}],
                   [{"end_time": 1.01, "seconds": 10.}]]
        for segments in invalid:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                movies.frame_schedule([0., .5, 1.], 12, 24, 1, {"segments": segments})
        _, plan = movies.playback_schedule([0., .5, 1.], 12, 24, 1,
                                           {"segments": [{"end_time": 1. + 5e-10, "seconds": 10.}]})
        self.assertEqual(plan["segments"][-1]["end_time"], 1.)


class MovieInputTests(unittest.TestCase):
    def model_metadata(self, model="fld"):
        return {"case": "marshak", "radiation_model": model,
                "radiation_integrator": movies.MODEL_INTEGRATORS[model],
                "radiation_algorithm_version": 1,
                "radiation_flux_semantics": movies.MODEL_FLUX[model]}

    def test_flux_semantics_distinguish_evolved_and_derived_fields(self):
        m1 = movies.model_contract(self.model_metadata("m1"))
        fld = movies.model_contract(self.model_metadata("fld"))
        self.assertEqual(m1["model_label"], "M1 RK1")
        self.assertEqual(fld["model_label"], "FLD backward Euler")
        self.assertEqual(fld["radiation_flux_semantics"], "derived_fld_flux")
        self.assertIn("radiation momentum is not evolved", fld["flux_description"])
        self.assertIn("independently evolved", m1["flux_description"])
        self.assertIn("see the final-time run report", m1["reference_description"])
        self.assertNotIn("no applicable", m1["reference_description"])

    def test_moving_fld_uses_lab_flux_and_v2_identity(self):
        for case in movies.COUPLED:
            metadata = dict(self.model_metadata(), case=case, radiation_algorithm_version=2,
                radiation_flux_semantics="derived_fld_lab_flux", radiation_frame="mixed",
                radiation_hydro_scheme="first_order_conservative", reference_kind="unavailable",
                reference_model="none", reference_exact_for_solver=False, reference_sampling="none")
            result = movies.model_contract(metadata)
            self.assertEqual(result["radiation_algorithm_version"], 2)
            self.assertIn("enthalpy", result["flux_description"])
            self.assertIn("lab-frame", result["flux_label"])
            for change in ({"radiation_frame": "lab"}, {"radiation_algorithm_version": 1},
                           {"radiation_flux_semantics": "derived_fld_flux"}):
                with self.subTest(case=case, change=change), self.assertRaises(ValueError):
                    movies.model_contract(dict(metadata, **change))

    def test_fld_reference_metadata_remains_qualified_and_separate_from_movie_frames(self):
        metadata = dict(self.model_metadata(), reference_kind="linear-diffusion-limit",
            reference_model="su-olson-linear-halfspace", reference_exact_for_solver=False)
        contract = movies.model_contract(metadata)
        self.assertFalse(contract["reference_exact_for_solver"])
        self.assertIn("not an exact solution", contract["reference_description"])
        self.assertIn("numerical states only", contract["reference_description"])
        for change in ({"reference_model": "m1-telegraph"}, {"reference_exact_for_solver": True},
                       {"reference_kind": "exact"}, {"reference_model": "periodic-constant-diffusion"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "diffusion-limit"):
                movies.model_contract(dict(metadata, **change))

    def test_mislabeled_or_historical_input_is_rejected(self):
        for change in ({"radiation_integrator": "ader2"}, {"radiation_model": "m1"},
                       {"radiation_flux_semantics": "evolved_m1_flux"}, {"radiation_algorithm_version": 6},
                       {"runtime_profile": "newton-recovery-2"}, {"case": "subcritical_shock"}):
            metadata = dict(self.model_metadata(), **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                movies.model_contract(metadata)
        with self.assertRaisesRegex(ValueError, "does not match"):
            movies.model_contract(self.model_metadata(), {"method": "m1", "model": "m1", "integrator": "euler"})

    def test_movie_manifest_requires_new_models_and_supported_case_matrix(self):
        manifest = {"schema": 2, "workflow": "m1-fld-radiation", "methods": ["m1", "fld"],
                    "runs": [{"case": "marshak", "level": 0, "method": "fld", "model": "fld",
                              "integrator": "backward_euler"}]}
        movies.validate_movie_manifest(manifest)
        movies.validate_movie_manifest(dict(manifest, methods=["fld"]))
        for case in movies.FLD_CASES:
            movies.validate_movie_manifest(dict(manifest, runs=[dict(manifest["runs"][0], case=case)]))
        for change in ({"schema": 1}, {"workflow": "level-major-radiation"},
                       {"runtime_profiles": {"baseline": {}}}, {"methods": ["euler", "ader2"]},
                       {"methods": []}, {"methods": ["m1"]}, {"methods": ["fld", "m1"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                movies.validate_movie_manifest(dict(manifest, **change))

    def test_five_stationary_fld_cases_keep_reference_contracts_separate(self):
        for case in ("streaming_wave", "streaming_front"):
            metadata = dict(self.model_metadata(), case=case, reference_kind="unavailable", reference_model="none",
                            reference_exact_for_solver=False, reference_sampling="none")
            contract = movies.model_contract(metadata)
            self.assertIn('M1 streaming profile is not an FLD reference', contract['reference_description'])
            self.assertFalse(contract['reference_exact_for_solver'])
            for updates in ({'reference_model': 'm1-advection'}, {'reference_exact_for_solver': True},
                            {'reference_sampling': 'cell averages'}, {'reference_kind': 'linear-diffusion-limit'}):
                with self.subTest(case=case, updates=updates), self.assertRaises(ValueError):
                    movies.model_contract(dict(metadata, **updates))
        sphere = dict(self.model_metadata(), case='equilibrium_sphere', reference_kind='linear-diffusion-limit',
                      reference_model='steady-gaussian-emitter', reference_exact_for_solver=False,
                      reference_sampling='cell averages')
        contract = movies.model_contract(sphere)
        self.assertEqual(contract['reference_model'], 'steady-gaussian-emitter')
        self.assertIn('not an exact solution', contract['reference_description'])

    def test_full_domain_policy_invalidates_cropped_cache_and_ignores_saved_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "run.json").write_text('{"length": 2}')
            silo = folder / "X.0.silo"
            silo.write_bytes(b"unchanged numerical data")
            args = argparse.Namespace(width=900, height=800, vectors=144, arrow_stride=0,
                                      fps=24, seconds=12, hold=1, visit="visit", full_domain=False)
            crop = {"x": [-1., -.75], "y": [-.125, .125], "reason": "Old boundary crop"}
            scope = "Full benchmark duration; cropped central maps and a full-domain synchronized temperature profile."
            run = {"case": "subcritical_shock", "movie_view": crop, "physical_scope": scope}
            cropped = movies.source_signature(folder, [silo], args, run)
            self.assertEqual(cropped["movie_view"], crop)
            self.assertEqual(cropped["physical_scope"], scope)
            args.full_domain = True
            full = movies.source_signature(folder, [silo], args, run)
            self.assertNotEqual(full, cropped)
            self.assertIsNone(full["movie_view"])
            self.assertTrue(full["full_domain"])
            self.assertEqual(full["physical_scope"], scope.replace("cropped central maps", "full-domain central maps"))
            self.assertEqual(full["inputs"], cropped["inputs"])
            # Once full domain is explicit, ignored crop edits cannot alter it.
            run["movie_view"] = {"x": [10, 11], "y": [10, 11]}
            self.assertEqual(movies.source_signature(folder, [silo], args, run), full)
            self.assertEqual(movies.movie_view({"length": 2}, movies.effective_movie_view(args, run["movie_view"])),
                             ([-1., 1., -1., 1.], None))

    def test_full_domain_scope_keeps_physical_endpoint_qualification(self):
        args = argparse.Namespace(full_domain=True)
        scope = "Early expansion through 4000 s. The original 22000 s late-time blast benchmark is outside this campaign window."
        self.assertEqual(movies.effective_physical_scope(args, {"case": "radiating_blast", "physical_scope": scope}), scope)
        self.assertIsNone(movies.effective_physical_scope(args, {}))

    def test_crop_validation_preserves_a_fixed_subset_of_physical_domain(self):
        metadata = {"length": 7e10}
        view = {"x": [-3.5e10, -1.4e10], "y": [-1.05e10, 1.05e10], "reason": "Shock and inner precursor"}
        bounds, normalized = movies.movie_view(metadata, view)
        self.assertEqual(bounds, [-3.5e10, -1.4e10, -1.05e10, 1.05e10])
        self.assertEqual(normalized, view)
        self.assertEqual(movies.movie_view(metadata, None), ([-3.5e10, 3.5e10, -3.5e10, 3.5e10], None))
        for invalid in ({"x": [-4e10, 0], "y": [0, 1]},
                        {"x": [1, 0], "y": [0, 1]},
                        {"x": [0, float("inf")], "y": [0, 1]}):
            with self.assertRaises(ValueError):
                movies.movie_view(metadata, invalid)

    def test_private_encoding_links_saved_states_and_only_writes_a_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frames = []
            for index, time in enumerate([0., .3, 1.]):
                image = folder / f"radiation_{index:06d}.png"
                image.write_bytes(b"existing image fixture")
                frames.append({"time": time, "radiation_image": image.name})
            source = folder / "frames.json"
            source.write_text(json.dumps({"status": "complete", "frames": frames, "panels": ["radiation"]}))
            args = argparse.Namespace(seconds=12, fps=24, hold=1, ffmpeg="ffmpeg")
            plan = movies.prepare_private_encoding(source, folder / "private", args)
            self.assertEqual(plan["playback"]["seconds"], 12.)
            self.assertFalse(plan["publication_allowed"])
            self.assertFalse(Path(plan["output"]).exists())
            self.assertTrue((folder/"private/radiation_000000.png").is_symlink())
            self.assertEqual((folder/"radiation_000000.png").read_bytes(), b"existing image fixture")

    def test_numerical_series_excludes_analytic_and_sorts_numeric_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("X.10.silo", "X.2.silo", "X.0.silo", "final.silo", "analytic.silo"):
                (folder / name).write_bytes(b"fixture")
            self.assertEqual([p.name for p in movies.numerical_silos(folder)],
                             ["X.0.silo", "X.2.silo", "X.10.silo", "final.silo"])

    def test_job_generation_is_source_only_and_retains_normalized_units(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            metadata = {**self.model_metadata(), "dx": 1/32, "length": 1, "time": 1,
                        "units": {"system": "dimensionless", "energy_density": "normalized energy density",
                                  "length": "normalized length", "time": "normalized time"}}
            visit = folder / "bin/visit"
            visit.parent.mkdir()
            visit.touch()
            palette = folder / "3.4.2/linux-x86_64/resources/colortables/viridis.ct"
            palette.parent.mkdir(parents=True)
            palette.write_text('<Object name="ColorTable"><Object name="ColorControlPointList">'
                               '<Object name="ColorControlPoint"><Field name="position">0</Field>'
                               '<Field name="colors">68 1 84 255</Field></Object>'
                               '<Object name="ColorControlPoint"><Field name="position">1</Field>'
                               '<Field name="colors">253 231 37 255</Field></Object></Object></Object>')
            args = argparse.Namespace(width=900, height=800, vectors=144, arrow_stride=0, visit=str(visit))
            request = movies.write_visit_job(metadata, [folder/"X.0.silo", folder/"final.silo"],
                                            folder/"private", args, "private diagnostic", coupled=False)
            self.assertEqual(request["time_units"], "normalized time")
            self.assertEqual(request["length_units"], "normalized length")
            self.assertFalse(request["coupled"])
            self.assertEqual(request["flux_label"], "Derived FLD flux")
            self.assertEqual(request["radiation_flux_semantics"], "derived_fld_flux")
            ast.parse((folder/"private/render.py").read_text())
            self.assertFalse((folder/"private/movie.mp4").exists())
            self.assertEqual(json.loads((folder/"private/request.json").read_text()), request)
            args.full_domain = True
            full_request = movies.write_visit_job(
                metadata, [folder/"X.0.silo", folder/"final.silo"], folder/"full", args,
                "full domain diagnostic", coupled=False,
                view={"x": [-10, -9], "y": [9, 10], "reason": "Ignored stale crop"})
            self.assertEqual(full_request["view_bounds"], [-.5, .5, -.5, .5])
            self.assertIsNone(full_request["movie_view"])
            self.assertEqual(full_request["view_policy"], "full_domain")
            self.assertTrue(full_request["full_domain_forced"])
            self.assertTrue(full_request["profile_full_domain"])
            ast.parse((folder/"full/render.py").read_text())

    def test_moving_fld_job_generation_preserves_coupled_panels_and_lab_flux(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            visit = folder / "bin/visit"
            visit.parent.mkdir()
            visit.touch()
            palette = folder / "3.4.2/linux-x86_64/resources/colortables/viridis.ct"
            palette.parent.mkdir(parents=True)
            palette.write_text('<Object name="ColorTable">'
                               '<Object name="ColorControlPoint"><Field name="position">0</Field>'
                               '<Field name="colors">68 1 84 255</Field></Object>'
                               '<Object name="ColorControlPoint"><Field name="position">1</Field>'
                               '<Field name="colors">253 231 37 255</Field></Object></Object>')
            args = argparse.Namespace(width=900, height=800, vectors=144, arrow_stride=0,
                                      visit=str(visit), full_domain=True)
            for case in movies.COUPLED:
                with self.subTest(case=case):
                    metadata = dict(self.model_metadata(), case=case, radiation_algorithm_version=2,
                        radiation_flux_semantics="derived_fld_lab_flux", radiation_frame="mixed",
                        radiation_hydro_scheme="first_order_conservative", reference_kind="unavailable",
                        reference_model="none", reference_exact_for_solver=False, reference_sampling="none",
                        dx=1/32, length=1, time=1,
                        units={"system": "CGS", "energy_density": "erg/cm^3", "length": "cm", "time": "s"})
                    request = movies.write_visit_job(metadata, [folder/"X.0.silo", folder/"final.silo"],
                        folder/case, args, "moving FLD source-only test", coupled=True)
                    self.assertTrue(request["coupled"])
                    self.assertEqual(request["temperature_profile"], case in movies.SHOCKS)
                    self.assertEqual(request["radiation_algorithm_version"], 2)
                    self.assertEqual(request["radiation_flux_semantics"], "derived_fld_lab_flux")
                    self.assertEqual(request["flux_label"], "Derived lab-frame FLD flux")
                    self.assertEqual(request["radiation_frame"], "mixed")
                    self.assertEqual(request["time_units"], "s")
                    ast.parse((folder/case/"render.py").read_text())
                    self.assertFalse((folder/case/"movie.mp4").exists())
            with self.assertRaisesRegex(ValueError, "Stationary FLD"):
                movies.write_visit_job(self.model_metadata(), [], folder/"invalid", args,
                                       "invalid stationary coupled panel", coupled=True)
            self.assertFalse((folder/"invalid").exists())


if __name__ == "__main__":
    unittest.main()
