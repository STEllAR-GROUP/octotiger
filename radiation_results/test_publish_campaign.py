#!/usr/bin/env python3
"""Local website staging checks; synthetic data, no solver or network calls."""
import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import publish_campaign as publisher


class LocalWebsiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='m1-fld-website-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'campaign'
        self.root.mkdir()
        self.destination = Path(temporary.name) / 'site'
        self.run = {'id': 'l0-gaussian_pulse-fld', 'case': 'gaussian_pulse', 'level': 0,
                    'model': 'fld', 'method': 'fld', 'integrator': 'backward_euler', 'status': 'completed',
                    'directory': 'runs/l0/gaussian_pulse/fld'}
        self.manifest = {'schema': 2, 'workflow': 'm1-fld-radiation', 'active': False,
                         'runs': [self.run]}
        self.summary = {'cases': {'gaussian_pulse': [{'status': 'fld_only'}]}}
        for relative, text in {'index.html': '<h1>M1 RK1 and FLD</h1>', 'campaign.json': '{}',
                'commands.txt': 'prepared commands', 'comparisons/index.html': 'comparison',
                'comparisons/.matplotlib-cache/fontlist.json': 'private cache',
                '.campaign/runtime/octotiger': 'private binary',
                self.run['directory'] + '/index.html': 'model report',
                self.run['directory'] + '/gaussian_pulse.html': '<main><h4>Movies</h4><p class="muted">No completed movie is available for this resolution.</p></main>',
                self.run['directory'] + '/run.json': '{}',
                self.run['directory'] + '/slice.csv': 'er,er_ref\n1,1',
                self.run['directory'] + '/reference.bin': 'analytic reference',
                self.run['directory'] + '/L1.dat': 'reference norm',
                self.run['directory'] + '/final.silo': 'private checkpoint'}.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        for name, result in [('load', self.manifest), ('verify_snapshot', None),
                             ('validate_completed', {'origin': 'octotiger'})]:
            mock = patch.object(publisher, name, return_value=result)
            self.addCleanup(mock.stop); mock.start()
        mock = patch.object(publisher.compare_campaign, 'validate_manifest')
        self.addCleanup(mock.stop); mock.start()
        mock = patch.object(publisher.compare_campaign, 'refresh', return_value=self.summary)
        self.addCleanup(mock.stop); mock.start()

    def stage(self):
        return publisher.build_site(self.root, self.destination)

    def make_movie(self):
        folder = self.root / self.run['directory'] / 'gaussian_pulse/l0'
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {'case': 'gaussian_pulse', 'radiation_model': 'fld',
                    'radiation_integrator': 'backward_euler', 'radiation_algorithm_version': 1,
                    'radiation_flux_semantics': 'derived_fld_flux',
                    'reference_kind': 'linear-diffusion-limit',
                    'reference_model': 'periodic-constant-diffusion', 'reference_exact_for_solver': False}
        publisher.write_json(folder / 'run.json', metadata)
        self.run['run_metadata_sha256'] = publisher.file_hash(folder / 'run.json')
        self.movie_folder = folder / publisher.MOVIE_SUBDIR
        self.movie_folder.mkdir(parents=True)
        assets = {}
        for name in ('movie.mp4', 'poster.png'):
            path = self.movie_folder / name
            path.write_bytes(b'synthetic movie asset for staging tests')
            assets[name] = {'bytes': path.stat().st_size, 'sha256': publisher.file_hash(path)}
        manifest = {'schema': 2, 'status': 'complete', 'run_id': self.run['id'],
                    'case': 'gaussian_pulse', 'level': 0, 'method': 'fld',
                    'run_metadata_sha256': self.run['run_metadata_sha256'],
                    'assets': assets, 'frames': 3, 'source_snapshots': 3,
                    'displayed_snapshots': 3, 'snapshot_times': [0, .125, .25], 'time_units': 's',
                    'timing': 'Actual Silo time gaps; stored states held without solution interpolation; endpoint holds included',
                    'description': 'Numerical radiation energy & flux',
                    **publisher.model_contract(metadata, self.run)}
        publisher.write_json(self.movie_folder / 'campaign-movie.json', manifest)
        return manifest

    def test_local_stage_contains_reports_and_receipt_but_no_private_inputs(self):
        receipt = self.stage()
        self.assertFalse(receipt['remote_deployment'])
        self.assertEqual(receipt['status'], 'staged_locally')
        self.assertEqual(receipt['completed_runs'], 1)
        self.assertTrue((self.destination / self.run['directory'] / 'slice.csv').is_file())
        self.assertTrue((self.destination / self.run['directory'] / 'reference.bin').is_file())
        self.assertTrue((self.destination / self.run['directory'] / 'L1.dat').is_file())
        self.assertTrue(receipt['local_links_verified'])
        for relative in ('.campaign', 'comparisons/.matplotlib-cache',
                         self.run['directory'] + '/final.silo'):
            self.assertFalse((self.destination / relative).exists(), relative)
        saved = json.loads((self.destination / 'site-manifest.json').read_text())
        self.assertEqual(saved, receipt)
        for name, digest in receipt['file_sha256'].items():
            self.assertEqual(publisher.file_hash(self.destination / name), digest)

    def test_empty_active_and_synthetic_results_are_rejected(self):
        for update, text in (({'active': True}, 'current run'), ({'runs': []}, 'No completed')):
            original = copy.deepcopy(self.manifest)
            self.manifest.update(update)
            with self.assertRaisesRegex(ValueError, text):
                self.stage()
            self.manifest.clear(); self.manifest.update(original)
        for metadata in ({'origin': 'synthetic fixture'}, {'synthetic': True}):
            with patch.object(publisher, 'validate_completed', return_value=metadata):
                with self.assertRaisesRegex(ValueError, 'Synthetic'):
                    self.stage()
        self.assertFalse(self.destination.exists())

    def test_invalid_comparison_and_changed_provenance_prevent_staging(self):
        for status in ('invalid_output', 'incompatible'):
            self.summary['cases']['gaussian_pulse'][0]['status'] = status
            with self.assertRaisesRegex(ValueError, 'comparison output'):
                self.stage()
        with patch.object(publisher, 'verify_snapshot', side_effect=ValueError('changed snapshot')):
            with self.assertRaisesRegex(ValueError, 'changed snapshot'):
                self.stage()
        self.assertFalse(self.destination.exists())

    def test_existing_or_nested_site_is_rejected(self):
        self.destination.mkdir(); (self.destination / 'keep.txt').write_text('user data')
        with self.assertRaisesRegex(ValueError, 'new or empty'):
            self.stage()
        self.assertEqual((self.destination / 'keep.txt').read_text(), 'user data')
        for destination in (self.root, self.root / 'site'):
            with self.assertRaisesRegex(ValueError, 'separate'):
                publisher.build_site(self.root, destination)

    def test_symlinks_in_results_are_rejected(self):
        (self.root / self.run['directory'] / 'private.txt').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_campaign_lock_blocks_staging_and_is_held_during_copy(self):
        with (self.root / '.campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'locked'):
                self.stage()
        self.assertFalse(self.destination.exists())
        original = publisher.prepare_html_and_validate_links
        def verify_while_locked(site):
            with (self.root / '.campaign.lock').open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return original(site)
        with patch.object(publisher, 'prepare_html_and_validate_links', side_effect=verify_while_locked):
            self.stage()
        with (self.root / '.campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_reference_downloads_remain_live_and_checkpoint_links_become_visible_text(self):
        folder = self.root / self.run['directory']
        source = ('<!doctype html><html><body><a href="reference.bin">Analytic reference</a>'
                  '<a href="L1.dat">L1 norms</a><a href="final.silo">Final numerical Silo</a>'
                  '<a href="checkpoint/restart.silo"><b>Restart</b></a>'
                  '<a href="https://example.org/paper.silo">External reference</a>'
                  '<a href="#details">Details</a><p id="details">A &amp; B</p></body></html>')
        (folder / 'index.html').write_text(source)
        checkpoint = folder / 'checkpoint/private.json'
        checkpoint.parent.mkdir(); checkpoint.write_text('{}')
        self.stage()
        saved = (self.destination / self.run['directory'] / 'index.html').read_text()
        self.assertIn('href="reference.bin"', saved)
        self.assertIn('href="L1.dat"', saved)
        self.assertNotIn('href="final.silo"', saved)
        self.assertNotIn('href="checkpoint/restart.silo"', saved)
        self.assertIn('Final numerical Silo (local file; excluded from staged site)', saved)
        self.assertIn('<b>Restart</b> (local file; excluded from staged site)', saved)
        self.assertIn('href="https://example.org/paper.silo"', saved)
        self.assertIn('A &amp; B', saved)
        self.assertFalse((self.destination / self.run['directory'] / 'checkpoint').exists())
        self.assertEqual((folder / 'index.html').read_text(), source)

    def test_missing_assets_and_escaping_links_fail_without_partial_destination(self):
        page = self.root / 'index.html'
        for markup in ('<a href="missing.csv">Data</a>', '<img src="missing.png">',
                       '<video poster="missing.png"></video>', '<a href="../outside.txt">Outside</a>',
                       '<a href="#missing">Anchor</a>'):
            page.write_text(markup)
            with self.subTest(markup=markup), self.assertRaisesRegex(ValueError, 'public|Public'):
                self.stage()
            self.assertFalse(self.destination.exists())

    def test_encoded_paths_queries_and_cross_page_anchors_are_checked(self):
        (self.root / 'index.html').write_text('<a href="comparisons/index.html?view=full#metrics">Metrics</a>')
        (self.root / 'comparisons/index.html').write_text('<h1 id="metrics">Metrics</h1><img src="plot%20one.png">')
        (self.root / 'comparisons/plot one.png').write_bytes(b'fixture image')
        self.stage()
        self.assertTrue((self.destination / 'comparisons/plot one.png').is_file())

    def test_destination_populated_during_staging_is_not_overwritten(self):
        original = publisher.prepare_html_and_validate_links
        def concurrent_output(site):
            self.destination.mkdir()
            (self.destination / 'keep.txt').write_text('user data')
            return original(site)
        with patch.object(publisher, 'prepare_html_and_validate_links', side_effect=concurrent_output):
            with self.assertRaises(OSError):
                self.stage()
        self.assertEqual((self.destination / 'keep.txt').read_text(), 'user data')
        self.assertEqual([p.name for p in self.destination.iterdir()], ['keep.txt'])

    def test_movie_gallery_embeds_verified_assets_without_changing_run_inputs(self):
        self.make_movie()
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        scratch = self.movie_folder / 'renders/private/frame0000.png'
        scratch.parent.mkdir(parents=True); scratch.write_bytes(b'private rendering scratch')
        receipt = self.stage()
        self.assertEqual(receipt['completed_movies'], 1)
        self.assertEqual(receipt['movie_run_ids'], [self.run['id']])
        gallery = (self.destination / 'movies/index.html').read_text()
        self.assertIn('1 movies available from 1 completed runs', gallery)
        self.assertIn('3 stored states; 3 shown', gallery)
        self.assertIn('Zero-order hold; no solution interpolation', gallery)
        self.assertIn('0, 0.125, 0.25 s', gallery)
        self.assertIn('derived FLD diagnostic flux', gallery)
        self.assertIn('not an exact solution of nonlinear', gallery)
        self.assertIn('preload="none"', gallery)
        self.assertIn('controls', gallery)
        root_page = (self.destination / 'index.html').read_text()
        self.assertIn('Open the movie gallery', root_page)
        report = (self.destination / self.run['directory'] / 'gaussian_pulse.html').read_text()
        self.assertIn('<video controls', report)
        self.assertNotIn('No completed movie is available', report)
        staged = self.destination / self.movie_folder.relative_to(self.root)
        self.assertEqual(sorted(path.name for path in staged.iterdir()), ['campaign-movie.json', 'movie.mp4', 'poster.png'])
        self.assertTrue(receipt['local_links_verified'])
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_movie_asset_and_metadata_hash_mismatches_prevent_staging(self):
        original = self.make_movie()
        changed = copy.deepcopy(original)
        changed['run_metadata_sha256'] = '0' * 64
        publisher.write_json(self.movie_folder / 'campaign-movie.json', changed)
        with self.assertRaisesRegex(ValueError, 'match the completed run'):
            self.stage()
        publisher.write_json(self.movie_folder / 'campaign-movie.json', original)
        (self.movie_folder / 'movie.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'asset is missing or changed'):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_invalid_movie_model_timing_and_symlinks_are_rejected(self):
        original = self.make_movie()
        for updates, error in (({'radiation_flux_semantics': 'evolved_m1_flux'}, 'identity changed'),
                ({'snapshot_times': [0, .25, .125]}, 'stored-state movie timing'),
                ({'source_snapshots': 2}, 'stored-state movie timing'),
                ({'displayed_snapshots': 0}, 'stored-state movie timing'),
                ({'timing': 'linear interpolation'}, 'stored-state movie timing')):
            publisher.write_json(self.movie_folder / 'campaign-movie.json', {**original, **updates})
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, error):
                self.stage()
        publisher.write_json(self.movie_folder / 'campaign-movie.json', original)
        (self.movie_folder / 'movie.mp4').unlink()
        (self.movie_folder / 'movie.mp4').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(ValueError, 'asset is missing or changed'):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_unfinished_movie_cache_is_not_published_or_claimed(self):
        folder = self.root / self.run['directory'] / 'gaussian_pulse/l0' / publisher.MOVIE_SUBDIR
        folder.mkdir(parents=True)
        (folder / 'movie.mp4').write_bytes(b'unfinished')
        (folder / 'poster.png').write_bytes(b'unfinished')
        receipt = self.stage()
        self.assertEqual(receipt['completed_movies'], 0)
        self.assertFalse((self.destination / 'movies').exists())
        self.assertFalse((self.destination / folder.relative_to(self.root)).exists())

    def test_partial_movie_gallery_reports_available_count(self):
        self.make_movie()
        self.manifest['runs'].append({**self.run, 'id': 'l1-gaussian_pulse-fld', 'level': 1,
                                      'directory': 'runs/l1/gaussian_pulse/fld'})
        receipt = self.stage()
        self.assertEqual(receipt['completed_movies'], 1)
        self.assertEqual(receipt['completed_runs'], 2)
        gallery = (self.destination / 'movies/index.html').read_text()
        self.assertIn('1 movies available from 2 completed runs', gallery)
        self.assertIn('1 completed runs do not yet have a published movie', gallery)


if __name__ == '__main__':
    unittest.main()
