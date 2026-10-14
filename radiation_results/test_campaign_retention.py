#!/usr/bin/env python3
"""Synthetic retention receipts only; no Silo readers, renderer, or solver."""
import json
from pathlib import Path
import tempfile
import unittest

import campaign_retention as retention


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prepare_outputs(root, run, count=3):
    folder = retention.leaf(root, run)
    folder.mkdir(parents=True, exist_ok=True)
    times = [run['stop_time'] * index / (count - 1) for index in range(count)]
    for name in [*(f'X.{index}.silo' for index in range(count)), 'final.silo']:
        (folder / name).write_bytes(('synthetic ' + name).encode())
        data = folder / (name + '.data')
        data.mkdir(exist_ok=True)
        (data / '0.silo').write_bytes(('synthetic data ' + name).encode())
    (folder / 'checkpoint.final').write_bytes(b'preserve restart data')
    destination = folder / 'movies/radiation-energy-flux'
    destination.mkdir(parents=True, exist_ok=True)
    profiles, frames = [], []
    for index, time in enumerate(times):
        profile = destination / f'profile-{index}.json'
        write_json(profile, {'schema': 1, 'time': time, 'columns': ['x', 'er', 'fx', 'fy', 'fz', 'egas'],
                             'rows': [[0, 1, 0, 0, 0, 1]]})
        image = destination / f'frame-{index}.png'
        image.write_bytes(b'synthetic image')
        profiles.append({'time': time, **retention.file_record(folder, profile)})
        frames.append({'time': time, 'panel': 'radiation', **retention.file_record(folder, image)})
    for name in ('movie.mp4', 'poster.png'):
        (destination / name).write_bytes(b'synthetic verified media')
    settings = {'seconds': 60, 'fps': 24, 'full_domain': True}
    inputs = []
    for silo in [*(folder / f'X.{index}.silo' for index in range(count)), folder / 'final.silo']:
        for path in [silo, *Path(str(silo) + '.data').rglob('*')]:
            if path.is_file():
                inputs.append({'path': str(path.relative_to(folder)), 'size': path.stat().st_size,
                               'mtime_ns': path.stat().st_mtime_ns})
    movie = {'status': 'complete', 'run_metadata_sha256': run['run_metadata_sha256'],
        'source_snapshots': count, 'displayed_snapshots': count, 'snapshot_times': times,
        'signature': {**settings, 'inputs': inputs}, 'assets': {name: retention.file_record(destination, destination / name)
            for name in ('movie.mp4', 'poster.png')}}
    write_json(destination / 'campaign-movie.json', movie)
    write_json(destination / 'volume_integrals.json', {'schema': 1, 'status': 'verified',
        'scope': 'full-volume stored numerical states', 'integration': 'full-volume leaf-cell', 'records': [{'time': time,
            'volume': 1, 'physical_c': 1, 'gas_energy': 1, 'radiation_energy': 1,
            'gas_momentum': [0, 0, 0], 'radiation_momentum': [0, 0, 0]} for time in times]})
    extraction = {'schema': 1, 'status': 'verified', 'settings': settings,
        'run_metadata_sha256': run['run_metadata_sha256'], 'snapshot_times': times, 'profiles': profiles,
        'panels': ['radiation'], 'frames': frames,
        'volume_integrals': retention.file_record(folder, destination / 'volume_integrals.json'),
        'movie': retention.file_record(folder, destination / 'movie.mp4'),
        'movie_metadata': retention.file_record(folder, destination / 'campaign-movie.json')}
    write_json(destination / 'extraction.json', extraction)
    return extraction


class RetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='retention-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = {'id': 'l2-example-m1', 'directory': 'runs/m1', 'case': 'example', 'level': 2,
                    'stop_time': 1.5, 'run_metadata_sha256': 'synthetic',
                    'attempts': [{'pid': 100, 'process_start_ticks': '123'}]}
        self.manifest = {'snapshots': 3}
        self.owner = {'directory': 'runs/m1', 'claimed_utc': '2026-10-14T12:00:00Z',
                      'solver_identity': {'pid': 100, 'process_start_ticks': '123'}}
        self.extraction = prepare_outputs(self.root, self.run)
        self.folder = retention.leaf(self.root, self.run)

    def authorize(self):
        receipt = retention.make_receipt(self.root, self.manifest, self.run, self.owner)
        path = self.folder / 'retention.json'
        write_json(path, receipt)
        self.run['retention'] = {'path': str(path.relative_to(self.root)), 'sha256': retention.digest(path)}
        return receipt

    def test_only_intermediate_raw_is_removed_and_receipt_resumes(self):
        receipt = self.authorize()
        removed_bytes = retention.prune(self.root, self.run)
        self.assertEqual(removed_bytes, sum(record['bytes'] for record in receipt['removed_raw']))
        self.assertFalse((self.folder / 'X.1.silo').exists())
        self.assertFalse((self.folder / 'X.1.silo.data').exists())
        for name in ('X.0.silo', 'X.2.silo', 'final.silo', 'checkpoint.final'):
            self.assertTrue((self.folder / name).is_file())
        for record in receipt['extracted_artifacts']:
            self.assertTrue((self.folder / record['path']).is_file())
        retention.prune(self.root, self.run)
        cached = retention.validate_retained_cache(self.root, self.run, self.extraction['settings'])
        self.assertEqual(cached['source_snapshots'], 3)

    def remote_import(self):
        plan = self.root / '.campaign/plan.json'
        write_json(plan, {'synthetic': 'frozen plan'})
        prefix = self.run['case'] + '/l' + str(self.run['level']) + '/'
        files = {prefix + str(path.relative_to(self.folder)): {'sha256': retention.digest(path),
            'bytes': path.stat().st_size} for path in self.folder.rglob('*') if path.is_file()}
        receipt = {'schema': 1, 'status': 'complete', 'owner': 'octotiger-m1-fld-qb4-20261014',
            'nodes': 1, 'tasks': 1, 'job_id': '102', 'node': 'qb-compute1', 'run_id': self.run['id'],
            'run_metadata_sha256': self.run['run_metadata_sha256'], 'files': files}
        wrapper = {'schema': 1, 'status': 'verified', 'owner': receipt['owner'], 'run_id': self.run['id'],
            'directory': self.run['directory'], 'plan_sha256': retention.digest(plan),
            'imported_utc': '2026-10-14T12:00:00Z', 'receipt': receipt}
        path = self.root / '.imports' / (self.run['id'] + '.json')
        write_json(path, wrapper)
        self.run['remote_import'] = {'path': str(path.relative_to(self.root)), 'sha256': retention.digest(path)}
        self.run['attempts'] = []
        self.owner = retention.remote_owner(self.root, self.run)
        return path

    def test_verified_remote_import_owns_raw_without_local_pid(self):
        self.remote_import()
        self.assertNotIn('solver_identity', self.owner)
        self.authorize()
        retention.prune(self.root, self.run)
        self.assertFalse((self.folder / 'X.1.silo').exists())
        self.assertTrue((self.folder / 'X.0.silo').exists())
        self.assertTrue((self.folder / 'final.silo').exists())
        self.assertTrue((self.folder / 'checkpoint.final').exists())
        retention.validate_retention(self.root, self.run)

    def test_remote_receipt_tampering_and_fabricated_pid_never_authorize_removal(self):
        path = self.remote_import()
        self.owner['solver_identity'] = {'pid': 100, 'process_start_ticks': '123'}
        with self.assertRaisesRegex(ValueError, 'not a local solver PID'):
            self.authorize()
        self.owner.pop('solver_identity')
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            self.authorize()
        self.assertTrue((self.folder / 'X.1.silo').exists())

    def test_remote_raw_inventory_cannot_adopt_files_added_after_import(self):
        self.remote_import()
        (self.folder / 'checkpoint.extra').write_text('Unowned later checkpoint')
        with self.assertRaisesRegex(ValueError, 'verified imported file inventory'):
            self.authorize()
        self.assertTrue((self.folder / 'X.1.silo').exists())

    def test_no_receipt_means_no_deletion_and_no_cache_override(self):
        with self.assertRaisesRegex(ValueError, 'committed'):
            retention.prune(self.root, self.run)
        self.assertIsNone(retention.validate_retained_cache(self.root, self.run, {}))
        self.assertTrue((self.folder / 'X.1.silo').is_file())

    def test_receipt_reference_cannot_be_removed_to_bypass_validation(self):
        self.authorize()
        self.run.pop('retention')
        with self.assertRaisesRegex(ValueError, 'without its committed'):
            retention.validate_retention(self.root, self.run)

    def test_unowned_outputs_cannot_be_authorized(self):
        with self.assertRaisesRegex(ValueError, 'newly owned'):
            retention.make_receipt(self.root, self.manifest, self.run, {})

    def test_missing_profile_or_frame_blocks_authorization(self):
        path = self.folder / self.extraction['profiles'][1]['path']
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            self.authorize()
        self.assertTrue((self.folder / 'X.1.silo').exists())

    def test_missing_snapshot_and_missing_panel_are_rejected(self):
        path = self.folder / 'movies/radiation-energy-flux/extraction.json'
        self.extraction['frames'].pop()
        write_json(path, self.extraction)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.authorize()

    def test_fewer_than_all_requested_samples_are_rejected(self):
        self.manifest['snapshots'] = 121
        with self.assertRaisesRegex(ValueError, 'every requested'):
            self.authorize()

    def test_tampered_receipt_or_preserved_raw_or_artifact_blocks_resume(self):
        self.authorize()
        (self.folder / 'final.silo').write_bytes(b'changed endpoint')
        with self.assertRaisesRegex(ValueError, 'changed'):
            retention.prune(self.root, self.run)
        self.assertTrue((self.folder / 'X.1.silo').exists())

    def test_changed_cache_settings_refuse_regeneration_after_pruning(self):
        self.authorize(); retention.prune(self.root, self.run)
        with self.assertRaisesRegex(ValueError, 'cannot be rerendered'):
            retention.validate_retained_cache(self.root, self.run, {'seconds': 120})

    def test_symlink_and_parent_traversal_are_rejected(self):
        (self.folder / 'X.1.silo').unlink()
        (self.folder / 'X.1.silo').symlink_to(self.folder / 'final.silo')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.authorize()
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            retention.safe_path(self.folder, '../outside')

    def test_interrupted_removal_is_safe_and_idempotently_resumed(self):
        self.authorize()
        calls = 0
        def interrupted():
            nonlocal calls
            calls += 1
            if not (self.folder / 'X.1.silo').exists():
                raise RuntimeError('STOP marker')
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            retention.prune(self.root, self.run, interrupted)
        retention.prune(self.root, self.run)
        self.assertFalse((self.folder / 'X.1.silo.data').exists())
        self.assertTrue((self.folder / 'checkpoint.final').exists())

    def test_extra_uninventoried_file_is_never_removed(self):
        self.authorize()
        extra = self.folder / 'X.1.silo.data/keep-me'
        extra.write_text('unowned')
        with self.assertRaises(OSError):
            retention.prune(self.root, self.run)
        self.assertEqual(extra.read_text(), 'unowned')


if __name__ == '__main__':
    unittest.main()
