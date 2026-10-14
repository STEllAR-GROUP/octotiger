"""Offline synthetic imports only: no SSH, scheduler, solver, or renderer."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qb4 import import_completed as importer


def make_fixture(directory):
    root, batch = directory / 'campaign', directory / 'transferred'
    frozen = root / '.campaign'
    frozen.mkdir(parents=True)
    run = {'id': 'l2-gaussian_pulse-fld', 'case': 'gaussian_pulse', 'level': 2, 'method': 'fld',
        'model': 'fld', 'integrator': 'backward_euler', 'backend': 'cpu', 'directory': 'runs/fld',
        'stop_time': 1.5, 'hard_dt': .00625, 'status': 'pending', 'attempts': []}
    old = {**run, 'id': 'l2-gaussian_pulse-m1', 'method': 'm1', 'model': 'm1', 'integrator': 'euler',
        'directory': 'runs/m1', 'status': 'completed', 'run_metadata_sha256': 'local-unmodified'}
    manifest = {'schema': 2, 'workflow': 'm1-fld-radiation', 'methods': ['m1', 'fld'], 'levels': [2],
        'cases': ['gaussian_pulse'], 'unsupported': [], 'threads': 12, 'snapshots': 3,
        'minimum_free_gib': .001, 'memory_limit_gib': 1, 'original_build': '/local/build',
        'original_results_build': '/local/results', 'original_source': '/local/source',
        'active': False, 'status': 'ready', 'scope': 'Synthetic offline imports only',
        'executable_sha256': 'a' * 64, 'runtime_library_aliases': {},
        'runs': [old, run], 'preparation_checks': {'config_dry_runs': 2, 'solver_runs': 0, 'runtime_help_passed': True}}
    sources = {'src/example.cpp': hashlib.sha256(b'synthetic frozen source').hexdigest()}
    runtime = {'schema': 1, 'owner': importer.OWNER, 'nodes': 1, 'tasks': 1,
        'executable_sha256': 'b' * 64, 'source_commit': 'c' * 40, 'source_sha256': sources,
        'source_tree_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'solver_library_sha256': {'libhpx.so': 'd' * 64}, 'build': {'job_id': '101', 'node': 'qb-node1'}}
    manifest['remote_runtime'] = runtime
    importer.runner.write_json(frozen / 'plan.json', importer.runner.execution_plan(manifest))
    manifest['snapshot_sha256'] = {'.campaign/plan.json': importer.runner.file_hash(frozen / 'plan.json')}
    importer.runner.write_json(root / 'campaign.json', manifest)
    (root / 'STOP').write_text('Local solver must stay stopped.\n')
    local = root / old['directory']
    local.mkdir(parents=True)
    (local / 'keep-local.txt').write_text('Existing local result must not change.\n')
    folder = batch / run['case'] / 'l2'
    folder.mkdir(parents=True)
    config = {'radiation_model': 'fld', 'radiation_integrator': 'backward_euler',
        'max_level': '2', 'min_level': '2', 'stop_time': '1.5', 'hard_dt': '.00625',
        'problem': 'RADIATION_GAUSSIAN_PULSE', 'radiation_backend': 'cpu', 'radiation_reduced_c_factor': '1'}
    (folder / 'run.ini').write_text(''.join(key + '=' + value + '\n' for key, value in config.items()))
    metadata = {'status': 'complete', 'case': run['case'], 'level': 2, 'radiation_model': 'fld',
        'radiation_integrator': 'backward_euler', 'radiation_algorithm_version': 1,
        'radiation_flux_semantics': 'derived_fld_flux', 'executable_sha256': runtime['executable_sha256'],
        'solver_library_sha256': runtime['solver_library_sha256'], 'config': config, 'time': 1.5}
    importer.runner.write_json(folder / 'run.json', metadata)
    importer.runner.write_json(batch / 'batch.json', {'source_sha256': sources})
    (batch / '.runner.lock').touch()
    (folder / 'run.log').write_text('Synthetic completed solve; no simulation was launched.\n')
    (folder / 'radiation-conservation.csv').write_text('t,energy\n0,1\n1.5,1\n')
    (folder / 'checkpoint.final').write_bytes(b'preserve restart data')
    for name in ('X.0.silo', 'X.1.silo', 'X.2.silo', 'final.silo'):
        (folder / name).write_text('synthetic ' + name)
        shards = folder / (name + '.data')
        shards.mkdir()
        (shards / '0.silo').write_text('synthetic data ' + name)
    receipt = {key: value for key, value in runtime.items() if key != 'build'}
    receipt.update(status='complete', run_id=run['id'], case=run['case'], level=2, model='fld',
        integrator='backward_euler', job_id='102', node='qb-node2',
        config_sha256=importer.runner.file_hash(folder / 'run.ini'),
        run_metadata_sha256=importer.runner.file_hash(folder / 'run.json'))
    receipt['files'] = {str(path.relative_to(batch)): {'bytes': path.stat().st_size,
        'sha256': importer.runner.file_hash(path)} for path in batch.rglob('*') if path.is_file()}
    receipt_path = directory / 'completion.json'
    importer.runner.write_json(receipt_path, receipt)
    return root, batch, receipt_path, manifest, receipt


class ImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='qb-import-test-')
        self.addCleanup(temporary.cleanup)
        self.root, self.batch, self.receipt_path, self.manifest, self.receipt = make_fixture(Path(temporary.name))

    def main(self, *flags):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            code = importer.main(['--campaign', str(self.root), '--batch', str(self.batch),
                '--receipt', str(self.receipt_path), *flags])
        return code, json.loads(stream.getvalue())

    def files(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob('*') if path.is_file()}

    def test_default_is_read_only_and_does_not_launch_anything(self):
        before = self.files(self.root)
        with patch('subprocess.Popen', side_effect=AssertionError('No subprocess permitted')):
            code, plan = self.main()
        self.assertEqual(code, 0)
        self.assertTrue(plan['read_only'])
        self.assertTrue(plan['solver_stop_preserved'])
        self.assertEqual(before, self.files(self.root))

    def test_import_is_atomic_idempotent_and_preserves_local_result_and_stop(self):
        before = self.files(self.root / 'runs/m1')
        original_source = self.files(self.batch)
        self.assertEqual(self.main('--import')[1]['status'], 'imported')
        first = self.files(self.root)
        self.assertEqual(self.main('--import')[1]['status'], 'already-imported')
        self.assertEqual(first, self.files(self.root))
        self.assertEqual(before, self.files(self.root / 'runs/m1'))
        self.assertEqual(original_source, self.files(self.batch))
        run = importer.runner.load(self.root)['runs'][1]
        self.assertEqual(run['attempts'], [])
        self.assertEqual(importer.runner.expected_executable(self.root, importer.runner.load(self.root), run), 'b' * 64)
        self.assertEqual(importer.retention.remote_owner(self.root, run)['job_id'], '102')
        self.assertTrue((self.root / 'STOP').exists())

    def test_tampered_missing_extra_and_symlink_files_rejected_before_commit(self):
        folder = self.batch / 'gaussian_pulse/l2'
        for mutation, repair in (
            (lambda: (folder / 'run.log').write_text('tampered'),
             lambda: (folder / 'run.log').write_text('Synthetic completed solve; no simulation was launched.\n')),
            (lambda: (folder / 'extra').write_text('uninventoried'), lambda: (folder / 'extra').unlink()),
            (lambda: (folder / 'link').symlink_to(folder / 'final.silo'), lambda: (folder / 'link').unlink())):
            mutation()
            with self.assertRaises(ValueError):
                self.main('--import')
            repair()
            self.assertFalse((self.root / '.imports').exists())
            self.assertFalse((self.root / 'runs/fld').exists())

    def test_runtime_source_job_and_run_mismatches_rejected(self):
        for key, value in [('executable_sha256', 'e' * 64), ('source_commit', 'f' * 40),
                           ('nodes', 2), ('job_id', ''), ('node', '../bad'), ('model', 'm1'), ('status', 'running')]:
            bad = {**self.receipt, key: value}
            importer.runner.write_json(self.receipt_path, bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.main('--import')
        importer.runner.write_json(self.receipt_path, self.receipt)
        self.assertFalse((self.root / 'runs/fld').exists())

    def test_imported_result_retains_all_samples_and_validates_after_raw_pruning(self):
        from test_campaign_retention import prepare_outputs
        self.main('--import')
        manifest = importer.runner.load(self.root)
        run = manifest['runs'][1]
        prepare_outputs(self.root, run, manifest['snapshots'])
        owner = importer.retention.remote_owner(self.root, run)
        receipt = importer.retention.make_receipt(self.root, manifest, run, owner)
        folder = importer.retention.leaf(self.root, run)
        path = folder / 'retention.json'
        importer.runner.write_json(path, receipt)
        run['retention'] = {'path': str(path.relative_to(self.root)), 'sha256': importer.runner.file_hash(path)}
        importer.runner.write_json(self.root / 'campaign.json', manifest)
        importer.retention.prune(self.root, run)
        self.assertFalse((folder / 'X.1.silo').exists())
        for name in ('X.0.silo', 'X.2.silo', 'final.silo', 'checkpoint.final'):
            self.assertTrue((folder / name).is_file())
        importer.runner.validate_completed(self.root, manifest, run)
        self.assertEqual(self.main('--import')[1]['status'], 'already-imported')
        (folder / 'final.silo').write_text('corrupt retained endpoint')
        with self.assertRaises(ValueError):
            importer.runner.validate_completed(self.root, manifest, run)

    def test_partial_snapshot_inventory_is_not_completion(self):
        bad = copy.deepcopy(self.receipt)
        relative = 'gaussian_pulse/l2/X.1.silo'
        (self.batch / relative).unlink()
        del bad['files'][relative]
        importer.runner.write_json(self.receipt_path, bad)
        with self.assertRaisesRegex(ValueError, 'every numbered snapshot'):
            self.main('--import')

    def test_existing_completed_local_or_unowned_directory_never_overwritten(self):
        destination = self.root / 'runs/fld'
        destination.mkdir(parents=True)
        (destination / 'existing').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'refusing to adopt'):
            self.main('--import')
        self.assertEqual((destination / 'existing').read_text(), 'keep')
        self.manifest['runs'][1]['status'] = 'completed'
        importer.runner.write_json(self.root / 'campaign.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'completed local result'):
            self.main('--import')

    def test_commit_interruption_recovers_only_matching_verified_payload(self):
        with patch.object(importer.runner, 'save', side_effect=RuntimeError('synthetic interrupted commit')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted commit'):
                self.main('--import')
        self.assertTrue((self.root / 'runs/fld/gaussian_pulse/l2/final.silo').is_file())
        self.assertEqual(importer.runner.load(self.root)['runs'][1]['status'], 'pending')
        self.assertEqual(self.main('--import')[1]['status'], 'imported')

    def test_active_and_postprocess_stop_block_import_but_solver_stop_does_not(self):
        self.manifest['active'] = True
        importer.runner.write_json(self.root / 'campaign.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'active'):
            self.main('--import')
        self.manifest['active'] = False
        importer.runner.write_json(self.root / 'campaign.json', self.manifest)
        (self.root / 'POSTPROCESS_STOP').touch()
        with self.assertRaisesRegex(ValueError, 'POSTPROCESS_STOP'):
            self.main('--import')
        (self.root / 'POSTPROCESS_STOP').unlink()
        self.assertEqual(self.main('--import')[1]['status'], 'imported')

    def test_source_mutation_during_copy_never_commits(self):
        original = importer.shutil.copy2
        def altered(source, destination):
            result = original(source, destination)
            if source.name == 'final.silo':
                destination.write_text('changed after copy')
            return result
        with patch.object(importer.shutil, 'copy2', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'hash/size'):
                self.main('--import')
        self.assertFalse((self.root / 'runs/fld').exists())
        self.assertEqual(importer.runner.load(self.root)['runs'][1]['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
