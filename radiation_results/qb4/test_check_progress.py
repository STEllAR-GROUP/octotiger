#!/usr/bin/env python3
"""Offline safety checks: SSH, Slurm, transfers and publication are mocked."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('check_progress_under_test', Path(__file__).with_name('check_progress.py'))
progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progress)
RETENTION_SPEC = importlib.util.spec_from_file_location('retention_for_progress_tests',
    Path(__file__).resolve().parents[1] / 'campaign_retention.py')
retention_paths = importlib.util.module_from_spec(RETENTION_SPEC)
RETENTION_SPEC.loader.exec_module(retention_paths)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProgressSafety(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.directory = self.base / 'migration'
        self.directory.mkdir()
        self.root = self.base / 'campaign'
        (self.root / '.publication').mkdir(parents=True)
        (self.root / '.campaign/source/src').mkdir(parents=True)
        source = self.root / '.campaign/source/src/a.cpp'
        source.write_text('application source')
        source_map = {'src/a.cpp': digest(source)}
        self.runtime = {'schema': 1, 'owner': progress.OWNER, 'nodes': 1, 'tasks': 1,
            'source_commit': 'a' * 40, 'worker_sha256': 'b' * 64, 'executable_sha256': 'c' * 64,
            'source_sha256': source_map,
            'source_tree_sha256': hashlib.sha256(json.dumps(source_map, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'solver_library_sha256': {'libhpx.so.1': 'd' * 64}, 'build': {'job_id': '100', 'node': 'qb01'}}
        self.remote = self.base / 'remote'
        self.remote.mkdir()
        self.migration = {'owner': progress.OWNER, 'nodes': 1, 'local_campaign': str(self.root),
            'remote_root': str(self.remote), 'control_socket': str(self.base / 'existing-control.sock'),
            'source_commit': self.runtime['source_commit'], 'worker_sha256': self.runtime['worker_sha256']}
        progress.write(self.directory / 'migration.json', self.migration)
        (self.root / 'STOP').write_text('Local solves must remain stopped\n')
        self.run = {'id': 'l2-streaming_front-fld', 'case': 'streaming_front', 'level': 2,
            'model': 'fld', 'method': 'fld', 'integrator': 'backward_euler',
            'directory': 'runs/l2/streaming_front/fld', 'status': 'completed'}
        self.manifest = {'active': False, 'runs': [self.run], 'minimum_free_gib': 1}
        self.publication = {'status': 'completed', 'active': False, 'published': [self.run['id']]}
        self.save_publication()
        self.runner = Mock()
        self.runner.load.side_effect = lambda root: self.manifest
        self.runner.file_hash.side_effect = digest
        self.retention = Mock()
        self.retention.safe_path.side_effect = retention_paths.safe_path
        self.snapshot = {'build': {'status': 'complete'}, 'worker': None, 'receipts': {}}
        self.worker = {'status': 'waiting_for_publication', 'remote_runtime': self.runtime,
            'allocation': {'job_id': '200', 'node': 'qb02', 'nodes': 1, 'tasks': 1},
            'runs': {self.run['id']: {'status': 'complete'}}}
        self.original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.original_path))

    def save_publication(self):
        progress.write(self.root / '.publication/state.json', self.publication)

    def invoke(self, advance=True):
        with patch.dict(sys.modules, {'run_campaign': self.runner, 'campaign_retention': self.retention}), \
                patch.object(progress, 'connection', return_value=['ssh', 'fixture']), \
                patch.object(progress, 'capture', return_value=copy.deepcopy(self.snapshot)), \
                patch.object(progress, 'publish') as publish, patch.object(progress, 'collect') as collect, \
                patch.object(progress, 'submit_next', return_value={'returncode': 0, 'job_id': '201',
                                                                  'status': 'submitted'}) as submit:
            return progress.check(self.directory, advance), publish, collect, submit

    def ledger(self):
        (self.base / '.codex').mkdir(exist_ok=True)
        value = {'active_connection': {'owner': progress.OWNER, 'status': 'connected',
                    'control_path': self.migration['control_socket']},
                 'current_task': {'owner': progress.OWNER, 'task_attempts_reserved': 1,
                                  'user_authorized_task_maximum_logins': 1}}
        progress.write(self.base / '.codex/qb4-access-state.json', value)
        return value

    def test_existing_master_is_reused_even_at_login_ceiling_without_fallback(self):
        self.ledger()
        with patch.object(progress.Path, 'home', return_value=self.base), \
                patch.object(progress.subprocess, 'run', return_value=Mock(returncode=0)) as run:
            argv = progress.connection(self.migration)
        self.assertIn('ControlMaster=no', argv)
        self.assertIn('BatchMode=yes', argv)
        self.assertIn('ProxyCommand=/bin/false', argv)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], [*argv, '-O', 'check', progress.HOST])

    def test_lost_master_fails_without_retry_or_authentication(self):
        self.ledger()
        with patch.object(progress.Path, 'home', return_value=self.base), \
                patch.object(progress.subprocess, 'run', side_effect=subprocess.CalledProcessError(255, ['ssh'])) as run, \
                self.assertRaises(subprocess.CalledProcessError):
            progress.connection(self.migration)
        self.assertEqual(run.call_count, 1)
        self.assertIn('ProxyCommand=/bin/false', run.call_args.args[0])

    def test_foreign_disconnected_or_over_budget_ledger_cannot_contact_host(self):
        original = self.ledger()
        for group, key, value in [('active_connection', 'owner', 'other'),
                ('active_connection', 'status', 'closed'), ('active_connection', 'control_path', '/other'),
                ('current_task', 'owner', 'other'), ('current_task', 'task_attempts_reserved', 2)]:
            ledger = copy.deepcopy(original)
            ledger[group][key] = value
            progress.write(self.base / '.codex/qb4-access-state.json', ledger)
            with self.subTest(key=key, value=value), patch.object(progress.Path, 'home', return_value=self.base), \
                    patch.object(progress.subprocess, 'run') as run, self.assertRaises(ValueError):
                progress.connection(self.migration)
            run.assert_not_called()

    def test_remote_python_preserves_fail_closed_ssh_options_and_one_call(self):
        argv = progress.ssh_arguments(self.migration['control_socket'])
        with patch.object(progress.subprocess, 'run', return_value=Mock(stdout='{"ok":true}')) as run:
            self.assertEqual(progress.remote_python(argv, 'print("test")'), {'ok': True})
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[:len(argv)], argv)
        self.assertIn('ProxyCommand=/bin/false', command)

    def test_missing_local_stop_rejects_before_any_connection(self):
        (self.root / 'STOP').unlink()
        with patch.dict(sys.modules, {'run_campaign': self.runner, 'campaign_retention': self.retention}), \
                patch.object(progress, 'connection') as connect, self.assertRaises(ValueError):
            progress.check(self.directory, True)
        connect.assert_not_called()

    def test_running_local_record_blocks_runtime_pinning(self):
        for active, status in ((True, 'completed'), (False, 'running')):
            self.manifest['active'] = active
            self.run['status'] = status
            self.runner.save.reset_mock()
            with self.subTest(active=active, status=status), self.assertRaises(ValueError):
                progress.pin_runtime(self.root, self.migration, self.runtime, self.runner)
            self.runner.save.assert_not_called()

    def test_runtime_source_worker_and_previous_pin_cannot_change(self):
        for mutation in ({'source_commit': 'f' * 40}, {'worker_sha256': 'f' * 64},
                         {'source_sha256': {'src/a.cpp': 'f' * 64}}):
            bad = {**self.runtime, **mutation}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                progress.pin_runtime(self.root, self.migration, bad, self.runner)
        self.manifest['remote_runtime'] = {**self.runtime, 'executable_sha256': 'f' * 64}
        with self.assertRaisesRegex(ValueError, 'runtime changed'):
            progress.pin_runtime(self.root, self.migration, self.runtime, self.runner)
        self.runner.save.assert_not_called()

    def test_read_only_or_active_publication_does_not_advance(self):
        self.snapshot['worker'] = self.worker
        self.snapshot['receipts'] = {self.run['id']: {'run_id': self.run['id']}}
        for advance, active in ((False, False), (True, True)):
            self.publication['active'] = active
            self.save_publication()
            result, publish, collect, submit = self.invoke(advance)
            self.assertIn(result['status'], ('checked', 'publication-active'))
            publish.assert_not_called()
            collect.assert_not_called()
            submit.assert_not_called()

    def test_waiting_without_completion_receipt_cannot_submit(self):
        self.snapshot['worker'] = self.worker
        with self.assertRaises(ValueError):
            self.invoke()

    def test_running_local_record_blocks_all_advancement_without_worker(self):
        self.run['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'Local solver'):
            self.invoke()

    def test_receipt_mapping_cannot_credit_a_different_completed_run(self):
        self.snapshot['worker'] = self.worker
        self.snapshot['receipts'] = {self.run['id']: {'run_id': 'l2-marshak-fld'}}
        with self.assertRaises(ValueError):
            self.invoke()

    def test_remote_receipt_cannot_relabel_an_already_completed_local_run(self):
        self.snapshot['receipts'] = {self.run['id']: {'run_id': self.run['id']}}
        with self.assertRaises(ValueError):
            self.invoke()

    def test_changed_remote_receipt_is_rejected_after_duplicate_transfer_cleanup(self):
        self.run['remote_import'] = {'path': '.imports/' + self.run['id'] + '.json', 'sha256': 'a' * 64}
        path = self.root / self.run['remote_import']['path']
        path.parent.mkdir()
        progress.write(path, {'receipt': {'run_id': self.run['id'], 'proof': 'original'}})
        self.snapshot['receipts'] = {self.run['id']: {'run_id': self.run['id'], 'proof': 'changed'}}
        with self.assertRaises(ValueError):
            self.invoke()

    def incoming(self, name=None, content=None):
        name = name or self.run['id']
        folder = self.root / '.remote-incoming' / name
        folder.mkdir(parents=True)
        receipt = content or {'run_id': name, 'proof': 'committed'}
        progress.write(folder / 'completion.json', receipt)
        (folder / 'batch').mkdir()
        (folder / 'batch/raw-data').write_text('duplicate transfer')
        self.run['remote_import'] = {'path': '.imports/' + name + '.json', 'sha256': 'a' * 64}
        path = self.root / self.run['remote_import']['path']
        path.parent.mkdir(exist_ok=True)
        progress.write(path, {'receipt': {'run_id': name, 'proof': 'committed'}})
        return folder

    def test_cross_receipt_duplicate_cleanup_is_rejected(self):
        incoming = self.incoming(content={'run_id': self.run['id'], 'proof': 'different run bytes'})
        with patch.object(progress.shutil, 'rmtree') as remove, self.assertRaisesRegex(ValueError, 'receipt differs'):
            self.invoke()
        remove.assert_not_called()
        self.assertTrue((incoming / 'batch/raw-data').is_file())

    def test_unpublished_duplicate_is_preserved(self):
        incoming = self.incoming()
        self.publication['published'] = []
        self.save_publication()
        with patch.object(progress.shutil, 'rmtree') as remove:
            self.invoke()
        remove.assert_not_called()
        self.assertTrue(incoming.is_dir())

    def test_only_verified_published_matching_duplicate_is_cleaned(self):
        incoming = self.incoming()
        with patch.object(progress.shutil, 'rmtree') as remove:
            self.invoke()
        self.runner.validate_completed.assert_called_once_with(self.root, self.manifest, self.run)
        self.retention.validate_retention.assert_called_once_with(self.root, self.run)
        remove.assert_called_once_with(incoming)

    def test_completed_publication_advances_once_and_records_the_successor(self):
        incoming = self.incoming()
        receipt = progress.read(incoming / 'completion.json')
        self.snapshot.update(worker=self.worker, receipts={self.run['id']: receipt})
        result, publish, collect, submit = self.invoke()
        self.assertEqual(result['current_run_job_id'], '201')
        submit.assert_called_once()
        publish.assert_not_called()
        collect.assert_not_called()
        self.assertFalse(incoming.exists())
        saved = progress.read(self.directory / 'migration.json')
        self.assertEqual(saved['advanced_after_job'], '200')
        _, _, _, submit_again = self.invoke()
        submit_again.assert_not_called()

    def test_successor_waits_until_every_received_run_is_published(self):
        incoming = self.incoming()
        self.snapshot.update(worker=self.worker, receipts={self.run['id']: progress.read(incoming / 'completion.json')})
        self.publication['published'] = []
        self.save_publication()
        _, publish, _, submit = self.invoke()
        publish.assert_called_once()
        submit.assert_not_called()
        self.assertTrue(incoming.is_dir())

    def test_symlinked_incoming_parent_cannot_redirect_cleanup(self):
        incoming = self.incoming()
        parent = incoming.parent
        outside = self.base / 'outside-incoming'
        parent.rename(outside)
        parent.symlink_to(outside, target_is_directory=True)
        with patch.object(progress.shutil, 'rmtree') as remove, self.assertRaises(ValueError):
            self.invoke()
        remove.assert_not_called()
        self.assertTrue((outside / incoming.name / 'batch/raw-data').is_file())

    def test_symlinked_incoming_parent_cannot_redirect_transfer(self):
        outside = self.base / 'outside-incoming'
        outside.mkdir()
        (self.root / '.remote-incoming').symlink_to(outside, target_is_directory=True)
        receipt = {'run_id': self.run['id'], 'files': {}}
        with patch.dict(sys.modules, {'campaign_retention': self.retention}), \
                patch.object(progress.subprocess, 'run') as transfer, self.assertRaises(ValueError):
            progress.collect(self.root, self.migration, ['ssh', 'fixture'], receipt, self.runner)
        transfer.assert_not_called()
        self.assertFalse((outside / self.run['id']).exists())

    def run_submission_script(self, result=None, error=None, accounting='200|COMPLETED|0:0|1\n'):
        """Execute only generated Python control flow with sbatch fully mocked."""
        result = result or Mock(returncode=0, stdout='201;cluster\n', stderr='')
        def execute(ssh, script):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    exec(compile(script, '<offline-submission-script>', 'exec'), {})
                except SystemExit as exit:
                    if exit.code not in (None, 0):
                        raise
            return json.loads(output.getvalue())
        def scheduler(command, **kwargs):
            if command[0] == 'sacct':
                self.assertIn('--allocations', command)
                self.assertEqual(command[command.index('--jobs') + 1], '200')
                return Mock(returncode=0, stdout=accounting, stderr='')
            self.assertEqual(command[0], 'sbatch')
            if error is not None:
                raise error
            return result
        return patch.object(progress, 'remote_python', side_effect=execute), \
            patch.object(progress.subprocess, 'run', side_effect=scheduler)

    def test_submission_reservation_prevents_duplicate_sbatch(self):
        remote, subprocess_mock = self.run_submission_script()
        with remote, subprocess_mock as sbatch:
            first = progress.submit_next([], self.migration, self.worker)
            again = progress.submit_next([], self.migration, self.worker)
        self.assertEqual(first['job_id'], '201')
        self.assertEqual(first, again)
        self.assertEqual([call.args[0][0] for call in sbatch.call_args_list], ['sacct', 'sbatch'])
        self.assertFalse(any(arg.startswith('--dependency') for arg in sbatch.call_args.args[0]))
        self.assertEqual(first['previous_job_accounting'], ['200', 'COMPLETED', '0:0', '1'])

    def test_unknown_submission_outcome_remains_reserved_and_never_retries(self):
        remote, subprocess_mock = self.run_submission_script(error=OSError('connection lost during sbatch'))
        with remote, subprocess_mock as sbatch, self.assertRaises(OSError):
            progress.submit_next([], self.migration, self.worker)
        self.assertEqual([call.args[0][0] for call in sbatch.call_args_list], ['sacct', 'sbatch'])
        remote, subprocess_mock = self.run_submission_script()
        with remote, subprocess_mock as sbatch, self.assertRaisesRegex(RuntimeError, 'Unknown'):
            progress.submit_next([], self.migration, self.worker)
        sbatch.assert_not_called()

    def test_unfinished_failed_missing_or_multi_node_predecessor_never_submits(self):
        for accounting in ('200|RUNNING|0:0|1\n', '200|FAILED|1:0|1\n',
                '200|COMPLETED|1:0|1\n', '200|COMPLETED|0:0|2\n', '',
                '200|COMPLETED|0:0|1\n200|COMPLETED|0:0|1\n'):
            remote, subprocess_mock = self.run_submission_script(accounting=accounting)
            with self.subTest(accounting=accounting), remote, subprocess_mock as scheduler, \
                    self.assertRaisesRegex(RuntimeError, 'not confirmed completed'):
                progress.submit_next([], self.migration, self.worker)
            self.assertEqual([call.args[0][0] for call in scheduler.call_args_list], ['sacct'])
            self.assertFalse((self.remote / 'submissions/after-200.json').exists())

    def test_rejected_submission_cannot_be_automatically_retried(self):
        remote, subprocess_mock = self.run_submission_script(Mock(returncode=1, stdout='', stderr='denied'))
        with remote, subprocess_mock as sbatch:
            rejected = progress.submit_next([], self.migration, self.worker)
        self.assertEqual(rejected['returncode'], 1)
        remote, subprocess_mock = self.run_submission_script()
        with remote, subprocess_mock as sbatch, self.assertRaises(RuntimeError):
            progress.submit_next([], self.migration, self.worker)
        sbatch.assert_not_called()

    def test_success_exit_without_job_identity_is_not_claimed_submitted(self):
        remote, subprocess_mock = self.run_submission_script(Mock(returncode=0, stdout='not-a-job\n', stderr=''))
        with remote, subprocess_mock:
            try:
                result = progress.submit_next([], self.migration, self.worker)
            except (ValueError, RuntimeError):
                return
        self.assertNotEqual(result.get('status'), 'submitted')
        self.assertNotEqual(result.get('returncode'), 0)

    def test_foreign_existing_submission_receipt_is_not_reused(self):
        record = self.remote / 'submissions/after-200.json'
        record.parent.mkdir()
        progress.write(record, {'status': 'submitted', 'returncode': 0, 'job_id': '300',
            'dependency': '999', 'nodes': 1, 'tasks': 1, 'cpus': 64})
        remote, subprocess_mock = self.run_submission_script()
        with remote, subprocess_mock as sbatch, self.assertRaises((ValueError, RuntimeError)):
            progress.submit_next([], self.migration, self.worker)
        sbatch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
