#!/usr/bin/env python3
"""Offline worker tests: synthetic files/processes, never Slurm or a solver."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('one_node', Path(__file__).with_name('one_node_campaign.py'))
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
FULL = json.loads((Path(__file__).parents[1] / 'full-study.json').read_text())


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'campaign'
        self.source, self.build = self.root / 'source', self.root / 'build'
        (self.source / 'radiation_results').mkdir(parents=True)
        worker.write(self.source / 'radiation_results/full-study.json', FULL)
        self.plan = worker.make_plan(FULL)
        self.run = self.plan[0]
        self.identity = {'schema': 1, 'owner': worker.OWNER, 'nodes': 1, 'tasks': 1,
            'build': {'job_id': '1001', 'node': 'qbd-builder'}, 'source_commit': 'a' * 40,
            'source_sha256': {'CMakeLists.txt': 'b' * 64}, 'executable_sha256': 'c' * 64,
            'solver_library_sha256': {'libhpx.so': 'd' * 64}, 'worker_sha256': 'e' * 64}
        self.identity['source_tree_sha256'] = worker.canonical(self.identity['source_sha256'])
        self.job = {'nodes': 1, 'tasks': 1, 'threads': 64, 'node': 'qbd123', 'job_id': '1234'}

    def batch(self, run=None):
        run = run or self.run
        batch = self.output / run['directory']
        leaf = batch / run['case'] / f'l{run["level"]}'
        leaf.mkdir(parents=True)
        config = {'radiation_reduced_c_factor': '1', 'radiation_backend': 'cpu', 'radiation_model': run['model'],
            'radiation_integrator': run['integrator'], 'stop_time': str(run['stop_time']),
            'hard_dt': str(run['hard_dt'] / 2), 'min_level': str(run['level']),
            'max_level': str(run['level']), 'unigrid': 'on'}
        moving = run['model'] == 'fld' and run['case'] in worker.COUPLED
        meta = {'status': 'complete', 'case': run['case'], 'level': run['level'], 'time': run['stop_time'],
            'radiation_model': run['model'], 'radiation_integrator': run['integrator'],
            'radiation_algorithm_version': 2 if moving else 1,
            'radiation_flux_semantics': 'derived_fld_lab_flux' if moving else
                'derived_fld_flux' if run['model'] == 'fld' else 'evolved_m1_flux',
            'executable_sha256': self.identity['executable_sha256'], 'config': config,
            'command': [str(self.output / 'control/launch-solver'), str(self.build / 'octotiger'),
                        '--config_file=' + str(leaf / 'run.ini'), '--hpx:threads=64']}
        if moving:
            meta.update(radiation_frame='mixed', radiation_hydro_scheme='first_order_conservative')
        worker.write(leaf / 'run.json', meta)
        (leaf / 'run.ini').write_text(''.join(f'{key}={value}\n' for key, value in config.items()))
        (leaf / 'run.log').write_text('synthetic completed solver log\n')
        (leaf / 'radiation-conservation.csv').write_text('t,er\n0,1\n')
        # Actual cpp/run.cpp serialization: batch has no threads field; the
        # solver argv and live state record it. Keep these distinct fixtures.
        worker.write(batch / 'batch.json', {'snapshots': 121, 'workflow': 'radiation-results',
            'cases': [run['case']], 'levels': [run['level']], 'time': run['stop_time'],
            'executable_sha256': self.identity['executable_sha256'],
            'source_sha256': self.identity['source_sha256'],
            'options': {'radiation_model': run['model'], 'radiation_integrator': run['integrator']}})
        worker.write(batch / 'live.json', {'active': False, 'snapshots': 121, 'threads': 64,
            'runs': [{'case': run['case'], 'level': run['level'], 'stage': 'Ready'}]})
        (batch / '.runner.lock').touch()
        for name in [f'X.{index}.silo' for index in range(120)] + ['final.silo']:
            (leaf / name).write_bytes(b'synthetic Silo header')
            (leaf / (name + '.data')).mkdir()
            (leaf / (name + '.data') / '0.silo').write_bytes(b'synthetic shard')
        return batch, leaf

    def receipt(self, run=None):
        run = run or self.run
        batch, leaf = self.batch(run)
        result = worker.completion_receipt(batch, run, self.identity, self.job, {'pid': 7, 'process_start_ticks': '12'})
        path = self.output / 'receipts' / (run['id'] + '.json')
        worker.write(path, result)
        return result, path, batch, leaf

    def execute(self, *extra, spawn=None, signal_handlers=None, stop_effect=None, plan=None, job=None):
        signals = {} if signal_handlers is None else signal_handlers
        with patch.object(worker, 'allocation', return_value=job or self.job), \
             patch.object(worker, 'runtime_identity', return_value=self.identity), \
             patch.object(worker, 'memory_ceiling', return_value=2**40), \
             patch.object(worker, 'make_plan', return_value=plan or [self.run]), \
             patch.object(worker, 'check_resources'), \
             patch.object(worker, 'process_identity', return_value={'pid': 10, 'process_start_ticks': '20'}), \
             patch.object(worker, 'stop_owned', side_effect=stop_effect) as stopped, \
             patch.object(worker.subprocess, 'Popen', side_effect=spawn or AssertionError('No subprocess expected')) as launched, \
             patch.object(worker.signal, 'signal', side_effect=lambda sig, handler: signals.update({sig: handler})), \
             patch.object(worker.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            status = worker.main(['--source', str(self.source), '--build', str(self.build),
                                  '--output', str(self.output), *extra])
        return status, launched, stopped

    def test_full_matrix_exact_skips_and_physical_times(self):
        self.assertEqual(len(self.plan), 46)
        self.assertFalse(set(worker.SKIPPED) & {run['id'] for run in self.plan})
        for run in self.plan:
            self.assertEqual(run['snapshots'], 121)
            self.assertEqual(run['stop_time'], FULL['case_settings'].get(run['case'], {}).get('stop_time', .2))
        self.assertEqual(self.plan[0]['id'], 'l2-streaming_front-m1')

    def test_allocation_guard(self):
        env = {'SLURM_JOB_ID': '1234', 'SLURM_NNODES': '1', 'SLURM_NTASKS': '1',
               'SLURM_CPUS_PER_TASK': '64', 'SLURMD_NODENAME': 'qbd123'}
        self.assertEqual(worker.allocation(env, 'qbd123', 64), self.job)
        for key, value in (('SLURM_JOB_ID', ''), ('SLURM_NNODES', '2'), ('SLURM_NTASKS', '2'),
                           ('SLURM_CPUS_PER_TASK', '32'), ('SLURMD_NODENAME', 'login1')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.allocation(dict(env, **{key: value}), 'qbd123', 64)
        with self.assertRaises(ValueError):
            worker.allocation(env, 'qbd123', 32)

    def test_command_one_model_cpu_physical_c_no_render_or_build(self):
        for run in self.plan:
            argv = worker.command(self.source, self.build, self.build / 'radiation-results-tools',
                                  self.output, self.root / 'launcher', run)
            for key, value in (('--threads', '64'), ('--snapshots', '121'), ('--radiation-backend', 'cpu'),
                               ('--radiation-reduced-c-factor', '1'), ('--radiation-integrator', run['integrator'])):
                self.assertEqual(argv[argv.index(key) + 1], value)
            self.assertTrue({'--no-build', '--no-movies', '--no-open'} <= set(argv))
            self.assertNotIn('--resume', argv)

    def test_receipt_includes_every_file_and_hidden_empty_lock(self):
        receipt, path, batch, leaf = self.receipt()
        self.assertEqual(receipt['files']['.runner.lock']['bytes'], 0)
        self.assertEqual(receipt['config_sha256'], worker.digest(leaf / 'run.ini'))
        self.assertEqual(receipt['run_metadata_sha256'], worker.digest(leaf / 'run.json'))
        self.assertEqual(receipt['snapshot_inventory']['count'], 121)
        self.assertTrue(worker.verify_completed(self.output, self.run, self.identity))
        self.assertFalse(path.is_relative_to(batch))

    def test_actual_command_threads_and_live_state_are_required(self):
        batch, leaf = self.batch()
        original = worker.read(leaf / 'run.json')
        for argv in ([], original['command'][:-1], original['command'] + ['--hpx:threads=64'],
                     original['command'][:-1] + ['--hpx:threads=32'],
                     original['command'][:-1] + ['--hpx:threads', '64']):
            with self.subTest(argv=argv):
                worker.write(leaf / 'run.json', dict(original, command=argv))
                with self.assertRaisesRegex(ValueError, 'worker count'):
                    worker.completion_receipt(batch, self.run, self.identity, self.job)
        worker.write(leaf / 'run.json', original)
        live = worker.read(batch / 'live.json')
        for key, value in (('threads', 32), ('active', True), ('snapshots', 3), ('runs', [])):
            with self.subTest(key=key):
                worker.write(batch / 'live.json', dict(live, **{key: value}))
                with self.assertRaisesRegex(ValueError, 'inactive and complete'):
                    worker.completion_receipt(batch, self.run, self.identity, self.job)

    def failed_receipt(self, plan=None):
        def spawn(argv, **kwargs):
            self.batch()
            worker.write(self.output / 'solver-identities' / (self.run['id'] + '.json'),
                {'pid': 987654, 'process_start_ticks': '12', 'session_id': 10,
                 'job_id': self.job['job_id'], 'node': self.job['node']})
            return Mock(pid=10, returncode=0, poll=Mock(return_value=0))
        with patch.object(worker, 'completion_receipt', side_effect=KeyError('threads')):
            status, _, _ = self.execute(spawn=spawn, plan=plan)
        self.assertEqual(status, 75)
        state = worker.read(self.output / 'worker-state.json')
        self.assertEqual(state['runs'][self.run['id']]['job_id'], self.job['job_id'])
        return state

    def test_recovery_writes_only_receipt_with_original_solve_identity(self):
        plan = self.plan[:2]
        self.failed_receipt(plan)
        before = worker.inventory(self.output / self.run['directory'])
        recovery = dict(self.job, job_id='9999', node='qbd-other')
        status, launch, stopped = self.execute('--recover-completed', plan=plan, job=recovery)
        self.assertEqual(status, 0)
        launch.assert_not_called()
        stopped.assert_not_called()
        self.assertEqual(worker.inventory(self.output / self.run['directory']), before)
        path = self.output / 'receipts' / (self.run['id'] + '.json')
        receipt = worker.read(path)
        for key in ('job_id', 'node', 'nodes', 'tasks', 'threads'):
            self.assertEqual(receipt[key], self.job[key])
        self.assertEqual(receipt['solver_identity']['pid'], 987654)
        state = worker.read(self.output / 'worker-state.json')
        self.assertEqual(state['status'], 'waiting_for_publication')
        self.assertFalse(state['active'])
        record = state['runs'][self.run['id']]
        self.assertEqual(record['job_id'], self.job['job_id'])
        self.assertEqual(record['recovery']['allocation'], recovery)
        self.assertTrue(worker.verify_completed(self.output, self.run, self.identity))
        self.assertFalse((self.output / plan[1]['directory']).exists())
        saved = path.read_bytes()
        status, launch, _ = self.execute('--recover-completed', plan=plan, job=recovery)
        self.assertEqual(status, 75)  # Next run is absent; still never launch.
        launch.assert_not_called()
        self.assertEqual(path.read_bytes(), saved)
        self.assertEqual(worker.read(self.output / 'worker-state.json')['runs'][self.run['id']]['job_id'],
                         self.job['job_id'])

    def test_recovery_requires_original_record_and_solver_identity(self):
        original = self.failed_receipt()
        path = self.output / 'worker-state.json'
        for key in ('job_id', 'node', 'threads', 'started_utc', 'command', 'runner_identity'):
            with self.subTest(key=key):
                state = json.loads(json.dumps(original))
                state['runs'][self.run['id']].pop(key)
                worker.write(path, state)
                status, launch, _ = self.execute('--recover-completed', job=dict(self.job, node='other'))
                self.assertEqual(status, 75)
                launch.assert_not_called()
                self.assertFalse((self.output / 'receipts' / (self.run['id'] + '.json')).exists())
        worker.write(path, original)
        (self.output / 'solver-identities' / (self.run['id'] + '.json')).unlink()
        status, launch, _ = self.execute('--recover-completed')
        self.assertEqual(status, 75)
        launch.assert_not_called()

    def test_partial_or_foreign_source_is_not_recovered(self):
        original = self.failed_receipt()
        batch = self.output / self.run['directory']
        leaf = batch / self.run['case'] / f'l{self.run["level"]}'
        meta = worker.read(leaf / 'run.json')
        worker.write(leaf / 'run.json', dict(meta, status='running'))
        status, launch, _ = self.execute('--recover-completed', job=dict(self.job, node='other'))
        self.assertEqual(status, 75)
        launch.assert_not_called()
        worker.write(leaf / 'run.json', meta)
        worker.write(self.output / 'worker-state.json', original)
        document = worker.read(batch / 'batch.json')
        document['source_sha256']['CMakeLists.txt'] = 'f' * 64
        worker.write(batch / 'batch.json', document)
        status, launch, _ = self.execute('--recover-completed', job=dict(self.job, node='other'))
        self.assertEqual(status, 75)
        launch.assert_not_called()
        self.assertFalse((self.output / 'receipts' / (self.run['id'] + '.json')).exists())

    def test_recovery_rejects_foreign_solver_identity_and_active_writer(self):
        original = self.failed_receipt()
        state_path = self.output / 'worker-state.json'
        solver_path = self.output / 'solver-identities' / (self.run['id'] + '.json')
        solver = worker.read(solver_path)
        for key, value in (('job_id', '7777'), ('node', 'foreign'), ('session_id', 98765)):
            with self.subTest(key=key):
                worker.write(state_path, original)
                worker.write(solver_path, dict(solver, **{key: value}))
                status, launch, _ = self.execute('--recover-completed', job=dict(self.job, node='other'))
                self.assertEqual(status, 75)
                launch.assert_not_called()
        worker.write(state_path, original)
        worker.write(solver_path, solver)
        with (self.output / self.run['directory'] / '.runner.lock').open('r') as lock:
            worker.fcntl.flock(lock, worker.fcntl.LOCK_EX | worker.fcntl.LOCK_NB)
            status, launch, _ = self.execute('--recover-completed', job=dict(self.job, node='other'))
        self.assertEqual(status, 75)
        launch.assert_not_called()
        self.assertFalse((self.output / 'receipts' / (self.run['id'] + '.json')).exists())

    def test_recovery_on_original_node_refuses_live_original_process(self):
        self.failed_receipt()
        # execute() supplies this exact process identity as still present.
        status, launch, _ = self.execute('--recover-completed')
        self.assertEqual(status, 75)
        launch.assert_not_called()
        self.assertIn('still alive', worker.read(self.output / 'worker-state.json')['error'])

    def test_ordinary_existing_batch_refusal_keeps_original_solve_record(self):
        self.failed_receipt()
        status, launch, _ = self.execute()
        self.assertEqual(status, 75)
        launch.assert_not_called()
        self.assertEqual(worker.read(self.output / 'worker-state.json')['runs'][self.run['id']]['job_id'],
                         self.job['job_id'])

    def test_recovery_requires_existing_state_without_launch_or_new_output(self):
        with self.assertRaisesRegex(ValueError, 'existing worker state'):
            self.execute('--recover-completed')
        self.assertFalse(self.output.exists())

    def test_tampered_completed_files_are_never_reused(self):
        _, _, _, leaf = self.receipt()
        (leaf / 'X.0.silo.data/0.silo').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            worker.verify_completed(self.output, self.run, self.identity)

    def test_tampered_receipt_model_is_rejected(self):
        receipt, path, _, _ = self.receipt()
        receipt['model'] = 'fld'
        worker.write(path, receipt)
        with self.assertRaisesRegex(ValueError, 'changed'):
            worker.verify_completed(self.output, self.run, self.identity)

    def test_inventory_rejects_symlinks_and_special_files(self):
        self.output.mkdir()
        (self.output / 'link').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            worker.inventory(self.output)
        (self.output / 'link').unlink()
        os.mkfifo(self.output / 'pipe')
        with self.assertRaisesRegex(ValueError, 'Special'):
            worker.inventory(self.output)

    def test_missing_snapshot_refuses_completion(self):
        batch, leaf = self.batch()
        (leaf / 'X.14.silo').unlink()
        with self.assertRaisesRegex(ValueError, 'snapshots'):
            worker.completion_receipt(batch, self.run, self.identity, self.job)

    def test_moving_receipt_records_version2_and_lab_flux(self):
        run = next(run for run in self.plan if run['case'] == 'subcritical_shock' and run['model'] == 'fld')
        receipt, _, _, _ = self.receipt(run)
        self.assertEqual(receipt['model'], 'fld')
        self.assertTrue(worker.verify_completed(self.output, run, self.identity))

    def test_dry_run_writes_nothing_launches_nothing(self):
        status, launch, _ = self.execute('--dry-run')
        self.assertEqual(status, 0)
        launch.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_completed_run_is_preserved_without_launch(self):
        _, path, _, _ = self.receipt()
        original = path.read_bytes()
        status, launch, _ = self.execute()
        self.assertEqual(status, 0)
        self.assertEqual(path.read_bytes(), original)
        launch.assert_not_called()

    def test_partial_run_requires_continuation_without_overwriting(self):
        batch = self.output / self.run['directory']
        batch.mkdir(parents=True)
        (batch / 'partial.silo').write_bytes(b'preserved')
        status, launch, _ = self.execute()
        self.assertEqual(status, 75)
        self.assertEqual((batch / 'partial.silo').read_bytes(), b'preserved')
        self.assertEqual(worker.read(self.output / 'worker-state.json')['status'], 'needs_checkpoint_continuation')
        launch.assert_not_called()

    def test_success_writes_receipt_and_progress(self):
        def spawn(argv, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            self.batch()
            return Mock(pid=123, returncode=0, poll=Mock(return_value=0))
        status, launch, stopped = self.execute(spawn=spawn)
        self.assertEqual(status, 0)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(stopped.call_count, 1)
        self.assertTrue(worker.verify_completed(self.output, self.run, self.identity))
        self.assertFalse(worker.read(self.output / 'worker-state.json')['active'])

    def test_signal_stops_owned_child_preserves_partial(self):
        handlers = {}
        def spawn(argv, **kwargs):
            batch = self.output / self.run['directory']
            batch.mkdir(parents=True)
            (batch / 'X.0.silo').write_text('partial retained')
            child = Mock(pid=123, returncode=None)
            def poll():
                handlers[signal.SIGUSR1](signal.SIGUSR1, None)
                return None
            child.poll.side_effect = poll
            return child
        status, _, stopped = self.execute(spawn=spawn, signal_handlers=handlers)
        self.assertEqual(status, 75)
        self.assertEqual(stopped.call_count, 1)
        self.assertEqual(worker.read(self.output / 'worker-state.json')['status'], 'needs_checkpoint_continuation')
        self.assertTrue((self.output / self.run['directory'] / 'X.0.silo').exists())
        self.assertFalse((self.output / 'receipts' / (self.run['id'] + '.json')).exists())

    def test_failed_drain_keeps_active_identity(self):
        def spawn(argv, **kwargs):
            return Mock(pid=123, returncode=0, poll=Mock(return_value=0))
        with self.assertRaisesRegex(RuntimeError, 'did not drain'):
            self.execute(spawn=spawn, stop_effect=RuntimeError('did not drain'))
        self.assertTrue(worker.read(self.output / 'worker-state.json')['active'])

    def test_session_drain_signals_only_verified_child_groups(self):
        child = Mock(pid=123)
        with patch.object(worker, 'session_members', return_value={124: {'group': 124, 'rss': 0}}), \
                patch.object(worker.os, 'killpg') as kill, \
                patch.object(worker.time, 'monotonic', side_effect=[0, 9, 10, 16]):
            with self.assertRaisesRegex(RuntimeError, 'did not drain'):
                worker.stop_owned(child)
        self.assertEqual([call.args for call in kill.call_args_list], [(124, signal.SIGTERM), (124, signal.SIGKILL)])
        with self.assertRaisesRegex(ValueError, 'worker session'):
            worker.stop_owned(Mock(pid=os.getsid(0)))

    def test_worker_lock_prevents_two_supervisors(self):
        self.output.mkdir()
        with (self.output / '.worker.lock').open('a') as held:
            worker.fcntl.flock(held, worker.fcntl.LOCK_EX | worker.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.execute()

    def test_low_storage_refuses_launch(self):
        with patch.object(worker.shutil, 'disk_usage', return_value=Mock(free=1024)):
            with self.assertRaisesRegex(ValueError, 'Storage reserve'):
                worker.check_resources(self.root, 2**40, 2**30)

    def test_run_limit_releases_node_then_skips_completed_on_next_invocation(self):
        plan = self.plan[:2]
        launched_ids = []
        def spawn(argv, **kwargs):
            model = argv[argv.index('--radiation-model') + 1]
            run = next(run for run in plan if run['model'] == model)
            launched_ids.append(run['id'])
            self.batch(run)
            return Mock(pid=123, returncode=0, poll=Mock(return_value=0))
        status, launch, stopped = self.execute('--run-limit', '1', spawn=spawn, plan=plan)
        self.assertEqual(status, 0)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(stopped.call_count, 1)
        state = worker.read(self.output / 'worker-state.json')
        self.assertEqual(state['status'], 'waiting_for_publication')
        self.assertFalse(state['active'])
        self.assertFalse((self.output / plan[1]['directory']).exists())
        receipt = self.output / 'receipts' / (plan[0]['id'] + '.json')
        original = receipt.read_bytes()
        status, launch, _ = self.execute('--run-limit', '1', spawn=spawn, plan=plan)
        self.assertEqual(status, 0)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launched_ids, [run['id'] for run in plan])
        self.assertEqual(receipt.read_bytes(), original)
        self.assertEqual(worker.read(self.output / 'worker-state.json')['status'], 'waiting_for_publication')
        status, launch, _ = self.execute('--run-limit', '1', plan=plan)
        self.assertEqual(status, 0)
        launch.assert_not_called()
        self.assertEqual(worker.read(self.output / 'worker-state.json')['status'], 'complete')

    def test_run_limit_requires_positive_count(self):
        for value in ('0', '-1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be positive'):
                self.execute('--run-limit', value)
        self.assertFalse(self.output.exists())

    def build_files(self):
        tools = self.build / 'radiation-results-tools'
        tools.mkdir(parents=True)
        binaries = {'executable': self.build / 'octotiger', 'results': tools / 'radiation-results',
                    'generator': tools / 'gen_radiation_reference'}
        for name, path in binaries.items():
            path.write_text('synthetic ' + name)
            path.chmod(0o755)
        manifest = self.build / 'qb4-build-manifest.txt'
        manifest.write_text('job=1001\nnode=qbd-builder\njobs=16\n' + ''.join(
            worker.digest(path) + '  ' + str(path) + '\n' for path in binaries.values()))
        worker.write(self.build.parent / 'build-progress.json', {'schema': 1, 'owner': worker.OWNER,
            'phase': 'build', 'status': 'completed', 'job_id': '1001', 'nodes': 1, 'tasks': 1,
            'node': 'qbd-builder.cluster'})
        (self.build / 'CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL=' + str(self.source) + '\n')
        return binaries, manifest

    def test_build_identity_binds_completed_job_node_paths_and_hashes(self):
        binaries, manifest = self.build_files()
        result = worker.build_identity(self.build, binaries)
        self.assertEqual(result['build'], {'job_id': '1001', 'node': 'qbd-builder'})
        self.assertEqual(result['build_evidence']['manifest']['sha256'], worker.digest(manifest))
        self.assertIn('progress', result['build_evidence'])

    def test_build_manifest_fallback_still_pins_current_artifacts(self):
        binaries, _ = self.build_files()
        (self.build.parent / 'build-progress.json').unlink()
        self.assertEqual(worker.build_identity(self.build, binaries)['build']['job_id'], '1001')
        binaries['executable'].write_text('different executable')
        with self.assertRaisesRegex(ValueError, 'current artifact'):
            worker.build_identity(self.build, binaries)

    def test_incomplete_or_different_builder_is_rejected(self):
        binaries, _ = self.build_files()
        path = self.build.parent / 'build-progress.json'
        original = worker.read(path)
        for key, value in (('status', 'running'), ('job_id', '1002'), ('node', 'other-node'),
                           ('nodes', 2), ('tasks', 2), ('phase', 'solve'), ('owner', 'foreign')):
            with self.subTest(key=key):
                worker.write(path, dict(original, **{key: value}))
                with self.assertRaises(ValueError):
                    worker.build_identity(self.build, binaries)

    def test_build_manifest_cannot_substitute_artifact_from_other_directory(self):
        binaries, manifest = self.build_files()
        manifest.write_text(manifest.read_text().replace(str(binaries['executable']), '/other/build/octotiger'))
        with self.assertRaisesRegex(ValueError, 'current artifact'):
            worker.build_identity(self.build, binaries)

    def test_runtime_identity_has_importer_schema_and_verified_build(self):
        binaries, _ = self.build_files()
        library = self.root / 'libhpx.so'
        library.write_text('synthetic HPX library')
        sources = {key: self.identity[key] for key in ('source_commit', 'source_sha256', 'source_tree_sha256')}
        with patch.object(worker, 'source_identity', return_value=sources), \
                patch.object(worker.subprocess, 'check_output', return_value=f'libhpx.so => {library} (0x0)\n'):
            result = worker.runtime_identity(self.source, self.build, binaries['results'].parent)
            self.assertEqual({key: result[key] for key in ('schema', 'nodes', 'tasks')},
                             {'schema': 1, 'nodes': 1, 'tasks': 1})
            self.assertEqual(result['build'], self.identity['build'])
            self.assertEqual(result['executable_sha256'], worker.digest(binaries['executable']))
            (self.build / 'CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL=/other/source\n')
            with self.assertRaisesRegex(ValueError, 'selected source'):
                worker.runtime_identity(self.source, self.build, binaries['results'].parent)

    def test_actual_receipt_matches_shared_collector_contract(self):
        spec = importlib.util.spec_from_file_location('collector_contract', Path(__file__).parents[1] / 'run_campaign.py')
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        receipt, _, _, _ = self.receipt()
        self.assertEqual(campaign.validate_remote_receipt({'remote_runtime': self.identity}, self.run, receipt), receipt)

    def test_progress_reads_only_tail_and_reports_latest_endpoint(self):
        batch, leaf = self.batch()
        (leaf / 'run.log').write_text('old text\n' * 10000 +
            'FLD accepted t=1.25 dt=0.25 substeps=1\nTS 4:: t: 1.500000e+00, dt: 0.25\n')
        progress = worker.solver_progress(self.output, self.run)
        self.assertEqual(progress['simulation_time'], 1.5)
        (leaf / 'run.log').write_text('FLD accepted t=2 dt=0.125 substeps=1\n')
        self.assertEqual(worker.solver_progress(self.output, self.run)['simulation_time'], 2.125)
        (leaf / 'run.log').write_text('initializing\n')
        self.assertIsNone(worker.solver_progress(self.output, self.run))


if __name__ == '__main__':
    unittest.main()
