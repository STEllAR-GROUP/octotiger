#!/usr/bin/env python3
"""Publication sequencing tests: mocked children only, no solver or network."""
import contextlib
import copy
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_published_campaign as published
from test_campaign_retention import prepare_outputs


class PublishedCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='published-campaign-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'campaign'
        frozen = self.root / '.campaign'
        frozen.mkdir(parents=True)
        for name in (*published.SCRIPTS, published.POLICY_NAME):
            shutil.copyfile(Path(published.__file__).with_name(name), frozen / name)
        self.manifest = {'schema': 2, 'workflow': 'm1-fld-radiation', 'methods': ['m1', 'fld'],
            'levels': [2], 'cases': ['gaussian_pulse'], 'unsupported': [], 'threads': 2, 'snapshots': 121,
            'minimum_free_gib': .001, 'memory_limit_gib': 1, 'original_build': '/synthetic/build',
            'original_results_build': '/synthetic/results', 'original_source': '/synthetic/source',
            'active': False, 'status': 'prepared', 'scope': 'synthetic sequencing tests', 'executable_sha256': 'synthetic',
            'runtime_library_aliases': {}, 'runs': [], 'preparation_checks':
                {'config_dry_runs': 2, 'solver_runs': 0, 'runtime_help_passed': True}}
        for model, integrator in [('m1', 'euler'), ('fld', 'backward_euler')]:
            self.manifest['runs'].append({'id': 'l2-gaussian_pulse-' + model, 'case': 'gaussian_pulse',
                'level': 2, 'method': model, 'model': model, 'integrator': integrator, 'backend': 'cpu',
                'directory': 'runs/' + model, 'stop_time': 1.5, 'hard_dt': .00625,
                'status': 'pending', 'attempts': []})
        published.runner.write_json(frozen / 'plan.json', published.runner.execution_plan(self.manifest))
        self.manifest['snapshot_sha256'] = {str(path.relative_to(self.root)): published.runner.file_hash(path)
            for path in frozen.iterdir() if path.is_file()}
        self.save_manifest()
        self.events = []
        self.phases = patch.object(published, 'phase', side_effect=self.fake_phase)
        self.phases.start()
        self.addCleanup(self.phases.stop)
        free = patch.object(published.shutil, 'disk_usage', return_value=Mock(free=2**50))
        free.start(); self.addCleanup(free.stop)

    def save_manifest(self):
        published.runner.write_json(self.root / 'campaign.json', self.manifest)

    def mark_complete(self, index):
        self.manifest = published.runner.load(self.root)
        run = self.manifest['runs'][index]
        folder = self.root / run['directory'] / run['case'] / ('l' + str(run['level']))
        metadata = {'status': 'complete', 'case': run['case'], 'level': run['level'],
            'time': run['stop_time'], 'radiation_model': run['model'], 'radiation_integrator': run['integrator'],
            'radiation_algorithm_version': 1, 'radiation_flux_semantics': published.runner.FLUX[run['model']],
            'executable_sha256': 'synthetic'}
        published.runner.write_json(folder / 'run.json', metadata)
        run.update(status='completed', run_metadata_sha256=published.runner.file_hash(folder / 'run.json'))
        self.save_manifest()

    def fake_phase(self, root, manifest, state, name, command, stage=None):
        self.events.append((name, command))
        if name == 'solver':
            self.assertEqual(command[-2:], ['--run-limit', '1'])
            self.manifest = published.runner.load(root)
            index = next(i for i, run in enumerate(self.manifest['runs']) if run['status'] != 'completed')
            run = self.manifest['runs'][index]
            identity = {'pid': 100 + index, 'process_start_ticks': str(1000 + index)}
            run['attempts'].append(identity)
            self.save_manifest()
            published.prove_run_ownership(root, state,
                {identity['pid']: {'start_ticks': identity['process_start_ticks']}})
            self.mark_complete(index)
        elif name == 'movie':
            self.assertEqual(command[command.index('--seconds') + 1], '60')
            self.assertIn('--full-domain', command)
            current = published.runner.load(root)
            run_id = command[command.index('--run-id') + 1]
            run = next(item for item in current['runs'] if item['id'] == run_id)
            if not run.get('retention'):
                prepare_outputs(root, run, current['snapshots'])
        elif name == 'stage':
            self.assertFalse(stage.exists(), 'staging must use a fresh unique directory')
            stage.mkdir(parents=True)
            (stage / 'index.html').write_text('synthetic publication')
        elif name == 'deploy':
            self.assertIn('--deploy', command)
            count = sum(run['status'] == 'completed' for run in published.runner.load(root)['runs'])
            published.runner.write_json(stage / 'publication-manifest.json', {'completed_runs': count})
            generation = 'm1-fld-20261013-20261014T010203Z-' + stage.name[:8]
            receipt = {'status': 'verified', 'generation': generation, 'stage': str(stage),
                'completed_runs': count, 'manifest_sha256': published.runner.file_hash(stage / 'publication-manifest.json'),
                'url': 'https://example.invalid/campaign/'}
            published.runner.write_json(root / '.deployment' / (generation + '.json'), receipt)
            published.runner.write_json(root / '.deployment/latest-receipt.json', receipt)

    def main(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = published.main(['--campaign', str(self.root), *arguments])
        return code, output.getvalue()

    def test_default_plan_and_status_are_read_only(self):
        self.mark_complete(0)
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        for arguments in ((), ('--plan',), ('--status',)):
            code, output = self.main(*arguments)
            plan = json.loads(output)
            self.assertEqual(code, 0)
            self.assertEqual(plan['next_action'], 'publish')
            self.assertTrue(plan['explicit_start_required'])
        after = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(self.events, [])
        self.assertFalse((self.root / '.publication').exists())
        self.assertFalse((self.root / '.publication.lock').exists())

    def test_explicit_start_alternates_one_run_movie_stage_deploy(self):
        self.assertEqual(self.main('--start')[0], 0)
        self.assertEqual([name for name, _ in self.events], ['solver', 'movie', 'stage', 'deploy'] * 2)
        state = published.load_state(self.root)
        self.assertEqual(state['status'], 'completed')
        self.assertFalse(state['active'])
        self.assertEqual(len(state['published']), 2)
        stages = self.root.parent / (self.root.name + '-publication-stages')
        self.assertEqual([path.name for path in stages.iterdir()], [state['latest_verified_attempt']])
        self.assertTrue(all(Path(saved['receipt']['path']).is_file() for saved in state['published'].values()))
        self.events.clear()
        self.assertEqual(self.main('--start')[0], 0)
        self.assertEqual(self.events, [])

    def test_completed_unpublished_results_are_published_before_next_solver(self):
        self.mark_complete(0)
        self.assertEqual(self.main('--start')[0], 0)
        self.assertEqual([name for name, _ in self.events], ['movie', 'stage', 'deploy', 'solver', 'movie', 'stage', 'deploy'])

    def test_multiple_completed_unpublished_results_are_rendered_before_one_publish(self):
        self.mark_complete(0); self.mark_complete(1)
        self.assertEqual(self.main('--start')[0], 0)
        self.assertEqual([name for name, _ in self.events], ['movie', 'movie', 'stage', 'deploy'])
        for run in published.runner.load(self.root)['runs']:
            self.assertNotIn('retention', run)
            self.assertTrue((published.retention.leaf(self.root, run) / 'X.1.silo').exists())

    def test_owned_runs_are_pruned_only_after_movie_verification(self):
        self.main('--start')
        for run in published.runner.load(self.root)['runs']:
            self.assertIn('retention', run)
            folder = published.retention.leaf(self.root, run)
            self.assertFalse((folder / 'X.1.silo').exists())
            self.assertTrue((folder / 'X.0.silo').exists())
            self.assertTrue((folder / 'final.silo').exists())
            self.assertTrue((folder / 'checkpoint.final').exists())
            published.runner.validate_completed(self.root, published.runner.load(self.root), run)

    def test_unverified_extraction_blocks_cleanup_publish_and_next_solver(self):
        def invalid_extraction(*args, **kwargs):
            self.fake_phase(*args, **kwargs)
            if args[3] == 'movie':
                run = published.runner.load(self.root)['runs'][0]
                (published.retention.leaf(self.root, run) / 'movies/radiation-energy-flux/profile-1.json').unlink()
        with patch.object(published, 'phase', side_effect=invalid_extraction):
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                self.main('--start')
        run = published.runner.load(self.root)['runs'][0]
        self.assertNotIn('retention', run)
        self.assertTrue((published.retention.leaf(self.root, run) / 'X.1.silo').exists())
        self.assertEqual([name for name, _ in self.events], ['solver', 'movie'])

    def test_raw_run_capacity_is_checked_in_addition_to_reserve(self):
        with patch.object(published.shutil, 'disk_usage', return_value=Mock(free=.5 * 2**30)):
            with self.assertRaisesRegex(RuntimeError, 'next raw run'):
                self.main('--start')
        self.assertEqual(self.events, [])

    def test_external_attempt_cannot_be_adopted_for_raw_removal(self):
        def external_attempt(*args, **kwargs):
            self.fake_phase(*args, **kwargs)
            if args[3] == 'solver':
                state = args[2]
                for owner in state.get('owned_runs', {}).values():
                    owner.pop('solver_identity', None)
                published.save_state(self.root, state)
        with patch.object(published, 'phase', side_effect=external_attempt):
            with self.assertRaisesRegex(ValueError, 'owned process tree'):
                self.main('--start')
        self.assertEqual([name for name, _ in self.events], ['solver', 'movie'])

    def test_publication_failure_pauses_and_resume_finishes_it_first(self):
        original = self.fake_phase
        def fail_deploy(*args, **kwargs):
            if args[3] == 'deploy':
                self.events.append(('deploy_failed', args[4]))
                raise RuntimeError('synthetic upload failure')
            return original(*args, **kwargs)
        with patch.object(published, 'phase', side_effect=fail_deploy):
            with self.assertRaisesRegex(RuntimeError, 'upload failure'):
                self.main('--start')
        state = published.load_state(self.root)
        self.assertEqual(state['status'], 'paused')
        self.assertFalse(state['active'])
        self.assertEqual([run['status'] for run in published.runner.load(self.root)['runs']], ['completed', 'pending'])
        failed_stage = published.stage_path(self.root, state['attempts'][0])
        self.events.clear()
        self.assertEqual(self.main('--start')[0], 0)
        self.assertEqual([name for name, _ in self.events][:4], ['movie', 'stage', 'deploy', 'solver'])
        self.assertFalse(failed_stage.exists())

    def test_verified_deploy_receipt_recovers_bookkeeping_without_reupload(self):
        def interrupted_after_deploy(*args, **kwargs):
            self.fake_phase(*args, **kwargs)
            if args[3] == 'deploy':
                raise RuntimeError('interrupted after verified deployment')
        self.mark_complete(0); self.mark_complete(1)
        with patch.object(published, 'phase', side_effect=interrupted_after_deploy):
            with self.assertRaisesRegex(RuntimeError, 'after verified'):
                self.main('--start')
        self.events.clear()
        self.assertEqual(self.main('--start')[0], 0)
        self.assertEqual(self.events, [])
        self.assertEqual(len(published.load_state(self.root)['published']), 2)

    def test_movie_failure_prevents_upload_and_further_simulation(self):
        def fail_movie(*args, **kwargs):
            if args[3] == 'movie':
                raise RuntimeError('synthetic movie failure')
            self.fake_phase(*args, **kwargs)
        with patch.object(published, 'phase', side_effect=fail_movie):
            with self.assertRaisesRegex(RuntimeError, 'movie failure'):
                self.main('--start')
        self.assertEqual([name for name, _ in self.events], ['solver'])

    def test_stop_and_active_states_prevent_all_work(self):
        (self.root / 'STOP').write_text('paused by user')
        with self.assertRaisesRegex(ValueError, 'STOP'):
            self.main('--start')
        (self.root / 'STOP').unlink()
        self.manifest['active'] = True; self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Campaign active'):
            self.main('--start')
        self.manifest['active'] = False; self.save_manifest()
        state = published.load_state(self.root); state['active'] = True
        published.save_state(self.root, state)
        with self.assertRaisesRegex(ValueError, 'wrapper active'):
            self.main('--start')
        self.assertEqual(self.events, [])

    def test_incomplete_preparation_and_changed_frozen_policy_are_rejected(self):
        self.manifest['preparation_checks']['config_dry_runs'] = 0; self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'preparation'):
            self.main('--start')
        self.manifest['preparation_checks']['config_dry_runs'] = 2; self.save_manifest()
        with (self.root / '.campaign' / published.POLICY_NAME).open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            self.main('--start')
        self.assertEqual(self.events, [])

    def test_full_wrapper_lock_blocks_duplicate_start(self):
        with (self.root / '.publication.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.main('--start')
        self.assertEqual(self.events, [])

    def test_changed_publication_receipt_blocks_silent_skip(self):
        self.main('--start')
        self.events.clear()
        saved = next(iter(published.load_state(self.root)['published'].values()))
        Path(saved['receipt']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            self.main('--start')
        self.assertEqual(self.events, [])

    def test_storage_reserve_is_checked_before_any_child(self):
        with patch.object(published.shutil, 'disk_usage', return_value=Mock(free=0)):
            with self.assertRaisesRegex(RuntimeError, 'storage'):
                self.main('--start')
        self.assertEqual(self.events, [])

    def test_owned_processes_include_nested_and_orphaned_sessions_only(self):
        sessions = {100}
        table = {100: {'parent': 1, 'session': 100}, 101: {'parent': 100, 'session': 101},
                 102: {'parent': 101, 'session': 101}, 400: {'parent': 1, 'session': 400}}
        with patch.object(published, 'process_table', return_value=table), patch.object(published.os, 'getsid', return_value=999):
            self.assertEqual(set(published.owned_processes(100, sessions)), {100, 101, 102})
            self.assertEqual(sessions, {100, 101})
            del table[100]
            table[101]['parent'] = 1
            self.assertEqual(set(published.owned_processes(100, sessions)), {101, 102})

    def test_new_orphan_session_is_adopted_after_wrapper_exits_between_polls(self):
        table = {101: {'parent': 999, 'session': 101, 'start_ticks': '200'},
                 102: {'parent': 101, 'session': 101, 'start_ticks': '201'},
                 400: {'parent': 999, 'session': 400, 'start_ticks': '100'},
                 500: {'parent': 1, 'session': 500, 'start_ticks': '202'}}
        with patch.object(published, 'process_table', return_value=table), \
                patch.object(published.os, 'getpid', return_value=999), \
                patch.object(published.os, 'getsid', return_value=999):
            sessions = published.OwnedSessions()
            sessions.baseline = {(400, '100')}
            sessions.add(100)
            self.assertEqual(set(published.owned_processes(100, sessions)), {101, 102})
            self.assertEqual(sessions, {100, 101})

    def test_dead_wrapper_drains_discovered_nested_sessions_without_outside_signals(self):
        child = Mock(pid=100, returncode=0)
        child.poll.return_value = 0
        signals = []
        ticks = iter(range(0, 100, 2))
        def members(pid, sessions):
            return {} if signals else {101: {'state': 'S', 'parent': 1, 'session': 101}}
        with patch.object(published, 'owned_processes', side_effect=members), \
                patch.object(published.runner, 'signal_session', side_effect=lambda sid, sig: signals.append((sid, sig))), \
                patch.object(published.time, 'monotonic', side_effect=lambda: next(ticks)), \
                patch.object(published.time, 'sleep'):
            published.drain_child(child, {100, 101})
        self.assertEqual({sid for sid, _ in signals}, {100, 101})
        self.assertTrue(all(sig == signal.SIGTERM for _, sig in signals))
        child.wait.assert_called_once()

    def test_phase_memory_failure_drains_child_and_records_log(self):
        self.phases.stop()
        state = published.load_state(self.root); state['active'] = True
        child = Mock(pid=100, returncode=None)
        with patch.object(published.subprocess, 'Popen', return_value=child) as popen, \
                patch.object(published, 'owned_processes', return_value={100: {'rss': 2**32}}), \
                patch.object(published, 'drain_child') as drain:
            with self.assertRaisesRegex(RuntimeError, 'memory budget'):
                published.phase(self.root, self.manifest, state, 'movie', ['mock-renderer'])
        drain.assert_called_once()
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertNotIn('child', state)
        self.assertTrue(list((self.root / '.publication/logs').glob('*.log')))

    def test_phase_failed_cleanup_retains_active_child_for_manual_review(self):
        self.phases.stop()
        state = published.load_state(self.root); state['active'] = True
        child = Mock(pid=100, returncode=1)
        child.poll.return_value = 1
        with patch.object(published.subprocess, 'Popen', return_value=child), \
                patch.object(published, 'owned_processes', return_value={}), \
                patch.object(published, 'drain_child', side_effect=RuntimeError('did not exit')):
            with self.assertRaisesRegex(RuntimeError, 'did not exit'):
                published.phase(self.root, self.manifest, state, 'movie', ['mock-renderer'])
        self.assertIn('child', published.load_state(self.root))
        self.assertTrue(published.load_state(self.root)['active'])


if __name__ == '__main__':
    unittest.main()
