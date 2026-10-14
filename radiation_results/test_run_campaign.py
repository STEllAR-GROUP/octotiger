#!/usr/bin/env python3
"""Preparation and process-safety checks; every subprocess is mocked."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('campaign_under_test', Path(__file__).with_name('run_campaign.py'))
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class CampaignSafety(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='m1-fld-campaign-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'prepared'
        self.checked_commands = []
        self.children = []

    def freeze(self, root, manifest):
        """Small immutable input tree; no binaries are copied or executed."""
        frozen = root / '.campaign'
        runtime = frozen / 'runtime'
        runtime.mkdir(parents=True)
        campaign.write_json(frozen / 'plan.json', campaign.execution_plan(manifest))
        for name in ('octotiger', 'radiation-results', 'gen_radiation_reference', 'libhpx.so.1'):
            (runtime / name).write_text('synthetic input ' + name)
        (runtime / 'libhpx.so').symlink_to('libhpx.so.1')
        (frozen / 'compare_campaign.py').write_text('# frozen report fixture\n')
        (frozen / 'run_campaign.py').write_bytes(Path(campaign.__file__).read_bytes())
        manifest['runtime_library_aliases'] = {'libhpx.so': 'libhpx.so.1'}
        manifest['executable_sha256'] = campaign.file_hash(runtime / 'octotiger')
        manifest['snapshot_sha256'] = {str(path.relative_to(root)): campaign.file_hash(path)
            for path in frozen.rglob('*') if path.is_file()}

    def check_command(self, command, **kwargs):
        self.checked_commands.append(command)
        # This assertion fails before any real child can be started if preparation
        # accidentally uses a simulation command instead of a parser/help check.
        self.assertTrue('--dry-run' in command or command[-1] == '--help', command)
        self.assertTrue(kwargs['check'])
        return subprocess.CompletedProcess(command, 0)

    def main(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return campaign.main(list(arguments))

    def prepare(self, *extra):
        with patch.object(campaign, 'snapshot', self.freeze), \
                patch.object(campaign.subprocess, 'run', side_effect=self.check_command), \
                patch.object(campaign.subprocess, 'Popen', side_effect=AssertionError('solver launch during preparation')):
            return self.main('--output', str(self.root), '--prepare', *extra)

    def write_manifest(self, manifest):
        campaign.write_json(self.root / 'campaign.json', manifest)

    def start(self):
        with patch.object(campaign.subprocess, 'Popen', side_effect=self.fake_child) as spawned, \
                patch.object(campaign, 'stop_child') as stopped, \
                patch.object(campaign, 'refresh_comparisons') as refreshed, \
                patch.object(campaign.shutil, 'disk_usage', return_value=Mock(free=2**50)), \
                patch.object(campaign, 'process_session_rss', return_value=0), \
                patch.object(campaign.time, 'sleep'):
            result = self.main('--resume', str(self.root), '--start')
        return result, spawned, stopped, refreshed

    def fake_child(self, command, **kwargs):
        self.assertTrue(kwargs['start_new_session'])
        self.assertNotIn('--dry-run', command)
        self.assertTrue(all(child.returncode == 0 for child in self.children), 'overlapping children')
        def argument(name):
            return command[command.index(name) + 1]
        folder = Path(argument('--output')) / command[2] / ('l' + command[3])
        folder.mkdir(parents=True)
        model = argument('--radiation-model')
        campaign.write_json(folder / 'run.json', {'status': 'complete', 'case': command[2],
            'level': int(command[3]), 'time': float(argument('--time')), 'radiation_model': model,
            'radiation_integrator': argument('--radiation-integrator'), 'radiation_algorithm_version': 1,
            'radiation_flux_semantics': campaign.FLUX[model],
            'executable_sha256': campaign.file_hash(Path(argument('--exe')))})
        child = Mock(pid=os.getpid(), returncode=None)
        states = iter((None, 0))
        def poll():
            result = next(states)
            child.returncode = result
            return result
        child.poll.side_effect = poll
        self.children.append(child)
        return child

    def test_prepare_and_resume_dry_run_never_start_a_solver(self):
        self.assertEqual(self.prepare(), 0)
        manifest = campaign.load(self.root)
        self.assertEqual(len(manifest['runs']), 8)
        self.assertEqual(manifest['authorization'], 'awaiting_explicit_start')
        self.assertEqual(manifest['status'], 'prepared')
        self.assertFalse(manifest['active'])
        self.assertEqual(manifest['preparation_checks']['solver_runs'], 0)
        self.assertTrue(manifest['preparation_checks']['runtime_help_passed'])
        self.assertEqual(sum('--dry-run' in command for command in self.checked_commands), 8)
        self.assertFalse((self.root / 'runs').exists())
        self.assertIn('Cross-model differences are not truncation-error estimates', (self.root / 'index.html').read_text())
        with patch.object(campaign.subprocess, 'run', side_effect=self.check_command), \
                patch.object(campaign.subprocess, 'Popen') as launch:
            self.assertEqual(self.main('--resume', str(self.root), '--dry-run'), 0)
            launch.assert_not_called()
        self.assertEqual(sum('--dry-run' in command for command in self.checked_commands), 16)

    def test_new_dry_run_writes_nothing_and_runs_no_external_program(self):
        with patch.object(campaign, 'snapshot') as snapshot, \
                patch.object(campaign.subprocess, 'run') as checked, \
                patch.object(campaign.subprocess, 'Popen') as launched:
            self.assertEqual(self.main('--output', str(self.root), '--dry-run'), 0)
            snapshot.assert_not_called(); checked.assert_not_called(); launched.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_start_requires_resume_and_an_explicit_action(self):
        with patch.object(campaign.subprocess, 'Popen') as launched:
            for arguments in (('--output', str(self.root), '--start'), ('--output', str(self.root)),
                    ('--resume', str(self.root))):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    self.main(*arguments)
            launched.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_failed_preparation_cannot_be_started_until_checks_pass(self):
        with patch.object(campaign, 'snapshot', self.freeze), \
                patch.object(campaign.subprocess, 'run', side_effect=subprocess.CalledProcessError(2, ['dry-run'])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.main('--output', str(self.root), '--prepare')
        with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(ValueError, 'Preparation is incomplete'):
            self.main('--resume', str(self.root), '--start')
        launched.assert_not_called()
        with patch.object(campaign.subprocess, 'run', side_effect=self.check_command):
            self.assertEqual(self.main('--resume', str(self.root), '--dry-run'), 0)
        self.assertTrue(campaign.load(self.root)['preparation_checks']['runtime_help_passed'])

    def test_start_requires_all_checks_including_runtime_help(self):
        self.prepare()
        original = campaign.load(self.root)
        for field, value in (('config_dry_runs', 0), ('solver_runs', 1), ('runtime_help_passed', False)):
            with self.subTest(field=field):
                manifest = copy.deepcopy(original)
                manifest['preparation_checks'][field] = value
                self.write_manifest(manifest)
                with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(ValueError, 'Preparation is incomplete'):
                    self.main('--resume', str(self.root), '--start')
                launched.assert_not_called()

    def test_active_or_running_crash_record_is_not_relaunched(self):
        self.prepare()
        original = campaign.load(self.root)
        for active in (True, False):
            with self.subTest(active=active):
                manifest = copy.deepcopy(original)
                manifest['active'] = active
                if not active:
                    manifest['runs'][0]['status'] = 'running'
                self.write_manifest(manifest)
                with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(ValueError, 'may still be alive'):
                    self.main('--resume', str(self.root), '--start')
                launched.assert_not_called()
                self.assertEqual(campaign.load(self.root), manifest)

    def test_saved_execution_settings_are_immutable(self):
        self.prepare()
        original = campaign.load(self.root)
        mutations = (lambda m: m.update(threads=7), lambda m: m['runs'][0].update(stop_time=3.),
            lambda m: m['runs'][0].update(directory='../escape'),
            lambda m: m['runs'][0].update(integrator='ader2'))
        for mutate in mutations:
            changed = copy.deepcopy(original); mutate(changed); self.write_manifest(changed)
            with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(ValueError, 'execution settings changed'):
                self.main('--resume', str(self.root), '--start')
            launched.assert_not_called()
        self.write_manifest(original)
        with self.assertRaises(SystemExit):
            self.main('--resume', str(self.root), '--start', '--threads', '3')

    def test_frozen_executable_change_is_rejected_before_start(self):
        self.prepare()
        (self.root / '.campaign/runtime/octotiger').write_text('changed executable')
        with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            self.main('--resume', str(self.root), '--start')
        launched.assert_not_called()

    def test_changed_orchestrator_cannot_execute_a_frozen_plan(self):
        self.prepare()
        saved = self.root / '.campaign/run_campaign.py'
        saved.write_text(saved.read_text() + '\n# different orchestration\n')
        with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaises(ValueError):
            campaign.verify_orchestrator(self.root)
        launched.assert_not_called()

    def test_host_interpreter_change_is_rejected_before_start(self):
        self.prepare()
        interpreter = self.root.parent / 'host-elf-loader'
        interpreter.write_text('original loader')
        manifest = campaign.load(self.root)
        manifest['host_elf_interpreters'] = {str(interpreter): campaign.file_hash(interpreter)}
        self.write_manifest(manifest)
        campaign.verify_snapshot(self.root, manifest)
        interpreter.write_text('upgraded loader')
        with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(ValueError, 'ELF interpreter changed'):
            self.main('--resume', str(self.root), '--start')
        launched.assert_not_called()

    def test_supported_matrix_does_not_silently_substitute_m1_for_fld(self):
        args = campaign.parser().parse_args(['--output', str(self.root), '--dry-run', '--levels', '0',
            '--cases', *campaign.CASES])
        manifest = campaign.plan(args)
        self.assertEqual(len(manifest['runs']), 10)
        self.assertEqual(len(manifest['unsupported']), 6)
        self.assertEqual({run['case'] for run in manifest['runs'] if run['model'] == 'fld'},
            {'gaussian_pulse', 'marshak'})
        for run in manifest['runs']:
            command = campaign.command(self.root, manifest, run)
            self.assertEqual(command[command.index('--radiation-backend') + 1], 'cpu')
            self.assertEqual(command[command.index('--radiation-reduced-c-factor') + 1], '1')
            self.assertEqual(run['integrator'], 'euler' if run['model'] == 'm1' else 'backward_euler')
        args.cases = ['equilibrium_sphere']; args.models = ['fld']
        with self.assertRaisesRegex(ValueError, 'No supported'):
            campaign.plan(args)

    def test_resource_and_timestep_limits_apply_before_preparation(self):
        for arguments in (('--threads', '7'), ('--hard-dt', '1'), ('--time', 'nan'),
                ('--minimum-free-gib', '0'), ('--memory-limit-gib', 'inf')):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.main('--output', str(self.root), '--dry-run', *arguments)
        self.assertFalse(self.root.exists())

    def test_explicit_start_is_sequential_and_completed_resume_is_read_only(self):
        self.prepare()
        result, spawned, stopped, refreshed = self.start()
        self.assertEqual(result, 0)
        self.assertEqual(spawned.call_count, 8)
        self.assertEqual(stopped.call_count, 8)
        self.assertEqual(refreshed.call_count, 8)
        manifest = campaign.load(self.root)
        self.assertEqual(manifest['status'], 'completed')
        self.assertFalse(manifest['active'])
        self.assertTrue(all(run['status'] == 'completed' and run['run_metadata_sha256'] for run in manifest['runs']))
        self.assertTrue(all(run['attempts'][0]['process_start_ticks'] for run in manifest['runs']))
        second, spawned, stopped, _ = self.start()
        self.assertEqual(second, 0); spawned.assert_not_called(); stopped.assert_not_called()
        self.assertEqual([len(run['attempts']) for run in campaign.load(self.root)['runs']], [1] * 8)

    def test_completed_metadata_cannot_change_on_resume(self):
        self.prepare('--levels', '0', '--models', 'm1', '--cases', 'gaussian_pulse')
        self.start()
        manifest = campaign.load(self.root); run = manifest['runs'][0]
        path = self.root / run['directory'] / run['case'] / 'l0/run.json'
        metadata = json.loads(path.read_text()); metadata['extra'] = 'modified after validation'
        campaign.write_json(path, metadata)
        with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(ValueError, 'metadata changed'):
            self.main('--resume', str(self.root), '--start')
        launched.assert_not_called()

    def test_stop_file_blocks_start_without_discarding_prepared_state(self):
        self.prepare(); (self.root / 'STOP').touch()
        with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(ValueError, 'STOP is present'):
            self.main('--resume', str(self.root), '--start')
        launched.assert_not_called()
        self.assertEqual(campaign.load(self.root)['status'], 'prepared')

    def test_disk_guard_fails_before_any_child(self):
        self.prepare()
        with patch.object(campaign.shutil, 'disk_usage', return_value=Mock(free=1)), \
                patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(RuntimeError, 'Free storage'):
            self.main('--resume', str(self.root), '--start')
        launched.assert_not_called()
        self.assertEqual(campaign.load(self.root)['status'], 'failed')

    def test_session_memory_limit_stops_the_child_and_records_interruption(self):
        self.prepare('--levels', '0', '--models', 'm1', '--cases', 'gaussian_pulse')
        with patch.object(campaign.subprocess, 'Popen', side_effect=self.fake_child), \
                patch.object(campaign, 'stop_child') as stopped, \
                patch.object(campaign.shutil, 'disk_usage', return_value=Mock(free=2**50)), \
                patch.object(campaign, 'process_session_rss', return_value=2**40), \
                self.assertRaisesRegex(RuntimeError, 'process session exceeded'):
            self.main('--resume', str(self.root), '--start')
        stopped.assert_called_once_with(self.children[0])
        manifest = campaign.load(self.root)
        self.assertEqual(manifest['runs'][0]['status'], 'interrupted')
        self.assertFalse(manifest['active'])

    def test_failed_session_cleanup_preserves_active_state_for_recovery(self):
        self.prepare('--levels', '0', '--models', 'm1', '--cases', 'gaussian_pulse')
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with patch.object(campaign.subprocess, 'Popen', side_effect=self.fake_child), \
                patch.object(campaign, 'stop_child', side_effect=RuntimeError('session still alive')), \
                patch.object(campaign.shutil, 'disk_usage', return_value=Mock(free=2**50)), \
                patch.object(campaign, 'process_session_rss', return_value=2**40), \
                self.assertRaisesRegex(RuntimeError, 'session still alive'):
            self.main('--resume', str(self.root), '--start')
        manifest = campaign.load(self.root)
        self.assertTrue(manifest['active'])
        self.assertEqual(manifest['runs'][0]['status'], 'running')
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        with patch.object(campaign.subprocess, 'Popen') as launched, self.assertRaisesRegex(ValueError, 'may still be alive'):
            self.main('--resume', str(self.root), '--start')
        launched.assert_not_called()


class ChildCleanup(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='campaign-proc-fixture-')
        self.addCleanup(self.temporary.cleanup)
        self.proc = Path(self.temporary.name)
        self.session = 7654321
        self.path_patch = patch.object(campaign, 'Path', side_effect=lambda value:
            self.proc if value == '/proc' else Path(value))
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def process(self, pid, group, session=None, rss=1, state='S'):
        fields = ['0'] * 22
        fields[0], fields[1], fields[2], fields[3] = state, '1', str(group), str(self.session if session is None else session)
        fields[19], fields[21] = '98765', str(rss)
        path = self.proc / str(pid) / 'stat'
        path.parent.mkdir(exist_ok=True)
        # The command field may contain spaces and closing parentheses.
        path.write_text(f'{pid} (mock ) worker) ' + ' '.join(fields) + '\n')
        return path

    def remove_killed_group(self, group, signum):
        if signum == signal.SIGKILL:
            for pid, process in campaign.session_processes(self.session).items():
                if process['group'] == group:
                    (self.proc / str(pid) / 'stat').unlink()

    def test_rss_counts_split_groups_and_orphans_only_in_the_owned_session(self):
        self.process(self.session, self.session, rss=2)
        self.process(self.session + 1, self.session + 1, rss=7)
        self.process(self.session + 2, self.session + 1, rss=3)
        self.process(self.session + 3, self.session + 3, session=9000000, rss=100000)
        malformed = self.proc / '9999999'; malformed.mkdir(); (malformed / 'stat').write_text('truncated')
        with patch.object(campaign.os, 'sysconf', return_value=4096):
            self.assertEqual(campaign.process_session_rss(self.session), 12 * 4096)
            (self.proc / str(self.session) / 'stat').unlink()
            self.assertEqual(campaign.process_session_rss(self.session), 10 * 4096,
                'exited wrapper must not hide the orphaned solver group')

    def test_dead_wrapper_cleanup_signals_all_owned_groups_and_no_other_session(self):
        solver_group, helper_group = self.session + 1, self.session + 2
        self.process(solver_group, solver_group)
        self.process(helper_group, helper_group)
        outside = self.process(self.session + 3, self.session + 3, session=9000000)
        child = Mock(pid=self.session, returncode=1)
        with patch.object(campaign.os, 'killpg', side_effect=self.remove_killed_group) as kill:
            campaign.stop_child(child)
        self.assertEqual({call.args for call in kill.call_args_list}, {
            (solver_group, signal.SIGTERM), (helper_group, signal.SIGTERM),
            (solver_group, signal.SIGKILL), (helper_group, signal.SIGKILL)})
        self.assertTrue(outside.exists())
        self.assertFalse(campaign.session_processes(self.session))
        self.assertEqual(child.wait.call_count, 2)

    def test_live_wrapper_timeout_escalates_for_separate_solver_group(self):
        solver_group = self.session + 1
        self.process(self.session, self.session)
        self.process(solver_group, solver_group)
        child = Mock(pid=self.session)
        child.wait.side_effect = [subprocess.TimeoutExpired('fake child', 10), 0]
        with patch.object(campaign.os, 'killpg', side_effect=self.remove_killed_group) as kill:
            campaign.stop_child(child)
        self.assertEqual(kill.call_args_list[0].args, (solver_group, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (self.session, signal.SIGTERM))
        self.assertIn((solver_group, signal.SIGKILL), [call.args for call in kill.call_args_list])
        self.assertFalse(campaign.session_processes(self.session))

    def test_cleanup_rescans_new_groups_and_does_not_reuse_stale_outside_group(self):
        old_group, new_group = self.session + 1, self.session + 2
        self.process(old_group, old_group)
        def signal_group(group, signum):
            if signum == signal.SIGTERM:
                self.process(old_group, old_group, session=9000000)
                self.process(new_group, new_group)
            else:
                self.remove_killed_group(group, signum)
        child = Mock(pid=self.session, returncode=1)
        with patch.object(campaign.os, 'killpg', side_effect=signal_group) as kill:
            campaign.stop_child(child)
        self.assertEqual([call.args for call in kill.call_args_list],
            [(old_group, signal.SIGTERM), (new_group, signal.SIGKILL)])
        self.assertTrue((self.proc / str(old_group) / 'stat').exists())

    def test_disappeared_groups_are_harmless(self):
        group = self.session + 1
        path = self.process(group, group)
        def disappear(group, signum):
            path.unlink()
            raise ProcessLookupError
        with patch.object(campaign.os, 'killpg', side_effect=disappear) as kill:
            campaign.stop_child(Mock(pid=self.session, returncode=0))
        self.assertEqual(kill.call_count, 1)
        self.assertFalse(campaign.session_processes(self.session))

    def test_cleanup_failure_keeps_recovery_explicit(self):
        self.process(self.session + 1, self.session + 1, state='D')
        with patch.object(campaign.os, 'killpg'), \
                patch.object(campaign.time, 'monotonic', side_effect=[0., 11.]), \
                self.assertRaisesRegex(RuntimeError, 'verify host processes before recovery'):
            campaign.stop_child(Mock(pid=self.session, returncode=0))

    def test_supervisor_session_is_never_signaled(self):
        with patch.object(campaign.os, 'killpg') as kill, \
                self.assertRaisesRegex(ValueError, 'not isolated'):
            campaign.signal_session(os.getsid(0), signal.SIGTERM)
        kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
