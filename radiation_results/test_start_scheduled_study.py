#!/usr/bin/env python3
"""The old scheduled entry point is now an explicit, guarded local wrapper."""
import contextlib
import importlib.util
import io
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

DIRECTORY = Path(__file__).resolve().parent


class ExplicitStudyEntryPoint(unittest.TestCase):
    def load(self, name, dependency):
        spec = importlib.util.spec_from_file_location(name + '_under_test', DIRECTORY / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'run_campaign': dependency}):
            spec.loader.exec_module(module)
        return module

    def test_import_does_not_schedule_or_start_any_work(self):
        delegate = Mock(return_value=0)
        for name in ('start_scheduled_study', 'run_local'):
            with self.subTest(name=name):
                module = self.load(name, types.SimpleNamespace(main=delegate))
                self.assertIs(module.main, delegate)
        delegate.assert_not_called()

    def test_script_entry_point_delegates_once_and_preserves_exit_status(self):
        for name in ('start_scheduled_study', 'run_local'):
            delegate = Mock(return_value=3)
            with self.subTest(name=name), patch.dict(sys.modules, {'run_campaign': types.SimpleNamespace(main=delegate)}), \
                    patch.object(sys, 'argv', [name + '.py', '--resume', '/tmp/prepared', '--dry-run']):
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(DIRECTORY / (name + '.py')), run_name='__main__')
                self.assertEqual(result.exception.code, 3)
                delegate.assert_called_once_with()

    def test_real_guard_requires_explicit_action_and_prepared_resume(self):
        spec = importlib.util.spec_from_file_location('study_campaign_fixture', DIRECTORY / 'run_campaign.py')
        campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
        wrapper = self.load('start_scheduled_study', campaign)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'never-created'
            with patch.object(campaign.subprocess, 'Popen') as launch, \
                    patch.object(campaign.subprocess, 'run') as external, \
                    contextlib.redirect_stderr(io.StringIO()):
                for arguments in (['--output', str(output)], ['--output', str(output), '--start']):
                    with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                        wrapper.main(arguments)
                launch.assert_not_called(); external.assert_not_called()
            self.assertFalse(output.exists())

    def test_local_wrapper_can_review_one_model_without_starting_it(self):
        spec = importlib.util.spec_from_file_location('local_campaign_fixture', DIRECTORY / 'run_campaign.py')
        campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
        wrapper = self.load('run_local', campaign)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'single-model'
            with patch.object(campaign.subprocess, 'Popen') as launch, \
                    patch.object(campaign.subprocess, 'run') as external, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(wrapper.main(['--output', str(output), '--dry-run', '--models', 'fld']), 0)
            import json
            manifest = json.loads(stdout.getvalue())
            self.assertEqual(manifest['methods'], ['fld'])
            self.assertTrue(all(run['model'] == 'fld' for run in manifest['runs']))
            self.assertFalse(output.exists())
            launch.assert_not_called(); external.assert_not_called()


if __name__ == '__main__':
    unittest.main()
