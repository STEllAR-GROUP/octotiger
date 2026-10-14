#!/usr/bin/env python3
"""Paired entry-point checks with mocked delegation; no simulations."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


def load_wrapper(dependency):
    spec = importlib.util.spec_from_file_location('paired_wrapper_under_test', Path(__file__).with_name('run_both.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'run_campaign': dependency}):
        spec.loader.exec_module(module)
    return module


class PairedEntryPoint(unittest.TestCase):
    def setUp(self):
        self.campaign = types.SimpleNamespace(main=Mock(return_value=0))
        self.wrapper = load_wrapper(self.campaign)

    def test_prepare_delegates_without_adding_execution_or_legacy_options(self):
        arguments = ['--output', '/tmp/reviewed-pair', '--prepare', '--levels', '0', '1']
        self.assertEqual(self.wrapper.main(arguments), 0)
        self.campaign.main.assert_called_once_with(arguments)
        self.assertNotIn('--start', self.campaign.main.call_args.args[0])
        self.assertNotIn('--radiation-integrator', self.campaign.main.call_args.args[0])

    def test_start_is_only_forwarded_when_explicit(self):
        arguments = ['--resume', '/tmp/reviewed-pair', '--start']
        self.campaign.main.return_value = 7
        self.assertEqual(self.wrapper.main(arguments), 7)
        self.campaign.main.assert_called_once_with(arguments)

    def test_pair_selection_cannot_be_overridden_or_reinterpreted_as_ader(self):
        for option in ('--models', '--radiation-model', '--radiation-integrator'):
            for arguments in ([option, 'm1'], [option + '=ader2']):
                with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, 'always selects M1 RK1 and FLD'):
                    self.wrapper.main(['--output', '/tmp/unused', '--prepare', *arguments])
        self.campaign.main.assert_not_called()

    def test_command_line_arguments_are_forwarded_without_implicit_start(self):
        arguments = ['--resume', '/tmp/prepared', '--dry-run']
        with patch.object(sys, 'argv', ['run_both.py', *arguments]):
            self.wrapper.main()
        self.campaign.main.assert_called_once_with(arguments)

    def test_actual_default_dry_run_prepares_both_models_without_writes(self):
        spec = importlib.util.spec_from_file_location('paired_campaign_fixture', Path(__file__).with_name('run_campaign.py'))
        campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
        wrapper = load_wrapper(campaign)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'never-created'
            with patch.object(campaign.subprocess, 'Popen') as launch, \
                    patch.object(campaign.subprocess, 'run') as external, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(wrapper.main(['--output', str(output), '--dry-run']), 0)
            import json
            manifest = json.loads(stdout.getvalue())
            self.assertEqual(manifest['methods'], ['m1', 'fld'])
            self.assertEqual({run['integrator'] for run in manifest['runs']}, {'euler', 'backward_euler'})
            self.assertEqual(manifest['authorization'], 'awaiting_explicit_start')
            self.assertFalse(output.exists())
            launch.assert_not_called(); external.assert_not_called()


if __name__ == '__main__':
    unittest.main()
