"""Regression tests for Step 07 harness defects; stand-ins are not physics evidence."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from verification_results import runner
from verification_results.adapters import native_suite, scenario, unified


class FinalValidationTests(unittest.TestCase):
    def test_scenario_failure_is_visible_and_outputs_are_isolated(self):
        descriptor = runner.descriptors()['hydro.sod.sod']
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'results'
            def fake(command, cwd, stdout, **kw):
                self.assertEqual(Path(cwd), out/'hydro.sod.sod')
                self.assertIn('--hpx:threads=3', command)
                stdout.write('incorrect density\n')
                (Path(cwd)/'final.silo').write_bytes(b'test fixture')
                return subprocess.CompletedProcess(command, 0)
            with patch.object(scenario.subprocess, 'run', side_effect=fake), patch.object(runner, 'git_value', return_value='test'):
                rc = scenario.execute([descriptor], ['rElEaSe', '--exe', '/bin/true', '--threads', '3', '--output', str(out)])
            self.assertEqual(rc, 1)
            result = json.loads((out/'verification.json').read_text())['tests'][0]
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['input_options']['max_level'], '1')
            self.assertTrue((out/'hydro.sod.sod/final.silo').exists())

    def test_zero_checks_and_zero_exit_cannot_certify_regression(self):
        descriptor = runner.descriptors()['hydro.amr.sod_big_amr']
        with tempfile.TemporaryDirectory() as tmp:
            code = scenario.execute([descriptor], ['--exe', '/bin/true', '--output', tmp])
            self.assertEqual(code, 3)
            self.assertEqual(json.loads((Path(tmp)/'verification.json').read_text())['tests'][0]['status'], 'conditional')

    def test_missing_executable_records_options_and_non_success(self):
        descriptor = runner.descriptors()['hydro.sod.sod']
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(scenario.execute([descriptor], ['--exe', '/nonexistent/octotiger', '--output', tmp]), 3)
            result = json.loads((Path(tmp)/'verification.json').read_text())['tests'][0]
            self.assertIn('command', result)
            self.assertIn('gravity=off', result['config_text'])
            self.assertTrue((Path(tmp)/'hydro.sod.sod/stdout.log').exists())

    def test_equals_syntax_smoke_accepts_shared_ctest_option_without_using_it(self):
        descriptor = runner.descriptors()['hydro.amr.sod_big_amr']
        with tempfile.TemporaryDirectory() as tmp:
            code = scenario.execute([descriptor], ['--exe=/bin/true', '--ctest=/nonexistent/ctest', '--output='+tmp])
            self.assertEqual(code, 3)
            result = json.loads((Path(tmp)/'verification.json').read_text())
            self.assertEqual(result['harness']['adapter'], 'scenario')
            self.assertFalse(result['execution']['ctest_used'])
            self.assertEqual(result['tests'][0]['status'], 'conditional')

    def test_equals_build_keeps_configured_route_and_rejects_executable_override(self):
        from verification_results.adapters import ctest_scenarios
        for arguments in (['--exe=/bin/true', '--build=/some/build'],
                          ['--exe', '/bin/true', '--build=/some/build'],
                          ['--exe=/bin/true', '--build', '/some/build']):
            with self.subTest(arguments=arguments), patch.object(ctest_scenarios, 'execute', return_value=3) as configured:
                self.assertEqual(scenario.execute([], arguments, plan=True), 3)
                configured.assert_called_once_with([], arguments, True)
        with self.assertRaisesRegex(ValueError, '--exe cannot replace'):
            scenario.execute([], ['--exe=/bin/true', '--build=/some/build'])

    def test_mixed_suite_explicit_executable_accepts_forwarded_ctest_option(self):
        available = runner.descriptors()
        selected = {key: available[key] for key in ('hydro.amr.sod_big_amr',
                    'gravity.rotating_star.rotating_star', 'radiation.ensman.equilibrium_sphere')}
        with tempfile.TemporaryDirectory() as tmp, patch.object(native_suite, 'execute', return_value=3):
            code = unified.execute(selected, ['Release', '--exe=/bin/true',
                                   '--ctest=/nonexistent/ctest', '--output='+tmp])
            self.assertEqual(code, 3)
            result = json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual([r['status'] for r in result['families']], ['conditional']*3)
            for family in ('hydro', 'gravity'):
                manifest = json.loads((Path(tmp)/family/'verification.json').read_text())
                self.assertFalse(manifest['execution']['ctest_used'])
                self.assertEqual(manifest['harness']['adapter'], 'scenario')

    def test_output_reuse_and_source_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'keep').write_text('data')
            with self.assertRaises(ValueError): scenario.execute([], ['--output', tmp])
        with self.assertRaises(runner.HarnessError): scenario.execute([], ['--output', str(runner.SOURCE_ROOT)])

    def test_mixed_suite_aggregates_failures_and_distinct_outputs(self):
        outputs = []
        def fake_scenario(inputs, args, **kw):
            outputs.append(runner.option_value(args, '--output'))
            self.assertNotIn('2', args)
            return 1 if inputs[0][1]['family'] == 'hydro' else 3
        def fake_native(inputs, args, **kw):
            outputs.append(runner.option_value(args, '--output'))
            self.assertEqual(args[:3], ['0', '1', '2'])
            return 0
        with tempfile.TemporaryDirectory() as tmp, patch.object(scenario, 'execute', side_effect=fake_scenario), patch.object(native_suite, 'execute', side_effect=fake_native):
            self.assertEqual(unified.execute(runner.descriptors(), ['0', '1', '2', 'Release', '--output', tmp]), 1)
            report = json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual([r['status'] for r in report['families']], ['failed', 'conditional', 'passed'])
            self.assertEqual(len(set(outputs)), 3)
            self.assertIn('radiation/report.html', (Path(tmp)/'index.html').read_text())

    def test_plan_all_and_family_are_read_only_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as text:
            out = Path(tmp)/'not-created'
            self.assertEqual(runner.main(['plan', 'all', '0', '1', '2', 'Debug', '--output', str(out)]), 0)
            self.assertFalse(out.exists())
            for name in ('hydro.sod.sod', 'gravity.sphere.self_gravitating_sphere', 'radiation.ensman.equilibrium_sphere'):
                self.assertIn(name, text.getvalue())
            self.assertEqual(runner.main(['plan', 'hydro', '--output', str(out)]), 0)

    def test_scenario_metadata_matches_inputs_and_schema_contract(self):
        for _, d in runner.descriptors().values():
            self.assertIn(d['build_type'], ('Debug', 'Release', 'RelWithDebInfo'))
            for key in ('reference_data', 'tolerance_policy'):
                self.assertIn('owner', d[key]); self.assertIn('description', d[key])
            if d['adapter']['name'] == 'octotiger_scenario':
                config = (runner.SOURCE_ROOT/d['parameters']['config']).read_text()
                self.assertIn('max_level='+str(d['resolution_levels'][0]), config)
        d = runner.descriptors()['gravity.sphere.self_gravitating_sphere'][1]
        pattern = d['parameters']['pass_regular_expressions'][1]
        self.assertRegex('pot 4.470774e-01 3.176879e+00', pattern)
        self.assertIn(pattern, (runner.SOURCE_ROOT/'test_problems/sphere/CMakeLists.txt').read_text())


if __name__ == '__main__': unittest.main()
