"""CTest integration evidence only: synthetic files are not physics results."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from verification_results import runner
from verification_results.adapters import ctest_scenarios as adapter, ctest_visuals, scenario

root = runner.sourceRoot
cmake = os.environ.get('OCTOTIGER_TEST_CMAKE') or shutil.which('cmake')
ctest = os.environ.get('OCTOTIGER_TEST_CTEST') or shutil.which('ctest')


class SelectionTests(unittest.TestCase):
    def test_stale_build_products_are_quarantined_before_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); build = root/'build'; output = root/'results'
            source = build/'test_problems/sod/final.silo'
            source.parent.mkdir(parents=True); source.write_bytes(b'valuable old state')
            output.mkdir()
            records = adapter.quarantineExisting([source], build, output)
            retained = output/records[0]['path']
            self.assertFalse(source.exists())
            self.assertEqual(retained.read_bytes(), b'valuable old state')
            self.assertEqual(adapter.hashFile(retained), records[0]['sha256'])
            self.assertTrue((output/'preexisting-build-output/manifest.json').is_file())

    def test_human_report_contains_plots_and_no_raw_json_dump(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            folder = output/'gravity.sphere.self_gravitating_sphere'/'legacy'
            raw = folder/'raw/case'; raw.mkdir(parents=True)
            (raw/'line.final.dat').write_text('0 1 2 3 4\n1 2 3 4 5\n')
            descriptor = runner.descriptors()['gravity.sphere.self_gravitating_sphere'][1]
            visual = ctest_visuals.visualize(folder, descriptor, None, 'ffmpeg')
            self.assertEqual(visual['status'], 'not_requested')
            self.assertTrue((folder/'plots/line-profile.png').is_file())
            manifest = {'execution': {'application_threads': 12}, 'tests': [{
                'identifier': 'gravity.sphere.self_gravitating_sphere',
                'descriptor': descriptor, 'status': 'passed', 'groups': [{
                    'name': 'legacy', 'status': 'passed', 'visualization': visual,
                    'execution': {'checks': [{'name': 'rho_regex', 'status': 'passed', 'seconds': '.1'}]},
                    'artifacts': []}]}]}
            ctest_visuals.writeReport(output, manifest)
            document = (output/'report.html').read_text()
            self.assertIn('line-profile.png', document)
            self.assertIn('rho_regex', document)
            self.assertNotIn('<pre>', document)

    def test_product_policy_renders_hydro_movie_but_not_gravity_movie(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visit = root/'visit'; visit.write_text('fixture')

            def fakeRun(command, **kwargs):
                if '-cli' in command:
                    render = Path(kwargs['cwd'])/'rendered'
                    records = []
                    fields = ('pot', 'rho') if 'gravity.' in str(kwargs['cwd']) else ('rho', 'egas')
                    for field in fields:
                        for frame in range(2):
                            image = render/field/f'frame-{frame:04d}.png'
                            image.parent.mkdir(parents=True, exist_ok=True)
                            image.write_bytes(b'png fixture')
                            records.append({'requested': field, 'variable': field, 'frame': frame,
                                            'database': 'fixture', 'path': str(image)})
                    (render/'visit-products.json').write_text(json.dumps(records))
                elif '-c:v' in command:
                    Path(command[-1]).write_bytes(b'mp4 fixture')
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            for identifier, expectsMovie in (('hydro.sod.sod', True),
                                               ('gravity.sphere.self_gravitating_sphere', False)):
                folder = root/identifier
                raw = folder/'raw/case'; raw.mkdir(parents=True)
                for name in ('X.0.silo', 'final.silo'):
                    (raw/name).write_text('fixture')
                descriptor = runner.descriptors()[identifier][1]
                with patch.object(ctest_visuals.subprocess, 'run', side_effect=fakeRun):
                    products = ctest_visuals.siloProducts(folder, descriptor, visit, 'ffmpeg')
                movies = [item for item in products if item['kind'] == 'video']
                self.assertEqual(bool(movies), expectsMovie)

    def test_requested_hpx_threads_override_environment_without_parallel_ctest(self):
        with patch.dict(os.environ, {'HPX_COMMANDLINE_OPTIONS': '--hpx:bind=balanced --hpx:threads=2'}):
            environment = adapter.hpxEnvironment(12)
        self.assertIn('--hpx:bind=balanced', environment['HPX_COMMANDLINE_OPTIONS'])
        self.assertIn('--hpx:threads=12', environment['HPX_COMMANDLINE_OPTIONS'])
        self.assertNotIn('--hpx:threads=2', environment['HPX_COMMANDLINE_OPTIONS'])

    def test_redirected_solver_log_is_discovered_for_live_following(self):
        registration = {'name': 'solver', 'command': ['sh', '-c', '/tmp/octotiger > solver.log'],
                        'properties': [{'name': 'WORKING_DIRECTORY', 'value': '/tmp/build/case'}]}
        self.assertEqual(adapter.redirectedLogs([registration], ['solver'], Path('/tmp/build')),
                         [Path('/tmp/build/case/solver.log')])

    def test_ctest_and_redirected_solver_output_are_streamed_and_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build, case, result = root/'build', root/'build/case', root/'result'
            case.mkdir(parents=True); result.mkdir()
            program = ('import pathlib,sys,time; '
                       'p=pathlib.Path("case/solver.log"); '
                       'p.write_text("octotiger line 1\\n"); time.sleep(.12); '
                       'p.open("a").write("octotiger line 2\\n"); '
                       'j=pathlib.Path(sys.argv[sys.argv.index("--output-junit")+1]); '
                       'j.write_text("<testsuite><testcase name=\\"solver\\"/></testsuite>"); '
                       'print("ctest progress",flush=True)')
            registration = {'name': 'solver',
                            'command': ['sh', '-c', '/tmp/octotiger > solver.log'],
                            'properties': [{'name': 'WORKING_DIRECTORY', 'value': str(case)}]}
            with contextlib.redirect_stdout(io.StringIO()) as output:
                record = adapter.runPhase([sys.executable, '-c', program], build,
                                           ['solver'], result, 'checks', [registration], 12)
            visible = output.getvalue()
            self.assertEqual(record['returncode'], 0)
            self.assertEqual(record['application_threads'], 12)
            self.assertIn('ctest progress', visible)
            self.assertIn('octotiger line 1', visible)
            self.assertIn('octotiger line 2', visible)
            self.assertIn('ctest progress', (result/'checks.log').read_text())

    def test_sod_big_and_sod_do_not_overlap(self):
        descriptors = runner.descriptors()
        import re
        def matches(key, name):
            selector = descriptors[key][1]['ctest']
            return bool(re.search(selector['include'], name) and not
                        (selector.get('exclude') and re.search(selector['exclude'], name)))
        self.assertTrue(matches('hydro.sod.sod', 'test_problems.gpu.sod_kokkos_cuda.rho_regex'))
        self.assertFalse(matches('hydro.sod.sod', 'test_problems.gpu.sod_big_kokkos_cuda.rho_regex'))
        self.assertTrue(matches('hydro.amr.sod_big_amr', 'test_problems.gpu.sod_big_kokkos_cuda.rho_regex'))
        self.assertFalse(matches('hydro.amr.sod_big_amr', 'test_problems.gpu.sod_kokkos_cuda.rho_regex'))

    def test_missing_build_is_conditional(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'result'
            code = scenario.execute([runner.descriptors()['hydro.sod.sod']],
                                    ['--build', str(Path(temporary) / 'missing'), '--output', str(out)])
            self.assertEqual(code, 3)
            result = json.loads((out / 'verification.json').read_text())
            self.assertEqual(result['tests'][0]['status'], 'conditional')
            self.assertIn('Configured CTest', result['tests'][0]['reason'])

    def test_junit_missing_and_disabled_checks_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            xml = Path(temporary) / 'result.xml'
            xml.write_text('<testsuite><testcase name="a"><skipped/></testcase></testsuite>')
            self.assertEqual(adapter.readJunit(xml, ['a'])[0]['status'], 'conditional')
            with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
                adapter.readJunit(xml, ['a', 'missing'])

    def test_unresolved_executable_and_silo_reference_are_explicit(self):
        groups = [{'registrations': [
            {'name': 'unresolved'},
            {'name': 'shell', 'command': ['sh', '-c', '/nonexistent/octotiger --runtime.config_file=x']},
            {'name': 'diff', 'command': ['/bin/true', '/nonexistent/reference.silo']},
        ]}]
        files, missing = adapter.registrationInputs(groups, Path('/configured/build'))
        self.assertEqual(len(missing), 3)
        self.assertTrue(any(value['kind'] == 'executable' for value in files.values()))

    def test_cleanup_outside_configured_build_is_refused(self):
        groups = [{'registrations': [{'name': 'cleanup',
                    'command': ['/usr/bin/cmake', '-E', 'remove', '/outside-build/final.silo'],
                    'properties': [{'name': 'FIXTURES_CLEANUP', 'value': ['sod']},
                                   {'name': 'WORKING_DIRECTORY', 'value': '/build/test_problems/sod'}]}]}]
        with self.assertRaisesRegex(ValueError, 'Cleanup escapes'):
            adapter.workDirectories(groups, Path('/build'))


@unittest.skipUnless(cmake and ctest, 'Set OCTOTIGER_TEST_CMAKE and OCTOTIGER_TEST_CTEST for real CTest harness integration')
class CTestIntegrationTests(unittest.TestCase):
    def configure(self, source, build, *extra):
        result = subprocess.run([cmake, '-S', str(source), '-B', str(build),
                                 '-DCMAKE_BUILD_TYPE=Release', *extra],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def fixture(self, temporary, *, failed=False, disabled=False):
        source, build = Path(temporary) / 'source', Path(temporary) / 'build'
        source.mkdir()
        cases = source / 'case'
        cases.mkdir()
        (source / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.21)\nproject(AdapterFixture NONE)\nenable_testing()\nadd_subdirectory(case)\n')
        generation = "from pathlib import Path; Path('raw.log').write_text('rho 1.0 2.0\\n'); Path('final.silo').write_text('" + ('rho.vals' if failed else 'synthetic data') + "'); Path('final.silo.data').mkdir(exist_ok=True); Path('final.silo.data/0.silo').write_text('synthetic field')"
        init = "from pathlib import Path; Path('rotating_star.bin').write_bytes(b'generated fixture')"
        # CMake bracket arguments preserve Python quoting exactly.
        (cases / 'CMakeLists.txt').write_text(f'''
add_test(NAME test_problems.rotating_star.init COMMAND "{sys.executable}" -c [=[{init}]=])
set_tests_properties(test_problems.rotating_star.init PROPERTIES FIXTURES_SETUP init)
foreach(variant IN ITEMS legacy kokkos)
  set(name test_problems.cpu.rotating_star_${{variant}})
  add_test(NAME ${{name}} COMMAND "{sys.executable}" -c [=[{generation}]=])
  set_tests_properties(${{name}} PROPERTIES FIXTURES_SETUP ${{name}} FIXTURES_REQUIRED init)
  add_test(NAME ${{name}}.rho_regex COMMAND ${{CMAKE_COMMAND}} -E cat raw.log)
  set_tests_properties(${{name}}.rho_regex PROPERTIES FIXTURES_REQUIRED ${{name}} PASS_REGULAR_EXPRESSION "rho 1.0 2.0")
  add_test(NAME ${{name}}.diff COMMAND ${{CMAKE_COMMAND}} -E cat final.silo)
  set_tests_properties(${{name}}.diff PROPERTIES FIXTURES_REQUIRED ${{name}} FAIL_REGULAR_EXPRESSION ".vals")
  add_test(NAME ${{name}}.fixture_cleanup COMMAND ${{CMAKE_COMMAND}} -E rm -f raw.log final.silo final.silo.data/0.silo)
  set_tests_properties(${{name}}.fixture_cleanup PROPERTIES FIXTURES_CLEANUP ${{name}})
endforeach()
add_test(NAME test_problems.rotating_star.init.fixture_cleanup COMMAND ${{CMAKE_COMMAND}} -E rm -f rotating_star.bin)
set_tests_properties(test_problems.rotating_star.init.fixture_cleanup PROPERTIES FIXTURES_CLEANUP init)
''')
        if disabled:
            with (cases / 'CMakeLists.txt').open('a') as stream:
                stream.write('set_tests_properties(test_problems.cpu.rotating_star_legacy.diff PROPERTIES DISABLED TRUE)\n')
        self.configure(source, build)
        return build

    def executeFixture(self, temporary, build, name='results', plan=False):
        output = Path(temporary) / name
        code = scenario.execute([runner.descriptors()['gravity.rotating_star.rotating_star']],
                                ['--build', str(build), '--ctest', ctest, '--output', str(output)], plan=plan)
        return code, output

    def test_real_ctest_all_checks_generator_and_cleanup_with_raw_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = self.fixture(temporary)
            code, output = self.executeFixture(temporary, build)
            self.assertEqual(code, 0)
            result = json.loads((output / 'verification.json').read_text())
            groups = result['tests'][0]['groups']
            self.assertEqual(len(groups), 2)
            for group in groups:
                self.assertEqual(group['status'], 'passed')
                self.assertEqual(len(group['execution']['checks']), 4)
                self.assertEqual(len(group['cleanup_execution']['checks']), 2)
                self.assertEqual(len(group['artifacts']), 4)
                location = output / result['tests'][0]['identifier'] / group['name']
                for artifact in group['artifacts']:
                    self.assertEqual(adapter.hashFile(location / artifact['path']), artifact['sha256'])
                self.assertIn('synthetic data', (location / 'raw/case/final.silo').read_text())
            self.assertFalse((build / 'case/final.silo').exists())
            self.assertFalse((build / 'case/rotating_star.bin').exists())

    def test_real_ctest_silo_regex_failure_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = self.fixture(temporary, failed=True)
            code, output = self.executeFixture(temporary, build)
            self.assertEqual(code, 1)
            result = json.loads((output / 'verification.json').read_text())
            self.assertEqual(result['status'], 'failed')
            for group in result['tests'][0]['groups']:
                self.assertEqual(group['status'], 'failed')
                self.assertEqual(group['cleanup_execution']['returncode'], 0)

    def test_disabled_registered_check_is_conditional(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = self.fixture(temporary, disabled=True)
            code, output = self.executeFixture(temporary, build)
            self.assertEqual(code, 3)
            self.assertEqual(json.loads((output / 'verification.json').read_text())['status'], 'conditional')

    def test_plan_read_only_and_existing_build_data_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = self.fixture(temporary)
            data = build / 'case/final.silo'
            data.write_text('irreplaceable previous result')
            with contextlib.redirect_stdout(io.StringIO()) as text:
                code, output = self.executeFixture(temporary, build, plan=True)
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            self.assertIn('rho_regex', text.getvalue())
            with self.assertRaisesRegex(ValueError, 'would be overwritten'):
                self.executeFixture(temporary, build)
            self.assertEqual(data.read_text(), 'irreplaceable previous result')

    def test_requested_configuration_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = self.fixture(temporary)
            with self.assertRaisesRegex(ValueError, 'requested Debug'):
                scenario.execute([runner.descriptors()['gravity.rotating_star.rotating_star']],
                                 ['Debug', '--build', str(build), '--ctest', ctest,
                                  '--output', str(Path(temporary) / 'results')])

    def test_archive_failure_keeps_build_raw_data_and_stops_later_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = self.fixture(temporary)
            with patch.object(adapter, 'archiveRaw', side_effect=OSError('simulated full disk')):
                code, output = self.executeFixture(temporary, build)
            self.assertEqual(code, 1)
            result = json.loads((output / 'verification.json').read_text())
            groups = result['tests'][0]['groups']
            self.assertEqual(groups[0]['status'], 'failed')
            self.assertEqual(groups[1]['status'], 'conditional')
            self.assertNotIn('execution', groups[1])
            self.assertNotIn('cleanup_execution', groups[0])
            self.assertTrue((build / 'case/final.silo').is_file())
            self.assertTrue((build / 'case/rotating_star.bin').is_file())

    def test_authoritative_registrations_inventory_and_sod_kernel_arguments(self):
        # Configure the actual legacy CMake registration files, not a rewritten
        # list. No Octo-TIGER build or numerical execution is claimed here.
        for grid in (8, 16):
            for backend in ('CPU', 'CUDA', 'HIP', 'SYCL'):
                with self.subTest(grid=grid, backend=backend), tempfile.TemporaryDirectory() as temporary:
                    source, build = Path(temporary) / 'source', Path(temporary) / 'build'
                    source.mkdir()
                    cmake = f'''cmake_minimum_required(VERSION 3.21)
project(LegacyRegistrations NONE)
enable_testing()
set(PROJECT_SOURCE_DIR "{root}")
set(PROJECT_BINARY_DIR "{build}")
set(OCTOTIGER_WITH_GRIDDIM {grid})
set(OCTOTIGER_WITH_KOKKOS ON)
set(OCTOTIGER_WITH_CUDA {'ON' if backend == 'CUDA' else 'OFF'})
set(OCTOTIGER_WITH_HIP {'ON' if backend == 'HIP' else 'OFF'})
set(Kokkos_ENABLE_SYCL {'ON' if backend == 'SYCL' else 'OFF'})
set(OCTOTIGER_SILODIFF_FAIL_PATTERN ".vals")
set(Silo_BROWSER /bin/true)
add_executable(gen_rotating_star_init IMPORTED)
set_target_properties(gen_rotating_star_init PROPERTIES IMPORTED_LOCATION /bin/true)
'''
                    for problem in ('sod', 'blast', 'star', 'sphere', 'rotating_star'):
                        cmake += f'add_subdirectory("{root}/test_problems/{problem}" "test_problems/{problem}")\n'
                    (source / 'CMakeLists.txt').write_text(cmake)
                    self.configure(source, build)
                    result = subprocess.run([ctest, '-C', 'Release', '--show-only=json-v1'],
                                            cwd=build, capture_output=True, text=True, check=True)
                    inventory = json.loads(result.stdout)
                    covered = set()
                    for _, descriptor in runner.descriptors().values():
                        if descriptor['family'] not in ('hydro', 'gravity'):
                            continue
                        groups = adapter.selectedGroups(inventory, descriptor['ctest'])
                        covered.update(t['name'] for group in groups for t in group['registrations'])
                    self.assertEqual(covered, {t['name'] for t in inventory['tests']})
                    cleanupNames = [t['name'] for t in inventory['tests']
                                     if t['name'] == 'test_problems.rotating_star.init.fixture_cleanup']
                    self.assertEqual(len(cleanupNames), 1)
                    for test in inventory['tests']:
                        if '.sod_' in test['name'] and adapter.values(test, 'FIXTURES_SETUP'):
                            self.assertIn('--hydro_host_kernel_type=', ' '.join(test['command']))
                            self.assertIn('--hydro_device_kernel_type=', ' '.join(test['command']))
                        if '.fixture_cleanup' in test['name'] and ('.sod_' in test['name'] or '.blast_' in test['name']):
                            self.assertNotIn('/test_problems/sphere/', ' '.join(test['command']))
                    evidence = os.environ.get('OCTOTIGER_CTEST_EVIDENCE')
                    if evidence:
                        target = Path(evidence)
                        target.mkdir(parents=True, exist_ok=True)
                        (target / f'griddim{grid}-{backend.lower()}-inventory.json').write_text(result.stdout)


class LegacySodShellTests(unittest.TestCase):
    def test_assignments_quoted_paths_and_fail_fast(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case = directory / 'sod'
                case.mkdir()
                (case / 'sod.ini').write_text('fixture only\n')
                (case / 'original.silo').write_text('fixture only\n')
                executable = directory / 'fake octotiger'
                comparator = directory / 'fake silodiff'
                executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > args.txt\n' + ('exit 7\n' if fail else 'touch final.silo\n'))
                comparator.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > diff-args.txt\n')
                executable.chmod(0o755)
                comparator.chmod(0o755)
                result = subprocess.run(['sh', str(root / 'test_problems/test_sod.sh'),
                                         str(executable), str(comparator)], cwd=directory,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 7 if fail else 0, result.stderr)
                self.assertEqual(
                    (case / 'args.txt').read_text(), '--runtime.config_file=sod.ini\n'
                )
                self.assertEqual((case / 'diff-args.txt').exists(), not fail)
                if not fail:
                    self.assertEqual((case / 'diff-args.txt').read_text().splitlines(),
                                     ['-A', '1.0e-10', '-R', '1.0e-10', 'original.silo', 'final.silo'])


if __name__ == '__main__':
    unittest.main()
