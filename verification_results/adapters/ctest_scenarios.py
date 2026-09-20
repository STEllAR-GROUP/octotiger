"""Execute configured legacy CTests without translating their numerical checks.

CTest >= 3.21 supplies the authoritative command/property inventory and JUnit
statuses. Each scenario variant runs serially with all its fixtures and checks;
cleanup is delayed only until raw products have been copied and hashed. A pass
means this configured registration set passed, not before/after equivalence or
coverage of build options which were not configured.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET


def properties(test):
    return {p['name']: p['value'] for p in test.get('properties', [])}


def values(test, key):
    value = properties(test).get(key, [])
    return [value] if isinstance(value, str) else value


def exact_pattern(names):
    if not names:
        raise ValueError('Refusing an empty CTest selection')
    return '^(' + '|'.join(re.escape(name) for name in sorted(names)) + ')$'


def selected_groups(inventory, selector):
    """Partition variant checks, then close transitive fixture/dependency edges."""
    tests = inventory['tests']
    by_name = {t['name']: t for t in tests}
    if len(by_name) != len(tests):
        raise ValueError('Duplicate registered CTest names')
    include = re.compile(selector['include'])
    exclude = re.compile(selector['exclude']) if selector.get('exclude') else None
    matching = [t for t in tests if include.search(t['name']) and
                not (exclude and exclude.search(t['name']))]
    roots = [t for t in matching if values(t, 'FIXTURES_SETUP') and
             not values(t, 'FIXTURES_CLEANUP')]
    groups = []
    covered = set()
    for root in roots:
        names = {t['name'] for t in matching if t['name'] == root['name'] or
                 t['name'].startswith(root['name'] + '.')}
        changed = True
        while changed:
            before = set(names)
            fixtures = set()
            for name in list(names):
                test = by_name[name]
                fixtures.update(values(test, 'FIXTURES_REQUIRED'))
                fixtures.update(values(test, 'FIXTURES_SETUP'))
                # CTest expands cleanup DEPENDS to every consumer of a shared
                # fixture. They are scheduling edges, not prerequisites to
                # rerun all other scenario variants before cleanup.
                dependencies = [] if values(test, 'FIXTURES_CLEANUP') else values(test, 'DEPENDS')
                for dependency in dependencies:
                    if dependency not in by_name:
                        raise ValueError(f'Missing registered dependency {dependency}')
                    names.add(dependency)
            for test in tests:
                if fixtures.intersection(values(test, 'FIXTURES_SETUP') +
                                         values(test, 'FIXTURES_CLEANUP')):
                    names.add(test['name'])
            changed = names != before
        entries = [t for t in tests if t['name'] in names]
        cleanup = [t['name'] for t in entries if values(t, 'FIXTURES_CLEANUP')]
        run = [t['name'] for t in entries if t['name'] not in cleanup]
        checks = [t for t in entries if values(t, 'PASS_REGULAR_EXPRESSION') or
                  values(t, 'FAIL_REGULAR_EXPRESSION')]
        if not checks:
            raise ValueError(f'{root["name"]}: registered scenario has no numerical checks')
        covered.update(names)
        groups.append({'name': root['name'], 'run': run, 'cleanup': cleanup,
                       'registrations': entries})
    omitted = {t['name'] for t in matching} - covered
    if omitted:
        raise ValueError('Unassigned registered checks: ' + ', '.join(sorted(omitted)))
    return groups


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def registration_inputs(groups, build):
    """Record resolved executable/reference provenance and missing prerequisites."""
    files, missing = {}, set()
    for group in groups:
        for test in group['registrations']:
            command = test.get('command', [])
            if not command:
                missing.add(f'{test["name"]}: executable unresolved by CTest')
                continue
            executable = command[0]
            if Path(executable).name in {'sh', 'bash'} and len(command) > 2 and command[1] == '-c':
                words = shlex.split(command[2])
                if words:
                    executable = words[0]
            resolved = shutil.which(executable)
            if not resolved:
                missing.add(f'{test["name"]}: executable unavailable: {executable}')
            else:
                path = Path(resolved).resolve()
                if str(path) not in files:
                    files[str(path)] = {'kind': 'executable', 'sha256': hash_file(path)}
            for argument in command[1:]:
                path = Path(argument)
                # Silo references outside the build are immutable inputs, not
                # products expected to be supplied by a preceding setup test.
                if path.is_absolute() and path.suffix == '.silo' and not path.is_relative_to(build):
                    if not path.is_file():
                        missing.add(f'{test["name"]}: Silo reference unavailable: {path}')
                    else:
                        if str(path) not in files:
                            files[str(path)] = {'kind': 'silo_reference', 'sha256': hash_file(path)}
    return files, sorted(missing)


def raw_files(directory):
    """Retain numerical data/logs, not compiler objects or generated build rules."""
    if not directory.is_dir():
        return []
    result = []
    for path in directory.iterdir():
        if path.is_symlink():
            raise ValueError(f'Refusing symlink in scenario outputs: {path}')
        if path.name in {'CMakeCache.txt', 'CMakeLists.txt', 'CTestTestfile.cmake',
                         'cmake_install.cmake', 'Makefile'}:
            continue
        if path.is_file() and path.suffix in {'.silo', '.bin', '.txt', '.log', '.dat', '.csv'}:
            result.append(path)
        elif path.is_dir() and path.name.endswith('.silo.data'):
            for child in path.rglob('*'):
                if child.is_symlink():
                    raise ValueError(f'Refusing symlink in raw Silo tree: {child}')
                if child.is_file():
                    result.append(child)
    return sorted(result)


def work_directories(groups, build):
    directories = set()
    for group in groups:
        for test in group['registrations']:
            directory = Path(properties(test).get('WORKING_DIRECTORY', build)).resolve()
            if not directory.is_relative_to(build) or directory == build:
                raise ValueError(f'Expected isolated scenario working directory inside build: {directory}')
            command = test.get('command', [])
            if (values(test, 'FIXTURES_CLEANUP') and len(command) >= 4 and
                    Path(command[0]).name == 'cmake' and command[1] == '-E' and
                    command[2] in {'remove', 'remove_directory', 'rm'}):
                for argument in command[3:]:
                    if argument.startswith('-'):
                        continue
                    target = (directory / argument).resolve()
                    if target == directory or not target.is_relative_to(directory):
                        raise ValueError(f'Cleanup escapes scenario directory; reconfigure fixed registrations: {target}')
            directories.add(directory)
    return sorted(directories)


def archive_raw(groups, build, folder):
    artifacts = []
    directories = {folder, folder.parent, folder.parent.parent}
    for directory in work_directories(groups, build):
        for path in raw_files(directory):
            target = folder / 'raw' / path.relative_to(build)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            digest = hash_file(path)
            if digest != hash_file(target):
                raise ValueError(f'Raw artifact changed while archiving: {path}')
            with target.open('rb') as stream:
                os.fsync(stream.fileno())
            directories.update(parent for parent in target.parents if parent.is_relative_to(folder))
            artifacts.append({'source': str(path), 'path': str(target.relative_to(folder)),
                              'bytes': target.stat().st_size, 'sha256': digest})
    # Raw data must be durable before a legacy cleanup is allowed to delete its
    # source. A sync failure propagates and leaves that source untouched.
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return artifacts


def read_junit(path, expected):
    root = ET.parse(path).getroot()
    tests = []
    for case in root.iter('testcase'):
        skipped = case.find('skipped') is not None or case.get('status') in {'notrun', 'disabled'}
        failed = case.find('failure') is not None or case.find('error') is not None
        tests.append({'name': case.attrib['name'], 'status': 'failed' if failed else
                      'conditional' if skipped else 'passed', 'seconds': case.get('time'),
                      'output': case.findtext('system-out', '')})
    names = [test['name'] for test in tests]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError(f'CTest result inventory mismatch: expected {sorted(expected)}, got {sorted(names)}')
    return tests


def run_phase(base, build, names, folder, phase):
    junit = folder / (phase + '.xml')
    command = [*base, '-R', exact_pattern(names), '-FA', '.*', '-j', '1',
               '--no-tests=error', '--verbose', '--output-junit', str(junit)]
    with (folder / (phase + '.log')).open('w') as stream:
        result = subprocess.run(command, cwd=build, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    last_log = build / 'Testing/Temporary/LastTest.log'
    if last_log.is_file():
        shutil.copy2(last_log, folder / (phase + '-LastTest.log'))
    record = {'command': command, 'returncode': result.returncode,
              'checks': read_junit(junit, names)}
    return record


def aggregate(records):
    statuses = [r['status'] for r in records]
    return 'failed' if 'failed' in statuses else 'conditional' if 'conditional' in statuses or not statuses else 'passed'


def execute(selected, arguments, plan=False):
    from verification_results import runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build_type', nargs='?', default='Release')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--root', type=Path, default=runner.SOURCE_ROOT)
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--ctest', default='ctest')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    opts = parser.parse_args(arguments)
    modes = {v.lower(): v for v in ('Debug', 'Release', 'RelWithDebInfo')}
    if opts.build_type.lower() not in modes or opts.threads != 1:
        raise ValueError('CTest uses serial fixtures (--threads=1); application threads remain in the authoritative registrations')
    if opts.exe:
        raise ValueError('--exe cannot replace commands in a configured CTest build; omit --build for conditional smoke only')
    mode = modes[opts.build_type.lower()]
    root = opts.root.expanduser().resolve()
    source = root / 'src/octotiger' if (root / 'src/octotiger/test_problems').is_dir() else root
    build = (opts.build or root / 'build/octotiger' / mode.lower()).expanduser().resolve()
    output = runner.safe_output(opts.output or runner.default_output('ctest'))
    if output.is_relative_to(build):
        raise ValueError('CTest reports must be outside the configured build tree')
    if not plan and output.exists() and any(output.iterdir()):
        raise ValueError('CTest output must be empty; choose a fresh directory')
    base = [opts.ctest, '-C', mode]
    manifest = {'schema_version': 1, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'source': {'commit': runner.git_value('rev-parse', 'HEAD'),
                           'dirty': bool(runner.git_value('status', '--porcelain')),
                           'scope': 'harness checkout; actual executable/reference hashes recorded per descriptor'},
                'harness': {'name': 'verification_results', 'adapter': 'configured_ctest'},
                'build': {'directory': str(build), 'requested_build_type': mode},
                'execution': {'arguments': arguments, 'ctest_parallelism': 1,
                              'application_threads': 'unchanged configured CTest command/environment'},
                'coverage': 'All selected configured variants/checks/fixtures; not unconfigured hardware or before/after numerical equivalence',
                'tests': []}
    unavailable = None
    inventory = {'tests': []}
    try:
        if not (build / 'CTestTestfile.cmake').is_file():
            raise FileNotFoundError('Configured CTest build unavailable; configure and build Octo-TIGER with tests and reference data')
        version = subprocess.run([opts.ctest, '--version'], capture_output=True, text=True, check=True)
        match = re.search(r'ctest version (\d+)\.(\d+)', version.stdout)
        if not match or tuple(map(int, match.groups())) < (3, 21):
            raise FileNotFoundError('CTest >= 3.21 is required for complete JUnit result accounting')
        manifest['build']['ctest_version'] = version.stdout.strip()
        query = subprocess.run([*base, '--show-only=json-v1'], cwd=build,
                               capture_output=True, text=True, check=True)
        inventory = json.loads(query.stdout)
        manifest['build']['registration_sources'] = [
            {'path': str(path), 'sha256': hash_file(path) if path.is_file() else None}
            for path in map(Path, inventory.get('backtraceGraph', {}).get('files', []))]
        cache = build / 'CMakeCache.txt'
        if cache.is_file():
            manifest['build']['cache_sha256'] = hash_file(cache)
            home = re.search(r'^CMAKE_HOME_DIRECTORY:[^=]+=(.*)$', cache.read_text(), re.M)
            manifest['build']['configured_source_directory'] = home[1] if home else None
            configured = re.search(r'^CMAKE_BUILD_TYPE:[^=]+=(.*)$', cache.read_text(), re.M)
            multi = re.search(r'^CMAKE_CONFIGURATION_TYPES:[^=]+=(.*)$', cache.read_text(), re.M)
            if configured and configured[1] and not multi and configured[1] != mode:
                raise ValueError(f'Build is {configured[1]}, requested {mode}; refusing mislabeled run')
    except FileNotFoundError as error:
        unavailable = str(error)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        unavailable = f'CTest inventory unavailable: {error}'
    all_groups = []
    for path, descriptor in selected:
        identifier = '.'.join(descriptor[k] for k in ('family', 'suite', 'name'))
        config = source / descriptor['parameters']['config']
        item = {'identifier': identifier, 'descriptor': descriptor,
                'descriptor_sha256': hash_file(path), 'status': 'planned',
                'config_text': config.read_text() if config.is_file() else None,
                'config_sha256': hash_file(config) if config.is_file() else None,
                'groups': []}
        if unavailable:
            item.update(status='conditional', reason=unavailable)
        elif not descriptor.get('ctest'):
            item.update(status='conditional', reason='No authoritative CTest selector registered')
        else:
            item['groups'] = selected_groups(inventory, descriptor['ctest'])
            if not item['groups']:
                item.update(status='conditional', reason='No matching scenarios enabled in this configured build')
            all_groups.extend(item['groups'])
        manifest['tests'].append(item)
    if plan:
        manifest['inventory'] = inventory
        print(json.dumps(manifest, indent=2))
        return 0
    # Existing solver files may be irreplaceable. Refuse to let legacy commands
    # overwrite them; use a fresh build/test tree instead.
    existing = [str(path) for directory in work_directories(all_groups, build)
                for path in raw_files(directory)]
    if existing:
        raise ValueError('Existing scenario output would be overwritten: ' + ', '.join(existing))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'ctest-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    halted = None
    for item in manifest['tests']:
        folder = output / item['identifier']
        folder.mkdir()
        item['registered_inputs'], missing = registration_inputs(item['groups'], build)
        if missing:
            item.update(status='conditional', reason='Missing configured prerequisites', missing_prerequisites=missing)
        if halted:
            item.update(status='conditional', reason=halted)
        for group in item['groups']:
            if missing or halted:
                group.update(status='conditional', reason=halted or 'Missing configured prerequisites')
                continue
            location = folder / group['name']
            location.mkdir()
            try:
                group['execution'] = run_phase(base, build, group['run'], location, 'checks')
                # Never run cleanup after an archive failure: raw evidence is
                # left in place, and the next invocation refuses to overwrite it.
                group['artifacts'] = archive_raw([group], build, location)
                if not group['artifacts']:
                    raise ValueError('No raw numerical/log artifacts were retained; refusing cleanup/certification')
                if group['cleanup']:
                    group['cleanup_execution'] = run_phase(base, build, group['cleanup'], location, 'cleanup')
                phases = [group['execution']] + ([group['cleanup_execution']] if group['cleanup'] else [])
                group['status'] = aggregate([check for phase in phases for check in phase['checks']])
                if any(phase['returncode'] != 0 for phase in phases):
                    group['status'] = 'failed'
            except (OSError, ValueError, ET.ParseError) as error:
                group.update(status='failed', reason=str(error))
                # Preserve logs/data even when CTest produced malformed results.
                try:
                    if not (location / 'raw').exists():
                        group['artifacts'] = archive_raw([group], build, location)
                except (OSError, ValueError) as archive_error:
                    group['archive_error'] = str(archive_error)
            (location / 'run.json').write_text(json.dumps(group, indent=2) + '\n')
            if group['status'] == 'failed' and ('cleanup_execution' not in group or
                    group['cleanup_execution']['returncode'] != 0):
                # Do not start a later variant over unarchived or uncleaned data.
                halted = 'Execution halted after an archive/cleanup failure; raw build data retained'
        if item['groups']:
            for group in item['groups']:
                group.setdefault('status', 'conditional')
            item['status'] = aggregate(item['groups'])
        (folder / 'run.json').write_text(json.dumps(item, indent=2) + '\n')
    manifest['status'] = aggregate(manifest['tests'])
    (output / 'verification.json').write_text(json.dumps(manifest, indent=2) + '\n')
    document = '<!doctype html><meta charset="utf-8"><h1>Configured CTest validation</h1><pre>' + html.escape(json.dumps(manifest, indent=2)) + '</pre>'
    for name in ('index.html', 'report.html'):
        (output / name).write_text(document)
    return {'passed': 0, 'failed': 1, 'conditional': 3}[manifest['status']]
