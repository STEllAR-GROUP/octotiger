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
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET


def properties(test):
    return {p['name']: p['value'] for p in test.get('properties', [])}


def values(test, key):
    value = properties(test).get(key, [])
    return [value] if isinstance(value, str) else value


def exactPattern(names):
    if not names:
        raise ValueError('Refusing an empty CTest selection')
    return '^(' + '|'.join(re.escape(name) for name in sorted(names)) + ')$'


def selectedGroups(inventory, selector):
    """Partition variant checks, then close transitive fixture/dependency edges."""
    tests = inventory['tests']
    byName = {t['name']: t for t in tests}
    if len(byName) != len(tests):
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
                test = byName[name]
                fixtures.update(values(test, 'FIXTURES_REQUIRED'))
                fixtures.update(values(test, 'FIXTURES_SETUP'))
                # CTest expands cleanup DEPENDS to every consumer of a shared
                # fixture. They are scheduling edges, not prerequisites to
                # rerun all other scenario variants before cleanup.
                dependencies = [] if values(test, 'FIXTURES_CLEANUP') else values(test, 'DEPENDS')
                for dependency in dependencies:
                    if dependency not in byName:
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
                  values(t, 'FAIL_REGULAR_EXPRESSION') or
                  'numerical' in values(t, 'LABELS')]
        if not checks:
            raise ValueError(f'{root["name"]}: registered scenario has no numerical checks')
        covered.update(names)
        groups.append({'name': root['name'], 'run': run, 'cleanup': cleanup,
                       'registrations': entries})
    omitted = {t['name'] for t in matching} - covered
    if omitted:
        raise ValueError('Unassigned registered checks: ' + ', '.join(sorted(omitted)))
    return groups


def hashFile(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def registrationInputs(groups, build):
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
                    files[str(path)] = {'kind': 'executable', 'sha256': hashFile(path)}
            for argument in command[1:]:
                path = Path(argument)
                # Silo references outside the build are immutable inputs, not
                # products expected to be supplied by a preceding setup test.
                if path.is_absolute() and path.suffix == '.silo' and not path.is_relative_to(build):
                    if not path.is_file():
                        missing.add(f'{test["name"]}: Silo reference unavailable: {path}')
                    else:
                        if str(path) not in files:
                            files[str(path)] = {'kind': 'silo_reference', 'sha256': hashFile(path)}
    return files, sorted(missing)


def rawFiles(directory):
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


def workDirectories(groups, build):
    """Return only the directories that own a selected scenario's products.

    Legacy Octo-TIGER CTests normally execute from the build root and put their
    products below ``test_problems/<case>``.  Treating the build root itself as
    a scenario directory rejected every such registration before CTest could
    run.  Cleanup registrations already name the product paths authoritatively,
    so use those paths to identify the isolated product directories.
    """
    directories = set()
    for group in groups:
        for test in group['registrations']:
            directory = Path(properties(test).get('WORKING_DIRECTORY', build)).resolve()
            if not directory.is_relative_to(build):
                raise ValueError(f'Expected isolated scenario working directory inside build: {directory}')
            if directory != build:
                directories.add(directory)
            command = test.get('command', [])
            if (values(test, 'FIXTURES_CLEANUP') and len(command) >= 4 and
                    Path(command[0]).name == 'cmake' and command[1] == '-E' and
                    command[2] in {'remove', 'remove_directory', 'rm'}):
                for argument in command[3:]:
                    if argument.startswith('-'):
                        continue
                    target = (directory / argument).resolve()
                    # Legacy CTests may execute from a nested CMake binary
                    # directory while their cleanup command names the shared
                    # test_problems/<case> product directory absolutely.
                    # It must remain within this configured build, but need
                    # not be a child of CTest's reported working directory.
                    if target == build or not target.is_relative_to(build):
                        raise ValueError(f'Cleanup escapes scenario directory; reconfigure fixed registrations: {target}')
                    # CMake's legacy registrations run in the build root but
                    # clean files in test_problems/<case>.  Archive/preflight
                    # that case directory, not the entire build tree.
                    directories.add(target if command[2] == 'remove_directory' else target.parent)
    return sorted(directories)


def archiveRaw(groups, build, folder):
    artifacts = []
    directories = {folder, folder.parent, folder.parent.parent}
    for directory in workDirectories(groups, build):
        for path in rawFiles(directory):
            target = folder / 'raw' / path.relative_to(build)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            digest = hashFile(path)
            if digest != hashFile(target):
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


def quarantineExisting(paths, build, output):
    """Preserve stale build-tree products before a new configured run.

    Interrupted legacy CTests can leave valuable Silo states and diagnostics in
    the configured build tree.  A fresh verification output directory should
    not require the user to delete those files by hand.  Copy, hash, sync, and
    only then unlink each source so the next CTest cannot overwrite evidence.
    """
    destination = output / 'preexisting-build-output'
    records = []
    for source in sorted(set(paths)):
        source = source.resolve()
        if not source.is_file() or not source.is_relative_to(build):
            raise ValueError(f'Refusing to quarantine unsafe build output: {source}')
        target = destination / source.relative_to(build)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f'Quarantine target already exists: {target}')
        digest = hashFile(source)
        shutil.copy2(source, target)
        if hashFile(target) != digest:
            raise ValueError(f'Quarantined artifact changed while copying: {source}')
        with target.open('rb') as stream:
            os.fsync(stream.fileno())
        descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        source.unlink()
        records.append({'source': str(source),
                        'path': str(target.relative_to(output)),
                        'bytes': target.stat().st_size, 'sha256': digest})
    (destination / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


def readJunit(path, expected):
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


def hpxEnvironment(threads):
    """Override HPX worker threads without changing CTest fixture parallelism."""
    environment = os.environ.copy()
    words = shlex.split(environment.get('HPX_COMMANDLINE_OPTIONS', ''))
    retained = []
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        if word == '--hpx:threads':
            skip = True
        elif not word.startswith('--hpx:threads='):
            retained.append(word)
    retained.append(f'--hpx:threads={threads}')
    environment['HPX_COMMANDLINE_OPTIONS'] = shlex.join(retained)
    return environment


def redirectedLogs(registrations, names, build):
    """Return files to which legacy shell registrations hide solver stdout."""
    answer = []
    selected = set(names)
    for test in registrations:
        if test['name'] not in selected:
            continue
        command = test.get('command', [])
        if len(command) < 3 or Path(command[0]).name not in {'sh', 'bash'} or command[1] != '-c':
            continue
        words = shlex.split(command[2])
        targets = []
        for index, word in enumerate(words):
            if word in {'>', '1>'} and index + 1 < len(words):
                targets.append(words[index + 1])
            elif word.startswith('>') and len(word) > 1:
                targets.append(word.lstrip('>'))
        directory = Path(properties(test).get('WORKING_DIRECTORY', build))
        for target in targets:
            path = Path(target)
            resolved = path if path.is_absolute() else directory / path
            if resolved not in answer:
                answer.append(resolved)
    return answer


def followLogs(paths, stop):
    offsets = {path: 0 for path in paths}
    while True:
        for path in paths:
            try:
                size = path.stat().st_size
                if size < offsets[path]:
                    offsets[path] = 0
                if size > offsets[path]:
                    with path.open(errors='replace') as stream:
                        stream.seek(offsets[path])
                        text = stream.read()
                        offsets[path] = stream.tell()
                    if text:
                        sys.stdout.write(text)
                        sys.stdout.flush()
            except (FileNotFoundError, OSError):
                pass
        if stop.is_set():
            return
        time.sleep(0.05)


def runPhase(base, build, names, folder, phase, registrations, threads):
    junit = folder / (phase + '.xml')
    command = [*base, '-R', exactPattern(names), '-FA', '.*', '-j', '1',
               '--no-tests=error', '--output-on-failure', '--output-junit', str(junit)]
    environment = hpxEnvironment(threads)
    paths = redirectedLogs(registrations, names, build)
    stop = threading.Event()
    follower = threading.Thread(target=followLogs, args=(paths, stop), daemon=True)
    follower.start()
    try:
        with (folder / (phase + '.log')).open('w') as stream:
            process = subprocess.Popen(command, cwd=build, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       env=environment)
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    stream.write(line)
                    stream.flush()
            finally:
                process.stdout.close()
            returncode = process.wait()
    finally:
        stop.set()
        follower.join()
    lastLog = build / 'Testing/Temporary/LastTest.log'
    if lastLog.is_file():
        shutil.copy2(lastLog, folder / (phase + '-LastTest.log'))
    record = {'command': command, 'returncode': returncode,
              'ctest_parallelism': 1, 'application_threads': threads,
              'HPX_COMMANDLINE_OPTIONS': environment['HPX_COMMANDLINE_OPTIONS'],
              'checks': readJunit(junit, names)}
    return record


def aggregate(records):
    statuses = [r['status'] for r in records]
    return 'failed' if 'failed' in statuses else 'conditional' if 'conditional' in statuses or not statuses else 'passed'


def execute(selected, arguments, plan=False):
    from verification_results import runner
    from verification_results.adapters import ctest_visuals
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('buildType', nargs='?', default='Release')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--root', type=Path, default=runner.sourceRoot)
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--ctest', default='ctest')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--visit', type=Path)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--output', type=Path)
    opts = parser.parse_args(arguments)
    modes = {v.lower(): v for v in ('Debug', 'Release', 'RelWithDebInfo')}
    if opts.buildType.lower() not in modes or opts.threads < 1:
        raise ValueError('Use a supported build type and a positive HPX thread count')
    if opts.exe:
        raise ValueError('--exe cannot replace commands in a configured CTest build; omit --build for conditional smoke only')
    mode = modes[opts.buildType.lower()]
    root = opts.root.expanduser().resolve()
    source = root / 'src/octotiger' if (root / 'src/octotiger/test_problems').is_dir() else root
    build = (opts.build or root / 'build/octotiger' / mode.lower()).expanduser().resolve()
    output = runner.safeOutput(opts.output or runner.defaultOutput('ctest'))
    if output.is_relative_to(build):
        raise ValueError('CTest reports must be outside the configured build tree')
    if not plan and output.exists() and any(output.iterdir()):
        raise ValueError('CTest output must be empty; choose a fresh directory')
    base = [opts.ctest, '-C', mode]
    manifest = {'schema_version': 1, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'source': {'commit': runner.gitValue('rev-parse', 'HEAD'),
                           'dirty': bool(runner.gitValue('status', '--porcelain')),
                           'scope': 'harness checkout; actual executable/reference hashes recorded per descriptor'},
                'harness': {'name': 'verification_results', 'adapter': 'configured_ctest'},
                'build': {'directory': str(build), 'requested_build_type': mode},
                'execution': {'arguments': arguments, 'ctest_parallelism': 1,
                              'application_threads': opts.threads,
                              'hpx_override': f'HPX_COMMANDLINE_OPTIONS=--hpx:threads={opts.threads}'},
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
            {'path': str(path), 'sha256': hashFile(path) if path.is_file() else None}
            for path in map(Path, inventory.get('backtraceGraph', {}).get('files', []))]
        cache = build / 'CMakeCache.txt'
        if cache.is_file():
            manifest['build']['cache_sha256'] = hashFile(cache)
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
    allGroups = []
    for path, descriptor in selected:
        identifier = '.'.join(descriptor[k] for k in ('family', 'suite', 'name'))
        config = source / descriptor['parameters']['config']
        item = {'identifier': identifier, 'descriptor': descriptor,
                'descriptor_sha256': hashFile(path), 'status': 'planned',
                'config_text': config.read_text() if config.is_file() else None,
                'config_sha256': hashFile(config) if config.is_file() else None,
                'groups': []}
        if unavailable:
            item.update(status='conditional', reason=unavailable)
        elif not descriptor.get('ctest'):
            item.update(status='conditional', reason='No authoritative CTest selector registered')
        else:
            item['groups'] = selectedGroups(inventory, descriptor['ctest'])
            if not item['groups']:
                item.update(status='conditional', reason='No matching scenarios enabled in this configured build')
            allGroups.extend(item['groups'])
        manifest['tests'].append(item)
    if plan:
        manifest['inventory'] = inventory
        print(json.dumps(manifest, indent=2))
        return 0
    # Preserve products left by an interrupted/older run.  The new result
    # directory is fresh, so it is a safe per-run quarantine and no evidence is
    # discarded merely to let CTest start again.
    existing = [path for directory in workDirectories(allGroups, build)
                for path in rawFiles(directory)]
    output.mkdir(parents=True, exist_ok=True)
    if existing:
        manifest['preexisting_build_output'] = quarantineExisting(existing, build, output)
        print(f'Quarantined {len(existing)} pre-existing build artifacts in '
              f'{output / "preexisting-build-output"}', flush=True)
    print(f'Configured CTest: fixture_parallelism=1 application_hpx_threads={opts.threads}',
          flush=True)
    (output / 'ctest-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    ctest_visuals.writeReport(output, manifest)
    halted = None
    for item in manifest['tests']:
        folder = output / item['identifier']
        folder.mkdir()
        item['registered_inputs'], missing = registrationInputs(item['groups'], build)
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
            group['status'] = 'running'
            item['status'] = 'running'
            ctest_visuals.writeReport(output, manifest)
            try:
                group['execution'] = runPhase(base, build, group['run'], location, 'checks',
                                               group['registrations'], opts.threads)
                # Never run cleanup after an archive failure: raw evidence is
                # left in place, and the next invocation refuses to overwrite it.
                group['artifacts'] = archiveRaw([group], build, location)
                if not group['artifacts']:
                    raise ValueError('No raw numerical/log artifacts were retained; refusing cleanup/certification')
                group['visualization'] = ctest_visuals.visualize(
                    location, item['descriptor'], opts.visit, opts.ffmpeg)
                if group['cleanup']:
                    group['cleanup_execution'] = runPhase(base, build, group['cleanup'], location, 'cleanup',
                                                           group['registrations'], opts.threads)
                phases = [group['execution']] + ([group['cleanup_execution']] if group['cleanup'] else [])
                group['status'] = aggregate([check for phase in phases for check in phase['checks']])
                if any(phase['returncode'] != 0 for phase in phases):
                    group['status'] = 'failed'
                if opts.visit and group['visualization']['status'] == 'failed':
                    group['status'] = 'failed'
                    group['reason'] = 'Required visual products failed: ' + group['visualization']['reason']
            except (OSError, ValueError, ET.ParseError) as error:
                group.update(status='failed', reason=str(error))
                # Preserve logs/data even when CTest produced malformed results.
                try:
                    if not (location / 'raw').exists():
                        group['artifacts'] = archiveRaw([group], build, location)
                except (OSError, ValueError) as archive_error:
                    group['archive_error'] = str(archive_error)
            (location / 'run.json').write_text(json.dumps(group, indent=2) + '\n')
            ctest_visuals.writeReport(output, manifest)
            if group['status'] == 'failed' and ('cleanup_execution' not in group or
                    group['cleanup_execution']['returncode'] != 0):
                # Do not start a later variant over unarchived or uncleaned data.
                halted = 'Execution halted after an archive/cleanup failure; raw build data retained'
        if item['groups']:
            for group in item['groups']:
                group.setdefault('status', 'conditional')
            item['status'] = aggregate(item['groups'])
        (folder / 'run.json').write_text(json.dumps(item, indent=2) + '\n')
        ctest_visuals.writeReport(output, manifest)
    manifest['status'] = aggregate(manifest['tests'])
    ctest_visuals.writeReport(output, manifest)
    return {'passed': 0, 'failed': 1, 'conditional': 3}[manifest['status']]
