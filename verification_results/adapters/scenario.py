"""Isolated scenario smoke runs; full legacy CTest equivalence is a separate gate."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess


def execute(selected, arguments, plan=False):
    # An explicit executable without a configured build retains the diagnostic
    # smoke interface. It never certifies the registered regression suite.
    optionNames = {argument.split('=', 1)[0] for argument in arguments}
    if '--exe' in optionNames and '--build' not in optionNames:
        return executeSmoke(selected, arguments, plan)
    from verification_results.adapters import ctest_scenarios
    return ctest_scenarios.execute(selected, arguments, plan)


def executeSmoke(selected, arguments, plan=False):
    from verification_results import runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('buildType', nargs='?', default='Release')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--root', type=Path, default=runner.sourceRoot)
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--ctest', help='Accepted for shared CLI compatibility; unused by diagnostic smoke runs')
    parser.add_argument('--visit', type=Path, help='Accepted for shared CLI compatibility; unused by diagnostic smoke runs')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='Accepted for shared CLI compatibility; unused by diagnostic smoke runs')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    opts = parser.parse_args(arguments)
    builds = {v.lower(): v for v in ('Debug', 'Release', 'RelWithDebInfo')}
    if opts.buildType.lower() not in builds or opts.threads < 1:
        raise ValueError('Use Debug/Release/RelWithDebInfo and positive threads; scenario resolutions are fixed by legacy inputs')
    buildType = builds[opts.buildType.lower()]
    output = runner.safeOutput(opts.output or runner.defaultOutput('run'))
    root = opts.root.expanduser().resolve()
    source = root/'src/octotiger' if (root/'src/octotiger/test_problems').is_dir() else root
    build = (opts.build or root/'build/octotiger'/buildType.lower()).expanduser().resolve()
    exe = (opts.exe or build/'octotiger').expanduser().resolve()
    manifest = {
        'schema_version': 1, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source': {'commit': runner.gitValue('rev-parse', 'HEAD'), 'dirty': bool(runner.gitValue('status', '--porcelain'))},
        'harness': {'name': 'verification_results', 'adapter': 'scenario'},
        'build': runner.compilerMetadata(str(build), root),
        'execution': {'thread_count': opts.threads, 'arguments': arguments, 'requested_build_type': buildType,
                      'ctest_used': False},
        'tests': [],
    }
    manifest['build']['requested_build_type'] = buildType
    for path, value in selected:
        p = value['parameters']; config = source/p['config']
        configText = config.read_text() if config.is_file() else None
        identifier = f"{value['family']}.{value['suite']}.{value['name']}"
        item = {'identifier': identifier, 'family': value['family'], 'regime': value['regime'],
                'descriptor': value, 'parameters': p, 'status': 'planned',
                'descriptor_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'config_text': configText,
                'config_sha256': hashlib.sha256(configText.encode()).hexdigest() if configText is not None else None,
                'command': [str(exe), f'--runtime.config_file={config}', f'--hpx:threads={opts.threads}', *p.get('arguments', [])],
                'working_directory': str(output/identifier),
                'coverage': 'Scenario smoke run only: incomplete regex subset, no Silo comparison or kernel-variant coverage; cannot establish migration equivalence'}
        item['input_options'] = dict(line.split('=', 1) for raw in (configText or '').splitlines()
                                     if (line := raw.split('#', 1)[0].strip()) and '=' in line)
        manifest['tests'].append(item)
    if plan:
        print(json.dumps(manifest, indent=2)); return 0
    if output.exists() and any(output.iterdir()):
        raise ValueError('Scenario output must be empty; choose a new directory')
    output.mkdir(parents=True, exist_ok=True)
    for item in manifest['tests']:
        folder = Path(item['working_directory']); folder.mkdir()
        log = folder/'stdout.log'; item['checks'] = []
        if not exe.is_file() or not os.access(exe, os.X_OK):
            item.update(status='conditional', reason='Octo-TIGER executable unavailable')
        elif item['config_text'] is None:
            item.update(status='failed', reason='Missing legacy configuration')
        elif 'problem.input_file' in item['input_options']:
            item.update(status='conditional', reason='Required generated input must be prepared by legacy CTest fixture; standalone scenario is not equivalent')
        else:
            try:
                item['executable_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
                with log.open('w') as stream:
                    item['returncode'] = subprocess.run(item['command'], cwd=folder, stdout=stream,
                                                       stderr=subprocess.STDOUT, check=False).returncode
                text = log.read_text(errors='replace')
                item['checks'] = [{'pattern': p, 'passed': bool(re.search(p, text))}
                                  for p in item['parameters'].get('pass_regular_expressions', [])]
                failed = item['returncode'] != 0 or any(not c['passed'] for c in item['checks'])
                item.update(status='failed' if failed else 'conditional',
                            reason='Scenario execution/check failed' if failed else item['coverage'])
            except OSError as error:
                item.update(status='failed', reason=str(error))
        if not log.exists(): log.write_text(item['reason']+'\n')
        (folder/'run.json').write_text(json.dumps(item, indent=2)+'\n')
    failed = any(x['status'] == 'failed' for x in manifest['tests'])
    manifest['status'] = 'failed' if failed else 'conditional'
    (output/'verification.json').write_text(json.dumps(manifest, indent=2)+'\n')
    document = '<!doctype html><meta charset="utf-8"><h1>Scenario validation</h1><pre>'+html.escape(json.dumps(manifest, indent=2))+'</pre>'
    for name in ('index.html', 'report.html'): (output/name).write_text(document)
    return 1 if failed else 3
