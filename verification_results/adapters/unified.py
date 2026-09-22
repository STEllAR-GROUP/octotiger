"""Dispatch mixed families into disjoint outputs and preserve every status."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def applicationCommand(opts, levels, build, output, sourceRoot, case):
    """Build the retained full-application radiation command."""
    from verification_results.adapters import radiation_results

    arguments = [*levels]
    if opts.build:
        arguments += ['--build', str(opts.build)]
    else:
        arguments += build
    if opts.root:
        arguments += ['--root', str(opts.root)]
    if opts.exe:
        arguments += ['--exe', str(opts.exe)]
    arguments += ['--threads', str(opts.threads), '--output', str(output),
                  '--ffmpeg', opts.ffmpeg, '--cxx', opts.cxx, '--no-open']
    if opts.visit:
        arguments += ['--visit', str(opts.visit)]
    mode = 'live' if opts.live else 'run'
    return radiation_results.command(sourceRoot, mode, case, arguments)


def runApplication(command, sourceRoot):
    return subprocess.run(command, cwd=sourceRoot, check=False,
                          env=os.environ.copy()).returncode


def writeProgress(out, results, runner, site):
    statuses = [record['status'] for record in results]
    summary = {
        'source_commit': runner.gitValue('rev-parse', 'HEAD'),
        'families': results,
        'status': ('failed' if 'failed' in statuses else
                   'running' if 'running' in statuses else
                   'conditional' if 'conditional' in statuses else
                   'passed' if statuses and all(value == 'passed' for value in statuses) else
                   'running'),
        'coverage': 'All registered descriptors dispatched; explicitly conditional cases remain unvalidated',
    }
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    site.write(out, runner.descriptors())


def familyStatus(records):
    statuses = {record['status'] for record in records}
    if 'failed' in statuses:
        return 'failed'
    if 'running' in statuses:
        return 'running'
    if 'conditional' in statuses:
        return 'conditional'
    return 'passed'


def execute(selected, arguments, plan=False):
    from verification_results import runner
    from verification_results.adapters import native_suite, scenario
    from verification_results.web import site
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('settings', nargs='*')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--ctest', default='ctest')
    parser.add_argument('--visit', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--no-open', action='store_true')
    opts = parser.parse_args(arguments)
    levels = [v for v in opts.settings if v.isdigit()]
    builds = [v for v in opts.settings if not v.isdigit()]
    if len(builds) > 1 or (builds and builds[0].lower() not in {'debug', 'release', 'relwithdebinfo'}):
        raise ValueError('Use one supported build type')
    if opts.threads < 1:
        raise ValueError('Mixed suite requires a positive thread count')
    if [int(x) for x in levels] != sorted(set(map(int, levels))) or any(int(x) > 5 for x in levels):
        raise ValueError('Mixed suite requires distinct increasing radiation levels 0..5')
    build = builds or ['Release']
    out = runner.safeOutput(opts.output or runner.defaultOutput('suite'))
    groups = []
    for family in ('hydro', 'gravity'):
        cases = [(key, path, d) for key, (path, d) in selected.items() if d['family'] == family]
        if not cases: continue
        args = [*build, '--threads', str(opts.threads), '--output', str(out/family)]
        args += ['--ctest', opts.ctest, '--ffmpeg', opts.ffmpeg]
        if opts.visit:
            args += ['--visit', str(opts.visit)]
        for flag in ('root', 'build', 'exe'):
            if getattr(opts, flag): args += ['--'+flag, str(getattr(opts, flag))]
        groups.append((family, 'configured', scenario.execute,
                       [(path, d) for _, path, d in cases], args))

    radiation = [(key, path, descriptor) for key, (path, descriptor) in selected.items()
                 if descriptor['family'] == 'radiation']
    native = [(key, descriptor) for key, _, descriptor in radiation
              if descriptor['adapter']['name'] != 'radiation_results']
    if native:
        args = [*levels, *build, '--threads', str(opts.threads),
                '--output', str(out/'radiation'), '--cxx', opts.cxx, '--ffmpeg', opts.ffmpeg]
        groups.append(('radiation', 'methods', native_suite.execute, native, args))
    application = [(key, descriptor) for key, _, descriptor in radiation
                   if descriptor['adapter']['name'] == 'radiation_results']
    if application:
        cases = [descriptor['adapter']['case'] for _, descriptor in application]
        case = 'all' if set(cases) == {'streaming_wave', 'streaming_front',
                                      'gaussian_pulse', 'equilibrium_sphere'} else cases[0]
        command = applicationCommand(opts, levels, build, out/'radiation'/'application',
                                       runner.sourceRoot, case)
        groups.append(('radiation', 'application', None, application, command))
    if plan:
        for _, component, callback, inputs, args in groups:
            if component == 'application':
                print(json.dumps({'component': component, 'tests': [key for key, _ in inputs],
                                  'command': args}, indent=2))
            else:
                callback(inputs, args, plan=True)
        return 0
    if out.exists() and any(out.iterdir()): raise ValueError('Suite output must be empty')
    out.mkdir(parents=True, exist_ok=True)
    print(f'Verification output: {out}', flush=True)
    print(f'Live report: {out/"index.html"}', flush=True)
    results = []
    components = []
    writeProgress(out, results, runner, site)
    if opts.live and not opts.no_open:
        try:
            subprocess.Popen(['xdg-open', str(out/'index.html')], cwd=out,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as error:
            print(f'Open the page manually: {error}', flush=True)
    for family, component, callback, inputs, args in groups:
        running = {'family': family, 'component': component, 'returncode': None,
                   'status': 'running'}
        components.append(running)
        familyRecords = [record for record in components if record['family'] == family]
        results = [record for record in results if record['family'] != family]
        results.append({'family': family, 'status': familyStatus(familyRecords),
                        'components': familyRecords})
        writeProgress(out, results, runner, site)
        try:
            if component == 'application':
                print('+ ' + ' '.join(map(str, args)), flush=True)
                code = runApplication(args, runner.sourceRoot)
            else:
                code = callback(inputs, args)
            running.update(returncode=code,
                           status='passed' if code == 0 else 'conditional' if code == 3 else 'failed')
        except Exception as error:
            message = f'{family}/{component} setup failed: {error}'
            # A top-level adapter error used to be visible only as a red tile
            # in the browser.  Emit it immediately as well as retaining it in
            # the live report, so a terminal run is diagnosable.
            print(message, flush=True)
            (out / f'{family}-{component}-error.log').write_text(message + '\n')
            running.update(returncode=1, status='failed', reason=message)
        familyRecords = [record for record in components if record['family'] == family]
        results = [record for record in results if record['family'] != family]
        results.append({'family': family, 'status': familyStatus(familyRecords),
                        'components': familyRecords})
        writeProgress(out, results, runner, site)
    return 1 if any(r['status'] == 'failed' for r in results) else 3 if any(r['status'] == 'conditional' for r in results) else 0
