"""Dispatch mixed families into disjoint outputs and preserve every status."""
import argparse
import json
from pathlib import Path


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
    opts = parser.parse_args(arguments)
    levels = [v for v in opts.settings if v.isdigit()]
    builds = [v for v in opts.settings if not v.isdigit()]
    if len(builds) > 1 or (builds and builds[0].lower() not in {'debug', 'release', 'relwithdebinfo'}):
        raise ValueError('Use one supported build type')
    if opts.threads != 1 or [int(x) for x in levels] != sorted(set(map(int, levels))) or any(int(x) > 5 for x in levels):
        raise ValueError('Mixed suite requires threads=1 and distinct increasing radiation levels 0..5')
    build = builds or ['Release']
    out = runner.safe_output(opts.output or runner.default_output('suite'))
    groups = []
    for family in ('hydro', 'gravity', 'radiation'):
        cases = [(key, path, d) for key, (path, d) in selected.items() if d['family'] == family]
        if not cases: continue
        args = [*build, '--threads', str(opts.threads), '--output', str(out/family)]
        if family == 'radiation':
            args = [*levels, *args, '--cxx', opts.cxx, '--ffmpeg', opts.ffmpeg]
            callback = native_suite.execute; inputs = [(key, d) for key, _, d in cases]
        else:
            args += ['--ctest', opts.ctest]
            for flag in ('root', 'build', 'exe'):
                if getattr(opts, flag): args += ['--'+flag, str(getattr(opts, flag))]
            callback = scenario.execute; inputs = [(path, d) for _, path, d in cases]
        groups.append((family, callback, inputs, args))
    if plan:
        for _, callback, inputs, args in groups: callback(inputs, args, plan=True)
        return 0
    if out.exists() and any(out.iterdir()): raise ValueError('Suite output must be empty')
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for family, callback, inputs, args in groups:
        try:
            code = callback(inputs, args)
            record = {'family': family, 'returncode': code,
                      'status': 'passed' if code == 0 else 'conditional' if code == 3 else 'failed'}
        except Exception as error:
            record = {'family': family, 'returncode': 1, 'status': 'failed', 'reason': repr(error)}
        results.append(record)
        summary = {'source_commit': runner.git_value('rev-parse', 'HEAD'), 'families': results,
                   'status': 'failed' if any(r['status'] == 'failed' for r in results) else 'conditional' if any(r['status'] == 'conditional' for r in results) else 'passed',
                   'coverage': 'All registered descriptors dispatched; conditional application/Ensman cases remain unvalidated'}
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        site.write(out, runner.descriptors())
    return 1 if any(r['status'] == 'failed' for r in results) else 3 if any(r['status'] == 'conditional' for r in results) else 0
