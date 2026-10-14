#!/usr/bin/env python3
"""Prepare and explicitly start local M1 RK1 / stationary FLD comparisons.

Preparation freezes sources, executables, libraries and scripts and writes a
local website. Only --resume DIR --start starts solver processes. The default
pilot has Gaussian and Marshak, levels 0/1, two workers and physical light
speed. Unsupported FLD cases are shown explicitly, never simulated as M1.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
CASES = ('streaming_wave', 'streaming_front', 'gaussian_pulse', 'equilibrium_sphere',
         'subcritical_shock', 'supercritical_shock', 'radiating_blast', 'marshak')
PAIRED_CASES = ('gaussian_pulse', 'marshak')
FLD_CASES = ('gaussian_pulse', 'equilibrium_sphere', 'marshak')
FLD_UNAVAILABLE = {
    'streaming_wave': 'Zero-opacity FLD iteration has not converged; transparent plateau treatment requires a solver extension',
    'streaming_front': 'Zero-opacity FLD iteration has not converged; transparent plateau treatment requires a solver extension',
    'subcritical_shock': 'Moving-gas radiation hydrodynamics is required; the current FLD solver is stationary',
    'supercritical_shock': 'Moving-gas radiation hydrodynamics is required; the current FLD solver is stationary',
    'radiating_blast': 'Moving-gas radiation hydrodynamics is required; the current FLD solver is stationary',
}
METHODS = ('m1', 'fld')
INTEGRATORS = {'m1': 'euler', 'fld': 'backward_euler'}
LABELS = {'m1': 'M1 RK1', 'fld': 'FLD backward Euler'}
FLUX = {'m1': 'evolved_m1_flux', 'fld': 'derived_fld_flux'}
WORKFLOW = 'm1-fld-radiation'
SCHEMA = 2


def execution_plan(manifest):
    """Fields which must never change after preparation, apart from run state."""
    keys = ('schema', 'workflow', 'methods', 'levels', 'cases', 'unsupported', 'threads', 'snapshots',
            'minimum_free_gib', 'memory_limit_gib', 'original_build', 'original_results_build', 'original_source')
    result = {key: manifest[key] for key in keys}
    run_keys = ('id', 'case', 'level', 'method', 'model', 'integrator', 'backend', 'directory', 'stop_time', 'hard_dt')
    result['runs'] = [{key: run[key] for key in run_keys} for run in manifest['runs']]
    return result


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(root):
    data = json.loads((root / 'campaign.json').read_text())
    if data.get('schema') != SCHEMA or data.get('workflow') != WORKFLOW:
        raise ValueError('This command requires a new M1/FLD campaign; historical campaigns retain their frozen scripts')
    return data


def plan(args):
    available = os.cpu_count() or 1
    if not 1 <= args.threads <= available:
        raise ValueError(f'--threads must be 1..{available}, within the available CPU count')
    if args.snapshots < 3:
        raise ValueError('At least three snapshots are required')
    for name in ('time', 'hard_dt', 'minimum_free_gib', 'memory_limit_gib'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and positive')
    if args.hard_dt > args.time:
        raise ValueError('--hard-dt must not exceed --time')
    models = [model for model in METHODS if model in args.models]
    cases = [case for case in CASES if case in args.cases]
    levels = sorted(set(args.levels))
    runs, unsupported = [], []
    for level in levels:
        for case in cases:
            for model in models:
                if model == 'fld' and case not in FLD_CASES:
                    unsupported.append({'case': case, 'level': level, 'model': model,
                        'reason': FLD_UNAVAILABLE[case]})
                    continue
                settings = args.case_settings.get(case, {})
                stop_time = settings.get('stop_time', args.time)
                hard_dt = settings.get('hard_dt', args.hard_dt)
                if not all(math.isfinite(value) and value > 0 for value in (stop_time, hard_dt)) or hard_dt > stop_time:
                    raise ValueError(f'Invalid time/timestep settings for {case}')
                runs.append({'id': f'l{level}-{case}-{model}', 'case': case, 'level': level,
                    'method': model, 'model': model, 'integrator': INTEGRATORS[model], 'backend': 'cpu',
                    'directory': f'runs/l{level}/{case}/{model}', 'status': 'pending', 'attempts': [],
                    'stop_time': stop_time, 'hard_dt': hard_dt})
    if not runs:
        raise ValueError('No supported model/problem combinations selected')
    return {'schema': SCHEMA, 'workflow': WORKFLOW, 'created_utc': now(), 'status': 'prepared',
        'active': False, 'authorization': 'awaiting_explicit_start', 'methods': models,
        'levels': levels, 'cases': cases, 'runs': runs, 'unsupported': unsupported,
        'threads': args.threads, 'snapshots': args.snapshots, 'minimum_free_gib': args.minimum_free_gib,
        'memory_limit_gib': args.memory_limit_gib, 'original_build': str(args.build.resolve()),
        'original_results_build': str(args.results_build.resolve()), 'original_source': str(ROOT),
        'scope': args.study_scope + ' Model discrepancies are not FLD error estimates. '
                 'FLD uses a serial root-gathered multigrid solve. No moving-gas or scaling claim.',
        'publication': {'status': 'local_only'}, 'require_comparisons': True}


def libraries(executable, env=None):
    output = subprocess.check_output(['ldd', str(executable)], text=True, env=env)
    if 'not found' in output:
        raise ValueError(f'Unresolved libraries: {executable}')
    return sorted({Path(match) for match in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(', output, re.M)})


def environment(root, threads=2):
    result = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    # Preserve one complete runtime, including HPX's dlopen alias.
    result['LD_LIBRARY_PATH'] = str(root / '.campaign/runtime')
    result.pop('LD_PRELOAD', None)
    return result


def snapshot(root, manifest):
    frozen = root / '.campaign'; source = frozen / 'source'; runtime = frozen / 'runtime'
    source.mkdir(parents=True); runtime.mkdir()
    write_json(frozen / 'plan.json', execution_plan(manifest))
    for name in ('src', 'octotiger', 'frontend', 'test_problems', 'cmake'):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'CMakeLists.txt', source / 'CMakeLists.txt')
    for name in ('configs', 'cpp'):
        shutil.copytree(ROOT / 'radiation_results' / name, source / 'radiation_results' / name)
    for name in ('run_campaign.py', 'compare_campaign.py', 'campaign_movies.py', 'publish_campaign.py',
                 'campaign_profile_plots.py', 'local-study.json', 'full-study.json', 'movie-study.json'):
        shutil.copy2(ROOT / 'radiation_results' / name, frozen / name)
    build = Path(manifest['original_build']); results = Path(manifest['original_results_build'])
    shutil.copy2(build / 'CMakeCache.txt', runtime / 'CMakeCache.txt')
    manifest['runtime_dependencies'] = {}
    for name, binary in {'octotiger': build / 'octotiger', 'radiation-results': results / 'radiation-results',
                         'gen_radiation_reference': results / 'gen_radiation_reference'}.items():
        shutil.copy2(binary, runtime / name)
        for library in libraries(binary):
            digest = file_hash(library)
            old = manifest['runtime_dependencies'].get(library.name)
            if old and old['sha256'] != digest:
                raise ValueError(f'Conflicting library {library.name}')
            manifest['runtime_dependencies'][library.name] = {'original_path': str(library), 'sha256': digest}
            shutil.copy2(library, runtime / library.name)
    versions = [p.name for p in runtime.glob('libhpx.so.*') if re.fullmatch(r'libhpx\.so\.\d+', p.name)]
    if len(versions) != 1:
        raise ValueError('Expected one HPX versioned library for the shared dlopen alias')
    (runtime / 'libhpx.so').symlink_to(versions[0])
    manifest['runtime_library_aliases'] = {'libhpx.so': versions[0]}
    launcher = frozen / 'launch-solver'
    launcher.write_text('#!/bin/sh\nexec "$@" --hpx:bind=none\n'); launcher.chmod(0o755)
    manifest['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest['executable_sha256'] = file_hash(runtime / 'octotiger')
    manifest['snapshot_sha256'] = {str(p.relative_to(root)): file_hash(p)
                                   for p in sorted(frozen.rglob('*')) if p.is_file()}
    manifest['host_elf_interpreters'] = {}
    for binary in ('octotiger', 'radiation-results', 'gen_radiation_reference'):
        resolved = libraries(runtime / binary, environment(root))
        original_names = manifest['runtime_dependencies']
        for path in resolved:
            if path.parent.resolve() == runtime.resolve():
                continue
            # The kernel resolves PT_INTERP before LD_LIBRARY_PATH applies.
            # Preserve the executable bytes and pin this one host dependency.
            if re.fullmatch(r'ld-(?:linux|musl)[\w.-]*\.so(?:\.\d+)*', path.name) and path.name in original_names:
                digest = file_hash(path)
                if digest == original_names[path.name]['sha256']:
                    manifest['host_elf_interpreters'][str(path)] = digest
                    continue
            raise ValueError(f'Frozen {binary} resolves outside its saved runtime: {path}')
    for path in frozen.rglob('*'):
        if path.is_file() and not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)


def verify_snapshot(root, manifest):
    if execution_plan(manifest) != json.loads((root / '.campaign/plan.json').read_text()):
        raise ValueError('Prepared execution settings changed; prepare a new campaign')
    for relative, digest in manifest['snapshot_sha256'].items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or file_hash(path) != digest:
            raise ValueError(f'Frozen input changed: {relative}')
    for name, target in manifest['runtime_library_aliases'].items():
        alias = root / '.campaign/runtime' / name
        if not alias.is_symlink() or os.readlink(alias) != target:
            raise ValueError('HPX runtime alias changed')
    for path, digest in manifest.get('host_elf_interpreters', {}).items():
        if file_hash(path) != digest:
            raise ValueError('Host ELF interpreter changed; prepare a new campaign')


def verify_orchestrator(root):
    if file_hash(__file__) != file_hash(root / '.campaign/run_campaign.py'):
        raise ValueError('Campaign runner changed; use the frozen .campaign/run_campaign.py to resume')


def command(root, manifest, run, dry_run=False):
    base = root / '.campaign'; runtime = base / 'runtime'
    result = [str(runtime / 'radiation-results'), 'run', run['case'], str(run['level']),
        '--root', str(base / 'source'), '--build', str(runtime), '--exe', str(runtime / 'octotiger'),
        '--generator', str(runtime / 'gen_radiation_reference'), '--launcher', str(base / 'launch-solver'),
        '--output', str(root / run['directory']), '--threads', str(manifest['threads']), '--jobs', '2',
        '--snapshots', str(manifest['snapshots']), '--time', str(run['stop_time']), '--hard-dt', str(run['hard_dt']),
        '--radiation-model', run['model'], '--radiation-integrator', run['integrator'],
        '--radiation-backend', 'cpu', '--radiation-reduced-c-factor', '1', '--no-build', '--no-open', '--no-movies']
    if dry_run:
        result.append('--dry-run')
    return result


def render(root, manifest):
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for run in manifest['runs']:
        url = run['directory'] + '/index.html'
        label = f'{run["case"]} · L{run["level"]}'
        title = f'<a href="{esc(url)}">{esc(label)}</a>' if (root / url).is_file() else esc(label)
        rows.append(f'<tr><td>{title}</td><td>{LABELS[run["model"]]}</td><td>{esc(run["status"])}</td>'
                    f'<td>{run["stop_time"]:g}</td><td>{run["hard_dt"]:g}</td></tr>')
    unsupported = ''.join(f'<li>{esc(case)}: {esc(FLD_UNAVAILABLE[case])}.</li>'
                          for case in manifest['cases'] if case not in FLD_CASES)
    comparison = '<a href="comparisons/index.html">Model comparison plots</a>' if (root / 'comparisons/index.html').is_file() else 'Comparison plots will appear after matching runs complete.'
    atomic_text(root / 'index.html', f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 RK1 and FLD radiation tests</title>
<style>body{{font:16px/1.6 system-ui;color:#16303b;background:#eff4f5;margin:0}}main{{max-width:1100px;margin:2em auto;background:white;padding:2em}}table{{width:100%;border-collapse:collapse}}td,th{{padding:.6em;text-align:left;border-bottom:1px solid #ccd}}a{{color:#067078}}.note{{border-left:4px solid #bc7b28;padding:1em;background:#fff5df}}.scroll{{overflow-x:auto}}</style><main>
<h1>M1 RK1 and FLD</h1><p class="note">Status: {esc(manifest['status'])}. {esc(manifest['scope'])}</p>
<p>M1 evolves radiation energy and flux using RK1 with radiation substeps. FLD solves energy and material heating together implicitly; its flux is a derived diagnostic. Cross-model differences are not truncation-error estimates.</p>
<p>{comparison}</p><div class="scroll"><table><tr><th>Problem</th><th>Model / integrator</th><th>Status</th><th>Final time</th><th>Maximum timestep</th></tr>{''.join(rows)}</table></div>
<p>Time is in seconds except for normalized Marshak time. Gaussian, equilibrium sphere and Marshak have analytic diffusion-limit references for FLD energy and flux; Marshak also has material energy. These are not exact nonlinear LP FLD solutions. Transparent streaming cases compare closures and conservation without claiming an analytic FLD profile. Reference discrepancies remain separate from cross-model differences. Prepared runs are not validation evidence.</p>
<h2>Supported scope</h2><p>Gaussian, sphere and Marshak support both models with physical light speed on CPU. The other FLD combinations are excluded from execution for the reasons below; their M1 runs remain available:</p><ul>{unsupported}</ul>
<p><a href="campaign.json">Plan and provenance</a> · <a href="commands.txt">Prepared commands</a></p>
<p>Execution requires a separate explicit start. This workflow does not publish to a remote website.</p></main></html>''')


def save(root, manifest):
    manifest['updated_utc'] = now(); write_json(root / 'campaign.json', manifest)
    atomic_text(root / 'commands.txt', '\n'.join(shlex.join(command(root, manifest, r)) for r in manifest['runs']) + '\n')
    render(root, manifest)


def validate_completed(root, manifest, run):
    folder = root / run['directory'] / run['case'] / f'l{run["level"]}'
    meta = json.loads((folder / 'run.json').read_text())
    expected = {'status': 'complete', 'case': run['case'], 'level': run['level'],
        'radiation_model': run['model'], 'radiation_integrator': run['integrator'],
        'radiation_algorithm_version': 1, 'radiation_flux_semantics': FLUX[run['model']],
        'executable_sha256': manifest['executable_sha256']}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Completed run identity/provenance differs: {run["id"]}')
    if not math.isclose(meta['time'], run['stop_time'], rel_tol=2e-12):
        raise ValueError('Final time differs from the prepared plan')
    digest = file_hash(folder / 'run.json')
    if run.get('run_metadata_sha256', digest) != digest:
        raise ValueError('Previously validated run metadata changed')
    run['run_metadata_sha256'] = digest
    return meta


def refresh_comparisons(root, manifest):
    with (root / 'comparison.log').open('a') as stream:
        subprocess.run([sys.executable, str(root / '.campaign/compare_campaign.py'), str(root)],
                       cwd=root, env=environment(root), stdout=stream, stderr=subprocess.STDOUT, check=True)


def session_processes(session):
    """Snapshot members of our start_new_session child, across process groups."""
    result = {}
    page_size = os.sysconf('SC_PAGE_SIZE')
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
            # Fields after comm start at state (3); pgrp=5, session=6, rss=24.
            # rr::execute creates another process group for the solver, while
            # retaining this session even if its wrapper exits first.
            if int(fields[3]) == session:
                group = int(fields[2])
                if group > 0:
                    result[int(stat.parent.name)] = {'group': group, 'state': fields[0],
                        'rss': max(0, int(fields[21])) * page_size}
        except (OSError, ValueError, IndexError):
            continue
    return result


def process_session_rss(session):
    return sum(process['rss'] for process in session_processes(session).values())


def signal_session(session, signum):
    if session <= 1 or session == os.getsid(0):
        raise ValueError('Refusing to signal a session not isolated from the campaign supervisor')
    groups = {process['group'] for process in session_processes(session).values()
              if process['state'] != 'Z'}
    # Leave the wrapper until last so it can reap its children during shutdown.
    for group in sorted(groups, key=lambda value: (value == session, value)):
        # Recheck membership before each signal. Do not signal a stale group
        # merely because it existed in an earlier snapshot of the session.
        if not any(process['group'] == group and process['state'] != 'Z'
                   for process in session_processes(session).values()):
            continue
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass


def stop_child(child):
    # The C++ runner moves each solver into its own process group. All remain
    # members of this owned session, including orphans of an exited wrapper.
    session = child.pid
    signal_session(session, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    signal_session(session, signal.SIGKILL)
    child.wait(timeout=10)
    deadline = time.monotonic() + 10
    while any(process['state'] != 'Z' for process in session_processes(session).values()):
        if time.monotonic() >= deadline:
            raise RuntimeError('Solver session did not exit after cleanup; verify host processes before recovery')
        time.sleep(.05)


def execute(root, manifest, run_limit=None):
    checks = manifest.get('preparation_checks', {})
    if checks.get('config_dry_runs') != len(manifest['runs']) or checks.get('solver_runs') != 0 or not checks.get('runtime_help_passed'):
        raise ValueError('Preparation is incomplete; finish --resume DIR --dry-run before starting')
    if manifest.get('active') or any(run['status'] == 'running' for run in manifest['runs']):
        raise ValueError('A prior attempt may still be alive; verify its host PID/start identity before recovery')
    if (root / 'STOP').exists():
        raise ValueError('STOP is present; remove it only when explicitly resuming the campaign')
    manifest.update(active=True, status='running', authorization='explicit_start', started_utc=now())
    manifest['last_start_run_limit'] = run_limit
    newly_completed = 0
    child = None
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for sig in handlers:
        signal.signal(sig, interrupted)
    try:
        for run in manifest['runs']:
            if run['status'] == 'completed':
                validate_completed(root, manifest, run); continue
            if (root / 'STOP').exists():
                raise KeyboardInterrupt
            verify_snapshot(root, manifest)
            if shutil.disk_usage(root).free < manifest['minimum_free_gib'] * 2**30:
                raise RuntimeError('Free storage fell below the saved reserve')
            destination = root / run['directory']
            if destination.exists():
                archived = root / 'attempts' / run['id'] / str(len(run['attempts']))
                archived.parent.mkdir(parents=True, exist_ok=True); destination.rename(archived)
            run.update(status='running'); run['attempts'].append({'started_utc': now()}); save(root, manifest)
            logs = root / 'logs'; logs.mkdir(exist_ok=True)
            with (logs / f'{run["id"]}-{len(run["attempts"])}.log').open('w') as stream:
                child = subprocess.Popen(command(root, manifest, run), cwd=root, env=environment(root),
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                run['pid'] = child.pid
                run['attempts'][-1]['pid'] = child.pid
                run['attempts'][-1]['process_session_id'] = child.pid
                run['attempts'][-1]['process_start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                save(root, manifest)
                while child.poll() is None:
                    if (root / 'STOP').exists():
                        raise KeyboardInterrupt
                    if shutil.disk_usage(root).free < manifest['minimum_free_gib'] * 2**30:
                        raise RuntimeError('Free storage fell below the saved reserve')
                    if process_session_rss(child.pid) > manifest['memory_limit_gib'] * 2**30:
                        raise RuntimeError('Solver process session exceeded the saved memory budget')
                    time.sleep(.25)
                if child.returncode:
                    raise RuntimeError(f'{run["id"]} failed with exit status {child.returncode}; see logs')
            stop_child(child)
            child = None
            validate_completed(root, manifest, run)
            run.update(status='completed'); run.pop('pid', None)
            run['attempts'][-1]['completed_utc'] = now(); save(root, manifest)
            refresh_comparisons(root, manifest)
            newly_completed += 1
            if run_limit is not None and newly_completed >= run_limit:
                break
        manifest['status'] = 'completed' if all(run['status'] == 'completed' for run in manifest['runs']) else 'ready'
    except KeyboardInterrupt:
        manifest['status'] = 'stopped'
        atomic_text(root / 'STOP', 'Stopped; requires explicit resume.\n')
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        try:
            if child is not None:
                stop_child(child)
            for run in manifest['runs']:
                if run['status'] == 'running':
                    run.update(status='interrupted'); run.pop('pid', None)
            manifest['active'] = False; save(root, manifest)
        finally:
            # A cleanup failure preserves the last active manifest so a later
            # start refuses overlapping work, but must still restore handlers.
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return 0 if manifest['status'] in ('completed', 'ready') else 1


def parser(arguments=None):
    p = argparse.ArgumentParser(description=__doc__)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument('--output', type=Path); target.add_argument('--resume', type=Path)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true'); action.add_argument('--dry-run', action='store_true')
    action.add_argument('--start', action='store_true'); action.add_argument('--refresh-only', action='store_true')
    p.add_argument('--run-limit', type=int,
        help='With --resume --start, complete at most N new runs, then return ready without a STOP file')
    p.add_argument('--preset', choices=('pilot', 'full', 'movies'), default='pilot',
        help='pilot: two short cases; full: eight benchmarks; movies: longer Gaussian/Marshak evolution with 121 states')
    p.add_argument('--levels', nargs='+', type=int, choices=range(6), default=[0, 1])
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(PAIRED_CASES))
    p.add_argument('--models', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--build', type=Path, default=ROOT / 'cmake-build-hpx2rc-radiation')
    p.add_argument('--results-build', type=Path, default=ROOT / 'radiation_results/.build')
    p.add_argument('--threads', type=int, default=2,
        help='HPX worker count, at most the available CPU count; frozen during preparation')
    p.add_argument('--snapshots', type=int, default=3)
    p.add_argument('--time', type=float, default=.25); p.add_argument('--hard-dt', type=float, default=.03125)
    p.add_argument('--minimum-free-gib', type=float, default=10); p.add_argument('--memory-limit-gib', type=float, default=8)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--preset', choices=('pilot', 'full', 'movies'), default='pilot')
    selected, _ = selector.parse_known_args(arguments if arguments is not None else [])
    preset_file = {'pilot': 'local-study.json', 'full': 'full-study.json', 'movies': 'movie-study.json'}[selected.preset]
    preset = json.loads(Path(__file__).with_name(preset_file).read_text())
    if preset.get('schema') != SCHEMA or preset.get('workflow') != WORKFLOW or preset.get('radiation_reduced_c_factor') != 1:
        raise ValueError(f'{preset_file} must describe the physical-c M1/FLD workflow')
    p.set_defaults(**{key: preset[key] for key in ('cases', 'levels', 'threads', 'snapshots',
        'hard_dt', 'minimum_free_gib', 'memory_limit_gib')}, time=preset['stop_time'], models=list(preset['models']),
        case_settings=preset.get('case_settings', {}), study_scope=preset['scope'])
    return p


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv); p = parser(arguments); args = p.parse_args(arguments)
    explicit = {argument.split('=')[0] for argument in arguments if argument.startswith('--')}
    args.case_settings = {case: {key: value for key, value in settings.items()
        if not ((key == 'stop_time' and '--time' in explicit) or (key == 'hard_dt' and '--hard-dt' in explicit))}
        for case, settings in args.case_settings.items()}
    if args.start and not args.resume:
        p.error('--start requires an already prepared --resume directory')
    if args.resume and args.prepare:
        p.error('Use --dry-run to verify an existing prepared campaign')
    if args.refresh_only and not args.resume:
        p.error('--refresh-only requires --resume')
    if args.run_limit is not None and (not args.resume or not args.start or args.run_limit <= 0):
        p.error('--run-limit must be positive and requires --resume DIR --start')
    if args.resume:
        allowed = {'--resume', '--dry-run', '--start', '--refresh-only', '--run-limit'}
        if any(a.split('=')[0] not in allowed for a in arguments if a.startswith('--')):
            p.error('Resume uses saved settings; do not override the prepared plan')
    root = (args.resume or args.output).resolve(); manifest = load(root) if args.resume else plan(args)
    if args.dry_run and not args.resume:
        print(json.dumps(manifest, indent=2)); return 0
    if not args.resume:
        if root.exists() and any(root.iterdir()):
            p.error('--output must be new or empty')
        root.mkdir(parents=True, exist_ok=True)
    with (root / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            manifest = load(root)
        if not args.resume:
            snapshot(root, manifest); save(root, manifest)
        verify_snapshot(root, manifest)
        verify_orchestrator(root)
        if args.prepare or args.dry_run:
            # C++ --dry-run checks configs and units; it does not launch the solver.
            for run in manifest['runs']:
                subprocess.run(command(root, manifest, run, True), cwd=root, env=environment(root), check=True)
            with (root / 'runtime-help.log').open('w') as stream:
                subprocess.run([str(root / '.campaign/runtime/octotiger'), '--help'], cwd=root,
                    env=environment(root), stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=30)
            manifest['preparation_checks'] = {'config_dry_runs': len(manifest['runs']), 'solver_runs': 0,
                                               'runtime_help_passed': True, 'snapshot_verified_utc': now()}
            save(root, manifest); return 0
        if args.refresh_only:
            refresh_comparisons(root, manifest); save(root, manifest); return 0
        return execute(root, manifest, args.run_limit)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
