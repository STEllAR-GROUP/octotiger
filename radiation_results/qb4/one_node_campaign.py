#!/usr/bin/env python3
"""Solve the remaining M1/FLD study in one existing QB allocation.

No login, submission, rendering, upload, deletion or automatic checkpoint
continuation is performed. SIGUSR1/SIGTERM/SIGINT stop only owned children;
partial output requires an explicit checkpoint-continuation follow-up.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

OWNER = 'octotiger-m1-fld-qb4-20261014'
THREADS = 64
CASES = ('streaming_wave', 'streaming_front', 'gaussian_pulse', 'equilibrium_sphere',
         'subcritical_shock', 'supercritical_shock', 'radiating_blast', 'marshak')
COUPLED = frozenset(CASES[4:7])
METHODS = {'m1': 'euler', 'fld': 'backward_euler'}
SKIPPED = ('l2-streaming_wave-m1', 'l2-streaming_wave-fld')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def allocation(env=None, hostname=None, affinity=None):
    env = os.environ if env is None else env
    hostname = socket.gethostname().split('.')[0] if hostname is None else hostname
    affinity = len(os.sched_getaffinity(0)) if affinity is None else affinity
    require(re.fullmatch(r'[1-9][0-9]*', env.get('SLURM_JOB_ID', '')), 'An allocated Slurm job is required')
    require(env.get('SLURM_NNODES') == '1' and env.get('SLURM_NTASKS') == '1',
            'Exactly one allocated node and one task are required')
    require(env.get('SLURM_CPUS_PER_TASK') == str(THREADS) and affinity >= THREADS,
            'Exactly 64 allocated CPUs and an affinity covering them are required')
    require(env.get('SLURMD_NODENAME', '').split('.')[0] == hostname and hostname,
            'Run on the allocated compute node, never the login host')
    return {'job_id': env['SLURM_JOB_ID'], 'node': hostname, 'nodes': 1, 'tasks': 1, 'threads': THREADS}


def make_plan(document):
    require(document.get('schema') == 2 and document.get('workflow') == 'm1-fld-radiation' and
            document.get('models') == METHODS and document.get('cases') == list(CASES) and
            document.get('levels') == [2, 3, 4] and document.get('snapshots') == 121 and
            document.get('radiation_reduced_c_factor') == 1, 'Unexpected full study contract')
    result = []
    for level in document['levels']:
        for case in CASES:
            settings = document.get('case_settings', {}).get(case, {})
            stop = settings.get('stop_time', document['stop_time'])
            dt = settings.get('hard_dt', document['hard_dt'])
            require(all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                        for value in (stop, dt)) and dt <= stop, 'Invalid physical time or timestep')
            for model, integrator in METHODS.items():
                run_id = f'l{level}-{case}-{model}'
                if run_id not in SKIPPED:
                    result.append({'id': run_id, 'case': case, 'level': level, 'model': model,
                                   'integrator': integrator, 'stop_time': stop, 'hard_dt': dt,
                                   'snapshots': 121, 'directory': 'runs/' + run_id})
    require(len(result) == 46, 'Expected exactly 46 remaining runs')
    return result


def inventory(folder):
    result = {}
    for path in sorted(Path(folder).rglob('*')):
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), f'Symlink in run inventory: {path}')
        if stat.S_ISDIR(mode):
            continue
        require(stat.S_ISREG(mode), f'Special file in run inventory: {path}')
        result[path.relative_to(folder).as_posix()] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    return result


def source_identity(source):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=source).decode().split('\0')
    files = {}
    for name in sorted(filter(None, names)):
        path = source / name
        if path.is_dir():  # A gitlink is not application source; dependencies have runtime hashes below.
            continue
        require(path.is_file() and path.resolve().is_relative_to(source), f'Missing or escaping source {name}')
        files[name] = digest(path)
    require(files and 'CMakeLists.txt' in files, 'Missing tracked application source')
    return {'source_commit': commit, 'source_tree_sha256': canonical(files), 'source_sha256': files}


def build_identity(build, binaries):
    """Bind the completed build allocation to these exact executable paths."""
    manifest = build / 'qb4-build-manifest.txt'
    require(manifest.is_file() and not manifest.is_symlink(), 'Missing regular QB build manifest')
    metadata, artifacts = {}, {}
    for line in manifest.read_text().splitlines():
        artifact = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line)
        if artifact:
            path = Path(artifact[2])
            require(path.is_absolute(), 'Build manifest artifact must use an absolute path')
            path = path.resolve()
            require(path not in artifacts, 'Duplicate build artifact identity')
            artifacts[path] = artifact[1]
        elif '=' in line:
            key, value = line.split('=', 1)
            require(key not in metadata, 'Duplicate build manifest metadata')
            metadata[key] = value
    job, node = metadata.get('job', ''), metadata.get('node', '')
    require(re.fullmatch(r'[1-9][0-9]*', job) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', node),
            'Invalid build job/node identity')
    for path in binaries.values():
        require(path.resolve() in artifacts and artifacts[path.resolve()] == digest(path),
                f'Build manifest does not match current artifact: {path}')
    evidence = {'manifest': {'path': str(manifest), 'sha256': digest(manifest)}}
    progress_path = build.parent / 'build-progress.json'
    if progress_path.exists():
        require(progress_path.is_file() and not progress_path.is_symlink(), 'Invalid build progress file')
        progress = read(progress_path)
        require(progress.get('schema') == 1 and progress.get('owner') == OWNER and
                progress.get('phase') == 'build' and progress.get('status') == 'completed' and
                type(progress.get('nodes')) is int and progress['nodes'] == 1 and
                type(progress.get('tasks')) is int and progress['tasks'] == 1,
                'Build progress must record a completed one-node build and checks')
        require(str(progress.get('job_id')) == job and isinstance(progress.get('node'), str) and
                progress['node'].split('.')[0] == node.split('.')[0],
                'Build progress job/node does not match the build manifest')
        evidence['progress'] = {'path': str(progress_path), 'sha256': digest(progress_path)}
    return {'build': {'job_id': job, 'node': node}, 'build_evidence': evidence}


def runtime_identity(source, build, tools):
    binaries = {'executable': build / 'octotiger', 'results': tools / 'radiation-results',
                'generator': tools / 'gen_radiation_reference'}
    result = {'schema': 1, 'owner': OWNER, 'nodes': 1, 'tasks': 1,
              **source_identity(source), 'worker_sha256': digest(__file__)}
    for name, path in binaries.items():
        require(path.is_file() and os.access(path, os.X_OK), f'Missing built executable {path}')
        result[name + '_sha256'] = digest(path)
    result.update(build_identity(build, binaries))
    output = subprocess.check_output(['ldd', str(binaries['executable'])], text=True)
    require('not found' not in output, 'Unresolved solver shared library')
    paths = re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(', output, re.M)
    libraries = {}
    for value in paths:
        path = Path(value)
        value = digest(path)
        require(path.name not in libraries or libraries[path.name] == value, 'Conflicting solver library basename')
        libraries[path.name] = value
    require(any(name.startswith('libhpx') for name in libraries), 'Solver is not linked against the expected HPX runtime')
    result['solver_library_sha256'] = libraries
    home = re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$', (build / 'CMakeCache.txt').read_text(), re.M)
    require(home is not None and Path(home[1]).resolve() == source.resolve(),
            'Build cache does not belong to the selected source checkout')
    result['build_cache_sha256'] = digest(build / 'CMakeCache.txt')
    return result


def command(source, build, tools, output, launcher, run):
    return [str(tools / 'radiation-results'), 'run', run['case'], str(run['level']),
            '--root', str(source), '--build', str(build), '--exe', str(build / 'octotiger'),
            '--generator', str(tools / 'gen_radiation_reference'), '--launcher', str(launcher),
            '--output', str(output / run['directory']), '--threads', str(THREADS), '--jobs', '2',
            '--snapshots', '121', '--time', str(run['stop_time']), '--hard-dt', str(run['hard_dt']),
            '--radiation-model', run['model'], '--radiation-integrator', run['integrator'],
            '--radiation-backend', 'cpu', '--radiation-reduced-c-factor', '1',
            '--no-build', '--no-movies', '--no-open']


def process_identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'process_start_ticks': fields[19]}


def session_members(session):
    members = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[3]) == session and fields[0] != 'Z':
                members[int(path.parent.name)] = {'group': int(fields[2]),
                    'rss': max(0, int(fields[21])) * os.sysconf('SC_PAGE_SIZE')}
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(child):
    require(child.pid > 1 and child.pid != os.getsid(0), 'Refusing to signal the worker session')
    for signum, seconds in ((signal.SIGTERM, 8), (signal.SIGKILL, 5)):
        groups = {value['group'] for value in session_members(child.pid).values()}
        for group in sorted(groups, key=lambda value: value == child.pid):
            if any(value['group'] == group for value in session_members(child.pid).values()):
                try:
                    os.killpg(group, signum)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            child.poll()
            if not session_members(child.pid):
                child.wait(timeout=1)
                return
            time.sleep(.1)
    raise RuntimeError('Owned child session did not drain; preserve active state and inspect before continuing')


def memory_ceiling(requested=None):
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    total = int(memory['MemTotal'].split()[0]) * 1024
    allocated = int(os.environ.get('SLURM_MEM_PER_NODE', '0')) * 1024**2
    if not allocated and os.environ.get('SLURM_MEM_PER_CPU', '').isdigit():
        allocated = int(os.environ['SLURM_MEM_PER_CPU']) * THREADS * 1024**2
    cap = .85 * min(total, allocated or total)
    value = requested * 1024**3 if requested is not None else cap
    require(math.isfinite(value) and 1024**3 <= value <= cap, 'Memory limit exceeds the conservative node/allocation allowance')
    return int(value)


def check_resources(output, memory_limit, minimum_free, session=None):
    require(shutil.disk_usage(output).free >= minimum_free, 'Storage reserve exhausted; preserve partial output')
    if session is not None:
        require(sum(item['rss'] for item in session_members(session).values()) <= memory_limit,
                'Owned process session exceeded its memory allowance')


def solver_progress(output, run):
    path = output / run['directory'] / run['case'] / f'l{run["level"]}' / 'run.log'
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - 65536))
            tail = stream.read().decode(errors='replace')
    except OSError:
        return None
    last = None
    for match in re.finditer(r'FLD accepted t=(\S+) dt=(\S+)|TS \d+:: t: ([^,\s]+)', tail):
        try:
            value = float(match[3]) if match[3] is not None else float(match[1]) + float(match[2])
            if math.isfinite(value) and value >= 0:
                last = value
        except ValueError:
            continue
    if last is None:
        return None
    return {'updated_utc': now(), 'simulation_time': last, 'stop_time': run['stop_time'],
            'time_units': 'normalized' if run['case'] == 'marshak' else 's',
            'fraction_of_simulated_duration': min(1., last / run['stop_time']),
            'source': 'solver run.log', 'log_bytes': size}


def completion_receipt(batch, run, identity, job, solver_identity=None):
    leaf = batch / run['case'] / f'l{run["level"]}'
    meta, batch_meta = read(leaf / 'run.json'), read(batch / 'batch.json')
    moving = run['model'] == 'fld' and run['case'] in COUPLED
    expected = {'status': 'complete', 'case': run['case'], 'level': run['level'],
                'radiation_model': run['model'], 'radiation_integrator': run['integrator'],
                'radiation_algorithm_version': 2 if moving else 1,
                'radiation_flux_semantics': ('derived_fld_lab_flux' if moving else
                    'derived_fld_flux' if run['model'] == 'fld' else 'evolved_m1_flux'),
                'executable_sha256': identity['executable_sha256']}
    require(all(meta.get(key) == value for key, value in expected.items()), 'Run identity or completion metadata mismatch')
    require(math.isclose(meta['time'], run['stop_time'], rel_tol=2e-12) and
            batch_meta.get('snapshots') == 121, 'Wrong time or snapshot count')
    # The C++ runner records threads in run.json's actual solver command and
    # live.json, not in batch.json or batch.json.options.
    argv = meta.get('command')
    require(isinstance(argv, list) and all(isinstance(value, str) for value in argv),
            'Missing recorded solver command')
    thread_flags = [value for value in argv if value.startswith('--hpx:threads')]
    require(type(job.get('threads')) is int and job['threads'] > 0 and
            thread_flags == [f'--hpx:threads={job["threads"]}'], 'Wrong recorded solver worker count')
    live = read(batch / 'live.json')
    require(live.get('active') is False and live.get('threads') == job['threads'] and
            live.get('snapshots') == 121 and len(live.get('runs', [])) == 1 and
            all(live['runs'][0].get(key) == value for key, value in
                {'case': run['case'], 'level': run['level'], 'stage': 'Ready'}.items()),
            'C++ batch is not inactive and complete with the requested worker count')
    require(batch_meta.get('workflow') == 'radiation-results' and
            batch_meta.get('cases') == [run['case']] and batch_meta.get('levels') == [run['level']] and
            batch_meta.get('executable_sha256') == identity['executable_sha256'] and
            math.isclose(batch_meta['time'], run['stop_time'], rel_tol=2e-12),
            'C++ batch identity differs from the requested run')
    config = {}
    for line in (leaf / 'run.ini').read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()
    require(config == meta['config'], 'Run config bytes do not describe the saved metadata')
    require(float(config['radiation_reduced_c_factor']) == 1 and config['radiation_backend'] == 'cpu' and
            config['radiation_model'] == run['model'] and config['radiation_integrator'] == run['integrator'] and
            float(config['stop_time']) == run['stop_time'] and
            0 < float(config['hard_dt']) <= run['hard_dt'] * (1 + 2e-12) and
            int(config['min_level']) == run['level'] and int(config['max_level']) == run['level'] and
            config.get('unigrid') == 'on', 'Run changed frozen physics, mesh or timestep cap')
    require(bool(batch_meta['source_sha256']) and
            all(identity['source_sha256'].get(name) == value for name, value in batch_meta['source_sha256'].items()),
            'C++ runner source hashes differ from the worker source inventory')
    numbered = sorted(int(path.name.split('.')[1]) for path in leaf.glob('X.*.silo')
                      if re.fullmatch(r'X\.[0-9]+\.silo', path.name))
    require(numbered == list(range(len(numbered))) and len(numbered) + 1 >= 121,
            'Missing or noncontiguous requested numerical snapshots')
    headers = [leaf / f'X.{index}.silo' for index in numbered] + [leaf / 'final.silo']
    require(all(path.is_file() and path.stat().st_size > 0 and
                any(path.with_name(path.name + '.data').rglob('*.silo')) for path in headers),
            'Missing checkpoint header or shard data')
    if moving:
        require(meta.get('radiation_frame') == 'mixed' and meta.get('radiation_hydro_scheme') == 'first_order_conservative',
                'Missing moving FLD equation metadata')
    return {'schema': 1, 'owner': OWNER, 'status': 'complete', 'completed_utc': now(), **job,
            'run_id': run['id'], 'case': run['case'], 'level': run['level'], 'model': run['model'],
            'integrator': run['integrator'], 'stop_time': run['stop_time'], 'hard_dt': run['hard_dt'],
            'snapshots': 121, **identity, 'solver_identity': solver_identity,
            'configured_hard_dt': float(config['hard_dt']),
            'snapshot_inventory': {'headers': [path.relative_to(batch).as_posix() for path in headers],
                                   'count': len(headers), 'physical_times': 'not read; validate during extraction'},
            'config_sha256': digest(leaf / 'run.ini'), 'run_metadata_sha256': digest(leaf / 'run.json'),
            'files': inventory(batch)}


def verify_completed(output, run, identity):
    path = output / 'receipts' / (run['id'] + '.json')
    if not path.exists():
        return False
    receipt = read(path)
    require(receipt.get('schema') == 1 and receipt.get('owner') == OWNER and receipt.get('status') == 'complete' and
            receipt.get('run_id') == run['id'] and receipt.get('nodes') == 1 and receipt.get('tasks') == 1,
            'Invalid existing completion receipt')
    require(all(receipt.get(key) == value for key, value in identity.items()), 'Completed runtime/source identity changed')
    require(re.fullmatch(r'[1-9][0-9]*', receipt.get('job_id', '')) and receipt.get('node'),
            'Invalid completed job/node identity')
    expected = completion_receipt(output / run['directory'], run, identity,
        {key: receipt[key] for key in ('nodes', 'tasks', 'threads', 'job_id', 'node')}, receipt.get('solver_identity'))
    require(all(receipt.get(key) == value for key, value in expected.items() if key != 'completed_utc'),
            'Completed receipt or files changed; refusing to overwrite')
    return True


def recover_receipt(output, run, identity, record, expected_command, recovery_job):
    """Validate finished science while retaining its original allocation identity."""
    require(isinstance(record, dict) and record.get('started_utc') and
            record.get('command') == expected_command, 'Missing or changed original solve record')
    original = {key: record.get(key) for key in ('nodes', 'tasks', 'threads', 'job_id', 'node')}
    require(original['nodes'] == 1 and original['tasks'] == 1 and original['threads'] == THREADS and
            isinstance(original['job_id'], str) and re.fullmatch(r'[1-9][0-9]*', original['job_id']) and
            isinstance(original['node'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', original['node']),
            'Invalid original solve allocation; recovery cannot supply a replacement')
    identity_path = output / 'solver-identities' / (run['id'] + '.json')
    require(identity_path.is_file() and not identity_path.is_symlink(), 'Missing original solver identity')
    solver = read(identity_path)
    runner = record.get('runner_identity', {})
    for value in (runner, solver):
        require(type(value.get('pid')) is int and value['pid'] > 1 and
                isinstance(value.get('process_start_ticks'), str) and
                re.fullmatch(r'[0-9]+', value['process_start_ticks']), 'Invalid original process identity')
    require(solver.get('job_id') == original['job_id'] and
            solver.get('node', '').split('.')[0] == original['node'].split('.')[0] and
            solver.get('session_id') == runner['pid'], 'Original solver and runner allocation/session disagree')
    if original['node'].split('.')[0] == recovery_job['node'].split('.')[0]:
        for value in (runner, solver):
            try:
                live_identity = process_identity(value['pid'])
            except (FileNotFoundError, ProcessLookupError):
                continue
            require(live_identity != {key: value[key] for key in ('pid', 'process_start_ticks')},
                    'Original solve process is still alive')
        require(not session_members(runner['pid']), 'Original solve session is still alive')
    batch = output / run['directory']
    # The C++ runner holds this same lock. Never validate an active writer.
    lock_path = batch / '.runner.lock'
    require(lock_path.is_file() and not lock_path.is_symlink(), 'Missing original C++ runner lock')
    with lock_path.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return completion_receipt(batch, run, identity, original, solver)


def solver_launch(arguments):
    allocation()
    require(len(arguments) >= 2 and str(Path(arguments[0]).resolve()) == os.environ.get('QB_WORKER_EXE') and
            '--hpx:threads=64' in arguments, 'Unexpected solver invocation')
    identity_path = Path(os.environ['QB_WORKER_SOLVER_IDENTITY'])
    require(not identity_path.exists(), 'A solver already started for this run')
    write(identity_path, {**process_identity(os.getpid()), 'session_id': os.getsid(0),
                          'job_id': os.environ['SLURM_JOB_ID'], 'node': socket.gethostname().split('.')[0]})
    os.execv(arguments[0], [*arguments, '--hpx:ignore-batch-env', '--hpx:localities=1',
                           '--hpx:ini=hpx.parcel.bootstrap=mpi', '--hpx:ini=hpx.parcel.mpi.enable=1',
                           '--hpx:ini=hpx.parcel.tcp.enable!=0', '--hpx:ini=hpx.stacks.use_guard_pages=0'])


def execute(args):
    require(args.run_limit is None or args.run_limit > 0, '--run-limit must be positive')
    job = allocation()
    source, build, output = args.source.resolve(), args.build.resolve(), args.output.resolve()
    tools = (args.tools or build / 'radiation-results-tools').resolve()
    document = read(source / 'radiation_results/full-study.json')
    plan = make_plan(document)
    identity = runtime_identity(source, build, tools)
    memory_limit = memory_ceiling(args.memory_limit_gib)
    require(math.isfinite(args.minimum_free_gib) and args.minimum_free_gib >= 20, 'At least 20 GiB storage reserve required')
    minimum_free = int(args.minimum_free_gib * 1024**3)
    if args.dry_run:
        print(json.dumps({'allocation': job, 'runs': plan, 'skipped_completed_locally': SKIPPED,
                          'remote_runtime': identity, 'memory_limit_bytes': memory_limit,
                          'run_limit': args.run_limit, 'recover_completed': args.recover_completed}, indent=2))
        return 0
    require(not args.recover_completed or (output / 'worker-state.json').is_file(),
            'Recovery requires an existing worker state and original solve record')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / 'worker-state.json'
        fixed = {'schema': 1, 'owner': OWNER, 'plan': plan, 'plan_sha256': canonical(plan),
                 'full_study_sha256': digest(source / 'radiation_results/full-study.json'),
                 'remote_runtime': identity, 'skipped_completed_locally': list(SKIPPED)}
        state = read(state_path) if state_path.exists() else {**fixed, 'runs': {}, 'active': False}
        require(all(state.get(key) == value for key, value in fixed.items()), 'Worker inputs changed; use a new campaign')
        require(not state.get('active'), 'Prior active worker requires explicit process-identity recovery')
        for path in ('logs', 'receipts', 'solver-identities', 'control'):
            (output / path).mkdir(exist_ok=True)
        launcher = output / 'control/launch-solver'
        launcher_text = '#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(Path(__file__).resolve())) + ' --launch-solver "$@"\n'
        if launcher.exists():
            require(launcher.read_text() == launcher_text, 'Solver launcher changed')
        else:
            launcher.write_text(launcher_text)
            launcher.chmod(0o755)
        interrupted = []
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
        def stop(signum, frame):
            interrupted.append(signum)
        for sig in handlers:
            signal.signal(sig, stop)
        child, current, newly_completed = None, None, 0
        state.update(active=True, status='running', allocation=job, worker_identity=process_identity(os.getpid()),
                     last_start_run_limit=args.run_limit, last_start_recover_completed=args.recover_completed)
        write(state_path, state)
        try:
            for run in plan:
                if interrupted:
                    state.update(status='stopped_before_next_run', stop_signal=interrupted[0])
                    break
                if verify_completed(output, run, identity):
                    state['runs'].setdefault(run['id'], {}).update(
                        status='complete', receipt='receipts/' + run['id'] + '.json')
                    write(state_path, state)
                    continue
                current = run
                batch = output / run['directory']
                if args.recover_completed:
                    require(batch.is_dir(), 'No completed batch to recover; recovery never launches science')
                    require(runtime_identity(source, build, tools) == identity, 'Runtime or source changed before recovery')
                    record = state['runs'].get(run['id'])
                    receipt = recover_receipt(output, run, identity, record,
                        command(source, build, tools, output, launcher, run), job)
                    require(not interrupted, 'Interrupted during recovery; completed files preserved')
                    require(runtime_identity(source, build, tools) == identity, 'Runtime or source changed during recovery')
                    receipt_path = output / 'receipts' / (run['id'] + '.json')
                    require(not receipt_path.exists(), 'Refusing to overwrite a completion receipt')
                    write(receipt_path, receipt)
                    record.update(status='complete', completed_utc=receipt['completed_utc'],
                        receipt='receipts/' + run['id'] + '.json', receipt_sha256=digest(receipt_path),
                        recovery={'allocation': job, 'validated_utc': now(), 'action': 'receipt only; no solver launched'})
                    state['status'] = 'waiting_for_publication'
                    current = None
                    break
                if batch.exists():
                    state['runs'].setdefault(run['id'], {}).update(status='needs_checkpoint_continuation',
                        reason='Existing incomplete output is preserved; never restart it from zero automatically')
                    state['status'] = 'needs_checkpoint_continuation'
                    break
                require(runtime_identity(source, build, tools) == identity, 'Runtime or source changed before launch')
                # One L4 run can require about 43 GiB of raw states; reserve that
                # before beginning, in addition to the continuously checked floor.
                raw_gib = {2: 1, 3: 7, 4: 48}[run['level']]
                check_resources(output, memory_limit, minimum_free + raw_gib * 1024**3)
                identity_path = output / 'solver-identities' / (run['id'] + '.json')
                require(not identity_path.exists(), 'Existing solver identity requires explicit recovery')
                env = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', QB_WORKER_EXE=str(build / 'octotiger'),
                           QB_WORKER_SOLVER_IDENTITY=str(identity_path))
                argv = command(source, build, tools, output, launcher, run)
                record = {'status': 'running', 'started_utc': now(), 'command': argv, **job}
                state['runs'][run['id']] = record
                write(state_path, state)
                with (output / 'logs' / (run['id'] + '.log')).open('x') as log:
                    child = subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    record['runner_identity'] = process_identity(child.pid)
                    write(state_path, state)
                    last_progress = time.monotonic()
                    while child.poll() is None:
                        if interrupted:
                            raise InterruptedError('Pre-timeout or termination signal received')
                        check_resources(output, memory_limit, minimum_free, child.pid)
                        if time.monotonic() - last_progress >= 30:
                            progress = solver_progress(output, run)
                            if progress is not None:
                                record['progress'] = progress
                            state['updated_utc'] = now()
                            write(state_path, state)
                            last_progress = time.monotonic()
                        time.sleep(2)
                    status = child.returncode
                    stop_owned(child)
                    child = None
                require(status == 0, f'Results runner exited {status}; partial files preserved')
                require(not interrupted, 'Interrupted after solver exit; completed files preserved for validation')
                require(runtime_identity(source, build, tools) == identity, 'Runtime or source changed during run')
                solver = read(identity_path) if identity_path.exists() else None
                receipt = completion_receipt(batch, run, identity, job, solver)
                receipt_path = output / 'receipts' / (run['id'] + '.json')
                require(not receipt_path.exists(), 'Refusing to overwrite a completion receipt')
                write(receipt_path, receipt)
                record.update(status='complete', completed_utc=now(), receipt='receipts/' + run['id'] + '.json',
                              receipt_sha256=digest(receipt_path))
                write(state_path, state)
                current = None
                newly_completed += 1
                if args.run_limit is not None and newly_completed >= args.run_limit:
                    state['status'] = 'waiting_for_publication'
                    break
            else:
                state['status'] = 'complete'
        except BaseException as error:
            if child is not None:
                stop_owned(child)
                child = None
            state.update(status='needs_checkpoint_continuation' if current else 'failed', error=str(error))
            if current:
                state['runs'].setdefault(current['id'], {}).update(status='needs_checkpoint_continuation', error=str(error))
        finally:
            if child is None:
                state['active'] = False
            state['updated_utc'] = now()
            write(state_path, state)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        return 0 if state['status'] in ('complete', 'waiting_for_publication') else 75


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments and arguments[0] == '--launch-solver':
        return solver_launch(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--tools', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-limit-gib', type=float)
    parser.add_argument('--minimum-free-gib', type=float, default=64)
    parser.add_argument('--run-limit', type=int,
                        help='Exit successfully for publication after this many new completed runs; verified skips do not count')
    parser.add_argument('--recover-completed', action='store_true',
                        help='Validate one finished batch and write its missing receipt using its original solve identity; never launch science')
    parser.add_argument('--dry-run', action='store_true', help='Validate allocation/inputs and print; launch nothing')
    return execute(parser.parse_args(arguments))


if __name__ == '__main__':
    sys.exit(main())
