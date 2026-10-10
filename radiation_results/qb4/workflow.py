#!/usr/bin/env python3
"""Offline-first QB4 plan: build/test, optional timing, full-time case jobs.

Preparing a plan never invokes Slurm. --submit is an explicit later action.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess

CASES = ('streaming_wave', 'streaming_front', 'gaussian_pulse', 'equilibrium_sphere',
         'radiating_blast', 'supercritical_shock', 'subcritical_shock')
LEVELS = (2, 3, 4, 5)
TIMES = {'radiating_blast': 22000.0, 'supercritical_shock': 7500.0,
         'subcritical_shock': 38300.0}

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def selection(config):
    supplied = config.get('run_selection')
    if supplied is not None and (not isinstance(supplied, list) or any(
            not isinstance(item, dict) or set(item) != {'case', 'level'} for item in supplied)):
        raise ValueError('run_selection must be a list of case/level objects')
    pairs = [(case, level) for level in LEVELS for case in CASES] if supplied is None else [
        (item['case'], item['level']) for item in supplied]
    if not pairs or len(set(pairs)) != len(pairs) or any(
            case not in CASES or type(level) is not int or level not in LEVELS for case, level in pairs):
        raise ValueError('run_selection must contain unique valid case/level objects')
    return sorted(pairs, key=lambda pair: (pair[1], CASES.index(pair[0])))


def adaptive_settings(document):
    """Translate the local profile coordinates into the runner's CGS domain.

    The transport sampler uses half-width 1; the website runner uses 3e10 cm.
    Field values are unchanged, so both gradient modes divide by this length.
    Coupled cases already use CGS. This conversion must happen exactly once.
    """
    if document is None:
        return {}
    rows = document.get('rows', [])
    if len(rows) != 28 or {(r['case'], r['finest_level']) for r in rows} != {
            (case, level) for case in CASES for level in LEVELS}:
        raise ValueError('AMR plan must contain exactly seven cases at levels 2..5')
    result = {}
    for row in rows:
        case, level = row['case'], row['finest_level']
        threshold = row['selected']['threshold']
        floor = row['floor']
        coarsen = row.get('coarsen_factor', .8)
        if row['base_level'] != level - 1 or row['field'] not in ('density', 'radiation_energy') or \
                row['mode'] not in ('absolute', 'relative') or type(row['buffer']) is not int or \
                not 0 <= row['buffer'] <= 2:
            raise ValueError('Invalid adjacent-level gradient settings in AMR plan')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (threshold, floor, coarsen)) or coarsen > 1:
            raise ValueError('AMR threshold/floor/coarsening factor must be finite and positive')
        divisor = 1. if case in TIMES else 3.e10
        result[(case, level)] = {
            'min_level': level - 1, 'max_level': level,
            'gradient_refine_field': row['field'], 'gradient_refine_mode': row['mode'],
            'gradient_refine_threshold': threshold / divisor, 'gradient_refine_floor': floor,
            'gradient_refine_buffer': row['buffer'], 'gradient_refine_coarsen_factor': coarsen,
            'local_threshold': threshold, 'threshold_length_conversion_cm': divisor,
            'local_verification': row.get('verification', {'status': 'offline_proposal_not_run'})}
    return result


def amr_arguments(settings):
    return [part for key, value in settings.items()
            if key == 'min_level' or key.startswith('gradient_refine_')
            for part in ('--' + key.replace('_', '-'), str(value))]


def tasks(config, amr_plan=None):
    result = [{'key': 'build', 'kind': 'build', 'nodes': 1,
               'cpus': config['build_cpus'], 'walltime': config['build_walltime'],
               'partition': config['build_partition']}]
    result.append({'key': 'tests', 'kind': 'tests', 'nodes': 1, 'cpus': 4,
                   'walltime': config.get('test_walltime', config['build_walltime']),
                   'partition': config['build_partition']})
    if config.get('timing_samples', False):
        for level in LEVELS:
            result.append({'key': 'timing-l' + str(level), 'kind': 'timing', 'level': level,
                           'nodes': config['nodes_by_level'][str(level)], 'cpus': config['threads'],
                           'walltime': config.get('timing_walltime', config['run_walltime']),
                           'partition': config['partition']})
    settings = adaptive_settings(amr_plan)
    for case, level in selection(config):
        n = 8 * 2**level
        stop = TIMES.get(case, config['radiation_time'])
        factor = config.get('reduced_c_factors', {}).get(case, 1.) if case in TIMES else 1.
        length = 2e13 if case == 'radiating_blast' else 7e10 if case in TIMES else 6e10
        dt = .25 * length / n / (3 * factor * 2.99792458e10)
        task = {'key': f'{case}-l{level}', 'kind': 'run', 'case': case,
                       'level': level, 'cells': n, 'time': stop,
                       'radiation_reduced_c_factor': factor,
                       'nodes': config['nodes_by_level'][str(level)],
                       'cpus': config['threads'], 'walltime': config['run_walltime'],
                       'partition': config['partition'],
                       'transport_step_ceiling_s': dt,
                       'minimum_transport_steps': math.ceil(stop / dt)}
        if settings:
            task['amr'] = settings[(case, level)]
            task['finest_transport_steps_estimate'] = task['minimum_transport_steps']
            task['transport_step_ceiling_s'] = 2 * dt
            task['minimum_transport_steps'] = math.ceil(stop / (2 * dt))
        odt = config.get('checkpoint_odt', {}).get(task['key'])
        if odt is not None:
            task['checkpoint_odt'] = odt
        result.append(task)
    if len(selection(config)) == 28:
        result.append({'key': 'collect', 'kind': 'collect', 'nodes': 1,
                   'cpus': config['plot_cpus'], 'walltime': config['plot_walltime'],
                   'partition': config['plot_partition']})
    return result

def validate(config, submitting=False):
    selection(config)
    for key in ('source_root', 'build_dir', 'output_root', 'environment_file'):
        if not Path(config[key]).is_absolute() or '\n' in config[key]:
            raise ValueError(f'{key} must be an absolute path without a newline')
    for key in ('threads', 'build_cpus', 'plot_cpus'):
        if type(config[key]) is not int or not 1 <= config[key] <= 64:
            raise ValueError(f'{key} must be in 1..64')
    for level in LEVELS:
        if type(config['nodes_by_level'][str(level)]) is not int or config['nodes_by_level'][str(level)] < 1:
            raise ValueError('Every level needs a positive node count')
    if type(config['snapshots']) is not int or config['snapshots'] < 3:
        raise ValueError('snapshots must be at least three')
    if type(config['no_silo']) is not bool:
        raise ValueError('no_silo must be true or false')
    if type(config['radiation_time']) not in (int, float) or not math.isfinite(config['radiation_time']) or config['radiation_time'] <= 0:
        raise ValueError('radiation_time must be positive')
    factors = config.get('reduced_c_factors', {})
    if not isinstance(factors, dict) or not set(factors).issubset(TIMES):
        raise ValueError('reduced_c_factors may only name coupled Ensman cases')
    for factor in factors.values():
        if type(factor) not in (int, float) or not math.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError('Reduced light-speed factors must be in (0, 1]')
    if type(config.get('timing_samples', False)) is not bool:
        raise ValueError('timing_samples must be true or false')
    cadence = config.get('checkpoint_odt', {})
    valid_keys = {case + '-l' + str(level) for case in CASES for level in LEVELS}
    if not isinstance(cadence, dict) or not set(cadence).issubset(valid_keys) or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in cadence.values()):
        raise ValueError('checkpoint_odt maps case-lN keys to positive physical seconds')
    if config.get('amr_plan') is not None and (not isinstance(config['amr_plan'], str) or
            not config['amr_plan'] or '\n' in config['amr_plan']):
        raise ValueError('amr_plan must be a file path')
    if submitting:
        if 'EDIT_ME' in json.dumps(config):
            raise ValueError('Replace every EDIT_ME placeholder before submission')
        for key in ('account', 'partition', 'build_partition', 'plot_partition', 'mpi_type'):
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', config[key]):
                raise ValueError(f'Invalid or missing {key}')
        for key in ('build_walltime', 'run_walltime', 'plot_walltime', 'test_walltime', 'timing_walltime'):
            if key not in config:
                continue
            if not re.fullmatch(r'(?:[0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}', config[key]):
                raise ValueError(f'Provide verified scheduler walltime in {key}')
        if config['mpi_type'] == 'none':
            raise ValueError('The solver needs a real MPI plugin, not none')
        if not Path(config['source_root']).is_dir():
            raise ValueError('Missing source checkout')
        if not Path(config['environment_file']).is_file():
            raise ValueError('Missing site environment file')

def prepare(config_path, directory):
    config = read(config_path)
    validate(config)
    if config.get('amr_plan') and config['no_silo'] and any(case in TIMES for case, level in selection(config)):
        raise ValueError('Adaptive coupled production runs must retain Silo checkpoints')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Plan directory must be empty')
    (directory/'logs').mkdir()
    document = None
    if config.get('amr_plan'):
        path = Path(config['amr_plan'])
        if not path.is_absolute():
            path = Path(config['source_root']) / path
        document = read(path)
        adaptive_settings(document)
    plan = {'config': config, 'tasks': tasks(config, document), 'directory': str(directory),
            'matrix_scope': 'full-28' if len(selection(config)) == 28 else 'selected-subset',
            'selected_runs': len(selection(config)), 'total_matrix_runs': 28}
    if document is not None:
        plan['amr_plan'] = document
        plan['amr_plan_sha256'] = fingerprint(document)
    write(directory/'plan.json', plan)
    workflow = str(Path(config['source_root'])/'radiation_results/qb4/workflow.py')
    for task in plan['tasks']:
        script = directory/(task['key']+'.sbatch')
        text = '#!/usr/bin/env bash\nset -euo pipefail\n'
        text += ': "${SLURM_JOB_ID:?Submit this job through workflow.py}"\n'
        text += 'export QB_ENV_FILE='+shlex.quote(config['environment_file'])+'\n'
        text += 'source "$QB_ENV_FILE"\n'
        text += 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FFTW_NUM_THREADS=1\n'
        text += ' '.join(shlex.quote(arg) for arg in
                         ['python3', workflow, 'worker', '--plan', str(directory/'plan.json'),
                          '--task', task['key']])+'\n'
        script.write_text(text)
        script.chmod(0o755)
    print(f'Prepared {len(plan["tasks"])} jobs without contacting Slurm: {directory}')
    for task in plan['tasks']:
        if task['kind'] == 'run':
            mesh = ('AMR L{}..{}, finest equivalent {}^3'.format(task['level'] - 1, task['level'], task['cells'])
                    if task.get('amr') else '{}^3'.format(task['cells']))
            print(f'{task["key"]}: {mesh}, {task["nodes"]} nodes x {task["cpus"]} workers, '
                  f't={task["time"]:g} s, alpha={task["radiation_reduced_c_factor"]:g}, '
                  f'>= {task["minimum_transport_steps"]:,} transport steps')
    return directory/'plan.json'

def load_plan(plan_path, submitting=False):
    plan = read(plan_path)
    validate(plan['config'], submitting=submitting)
    if plan.get('amr_plan') is not None and fingerprint(plan['amr_plan']) != plan.get('amr_plan_sha256'):
        raise ValueError('Frozen AMR plan digest differs; prepare a new plan')
    if bool(plan['config'].get('amr_plan')) != (plan.get('amr_plan') is not None):
        raise ValueError('Frozen AMR plan missing or unexpected')
    if plan['tasks'] != tasks(plan['config'], plan.get('amr_plan')):
        raise ValueError('Plan tasks differ from configuration; prepare a new plan')
    if Path(plan['directory']).resolve() != Path(plan_path).resolve().parent:
        raise ValueError('Plan moved from its prepared directory; prepare a new plan')
    return plan

def parse_sbatch_output(stdout, account=None):
    """Accept parsable output plus the observed QB4 Lua accounting announcements.

    A single final ID is mandatory. Announcement IDs must agree, and unfamiliar
    output is preserved for manual reconciliation instead of guessed or retried.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    final = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', lines[-1]) if lines else None
    if not final:
        raise RuntimeError('Ambiguous sbatch result: missing valid final job ID; inspect saved output and scheduler manually')
    job = final[1]
    patterns = (
        ('available', r'sbatch: [0-9]+(?:\.[0-9]+)? SUs available in ([A-Za-z0-9_.-]+)'),
        ('estimated', r'sbatch: [0-9]+(?:\.[0-9]+)? SUs estimated for this job\.'),
        ('submitted', r'sbatch: lua: Submitted job ([1-9][0-9]*)'))
    seen = set()
    for line in lines[:-1]:
        matched = False
        for kind, pattern in patterns:
            match = re.fullmatch(pattern, line)
            if match:
                if kind in seen or (kind == 'submitted' and match[1] != job) or (
                        kind == 'available' and account is not None and match[1] != account):
                    raise RuntimeError('Ambiguous sbatch result: conflicting or duplicate announcement; inspect saved output and scheduler manually')
                seen.add(kind)
                matched = True
                break
        if not matched:
            raise RuntimeError('Ambiguous sbatch result: unrecognized output before final ID; inspect saved output and scheduler manually')
    return job

def submit(plan_path):
    plan = load_plan(plan_path, submitting=True)
    directory = Path(plan['directory'])
    record_path = directory/'submission.json'
    try:
        fd = os.open(record_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
    except FileExistsError:
        raise ValueError('Submission record already exists; refusing duplicate jobs')
    record = []
    write(record_path, record)
    previous = None
    for task in plan['tasks']:
        cmd = ['sbatch', '--parsable', '--account='+plan['config']['account'],
               '--partition='+task['partition'], '--time='+task['walltime'],
               '--job-name=rad-'+task['key'], '--nodes='+str(task['nodes']),
               '--ntasks='+str(task['nodes']), '--ntasks-per-node=1',
               '--cpus-per-task='+str(task['cpus']),
               '--output='+str(directory/'logs'/(task['key']+'-%j.out')),
               '--error='+str(directory/'logs'/(task['key']+'-%j.err'))]
        if previous:
            cmd.append('--dependency=afterok:'+previous)
        cmd.append(str(directory/(task['key']+'.sbatch')))
        record.append({'task': task['key'], 'status': 'submitting', 'command': cmd})
        write(record_path, record)
        try:
            result = subprocess.run(cmd, check=True, universal_newlines=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            record[-1].update(status='submission-failed', error=str(error),
                              stdout=error.stdout, stderr=error.stderr, returncode=error.returncode)
            write(record_path, record)
            raise
        # Persist evidence before interpreting it: an unexpected response may
        # still mean the scheduler accepted a job, so it must never be retried.
        record[-1].update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        write(record_path, record)
        try:
            job = parse_sbatch_output(result.stdout, plan['config']['account'])
        except RuntimeError as error:
            record[-1].update(status='submission-ambiguous', error=str(error))
            write(record_path, record)
            raise
        record[-1].update(job_id=job, status='submitted')
        write(record_path, record)
        previous = job
        print(task['key'], job)

def worker(plan_path, task_key):
    plan = load_plan(plan_path)
    config = plan['config']
    task = next(t for t in plan['tasks'] if t['key'] == task_key)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Worker commands require a Slurm allocation')
    if socket.gethostname().split('.')[0] != os.environ.get('SLURMD_NODENAME', '').split('.')[0]:
        raise RuntimeError('Worker must execute on the allocated compute node, not a login shell')
    if int(os.environ.get('SLURM_NNODES', '0')) != task['nodes'] or int(os.environ.get('SLURM_NTASKS', '0')) != task['nodes']:
        raise RuntimeError('One task per allocated node is required')
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != task['cpus']:
        raise RuntimeError('Allocation CPU count differs from the plan')
    for name in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'FFTW_NUM_THREADS'):
        os.environ[name] = '1'
    source, build, output = (Path(config[k]) for k in ('source_root', 'build_dir', 'output_root'))
    runner = build/'radiation-results-tools/radiation-results'
    if task['kind'] == 'build':
        cmd = [str(source/'build.sh'), '--env', config['environment_file'],
               '--build-dir', str(build), '-j', str(task['cpus'])]
    elif task['kind'] == 'tests':
        cache = (build / 'CMakeCache.txt').read_text()
        match = re.search(r'^_?Python3_EXECUTABLE:[^=]+=([^\n]+)$', cache, re.MULTILINE)
        if not match:
            raise RuntimeError('Missing CMake-selected Python; rebuild with Python3_EXECUTABLE set')
        commands = [[match[1], '-c',
                     'import sys; assert sys.version_info >= (3,8), "Tests require Python >=3.8; rebuild with Python3_EXECUTABLE"'],
                    ['ctest', '--test-dir', str(build), '--output-on-failure', '--no-tests=error',
                     '-L', 'radiation|gradient', '-j', '1'],
                    ['ctest', '--test-dir', str(build / 'radiation-results-tools'),
                     '--output-on-failure', '--no-tests=error', '-R', '^radiation-results-', '-j', '1']]
    elif task['kind'] == 'timing':
        os.environ['QB_MPI_TYPE'] = config['mpi_type']
        cmd = ['python3', str(source / 'radiation_results/qb4/benchmark.py'), '--plan', str(plan_path),
               '--level', str(task['level'])]
    elif task['kind'] == 'run':
        if int(os.environ.get('SLURM_NNODES', '0')) != task['nodes']:
            raise RuntimeError('Allocation node count differs from the plan')
        os.environ['QB_MPI_TYPE'] = config['mpi_type']
        cmd = [str(runner), 'run', task['case'], str(task['level']), '--root', str(source),
               '--build', str(build), '--output', str(output/'batches'/task['key']),
               '--threads', str(task['cpus']), '--jobs', str(config['build_cpus']),
               '--snapshots', str(config['snapshots']), '--no-build', '--no-open', '--no-movies',
               '--launcher', str(source/'radiation_results/qb4/mpi-launch.sh')]
        if task['case'] not in TIMES:
            cmd += ['--time', str(config['radiation_time'])]
        else:
            cmd += ['--radiation-reduced-c-factor', str(task['radiation_reduced_c_factor'])]
        if task.get('amr'):
            cmd += amr_arguments(task['amr'])
        if 'checkpoint_odt' in task:
            cmd += ['--odt', str(task['checkpoint_odt'])]
        if config['no_silo']:
            cmd.append('--no-silo')
    else:
        # Collection is implemented separately so it can also be rerun manually.
        cmd = ['python3', str(source/'radiation_results/qb4/collect.py'), '--plan', str(plan_path)]
    if task['kind'] != 'tests':
        commands = [cmd]
    status_directory = output / 'workflow-status'
    status_directory.mkdir(parents=True, exist_ok=True)
    status_path = status_directory / (task_key + '.json')
    status = {'task': task, 'status': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
              'host': socket.gethostname(), 'commands': commands, 'plan_sha256': fingerprint(plan),
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': plan.get('matrix_scope', 'full-28'),
              'recovery': 'No automatic checkpoint continuation or retry; retain partial files for explicit recovery.'}
    write(status_path, status)
    try:
        for index, command in enumerate(commands):
            status['command_index'] = index
            write(status_path, status)
            if task['kind'] == 'tests':
                with (status_directory / ('tests-' + str(index) + '.log')).open('w') as log:
                    subprocess.run(command, cwd=source, check=True, stdout=log, stderr=subprocess.STDOUT)
            else:
                subprocess.run(command, cwd=source, check=True)
        status['status'] = 'complete'
    except Exception as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        status['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write(status_path, status)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode')
    sub.required = True
    p = sub.add_parser('prepare')
    p.add_argument('--config', required=True)
    p.add_argument('--directory', required=True)
    p.add_argument('--submit', action='store_true', help='Explicitly submit after preparing; default is offline only')
    p = sub.add_parser('submit')
    p.add_argument('--plan', required=True)
    p.add_argument('--submit', action='store_true', required=True)
    p = sub.add_parser('worker')
    p.add_argument('--plan', required=True)
    p.add_argument('--task', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        path = prepare(args.config, args.directory)
        if args.submit:
            submit(path)
    elif args.mode == 'submit':
        submit(args.plan)
    else:
        worker(args.plan, args.task)

if __name__ == '__main__':
    main()
