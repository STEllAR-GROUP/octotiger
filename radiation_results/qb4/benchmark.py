#!/usr/bin/env python3
"""Run three 20-step timing samples inside an existing compute allocation.

No connection, scheduler query/submission, full-duration run, or accuracy claim.
Source the matching QB environment before invoking this script.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from workflow import adaptive_settings, load_plan
from collect import amr_history

CASES = ('radiating_blast', 'supercritical_shock', 'subcritical_shock')
FACTORS = {'radiating_blast': 1., 'supercritical_shock': .1, 'subcritical_shock': .03}
STEPS = 20
C = 2.99792458e10
SCOPE = ('Timing sample only. Wall time includes startup, regridding, final CSV export, '
         'logging and shutdown. Extrapolation assumes the initial physical-time advance rate '
         'persists; later flow/timestep changes and machine variability can change the cost. '
         'This does not validate RSLA accuracy, factor convergence, or completion of the full benchmark.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, record):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def allocation():
    require(bool(re.fullmatch(r'[0-9]+', os.environ.get('SLURM_JOB_ID', ''))),
            'Run this benchmark in an allocated Slurm compute job')
    host = socket.gethostname().split('.')[0]
    require(host == os.environ.get('SLURMD_NODENAME', '').split('.')[0],
            'The benchmark must execute on the allocated compute node, not the login host')
    nodes = int(os.environ.get('SLURM_NNODES', '0'))
    tasks = int(os.environ.get('SLURM_NTASKS', '0'))
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    require(nodes > 0 and tasks == nodes, 'Require one MPI task per allocated node')
    require(1 <= cpus <= 64 and cpus <= len(os.sched_getaffinity(0)),
            'Requested HPX workers exceed the allocation, node size, or CPU affinity')
    require(bool(re.fullmatch(r'[A-Za-z0-9_.-]+', os.environ.get('QB_MPI_TYPE', ''))) and
            os.environ['QB_MPI_TYPE'] != 'none', 'Set QB_MPI_TYPE to the selected Slurm MPI plugin')
    return {'job_id': os.environ['SLURM_JOB_ID'], 'host': host, 'nodes': nodes,
            'tasks': tasks, 'threads_per_task': cpus, 'mpi_type': os.environ['QB_MPI_TYPE']}


def config(path):
    result = {}
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition('=')
        key, value = key.strip(), value.strip()
        require(separator and key and key not in result, 'Invalid/duplicate configuration key')
        result[key] = value
    return result


def timing(log, problem, target):
    marker = re.compile(r'^RADIATION_HYDRO_FINISHED (\w+) t=(\S+) requested=(\S+)\s*$')
    step = re.compile(r'^\s*([0-9]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
    final = []
    steps = {}
    for line in log.splitlines():
        match = marker.fullmatch(line)
        if match:
            require(match[1] == problem, 'Wrong final problem marker')
            final.append((float(match[2]), float(match[3])))
            continue
        match = step.fullmatch(line)
        if match:
            index = int(match[1])
            values = [float(match[i]) for i in range(2, 6)]
            require(index not in steps and all(math.isfinite(v) for v in values),
                    'Duplicate or nonfinite step timing record')
            require(values[1] > 0 and values[2] >= 0, 'Nonpositive timestep or negative elapsed time')
            steps[index] = values
    require(len(final) == 1, 'Missing or duplicate final hydro marker')
    actual, requested = final[0]
    require(math.isfinite(actual) and 0 < actual < target and requested == target,
            'The timing sample must advance time but remain short of the full target')
    require(set(steps) == set(range(1, STEPS + 1)), 'Expected exactly 20 executed steps')
    ordered = [steps[index] for index in range(1, STEPS + 1)]
    require(all(ordered[i][0] > ordered[i - 1][0] for i in range(1, STEPS)),
            'Step times are not increasing')
    require(math.isclose(ordered[-1][0], actual, rel_tol=2e-6), 'Step/marker final times disagree')
    return {'actual_physical_time_s': actual, 'executed_steps': len(steps),
            'mean_physical_timestep_s': actual / len(steps),
            'reported_step_wall_seconds': sum(row[2] for row in ordered),
            'last_ten_mean_reported_step_wall_seconds': sum(row[2] for row in ordered[-10:]) / 10.}


def benchmark(source, build, output, level, factors, amr_plan=None):
    resources = allocation()
    require(level in (2, 3, 4, 5), 'Level must be 2, 3, 4, or 5')
    require(set(factors) == set(CASES) and all(type(value) in (int, float) and
            math.isfinite(value) and 0 < value <= 1 for value in factors.values()),
            'Each coupled case needs a finite reduced-speed factor in (0, 1]')
    settings = adaptive_settings(amr_plan)
    source, build, output = (Path(path).resolve() for path in (source, build, output))
    executable = build / 'octotiger'
    launcher = source / 'radiation_results/qb4/mpi-launch.sh'
    require(executable.is_file() and os.access(executable, os.X_OK), 'Missing executable')
    require(launcher.is_file() and os.access(launcher, os.X_OK), 'Missing MPI launcher')
    require(re.search(r'^OCTOTIGER_WITH_GRIDDIM:[^=]+=8\s*$',
                      (build / 'CMakeCache.txt').read_text(), re.MULTILINE), 'Benchmark requires INX=8')
    configurations = {case: config(source / 'radiation_results/configs' / (case + '.ini')) for case in CASES}
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update({key: '1' for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',
                                             'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    report = {'status': 'running', 'scope': SCOPE, 'level': level, 'cells_per_side': 8 * 2**level,
              'resources': resources, 'source': str(source), 'build': str(build),
              'executable_sha256': digest(executable), 'launcher_sha256': digest(launcher),
              'cases': []}
    if amr_plan is not None:
        report['amr_plan'] = amr_plan
        report['adaptive'] = True
    report['source_sha256'] = {name: digest(source / name) for name in (
        'octotiger/radiation/m1.hpp', 'octotiger/radiation/opacities.hpp',
        'octotiger/options.hpp', 'src/options_processing.cpp', 'src/radiation/rad_grid.cpp',
        'src/node_server_actions_1.cpp', 'src/node_server_actions_2.cpp', 'src/node_server_actions_3.cpp',
        'octotiger/amr/gradient.hpp', 'src/grid.cpp',
        'src/test_problems/radiation/radiation.cpp', 'test_problems/radiation/hydro_benchmarks.hpp',
        'octotiger/test_problems/radiation/hydro_output.hpp', 'src/io/silo_in.cpp', 'src/io/silo_out.cpp')}
    write(output / 'benchmark.json', report)
    try:
        for case in CASES:
            folder = output / case
            folder.mkdir()
            # SliceOutput opts in only when this directory exists before startup.
            (folder / 'radiation-hydro-slices').mkdir()
            cfg = configurations[case]
            target = float(cfg['stop_time'])
            factor = factors[case]
            cfg.update(min_level=str(level), max_level=str(level), stop_step=str(STEPS - 1),
                       radiation_reduced_c_factor=str(factor), disable_diagnostics='on',
                       disable_analytic='on', disable_output='on', unigrid='on',
                       code_to_cm='1', code_to_s='1', code_to_g='1', datadir=str(folder),
                       odt=str(target))
            if settings:
                cfg.update({key: str(value) for key, value in settings[(case, level)].items()
                            if key in ('min_level', 'max_level') or key.startswith('gradient_refine_')})
                cfg['unigrid'] = 'off'
            cfg.pop('hard_dt', None)
            ini = folder / 'run.ini'
            ini.write_text(''.join('{}={}\n'.format(key, value) for key, value in sorted(cfg.items())))
            command = [str(launcher), str(executable), '--config_file=' + str(ini),
                       '--hpx:threads=' + str(resources['threads_per_task'])]
            record = {'case': case, 'status': 'running', 'scope': SCOPE,
                      'requested_full_time_s': target, 'requested_steps': STEPS,
                      'radiation_reduced_c_factor': factor, 'physical_c': C, 'transport_c': C * factor,
                      'configuration': cfg, 'configuration_sha256': digest(ini), 'command': command}
            report['cases'].append(record)
            write(folder / 'benchmark.json', record)
            write(output / 'benchmark.json', report)
            started = time.perf_counter()
            try:
                with (folder / 'run.log').open('w') as log:
                    subprocess.run(command, cwd=folder, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            finally:
                record['wall_seconds_including_startup'] = time.perf_counter() - started
            record.update(timing((folder / 'run.log').read_text(), cfg['problem'], target))
            if settings:
                record['amr_history'] = amr_history(folder / 'gradient-amr.csv', level,
                                                   record['actual_physical_time_s'])
            require(any((folder / 'radiation-hydro-slices').glob('slice-*.csv')),
                    'Missing final numerical hydro CSV output')
            require((folder / 'radiation-conservation.csv').is_file(), 'Missing conservation history')
            multiplier = target / record['actual_physical_time_s']
            record.update(status='timing-sample-complete',
                          estimated_full_duration_steps=math.ceil(STEPS * multiplier),
                          estimated_full_duration_wall_seconds=record['wall_seconds_including_startup'] * multiplier,
                          extrapolation='Linear initial-rate estimate only; startup is included and no full-duration completion is claimed.')
            write(folder / 'benchmark.json', record)
            write(output / 'benchmark.json', report)
            print('{}: {} steps, {:.8g} physical s, {:.4g} wall s (startup included)'.format(
                case, record['executed_steps'], record['actual_physical_time_s'],
                record['wall_seconds_including_startup']), flush=True)
        report['status'] = 'timing-samples-complete'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        if report['cases']:
            report['cases'][-1].update(status='failed', error=str(error))
            write(output / report['cases'][-1]['case'] / 'benchmark.json', report['cases'][-1])
        raise
    finally:
        write(output / 'benchmark.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', help='Use the frozen workflow AMR settings, factors, paths, and allocation')
    for name in ('source', 'build', 'output'):
        parser.add_argument('--' + name)
    parser.add_argument('--level', type=int, choices=(2, 3, 4, 5), required=True)
    parser.add_argument('--subcritical-factor', type=float, default=.03)
    parser.add_argument('--supercritical-factor', type=float, default=.1)
    parser.add_argument('--blast-factor', type=float, default=1.)
    args = parser.parse_args()
    if args.plan:
        if any((args.source, args.build, args.output)):
            parser.error('--plan supplies source, build and output')
        plan = load_plan(args.plan)
        cfg = plan['config']
        resources = allocation()
        require(resources['nodes'] == cfg['nodes_by_level'][str(args.level)] and
                resources['threads_per_task'] == cfg['threads'], 'Allocation differs from frozen timing plan')
        benchmark(cfg['source_root'], cfg['build_dir'],
                  Path(cfg['output_root']) / 'timings' / ('l' + str(args.level)), args.level,
                  {case: cfg.get('reduced_c_factors', {}).get(case, 1.) for case in CASES}, plan.get('amr_plan'))
    else:
        if not all((args.source, args.build, args.output)):
            parser.error('Supply --plan or all of --source, --build, --output')
        benchmark(args.source, args.build, args.output, args.level,
                  dict(subcritical_shock=args.subcritical_factor,
                       supercritical_shock=args.supercritical_factor, radiating_blast=args.blast_factor))


if __name__ == '__main__':
    main()
