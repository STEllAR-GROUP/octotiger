#!/usr/bin/env python3
"""Read-only checks for the single missing L2 M1 blast movie state.

This module neither launches a solver nor copies/renames Silo files. A native
two-step RK1 replay and a native output rewrite run separately under the job
owner's control. Original results and both original ledgers remain authoritative.
Silo times are supplied by a reader of the header's double ``cgs_time`` field.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


RUN_ID = 'l2-radiating_blast-m1'
SOURCE_SNAPSHOT = 'X.118.silo'
SUPPLEMENTAL_SNAPSHOT = 'X.119.silo'
STOP_TIME = 22000.0
PHYSICAL_C = 29979245800.0
ORIGINAL_JOB = {'job_id': '1102388', 'node': 'qbd172', 'nodes': 1, 'tasks': 1, 'threads': 64}
SCIENTIFIC_CAVEAT = (
    'Native RK1 field-state continuation from the saved M1 checkpoint; no SSP '
    'exact-archive or bitwise-continuation claim. The added state is evolved, '
    'not interpolated. Original final state and original radiation-conservation '
    'and momentum ledgers remain authoritative and are not replaced or rebased.'
)


def require(condition, message):
    if not condition:
        raise ValueError('Supplemental state: ' + message)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def directory(path):
    path = Path(path).absolute()
    require(path.is_dir() and not path.is_symlink(), 'expected a real directory: ' + str(path))
    require(path.resolve() == path, 'directory path traverses a symlink: ' + str(path))
    return path


def inventory_original(leaf):
    """Hash every regular file, including hidden files; reject all symlinks/devices."""
    leaf = directory(leaf)
    result = {}
    for parent, directories, names in os.walk(leaf, followlinks=False):
        for name in sorted(directories + names):
            path = Path(parent) / name
            info = path.lstat()
            require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode),
                    'symlink or special file in original tree: ' + str(path))
            if stat.S_ISREG(info.st_mode):
                digest = file_hash(path)
                after = path.stat()
                require((info.st_size, info.st_mtime_ns, info.st_ino) ==
                        (after.st_size, after.st_mtime_ns, after.st_ino),
                        'file changed while hashing: ' + str(path))
                result[path.relative_to(leaf).as_posix()] = {
                    'sha256': digest, 'bytes': info.st_size,
                    'mode': stat.S_IMODE(info.st_mode), 'mtime_ns': info.st_mtime_ns,
                }
    require(bool(result), 'empty original file inventory')
    return dict(sorted(result.items()))


def validate_unchanged(leaf, original_inventory):
    """Require the same complete inventory, hashes, sizes, modes and mtimes."""
    require(inventory_original(leaf) == original_inventory,
            'original files changed after recovery preparation')


def _config(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        require('=' in line, 'invalid configuration line')
        key, value = (item.strip() for item in line.split('=', 1))
        require(key and key not in result, 'duplicate or empty configuration key')
        result[key] = value
    return result


def _finite(value, description):
    require(not isinstance(value, bool), 'invalid ' + description)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('Supplemental state: invalid ' + description) from None
    require(math.isfinite(value), 'nonfinite ' + description)
    return value


def _snapshot_files(inventory, name):
    require(name in inventory and inventory[name]['bytes'] > 0,
            'missing or empty snapshot header: ' + name)
    shards = {key: value for key, value in inventory.items() if key.startswith(name + '.data/')}
    require(bool(shards) and all(key.endswith('.silo') and value['bytes'] > 0
                                for key, value in shards.items()),
            'missing or invalid snapshot shards: ' + name)
    return {name: inventory[name], **shards}


def validate_original(leaf, original_inventory, read_time):
    """Accept only the diagnosed completed run with X.119 absent, never a partial solve."""
    leaf = directory(leaf)
    validate_unchanged(leaf, original_inventory)
    for name in ('run.json', 'run.ini', 'run.log', 'radiation-conservation.csv', 'momentum-diagnostics.csv'):
        require(name in original_inventory and original_inventory[name]['bytes'] > 0,
                'missing original run artifact: ' + name)
    metadata = json.loads((leaf / 'run.json').read_text())
    expected = {'status': 'complete', 'case': 'radiating_blast', 'level': 2,
                'radiation_model': 'm1', 'radiation_integrator': 'euler',
                'radiation_algorithm_version': 1, 'radiation_flux_semantics': 'evolved_m1_flux'}
    require(all(metadata.get(key) == value for key, value in expected.items()),
            'original run/model/completion metadata differs')
    require(_finite(metadata.get('time'), 'configured final time') == STOP_TIME and
            _finite(metadata.get('physical_c'), 'physical light speed') == PHYSICAL_C and
            _finite(metadata.get('transport_c'), 'transport light speed') == PHYSICAL_C and
            _finite(metadata.get('radiation_reduced_c_factor'), 'reduced light speed') == 1,
            'original duration or physical light speed differs')
    require(re.fullmatch(r'[0-9a-f]{64}', str(metadata.get('executable_sha256', ''))) is not None,
            'missing original executable hash')
    config = _config(leaf / 'run.ini')
    require(config == metadata.get('config'), 'original INI and run metadata differ')
    expected_config = {'problem': 'RADIATION_BLAST_WAVE', 'radiation_model': 'm1',
        'radiation_integrator': 'euler', 'radiation_backend': 'cpu', 'radiation': 'on',
        'hydro': 'on', 'gravity': 'off', 'rad_implicit': 'on', 'periodic': 'off',
        'reflect_bc': 'off', 'inflow_bc': 'off', 'eos': 'IDEAL', 'unigrid': 'on',
        'min_level': '2', 'max_level': '2'}
    require(all(config.get(key) == value for key, value in expected_config.items()),
            'original physical configuration differs')
    expected_numeric = {'stop_time': STOP_TIME, 'radiation_reduced_c_factor': 1,
        'omega': 0, 'code_to_g': 1, 'code_to_cm': 1, 'code_to_s': 1,
        'cfl': .4, 'odt': STOP_TIME / 120, 'hard_dt': STOP_TIME / 120 / 5}
    require(all(_finite(config.get(key), key) == value for key, value in expected_numeric.items()),
            'original physical units, duration or timestep/capture settings differ')
    capture = metadata.get('movie_capture', {})
    require(capture.get('requested_snapshots') == 121 and capture.get('steps_per_output_check') == 5 and
            capture.get('odt') == expected_numeric['odt'] and capture.get('hard_dt') == expected_numeric['hard_dt'],
            'original capture metadata differs')
    numbered = sorted(name for name in original_inventory if '/' not in name and
                      re.fullmatch(r'X\.[0-9]+\.silo', name))
    require(set(numbered) == {f'X.{index}.silo' for index in range(119)},
            'expected exactly contiguous X.0..X.118, with only X.119 missing')
    names = [f'X.{index}.silo' for index in range(119)] + ['final.silo']
    require(not any(name.startswith('X.119.silo') for name in original_inventory),
            'partial supplemental snapshot already exists')
    times = []
    for name in names:
        _snapshot_files(original_inventory, name)
        times.append(_finite(read_time(leaf / name), 'Silo cgs_time'))
    require(times[0] == 0 and times[-1] == STOP_TIME and all(b > a for a, b in zip(times, times[1:])),
            'original snapshot times do not span the complete run in strict order')
    require(abs(times[-2] - (STOP_TIME - expected_numeric['odt'])) <= 32 * math.ulp(STOP_TIME),
            'source checkpoint is not the diagnosed last completed five-step batch')
    return {'run_id': RUN_ID, 'leaf': str(leaf), 'metadata': metadata,
            'snapshot_times': times, 'snapshot_names': names,
            'source_time': times[-2], 'final_time': times[-1]}


def validate_supplemental_time(original_times, supplemental_time):
    times = [_finite(value, 'original time') for value in original_times]
    time = _finite(supplemental_time, 'supplemental time')
    require(len(times) == 120 and all(b > a for a, b in zip(times, times[1:])),
            'expected 120 strictly increasing original times')
    require(times[-2] < time < times[-1] and time not in times,
            'supplemental time must be distinct and strictly between source and original final')
    return time


def _isolated(original, replay, export):
    original, replay = directory(original), directory(replay)
    export = Path(export).absolute()
    require(export.resolve() == export and (not export.exists() or
            (export.is_dir() and not export.is_symlink())), 'invalid export directory')
    paths = (original, replay, export)
    require(all(not a.is_relative_to(b) and not b.is_relative_to(a)
                for index, a in enumerate(paths) for b in paths[index + 1:]),
            'original, replay and export directories must be disjoint')
    return original, replay, export


def native_export_command(original_leaf, original_inventory, executable, replay_directory,
                          export_directory, *, threads=64):
    """Return argv AND required cwd; never execute or write a configuration.

    The native option is --output, although its C++ member is output_filename.
    Read paths are relative to cwd; output datadir must retain its trailing slash.
    """
    original, replay, export = _isolated(original_leaf, replay_directory, export_directory)
    validate_unchanged(original, original_inventory)
    require(not export.exists() or not any(export.iterdir()), 'export directory must be fresh and empty')
    require(type(threads) is int and threads == 64, 'recovery requires the original 64-worker allocation')
    executable = Path(executable).absolute()
    require(executable.is_file() and not executable.is_symlink(), 'missing regular frozen executable')
    metadata = json.loads((original / 'run.json').read_text())
    require(file_hash(executable) == metadata.get('executable_sha256'), 'frozen executable changed')
    replay_inventory = inventory_original(replay)
    _snapshot_files(replay_inventory, 'final.silo')
    return {'argv': [str(executable), '--config_file=' + str(original / 'run.ini'),
                     '--restart_filename=' + str(replay / 'final.silo'), '--output=X.119',
                     '--datadir=' + str(export) + '/', '--hpx:threads=64'],
            'cwd': str(replay), 'no_evolution': True,
            'expected_header': str(export / SUPPLEMENTAL_SNAPSHOT)}


def recovery_provenance(original_leaf, original_inventory, executable, replay_directory,
                        export_directory, original_job, recovery_job, read_time):
    """Verify immutable originals and actual times, then describe the supplemental files.

    This deliberately does not claim field equivalence or master/shard reference
    validation. The job owner must check those through Silo before installation.
    """
    original, replay, export = _isolated(original_leaf, replay_directory, export_directory)
    context = validate_original(original, original_inventory, read_time)
    require(original_job == ORIGINAL_JOB, 'original solve allocation differs')
    require(isinstance(recovery_job, dict) and
            re.fullmatch(r'[1-9][0-9]*', str(recovery_job.get('job_id', ''))) is not None and
            str(recovery_job['job_id']) != original_job['job_id'] and
            re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', str(recovery_job.get('node', ''))) is not None and
            all(type(recovery_job.get(key)) is int and recovery_job[key] == value
                for key, value in {'nodes': 1, 'tasks': 1, 'threads': 64}.items()),
            'invalid separate one-node recovery allocation')
    executable = Path(executable).absolute()
    require(executable.is_file() and not executable.is_symlink() and
            file_hash(executable) == context['metadata']['executable_sha256'], 'frozen executable changed')
    replay_inventory = inventory_original(replay)
    replay_files = _snapshot_files(replay_inventory, 'final.silo')
    export_inventory = inventory_original(export)
    exported_files = _snapshot_files(export_inventory, SUPPLEMENTAL_SNAPSHOT)
    require(not any(re.fullmatch(r'(?:X\.\d+|final)\.silo', name) and name != SUPPLEMENTAL_SNAPSHOT
                    for name in export_inventory), 'unexpected additional exported snapshot')
    replay_time = validate_supplemental_time(context['snapshot_times'], read_time(replay / 'final.silo'))
    exported_time = validate_supplemental_time(context['snapshot_times'], read_time(export / SUPPLEMENTAL_SNAPSHOT))
    require(exported_time == replay_time, 'native export changed the recovered physical time')
    validate_unchanged(original, original_inventory)
    return {'schema': 1, 'kind': 'native-supplemental-state', 'status': 'files-and-times-verified',
            'run_id': RUN_ID, 'original_job': dict(original_job), 'recovery_job': dict(recovery_job),
            'executable_sha256': context['metadata']['executable_sha256'],
            'original_run_metadata_sha256': original_inventory['run.json']['sha256'],
            'original_config_sha256': original_inventory['run.ini']['sha256'],
            'original_files': original_inventory, 'original_leaf': str(original),
            'source_checkpoint': {'name': SOURCE_SNAPSHOT, 'time': context['source_time'],
                                  'files': _snapshot_files(original_inventory, SOURCE_SNAPSHOT)},
            'replay': {'directory': str(replay), 'snapshot': 'final.silo', 'time': replay_time,
                       'files': replay_files, 'planned_stop_step': 1, 'planned_native_steps': 2},
            'native_export': {'directory': str(export), 'snapshot': SUPPLEMENTAL_SNAPSHOT,
                              'time': exported_time, 'files': exported_files, 'option': '--output=X.119'},
            'unchanged_physics': {'physical_c': PHYSICAL_C, 'model': 'm1', 'integrator': 'euler',
                                  'stop_time': STOP_TIME, 'requested_snapshots': 121},
            'completed_snapshot_times': context['snapshot_times'][:-1] + [exported_time, context['final_time']],
            'authoritative_ledgers': {name: original_inventory[name] for name in
                ('radiation-conservation.csv', 'momentum-diagnostics.csv')},
            'original_final': _snapshot_files(original_inventory, 'final.silo'),
            'required_external_validation': 'Verify the actual replay command and two native steps, read '
                'native-export master/shard references through Silo, and compare native fields with the '
                'replay state before installation; receipt integration is separate.',
            'scientific_caveat': SCIENTIFIC_CAVEAT}
