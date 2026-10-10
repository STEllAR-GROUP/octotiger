#!/usr/bin/env python3
"""Exercise opt-in gradient options, unchanged defaults, and Silo persistence.

Runs the real application sequentially with one HPX worker. The default check
compares omitted options with explicit field=none in this binary; it is not a
comparison against a pre-change executable. Pure production-path tests cover
the unchanged legacy refinement branch separately. Enabled threshold extremes
exercise base-only and all-fine meshes, not mixed-level AMR accuracy.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent / 'radiation'))
from test_restart import Silo


HYDRO_FIELDS = ('egas', 'tau', 'sx', 'sy', 'sz', 'rho_1', 'pot', 'lx', 'ly', 'lz')
GRAVITY_FIELDS = ('gx', 'gy', 'gz')
GRADIENT_METADATA = (
    'gradient_amr_options_version', 'gradient_refine_field', 'gradient_refine_mode',
    'gradient_refine_threshold', 'gradient_refine_floor', 'gradient_refine_buffer',
    'gradient_refine_coarsen_factor', 'gradient_min_level', 'gradient_max_level',
)


def compare_fields(reference, actual):
    report = {}
    for field, reference_blocks in reference.items():
        actual_blocks = actual[field]
        if reference_blocks.keys() != actual_blocks.keys():
            raise AssertionError(f'{field}: checkpoint block inventories differ')
        before, after = [], []
        for block in sorted(reference_blocks):
            if len(reference_blocks[block]) != len(actual_blocks[block]):
                raise AssertionError(f'{field}: array lengths differ in {block}')
            before.extend(reference_blocks[block])
            after.extend(actual_blocks[block])
        scale = max(abs(value) for value in before)
        difference = max(abs(a-b) for a, b in zip(before, after))
        if difference > 5e-11 * scale + 1e-15:
            raise AssertionError(f'{field}: maximum difference {difference} exceeds tolerance at scale {scale}')
        report[field] = {'cells': len(before), 'maximum_absolute_difference': difference,
                         'reference_scale': scale,
                         'relative_linf': difference / scale if scale else None,
                         'exact': before == after}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--silo-library')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    executable = args.executable.resolve()
    folder = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix='gradient-options-'))
    folder.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    print(f'Gradient option artifacts: {folder}', flush=True)
    base = {
        'problem': 'SOD', 'radiation': 'off', 'hydro': 'on', 'gravity': 'on',
        'omega': 0, 'unigrid': 'off', 'min_level': 1, 'max_level': 1,
        'xscale': 1, 'n_species': 1, 'eos': 'IDEAL', 'periodic': 'off',
        # The hard cap controls this tiny run. cfl=1 gives a two-step regrid
        # batch, so the driver's end-time cap reaches .0002 in exactly two steps.
        'cfl': 1, 'hard_dt': .0001, 'stop_time': .0002, 'stop_step': 1,
        'odt': 1, 'disable_output': 'off', 'disable_diagnostics': 'on',
        'disable_analytic': 'on', 'correct_am_hydro': 'off',
        'hydro_host_kernel_type': 'LEGACY', 'hydro_device_kernel_type': 'OFF',
        'monopole_device_kernel_type': 'OFF', 'multipole_device_kernel_type': 'OFF',
        'silo_num_groups': 1, 'code_to_g': 1, 'code_to_cm': 1, 'code_to_s': 1,
    }

    def invoke(name, updates=None, cli=None, restart=None, valid=True):
        output = folder / name
        output.mkdir()
        config = dict(base, datadir=str(output) + os.sep)
        config.update(updates or {})
        if restart:
            config['restart_filename'] = str(restart)
        ini = output / 'run.ini'
        ini.write_text(''.join(f'{key}={value}\n' for key, value in config.items()))
        command = [str(executable), '--config_file=' + str(ini), '--hpx:threads=1']
        command.extend(f'--{key}={value}' for key, value in (cli or {}).items())
        with (output / 'run.log').open('w') as log:
            result = subprocess.run(command, cwd=restart.parent if restart else output,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout if valid else 30)
        text = (output / 'run.log').read_text(errors='replace')
        if valid and (result.returncode or not (output / 'final.silo').is_file()):
            raise AssertionError(f'Valid run failed in {output}:\n{text[-5000:]}')
        if not valid and (result.returncode == 0 or 'gradient' not in text.lower() or (output / 'final.silo').exists()):
            raise AssertionError(f'Invalid options were not rejected before evolution in {output}:\n{text[-3000:]}')
        return output, result.returncode

    help_result = subprocess.run([str(executable), '--help', '--hpx:threads=1'],
                                 text=True, capture_output=True, timeout=30)
    for option in ('gradient_refine_field', 'gradient_refine_mode', 'gradient_refine_threshold',
                   'gradient_refine_floor', 'gradient_refine_buffer', 'gradient_refine_coarsen_factor'):
        if option not in help_result.stdout + help_result.stderr:
            raise AssertionError(f'Missing option help for {option}')

    invalid = (
        ('field', {'gradient_refine_field': 'bogus'}),
        ('mode', {'gradient_refine_mode': 'bogus'}),
        ('zero-threshold', {'gradient_refine_threshold': '0'}),
        ('nan-threshold', {'gradient_refine_threshold': 'nan'}),
        ('buffer', {'gradient_refine_buffer': '3'}),
        ('radiation-off', {'gradient_refine_field': 'radiation_energy'}),
        ('unigrid', {'unigrid': 'on'}),
        ('floor', {'gradient_refine_floor': '0'}),
        ('coarsen', {'gradient_refine_coarsen_factor': '1.01'}),
    )
    rejections = []
    for source in ('cli', 'config'):
        for label, values in invalid:
            selected = {'gradient_refine_field': 'density', **values}
            output, status = invoke(f'invalid-{source}-{label}',
                                    updates=selected if source == 'config' else None,
                                    cli=selected if source == 'cli' else None, valid=False)
            rejections.append({'source': source, 'case': label, 'returncode': status})

    silo = Silo(args.silo_library)

    def snapshot(output, gravity=False, gradient=False, expected_blocks=8):
        fields = HYDRO_FIELDS + (GRAVITY_FIELDS if gravity else ())
        names = ('problem', 'gravity', 'hydro') + (GRADIENT_METADATA if gradient else ())
        return silo.snapshot(output / 'final.silo', fields=fields, metadata_names=names,
                             block_count=expected_blocks)

    default, _ = invoke('default')
    explicit, _ = invoke('explicit-none', {'gradient_refine_field': 'none'})
    default_metadata, default_fields = snapshot(default, gravity=True)
    explicit_metadata, explicit_fields = snapshot(explicit, gravity=True)
    if default_metadata != explicit_metadata or default_metadata['gravity'] != 1 or default_metadata['hydro'] != 1:
        raise AssertionError('Default and explicit-none hydro/gravity metadata differ')
    if not math.isclose(default_metadata['time'], .0002, rel_tol=2e-13):
        raise AssertionError(f'Two-step default run ended at unexpected time {default_metadata["time"]}')
    unchanged = compare_fields(default_fields, explicit_fields)

    accepted = []
    restart_source = None
    restart_configuration = None
    restart_metadata = None
    for mode in ('absolute', 'relative'):
        for label, threshold, blocks in (('low', 1e-8, 64), ('high', 1e99, 8)):
            config = {'gravity': 'off', 'min_level': 1, 'max_level': 2,
                      'gradient_refine_field': 'density', 'gradient_refine_mode': mode,
                      'gradient_refine_threshold': threshold, 'gradient_refine_floor': .02,
                      'gradient_refine_buffer': 1, 'gradient_refine_coarsen_factor': .65}
            output, _ = invoke(f'{mode}-{label}', config)
            metadata, values = snapshot(output, gradient=True, expected_blocks=blocks)
            expected = {'gradient_amr_options_version': 1, 'gradient_refine_field': 1,
                        'gradient_refine_mode': int(mode == 'relative'),
                        'gradient_refine_threshold': threshold, 'gradient_refine_floor': .02,
                        'gradient_refine_buffer': 1, 'gradient_refine_coarsen_factor': .65,
                        'gradient_min_level': 1, 'gradient_max_level': 2}
            for name, value in expected.items():
                if metadata[name] != value:
                    raise AssertionError(f'{mode}-{label}: checkpoint {name}={metadata[name]} differs from {value}')
            accepted.append({'mode': mode, 'threshold': threshold, 'leaf_blocks': blocks,
                             'mesh': 'base-only' if blocks == 8 else 'all-fine', 'metadata': metadata})
            if mode == 'relative' and label == 'low':
                restart_source, restart_configuration, restart_metadata = output, config, metadata

    direct, _ = invoke('restart-direct', dict(restart_configuration, stop_time=.0004, stop_step=3))
    conflicting = {
        'gravity': 'off', 'stop_time': .0004, 'stop_step': 1, 'unigrid': 'on',
        'min_level': 0, 'max_level': 0, 'gradient_refine_field': 'none',
        'gradient_refine_mode': 'absolute', 'gradient_refine_threshold': 1e99,
        'gradient_refine_floor': .5, 'gradient_refine_buffer': 0,
        'gradient_refine_coarsen_factor': 1,
    }
    resumed, _ = invoke('restart-resumed', conflicting, restart=restart_source / 'final.silo')
    direct_metadata, direct_fields = snapshot(direct, gradient=True, expected_blocks=64)
    resumed_metadata, resumed_fields = snapshot(resumed, gradient=True, expected_blocks=64)
    for name in GRADIENT_METADATA:
        if resumed_metadata[name] != restart_metadata[name]:
            raise AssertionError(f'Restart did not restore saved {name}')
    if direct_metadata != resumed_metadata or not math.isclose(resumed_metadata['time'], .0004, rel_tol=2e-13):
        raise AssertionError('Restart endpoint metadata differs from uninterrupted run')
    restart_comparison = compare_fields(direct_fields, resumed_fields)

    report = {
        'status': 'pass', 'scope': 'One-worker application checks; default versus explicit none in current binary, threshold-extreme meshes, and checkpoint restoration. Not comparison with a pre-change binary or mixed-level AMR accuracy.',
        'default_equals_explicit_none': all(field['exact'] for field in unchanged.values()),
        'default_hydro_gravity_comparison': unchanged,
        'default_metadata': default_metadata,
        'rejected_options': rejections,
        'enabled_threshold_extremes': accepted,
        'restart_restored_metadata': {name: resumed_metadata[name] for name in GRADIENT_METADATA},
        'restart_fields': restart_comparison,
    }
    (folder / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'pass', 'report': str(folder/'validation.json'),
                      'invalid_options_rejected': len(rejections),
                      'default_equals_explicit_none': report['default_equals_explicit_none'],
                      'restart_maximum_absolute_difference': max(x['maximum_absolute_difference'] for x in restart_comparison.values())}, indent=2))


if __name__ == '__main__':
    main()
