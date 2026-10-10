#!/usr/bin/env python3
"""Collect all 28 completed QB4 runs into a new, self-contained static site.

Only local files and the already-built results check/plot commands are used.
No simulation, build, scheduler query, upload, or cluster connection is made.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit

from workflow import CASES, LEVELS, TIMES, fingerprint, tasks, validate

C = 2.99792458e10
ORIGIN = 'Octo-TIGER application'
UNITS = {'system': 'CGS', 'length': 'cm', 'time': 's', 'mass': 'g',
         'energy_density': 'erg/cm^3', 'flux': 'erg/(cm^2 s)', 'chi': '1/cm',
         'luminosity': 'erg/s', 'source': 'erg/(cm^3 s)'}
HASH = re.compile(r'[0-9a-f]{64}')
PROBLEMS = {case: 'RADIATION_' + case.upper() for case in CASES}
PROBLEMS['radiating_blast'] = 'RADIATION_BLAST_WAVE'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    require(path.is_file() and not path.is_symlink(), f'Missing regular file: {path}')
    require(path.stat().st_size > 0, f'Empty file: {path}')
    return path


def read_json(path):
    return json.loads(regular(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inside(path, root):
    return path == root or root in path.parents


def sha256(path):
    digest = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_ini(path):
    result = {}
    for line in regular(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition('=')
        key, value = key.strip(), value.strip()
        require(separator and key and key not in result, f'Invalid INI line in {path}: {line}')
        result[key] = value
    return result


def conservation_time(path, expected):
    """Stream the ledger before publication; the C++ reader checks its balance."""
    columns = ['t', 'volume', 'er', 'fx', 'fy', 'fz']
    columns += [f'{field}_{kind}' for kind in ('boundary', 'source')
                for field in ('er', 'fx', 'fy', 'fz')]
    previous = None
    count = 0
    with regular(path).open(newline='') as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == columns, f'Invalid conservation header: {path}')
        for row in reader:
            require(len(row) == len(columns), f'Incomplete conservation row: {path}')
            values = [float(value) for value in row]
            require(all(math.isfinite(value) for value in values), f'Nonfinite conservation row: {path}')
            actual = values[0]
            require(actual == 0 if previous is None else actual > previous,
                    f'Conservation time is not increasing from zero: {path}')
            previous = actual
            count += 1
    require(count >= 2 and math.isclose(previous, expected, rel_tol=4e-15, abs_tol=0),
            f'Conservation does not reach the planned final time {expected}: {path}')
    return previous


def amr_history(path, level, stop):
    """Validate every recorded leaf census without treating occupancy as accuracy.

    A census is emitted at initialization and regrids, not necessarily at the
    final output time. Counts cover the domain exactly; desired ratios may fail.
    """
    samples, outside, exceptions = 0, 0, 0
    previous, levels, last = None, {}, None

    def finish(stamp, leaves):
        coarse = leaves.get(str(level - 1), {}).get('leaf_blocks', 0)
        fine = leaves.get(str(level), {}).get('leaf_blocks', 0)
        require(coarse * 8 + fine == 8**level, f'AMR census does not cover domain: {path}')
        ratio = fine / coarse if coarse else None
        return {'last_regrid_time': stamp[0], 'last_regrid_step': stamp[1],
                'leaf_blocks': coarse + fine, 'leaf_cells': (coarse + fine) * 512,
                'levels': leaves}, ratio

    ratios = []
    with regular(path).open(newline='') as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == ['t', 'step', 'level', 'leaf_blocks', 'leaf_cells'],
                f'Invalid AMR census header: {path}')
        for row in reader:
            require(len(row) == 5, f'Incomplete AMR census: {path}')
            values = [float(v) for v in row]
            require(all(math.isfinite(v) for v in values) and all(v == int(v) for v in values[1:]),
                    f'Nonfinite/nonintegral AMR census: {path}')
            stamp = values[0], int(values[1])
            current, blocks, cells = map(int, values[2:])
            require(0 <= stamp[0] <= stop and stamp[1] >= 0 and
                    current in (level - 1, level) and blocks > 0 and cells == blocks * 512,
                    f'Invalid AMR leaf count/time/level: {path}')
            if stamp != previous:
                if previous is not None:
                    last, ratio = finish(previous, levels)
                    samples += 1
                    if ratio is not None:
                        ratios.append(ratio)
                    outside += ratio is None or not .1 <= ratio <= 1
                    exceptions += ratio is None and level <= 3
                    require(stamp[0] > previous[0] and stamp[1] > previous[1],
                            f'AMR census times/steps are not increasing: {path}')
                else:
                    require(stamp == (0, 0), f'Missing initial AMR census: {path}')
                previous, levels = stamp, {}
            require(str(current) not in levels, f'Duplicate AMR census level: {path}')
            levels[str(current)] = {'leaf_blocks': blocks, 'leaf_cells': cells}
    require(previous is not None, f'Empty AMR census: {path}')
    last, ratio = finish(previous, levels)
    samples += 1
    if ratio is not None:
        ratios.append(ratio)
    outside += ratio is None or not .1 <= ratio <= 1
    exceptions += ratio is None and level <= 3
    return {'samples': samples, 'full_mesh': last, 'last_fine_to_coarse_cell_ratio': ratio,
            'minimum_finite_cell_ratio': min(ratios) if ratios else None,
            'maximum_finite_cell_ratio': max(ratios) if ratios else None,
            'samples_outside_target_ratio': outside, 'allowed_coarse_all_fine_samples': exceptions,
            'scope': 'Observed occupancy only; ratio misses are retained and are not accuracy failures.'}


def inspect(plan):
    config = plan['config']
    validate(config)
    require(plan['tasks'] == tasks(config, plan.get('amr_plan')), 'Tasks differ from the frozen plan')
    if plan.get('amr_plan') is not None:
        require(fingerprint(plan['amr_plan']) == plan.get('amr_plan_sha256'), 'AMR plan digest differs')
    runs = [task for task in plan['tasks'] if task.get('kind') == 'run']
    expected = {(case, level) for case in CASES for level in LEVELS}
    require(len(runs) == 28 and {(t['case'], t['level']) for t in runs} == expected,
            'The plan must contain exactly seven cases at levels 2, 3, 4, and 5')
    output = Path(config['output_root'])
    require(output.is_dir() and not output.is_symlink(), 'Missing or symlinked output_root')
    build = Path(config['build_dir'])
    runner = regular(build / 'radiation-results-tools/radiation-results')
    require(os.access(runner, os.X_OK), f'Results tool is not executable: {runner}')
    executable_hash = sha256(build / 'octotiger')
    sources = None
    items = []
    for task in runs:
        case, level = task['case'], task['level']
        key = f'{case}-l{level}'
        stop = TIMES.get(case, config['radiation_time'])
        factor = config.get('reduced_c_factors', {}).get(case, 1.0)
        require(type(factor) in (int, float) and math.isfinite(factor) and 0 < factor <= 1,
                f'Invalid planned reduced light-speed factor: {key}')
        require(case in TIMES or factor == 1,
                f'Analytic radiation tests require the physical light speed: {key}')
        require(task.get('radiation_reduced_c_factor') == factor,
                f'Plan task has mismatched reduced light-speed factor: {key}')
        require(task['key'] == key and task['cells'] == 8 * 2**level and task['time'] == stop,
                f'Plan dimensions or final time differ from the full benchmark: {key}')
        batch_dir = output / 'batches' / key
        folder = batch_dir / case / f'l{level}'
        require(folder.is_dir() and inside(folder.resolve(), output.resolve()),
                f'Missing or external run directory: {folder}')
        require(not any(p.is_symlink() for p in (batch_dir, folder.parent, folder)),
                f'Symlinked run directory: {folder}')
        batch = read_json(batch_dir / 'batch.json')
        meta = read_json(folder / 'run.json')
        require(meta.get('status') == 'complete', f'Run is not complete: {key}')
        require(batch.get('cases') == [case] and batch.get('levels') == [level],
                f'Unexpected source batch membership: {key}')
        for field, value in {'case': case, 'level': level, 'inx': 8, 'cells': task['cells'],
                             'time': stop, 'c': C, 'units': UNITS, 'origin': ORIGIN,
                             'radiation_reduced_c_factor': factor, 'physical_c': C,
                             'transport_c': factor * C,
                             'executable_sha256': executable_hash}.items():
            require(meta.get(field) == value, f'Run {key} has mismatched {field}')
        for field, value in {'units': UNITS, 'origin': ORIGIN,
                             'executable_sha256': executable_hash}.items():
            require(batch.get(field) == value, f'Batch {key} has mismatched {field}')
        recorded_sources = batch.get('source_sha256')
        require(isinstance(recorded_sources, dict) and recorded_sources and
                all(isinstance(name, str) and isinstance(digest, str) and HASH.fullmatch(digest)
                    for name, digest in recorded_sources.items()), f'Missing source hashes: {key}')
        if sources is None:
            sources = recorded_sources
        require(recorded_sources == sources, f'Source provenance changed between runs: {key}')
        cfg = read_ini(folder / 'run.ini')
        require(cfg == meta.get('config'), f'run.ini differs from recorded configuration: {key}')
        require(cfg.get('problem') == PROBLEMS[case], f'Wrong problem: {key}')
        amr = task.get('amr')
        require(meta.get('adaptive', False) == bool(amr), f'Wrong adaptive mesh contract: {key}')
        for field, value in {'min_level': level - 1 if amr else level, 'max_level': level, 'stop_time': stop,
                             'code_to_g': 1, 'code_to_cm': 1, 'code_to_s': 1,
                             'radiation_reduced_c_factor': factor}.items():
            require(float(cfg.get(field, 'nan')) == value, f'Incorrect {field}: {key}')
        if amr:
            require(meta.get('min_level') == level - 1 and meta.get('max_level') == level and
                    cfg.get('unigrid') == 'off', f'Incorrect adaptive level bounds: {key}')
            for field, value in amr.items():
                if field.startswith('gradient_refine_'):
                    actual = cfg.get(field) if isinstance(value, str) else float(cfg.get(field, 'nan'))
                    require(actual == value, f'Incorrect {field}: {key}')
            history = amr_history(folder / 'gradient-amr.csv', level, stop)
            require(meta.get('mesh', {}).get('full_mesh') == history['full_mesh'],
                    f'Adaptive metadata differs from final recorded census: {key}')
        length = 2e13 if case == 'radiating_blast' else 7e10 if case in TIMES else 6e10
        require(meta.get('length') == length and meta.get('dx') == length / task['cells'],
                f'Incorrect domain or cell width: {key}')
        require(2 * float(cfg.get('xscale', 'nan')) == length, f'Incorrect domain: {key}')
        if case in TIMES:
            require(meta.get('reference_kind') == 'numerical-only' and meta.get('benchmark_time') == stop,
                    f'Missing coupled-benchmark scope: {key}')
        if case == 'gaussian_pulse':
            require(sha256(folder / 'reference.bin') == meta.get('reference_sha256'),
                    f'Gaussian reference changed: {key}')
        actual = conservation_time(folder / 'radiation-conservation.csv', stop)
        regular(folder / 'run.log')
        items.append((task, batch_dir, folder, meta, actual))
    return runner, executable_hash, sources, items


def copy_file(source, target):
    regular(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Copy rather than link: later changes to raw outputs must not alter the site.
    shutil.copy2(source, target)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.identifiers = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src', 'poster') and value:
                self.urls.append((tag, key, value))
            if (key == 'id' or (tag == 'a' and key == 'name')) and value:
                self.identifiers.add(value)


def check_links(site):
    """External citation anchors are allowed; page assets must be local files."""
    count = 0
    pages = {}
    for page in site.rglob('*.html'):
        parser = Links()
        parser.feed(page.read_text())
        pages[page.resolve()] = parser
    for page, parser in pages.items():
        for tag, attribute, url in parser.urls:
            parsed = urlsplit(url)
            if parsed.scheme in ('http', 'https') or parsed.netloc:
                require(tag == 'a' and attribute == 'href', f'External site asset: {page}: {url}')
                continue
            require(not parsed.scheme, f'Nonportable URL: {page}: {url}')
            path = unquote(parsed.path)
            require(not path.startswith(('/', '\\')) and '\\' not in path,
                    f'Nonrelative link: {page}: {url}')
            target = (page.parent / path).resolve() if path else page
            require(inside(target, site.resolve()) and target.is_file(),
                    f'Broken or external local link: {page}: {url}')
            if parsed.fragment and target in pages:
                require(unquote(parsed.fragment) in pages[target].identifiers,
                        f'Missing local anchor: {page}: {url}')
            count += 1
    require((site / 'index.html').is_file(), 'Renderer did not create index.html')
    for case in CASES:
        require((site / f'{case}.html').is_file(), f'Renderer did not create {case}.html')
    return count


def collect(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    output = Path(plan['config']['output_root'])
    require(output.is_dir(), 'No result output directory exists')
    # One local collector owns staging/publication, even if started twice.
    with (output / '.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (output / 'site').exists(), 'site already exists; it will not be overwritten')
        runner, executable_hash, sources, items = inspect(plan)
        environment = os.environ.copy()
        environment.update({key: '1' for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',
                                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
        with tempfile.TemporaryDirectory(prefix='.site-stage-', dir=output) as temporary:
            stage = Path(temporary)
            provenance = stage / 'provenance'
            provenance.mkdir()
            copy_file(plan_path, provenance / 'plan.json')
            records = []
            for task, batch_dir, folder, meta, actual in items:
                target = stage / task['case'] / f'l{task["level"]}'
                files = ['run.json', 'run.ini', 'run.log', 'radiation-conservation.csv']
                if task.get('amr'):
                    files.append('gradient-amr.csv')
                if task['case'] not in TIMES:
                    files += ['L1.dat', 'L2.dat', 'Linf.dat']
                for name in files:
                    copy_file(folder / name, target / name)
                require(read_json(target / 'run.json') == meta,
                        f'Run metadata changed during collection: {task["key"]}')
                slices = 'radiation-hydro-slices' if task['case'] in TIMES else 'radiation-slices'
                require((folder / slices).is_dir() and not (folder / slices).is_symlink(),
                        f'Missing regular slice directory: {folder / slices}')
                csv_files = list((folder / slices).glob('*.csv'))
                require(csv_files, f'No numerical slices: {folder}')
                for source in csv_files:
                    copy_file(source, target / slices / source.name)
                batch_name = task['key'] + '-batch.json'
                copy_file(batch_dir / 'batch.json', provenance / batch_name)
                records.append({'task': task['key'], 'actual_final_time': actual,
                                'run_metadata': str((target / 'run.json').relative_to(stage)),
                                'run_metadata_sha256': sha256(target / 'run.json'),
                                'source_batch': 'provenance/' + batch_name,
                                'source_batch_sha256': sha256(provenance / batch_name)})
                if task.get('amr'):
                    records[-1]['amr_history'] = amr_history(target / 'gradient-amr.csv', task['level'], actual)
            write_json(stage / 'batch.json', {
                'workflow': 'qb4-collected', 'origin': ORIGIN, 'cases': list(CASES),
                'levels': list(LEVELS), 'units': UNITS, 'executable_sha256': executable_hash,
                'source_sha256': sources, 'source_batches': records,
                'collection_note': 'All planned final outputs checked; this is not an independent accuracy acceptance test.'})
            for command in ('check', 'plot'):
                with (provenance / f'collection-{command}.log').open('w') as log:
                    subprocess.run([str(runner), command, str(stage)], check=True,
                                   cwd=Path(plan['config']['source_root']), env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
            # Plot data/scripts are portable as web assets only; remove scripts
            # embedding staging paths. The retained CSV/JSON files provide data.
            for path in stage.rglob('*'):
                require(not path.is_symlink(), f'Unexpected symlink in site: {path}')
                if path.is_file() and path.suffix in ('.gp', '.gnuplot'):
                    path.unlink()
                elif path.is_file():
                    require(path.suffix.lower() in ('.html', '.css', '.js', '.png', '.svg', '.pdf',
                                                  '.csv', '.json', '.ini', '.log', '.dat', '.txt'),
                            f'Unexpected site file: {path}')
            links = check_links(stage)
            write_json(provenance / 'collection.json', {
                'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
                'run_count': len(items), 'relative_links_checked': links,
                'plan_sha256': sha256(plan_path), 'results_tool_sha256': sha256(runner),
                'solver_sha256': executable_hash, 'runs': records,
                'validation_scope': 'Complete target-time data and report validation; no new simulation or independent physical validation.'})
            stage.rename(output / 'site')
        print(f'Collected and checked all 28 runs: {output / "site/index.html"}')
        return output / 'site'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    try:
        collect(args.plan)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'collect.py: {error}\n')


if __name__ == '__main__':
    main()
