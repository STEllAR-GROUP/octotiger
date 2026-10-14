#!/usr/bin/env python3
"""Audited retention of newly owned campaign outputs after verified extraction.

A receipt is immutable and is committed to campaign.json before any removal.
Only its explicit intermediate X.N.silo inventory can be removed. Raw endpoints,
all checkpoint files, numerical profiles, rendered frames and videos are retained.
"""
import hashlib
import json
import math
from pathlib import Path
import re


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def safe_path(folder, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or any(part in ('.', '..') for part in relative.parts):
        raise ValueError('Unsafe retention path')
    path = folder / relative
    for parent in (path, *path.parents):
        if parent == folder:
            break
        if parent.is_symlink():
            raise ValueError('Retention paths must not contain symlinks')
    path.resolve().relative_to(folder.resolve())
    return path


def file_record(folder, path):
    relative = str(path.relative_to(folder))
    path = safe_path(folder, relative)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Missing or empty retention artifact: ' + relative)
    return {'path': relative, 'sha256': digest(path), 'bytes': path.stat().st_size}


def check_file(folder, record, optional=False):
    path = safe_path(folder, record['path'])
    if optional and not path.exists():
        return
    if (not path.is_file() or path.stat().st_size == 0 or
            ('bytes' in record and path.stat().st_size != record['bytes']) or
            digest(path) != record.get('sha256')):
        raise ValueError('Retention artifact missing or changed: ' + record['path'])


def leaf(root, run):
    relative = str(Path(run['directory']) / run['case'] / ('l' + str(run['level'])))
    return safe_path(root, relative)


def same_times(actual, expected):
    return (len(actual) == len(expected) and all(type(a) in (int, float) and math.isfinite(a) and
        math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-14) for a, b in zip(actual, expected)))


def validate_extraction(folder, run, requested_snapshots, guard=lambda: None):
    path = folder / 'movies/radiation-energy-flux/extraction.json'
    extraction = json.loads(path.read_text())
    if (extraction.get('schema') != 1 or extraction.get('status') != 'verified' or
            extraction.get('run_metadata_sha256') != run['run_metadata_sha256'] or
            not isinstance(extraction.get('settings'), dict)):
        raise ValueError('Missing verified numerical/profile/frame extraction')
    times = extraction.get('snapshot_times', [])
    if (len(times) < requested_snapshots or len(times) < 2 or
            any(type(t) not in (int, float) or not math.isfinite(t) for t in times) or
            any(b <= a for a, b in zip(times, times[1:])) or
            not math.isclose(times[0], 0, abs_tol=1e-14) or
            not math.isclose(times[-1], run['stop_time'], rel_tol=2e-12)):
        raise ValueError('Extraction must retain every requested physical snapshot including both endpoints')
    profiles = extraction.get('profiles', [])
    if not same_times([record.get('time') for record in profiles], times):
        raise ValueError('Numerical profiles do not cover every snapshot')
    panels = extraction.get('panels', [])
    if not panels or 'radiation' not in panels or len(panels) != len(set(panels)):
        raise ValueError('Invalid extracted movie panels')
    frames = extraction.get('frames', [])
    if len(frames) != len(times) * len(panels):
        raise ValueError('Rendered frame coverage is incomplete')
    for panel in panels:
        if not same_times([r.get('time') for r in frames if r.get('panel') == panel], times):
            raise ValueError('Rendered frames do not cover every snapshot/panel')
    records = [*profiles, *frames, extraction['volume_integrals'], extraction['movie'], extraction['movie_metadata']]
    if len({r['path'] for r in records}) != len(records):
        raise ValueError('Extraction artifact paths must be unique')
    for record in records:
        guard()
        check_file(folder, record)
    for record in profiles:
        profile = json.loads(safe_path(folder, record['path']).read_text())
        columns, rows = profile.get('columns', []), profile.get('rows', [])
        if (profile.get('schema') != 1 or not same_times([profile.get('time')], [record['time']]) or
                not {'x', 'er', 'fx', 'fy', 'fz', 'egas'}.issubset(columns) or not rows or
                any(len(row) != len(columns) or any(type(x) not in (int, float) or not math.isfinite(x)
                    for x in row) for row in rows)):
            raise ValueError('Retained numerical profile is incomplete or nonfinite')
    integrals = json.loads(safe_path(folder, extraction['volume_integrals']['path']).read_text())
    totals = integrals.get('records', [])
    if (integrals.get('schema') != 1 or integrals.get('status') != 'verified' or
            integrals.get('scope') != 'full-volume stored numerical states' or
            integrals.get('integration') != 'full-volume leaf-cell' or
            not same_times([record.get('time') for record in totals], times)):
        raise ValueError('Full-volume integral extraction is incomplete')
    for total in totals:
        scalars = [total.get(key) for key in ('volume', 'physical_c', 'gas_energy', 'radiation_energy')]
        vectors = [total.get(key, []) for key in ('gas_momentum', 'radiation_momentum')]
        if (any(len(vector) != 3 for vector in vectors) or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in
                    [*scalars, *vectors[0], *vectors[1]]) or min(scalars[:2]) <= 0):
            raise ValueError('Full-volume energy/momentum integrals are nonfinite or incomplete')
    movie = json.loads(safe_path(folder, extraction['movie_metadata']['path']).read_text())
    if (movie.get('status') != 'complete' or movie.get('run_metadata_sha256') != run['run_metadata_sha256'] or
            movie.get('source_snapshots') != len(times) or movie.get('displayed_snapshots') != len(times) or
            not same_times(movie.get('snapshot_times', []), times) or
            {k: v for k, v in movie.get('signature', {}).items() if k != 'inputs'} != extraction['settings']):
        raise ValueError('Movie metadata differs from verified extraction')
    for name, record in movie.get('assets', {}).items():
        check_file(folder, {**record, 'path': str(Path(extraction['movie_metadata']['path']).parent / name)})
    if not movie.get('assets') or movie.get('assets', {}).get('movie.mp4', {}).get('sha256') != extraction['movie']['sha256']:
        raise ValueError('Movie asset verification missing')
    # Store an immutable list of every extracted artifact, including the JSON
    # metadata itself; later validation does not depend on raw Silo files.
    inventory = [file_record(folder, safe_path(folder, r['path'])) for r in records]
    inventory.append(file_record(folder, path))
    for name in movie['assets']:
        asset = safe_path(folder, str(Path(extraction['movie_metadata']['path']).parent / name))
        if str(asset.relative_to(folder)) not in {r['path'] for r in inventory}:
            inventory.append(file_record(folder, asset))
    return extraction, inventory


def raw_inventory(folder, guard=lambda: None):
    numbered = sorted((int(match[1]), path) for path in folder.iterdir()
        if (match := re.fullmatch(r'X\.(\d+)\.silo', path.name)))
    if len(numbered) < 2 or not (folder / 'final.silo').is_file():
        raise ValueError('Need initial/final raw Silo snapshots before retention')
    protected = {numbered[0][1].name, numbered[-1][1].name, 'final.silo'}
    names = {path.name for _, path in numbered}
    removable = {path.name for _, path in numbered if path.name not in protected}
    keep, remove = [], []
    for path in sorted(folder.rglob('*')):
        relative = path.relative_to(folder)
        first = relative.parts[0]
        # Exact top-level numeric outputs only; checkpoint paths never match.
        snapshot = first[:-5] if first.endswith('.data') else first
        raw = snapshot in names or first.startswith(('final.silo', 'checkpoint', 'chkpt', 'restart'))
        if not raw:
            continue
        if path.is_symlink():
            raise ValueError('Raw retention inventory contains a symlink')
        if path.is_file():
            guard()
            record = file_record(folder, path)
            (remove if snapshot in removable else keep).append(record)
    if not any(r['path'] == 'final.silo' for r in keep):
        raise ValueError('Final raw Silo was not inventoried')
    return keep, remove


def make_receipt(root, manifest, run, owner, guard=lambda: None):
    if owner.get('directory') != run['directory'] or not owner.get('claimed_utc'):
        raise ValueError('Raw cleanup is limited to a run newly owned by this wrapper')
    attempt = run.get('attempts', [{}])[-1] if run.get('attempts') else {}
    identity = {'pid': attempt.get('pid'), 'process_start_ticks': attempt.get('process_start_ticks')}
    if not all(identity.values()) or owner.get('solver_identity') != identity:
        raise ValueError('Raw cleanup requires verified solver process ownership')
    folder = leaf(root, run)
    extraction, artifacts = validate_extraction(folder, run, manifest['snapshots'], guard)
    movie = json.loads(safe_path(folder, extraction['movie_metadata']['path']).read_text())
    for record in movie['signature'].get('inputs', []):
        guard()
        path = safe_path(folder, record['path'])
        stat = path.stat()
        if stat.st_size != record['size'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError('Raw Silo input changed after extraction')
    preserved, removed = raw_inventory(folder, guard)
    raw_paths = {record['path'] for record in (*preserved, *removed)
                 if re.fullmatch(r'(X\.\d+|final)\.silo(?:\.data)?', Path(record['path']).parts[0])}
    if raw_paths != {record['path'] for record in movie['signature'].get('inputs', [])}:
        raise ValueError('Extraction does not identify every raw Silo input')
    return {'schema': 1, 'status': 'verified-extraction-removal-authorized', 'run_id': run['id'],
        'run_metadata_sha256': run['run_metadata_sha256'], 'owner': owner,
        'snapshot_times': extraction['snapshot_times'], 'requested_snapshots': manifest['snapshots'],
        'settings': extraction['settings'], 'preserved_raw': preserved, 'removed_raw': removed,
        'extracted_artifacts': artifacts,
        'movie_metadata': extraction['movie_metadata'], 'extraction': 'movies/radiation-energy-flux/extraction.json'}


def validate_retention(root, run, guard=lambda: None):
    pointer = run.get('retention')
    if pointer is None:
        if (leaf(root, run) / 'retention.json').exists():
            raise ValueError('Retention receipt exists without its committed campaign reference')
        return None
    folder = leaf(root, run)
    path = safe_path(root, pointer['path'])
    if path != folder / 'retention.json' or digest(path) != pointer.get('sha256'):
        raise ValueError('Retention receipt identity/hash changed')
    receipt = json.loads(path.read_text())
    if (receipt.get('schema') != 1 or receipt.get('status') != 'verified-extraction-removal-authorized' or
            receipt.get('run_id') != run['id'] or receipt.get('run_metadata_sha256') != run['run_metadata_sha256']):
        raise ValueError('Retention receipt belongs to different run metadata')
    attempt = run.get('attempts', [{}])[-1] if run.get('attempts') else {}
    identity = {'pid': attempt.get('pid'), 'process_start_ticks': attempt.get('process_start_ticks')}
    if receipt.get('owner', {}).get('solver_identity') != identity:
        raise ValueError('Retention receipt belongs to a different solver attempt')
    protected = receipt.get('preserved_raw', [])
    removed = receipt.get('removed_raw', [])
    if not protected or not any(r['path'] == 'final.silo' for r in protected):
        raise ValueError('Retention receipt is missing final raw state')
    inventory = [*protected, *removed, *receipt.get('extracted_artifacts', [])]
    if len({r['path'] for r in inventory}) != len(inventory):
        raise ValueError('Retention inventories overlap')
    for record in inventory:
        guard()
        check_file(folder, record, optional=record in removed)
    return receipt


def prune(root, run, guard=lambda: None):
    receipt = validate_retention(root, run, guard)
    if receipt is None:
        raise ValueError('Raw removal requires a committed retention receipt')
    folder = leaf(root, run)
    directories = set()
    for record in receipt['removed_raw']:
        guard()
        relative = Path(record['path'])
        if not re.fullmatch(r'X\.\d+\.silo(?:\.data)?', relative.parts[0]):
            raise ValueError('Only exact numeric intermediate Silo outputs may be removed')
        path = safe_path(folder, record['path'])
        check_file(folder, record, optional=True)
        if path.exists():
            path.unlink()
        directories.update(parent for parent in path.parents if parent != folder and folder in parent.parents)
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        guard()
        if directory.exists():
            directory.rmdir()  # Refuse un-inventoried extra contents rather than delete them.
    return sum(record['bytes'] for record in receipt['removed_raw'])


def validate_retained_cache(root, run, settings):
    receipt = validate_retention(root, run)
    if receipt is None:
        return None
    if settings != receipt['settings']:
        raise ValueError('Retained movie settings differ; deleted intermediate raw states cannot be rerendered')
    folder = leaf(root, run)
    return json.loads(safe_path(folder, receipt['movie_metadata']['path']).read_text())
