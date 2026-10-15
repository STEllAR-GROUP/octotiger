#!/usr/bin/env python3
"""Compare OctoTiger native checkpoint fields by leaf and Silo object identity.

This reader deliberately supports the current single-species, three-dimensional
Silo/HDF5 checkpoint layout only. It does not follow external HDF5 links, trust
raw dataset numbers, compare derived variables, or modify either checkpoint.
"""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

FIELDS = ('rho_1', 'egas', 'tau', 'sx', 'sy', 'sz', 'pot', 'lx', 'ly', 'lz', 'er', 'fx', 'fy', 'fz')
DEFAULT_TOLERANCE = 512 * np.finfo(np.float64).eps
REQUIRED_METADATA = (
    'version', 'code_to_g', 'code_to_cm', 'code_to_s', 'n_species', 'eos',
    'gravity', 'hydro', 'omega', 'problem', 'radiation', 'periodic', 'xscale',
    'radiation_options_version', 'radiation_reduced_c_factor', 'radiation_model',
    'radiation_integrator', 'radiation_algorithm_version', 'radiation_limiter',
    'radiation_flux_semantics', 'rad_implicit', 'rad_test_chi', 'rad_test_width',
    'rad_test_background', 'rad_test_amplitude', 'rad_test_luminosity')
COMPOSITION_METADATA = ('X', 'Z', 'atomic_mass', 'atomic_number')
FLD_METADATA = ('fld_boundary', 'fld_nonlinear_tolerance', 'fld_linear_tolerance',
                'fld_max_iterations', 'fld_max_linear_iterations')
FLD_HYDRO_METADATA = ('fld_hydro_unigrid', 'fld_hydro_reflect_bc', 'fld_hydro_cfl')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode('utf-8').rstrip('\0')
    if isinstance(value, str):
        return value.rstrip('\0')
    raise ValueError('Expected a Silo string')


def _hard_object(handle, path, h5py):
    """Resolve only hard links inside this HDF5 file, including parent groups."""
    require(isinstance(path, str) and '\\' not in path and ':' not in path,
            'Invalid internal HDF5 reference')
    parts = path.split('/')
    require(all(part not in ('.', '..') for part in parts), 'Traversing internal HDF5 reference')
    obj = handle['/']
    for part in filter(None, parts):
        require(isinstance(obj, h5py.Group), 'HDF5 reference traverses a non-group')
        require(isinstance(obj.get(part, getlink=True), h5py.HardLink),
                f'Nonlocal or missing HDF5 link: {path}')
        obj = obj[part]
    return obj


def _group_names(group, h5py):
    """Silo may store an intrinsic '..' backlink to the same-file parent."""
    for name in group:
        if name != '..':
            yield name
            continue
        link = group.get(name, getlink=True)
        require(isinstance(link, h5py.HardLink), 'Unsafe Silo parent backlink')
        target = group[name]
        require(isinstance(target, h5py.Group) and target.file.id == group.file.id and
                h5py.h5o.get_info(target.id).addr == h5py.h5o.get_info(group.parent.id).addr,
                'Silo parent backlink does not identify its same-file parent')


def _silo(obj, h5py):
    require(isinstance(obj, h5py.Datatype) and 'silo' in obj.attrs,
            f'Expected native Silo datatype object: {obj.name}')
    value = np.asarray(obj.attrs['silo'])
    require(value.size == 1 and value.dtype.names, f'Invalid Silo compound attributes: {obj.name}')
    record = value.reshape(-1)[0]
    return {key: record[key] for key in value.dtype.names}


def _strings(value, strict=True):
    array = np.asarray(value)
    if array.dtype.kind in 'iu' and array.dtype.itemsize == 1:
        return [array.tobytes().decode('utf-8', errors='strict' if strict else 'ignore')]
    require(array.dtype.kind in 'SUO', 'Silo reference list must contain strings or bytes')
    if not strict:
        return [bytes(item).decode('utf-8', errors='ignore') if isinstance(item, (bytes, np.bytes_))
                else str(item) for item in array.reshape(-1)]
    return [_text(item) for item in array.reshape(-1)]


def _reference_entries(handle, obj, member, h5py):
    meta = _silo(obj, h5py)
    require(member in meta, f'Missing {member} in {obj.name}')
    pointer = _text(meta[member])
    values = _hard_object(handle, pointer, h5py)
    require(isinstance(values, h5py.Dataset), f'Expected Silo string dataset: {pointer}')
    entries = []
    for item in _strings(values[()]):
        entries.extend(part for part in re.split('[;\0]', item) if part)
    require(entries, f'Empty reference list: {obj.name}')
    count_key = 'nblocks' if member == 'meshnames' else 'nvars'
    require(int(meta.get(count_key, -1)) == len(entries), f'Incorrect Silo reference count: {obj.name}')
    return entries


def _parse_reference(header, reference, expected_object=None):
    """Accept precisely HEADER.silo.data/NUMBER.silo:/OCTREE_LEAF/OBJECT."""
    require(reference.count(':') == 1 and '\\' not in reference, 'Invalid external Silo reference')
    filename, object_name = reference.split(':')
    parts = filename.split('/')
    require(len(parts) == 2 and parts[0] == header.name + '.data' and
            re.fullmatch(r'[0-9]+\.silo', parts[1]),
            f'Header references a foreign checkpoint shard: {reference}')
    match = re.fullmatch(r'/([1][0-7]*)/([A-Za-z_][A-Za-z_0-9]*)', object_name)
    require(match is not None, f'Invalid octree leaf/object reference: {reference}')
    leaf, field = match.groups()
    require(expected_object is None or field == expected_object, 'Header object name mismatch')
    data = header.parent / parts[0]
    shard = data / parts[1]
    require(data.is_dir() and not data.is_symlink() and shard.is_file() and not shard.is_symlink() and
            shard.resolve().parent == data.resolve(), 'Missing, symlinked, or escaping checkpoint shard')
    return shard, leaf, field


def _check_header_links(handle, header, h5py):
    """Reject hidden external/soft links and foreign references in string data."""
    seen = set()
    def visit(group):
        address = h5py.h5o.get_info(group.id).addr
        require(address not in seen, 'Repeated or cyclic HDF5 group link')
        seen.add(address)
        for name in _group_names(group, h5py):
            obj = _hard_object(handle, group.name.rstrip('/') + '/' + name, h5py)
            if isinstance(obj, h5py.Group):
                visit(obj)
            elif isinstance(obj, h5py.Dataset) and (obj.dtype.kind in 'SUO' or
                    obj.dtype.kind in 'iu' and obj.dtype.itemsize == 1):
                # Unused bytes following hostname's terminating NUL need not
                # be UTF-8. Actual Silo reference lists are decoded strictly.
                for text in _strings(obj[()], strict=False):
                    for entry in re.split('[;\0]', text):
                        if '.silo:' in entry:
                            _parse_reference(header, entry)
            elif isinstance(obj, h5py.Datatype) and 'silo' in obj.attrs:
                for value in _silo(obj, h5py).values():
                    if np.asarray(value).dtype.kind in 'SU':
                        for text in _strings(value):
                            for entry in re.split('[;\0]', text):
                                if '.silo:' in entry:
                                    _parse_reference(header, entry)
    visit(handle['/'])


def _numeric_dataset(handle, path, h5py, float64=False):
    obj = _hard_object(handle, path, h5py)
    require(isinstance(obj, h5py.Dataset), f'Expected numeric dataset: {path}')
    require(obj.dtype.kind in 'fiu', f'Non-numeric payload: {path}')
    if float64:
        require(obj.dtype.kind == 'f' and obj.dtype.itemsize == 8, f'Expected float64 payload: {path}')
    values = np.asarray(obj[()])
    require(values.size > 0 and np.isfinite(values).all(), f'Empty or nonfinite payload: {path}')
    return values


def _metadata(handle, h5py):
    result = {}
    for name in REQUIRED_METADATA:
        values = _numeric_dataset(handle, '/' + name, h5py)
        require(values.size == 1, f'Expected scalar checkpoint metadata: {name}')
        result[name] = values.reshape(-1)[0].item()
    require(result['n_species'] == 1 and result['radiation'] == 1 and result['radiation_model'] in (0, 1),
            'Only single-species radiation checkpoints are supported')
    moving_fld = result['radiation_model'] == 1 and result['hydro'] == 1
    require(result['radiation_options_version'] == 3 and result['hydro'] in (0, 1) and
            result['radiation_integrator'] == (3 if result['radiation_model'] else 0) and
            result['radiation_algorithm_version'] == (2 if moving_fld else 1) and
            result['radiation_flux_semantics'] == (2 if moving_fld else result['radiation_model']) and
            result['radiation_limiter'] == result['radiation_model'], 'Unsupported radiation model metadata')
    for name in COMPOSITION_METADATA:
        result[name] = _numeric_dataset(handle, '/' + name, h5py).tolist()
    if result['radiation_model'] == 1:
        for name in FLD_METADATA + (FLD_HYDRO_METADATA if result['hydro'] else ()):
            values = _numeric_dataset(handle, '/' + name, h5py)
            require(values.size == 1, f'Expected scalar checkpoint metadata: {name}')
            result[name] = values.reshape(-1)[0].item()
    require(all(result[name] > 0 for name in ('code_to_g', 'code_to_cm', 'code_to_s', 'xscale')) and
            0 < result['radiation_reduced_c_factor'] <= 1, 'Invalid physical units or transport speed')
    return result


def _native_fields(group, h5py):
    result = set()
    for name in _group_names(group, h5py):
        obj = _hard_object(group.file, group.name.rstrip('/') + '/' + name, h5py)
        if isinstance(obj, h5py.Datatype) and int(obj.attrs.get('silo_type', -1)) == 501:
            result.add(name)
    require(result == set(FIELDS), f'Native field set differs at leaf {group.name}: {sorted(result)}')
    return result


def _object_time(attrs, expected_time, label):
    require('dtime' in attrs, f'Missing physical object time: {label}')
    actual = np.asarray(attrs['dtime'])
    require(actual.size == 1 and actual.dtype.kind == 'f' and actual.dtype.itemsize == 8 and
            np.isfinite(actual).all(), f'Invalid physical object time: {label}')
    value = float(actual.reshape(-1)[0])
    require(abs(value - expected_time) <= DEFAULT_TOLERANCE * max(abs(value), abs(expected_time)),
            f'Shard object time differs from its header: {label}')


def _leaf(handle, name, h5py, physical_time):
    group = _hard_object(handle, '/' + name, h5py)
    require(isinstance(group, h5py.Group), f'Missing leaf group: {name}')
    _native_fields(group, h5py)
    mesh = _silo(_hard_object(handle, f'/{name}/quadmesh', h5py), h5py)
    _object_time(mesh, physical_time, name + '/quadmesh')
    require(int(mesh.get('ndims', 0)) == 3 and int(mesh.get('nspace', 0)) == 3 and
            int(mesh.get('coordtype', -1)) == 130, 'Expected a three-dimensional collinear quadmesh')
    dims = tuple(int(value) for value in np.asarray(mesh['dims']).reshape(-1))
    require(len(dims) == 3 and min(dims) > 1, 'Invalid quadmesh dimensions')
    coordinates = []
    for axis in range(3):
        values = _numeric_dataset(handle, _text(mesh[f'coord{axis}']), h5py, float64=True)
        require(values.shape == (dims[axis],) and np.all(np.diff(values) > 0),
                f'Invalid collinear coordinate array: leaf {name} axis {axis}')
        coordinates.append(values)
    fields = {}
    shapes = {}
    for field in FIELDS:
        attrs = _silo(_hard_object(handle, f'/{name}/{field}', h5py), h5py)
        _object_time(attrs, physical_time, name + '/' + field)
        shape = tuple(int(value) for value in np.asarray(attrs['dims']).reshape(-1))
        require(_text(attrs.get('meshid', b'')) == 'quadmesh' and int(attrs.get('ndims', 0)) == 3 and
                int(attrs.get('nvals', 0)) == 1 and int(attrs.get('centering', -1)) == 111 and
                shape == tuple(value - 1 for value in dims) and int(attrs.get('nels', -1)) == math.prod(shape),
                f'Invalid native field geometry: {name}/{field}')
        values = _numeric_dataset(handle, _text(attrs['value0']), h5py, float64=True)
        require(values.shape in ((math.prod(shape),), tuple(reversed(shape))),
                f'Invalid native field payload shape: {name}/{field}')
        fields[field] = values.reshape(tuple(reversed(shape)))
        shapes[field] = shape
    return {'coordinates': coordinates, 'fields': fields, 'shapes': shapes}


def load_snapshot(header):
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError('Semantic Silo comparison requires h5py') from error
    header = Path(header).absolute()
    require(header.name.endswith('.silo') and header.is_file() and not header.is_symlink(),
            'Expected a regular Silo header')
    with h5py.File(header, 'r') as master:
        _check_header_links(master, header, h5py)
        metadata = _metadata(master, h5py)
        time = _numeric_dataset(master, '/cgs_time', h5py, float64=True)
        require(time.size == 1 and time.reshape(-1)[0] >= 0, 'Invalid physical checkpoint time')
        expected = {}
        header_fields = {name for name in _group_names(master, h5py) if isinstance(master[name], h5py.Datatype) and
                         int(master[name].attrs.get('silo_type', -1)) == 521}
        require(header_fields == set(FIELDS), 'Native multivar header field set differs')
        references = _reference_entries(master, _hard_object(master, '/quadmesh', h5py), 'meshnames', h5py)
        for reference in references:
            shard, leaf, _ = _parse_reference(header, reference, 'quadmesh')
            require(leaf not in expected, f'Duplicate octree leaf: {leaf}')
            expected[leaf] = shard
        for field in FIELDS:
            refs = _reference_entries(master, _hard_object(master, '/' + field, h5py), 'varnames', h5py)
            field_leaves = {}
            for reference in refs:
                shard, leaf, _ = _parse_reference(header, reference, field)
                require(leaf not in field_leaves, f'Duplicate field leaf: {field}/{leaf}')
                field_leaves[leaf] = shard
            require(field_leaves == expected, f'Field header topology differs: {field}')
        leaves = {}
        for shard in sorted(set(expected.values())):
            with h5py.File(shard, 'r') as handle:
                actual_leaves = {name for name in handle if re.fullmatch(r'[1][0-7]*', name)}
                wanted = {name for name, path in expected.items() if path == shard}
                require(actual_leaves == wanted, 'Shard leaf topology differs from its header')
                for name in sorted(wanted):
                    require(not any(name[:length] in expected for length in range(1, len(name))),
                            'Octree contains both a leaf and its descendant')
                    leaves[name] = _leaf(handle, name, h5py, float(time.reshape(-1)[0]))
        require(leaves, 'Empty checkpoint topology')
        if 'leaf_count' in master:
            count = _numeric_dataset(master, '/leaf_count', h5py)
            require(count.size == 1 and count.reshape(-1)[0] == len(leaves), 'Checkpoint leaf_count disagrees')
        return {'header': str(header), 'time': float(time.reshape(-1)[0]), 'metadata': metadata, 'leaves': leaves}


def compare_loaded(original, replay, tolerance=DEFAULT_TOLERANCE):
    """Compare already validated native snapshots; shared by the HDF5 adapter."""
    require(isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and tolerance >= 0,
            'Comparison tolerance must be nonnegative and finite')
    report = {'schema': 1, 'passed': False, 'tolerance': float(tolerance),
              'original': original['header'], 'replay': replay['header'], 'errors': [], 'fields': {},
              'coordinates_equal': True, 'scaling': 'maximum absolute field value across all leaves of both snapshots'}
    errors = report['errors']
    if original['metadata'] != replay['metadata']:
        errors.append('Physical model metadata differs')
    report['metadata'] = original['metadata']
    scale = max(abs(original['time']), abs(replay['time']))
    time_difference = abs(original['time'] - replay['time'])
    report['time'] = {'original': original['time'], 'replay': replay['time'], 'difference': time_difference}
    if not math.isfinite(time_difference) or time_difference > tolerance * scale:
        errors.append('Physical checkpoint times differ')
    if set(original['leaves']) != set(replay['leaves']):
        errors.append('Octree leaf topology differs')
        return report
    report['leaf_count'] = len(original['leaves'])
    totals = {field: [np.longdouble(0), np.longdouble(0)] for field in FIELDS}
    invalid_fields = set()
    for name, left in original['leaves'].items():
        right = replay['leaves'][name]
        if len(left['coordinates']) != 3 or len(right['coordinates']) != 3 or any(
                not np.array_equal(a, b) for a, b in zip(left['coordinates'], right['coordinates'])):
            report['coordinates_equal'] = False
            errors.append(f'Quadmesh coordinates differ: {name}')
        if set(left['fields']) != set(FIELDS) or set(right['fields']) != set(FIELDS):
            errors.append(f'Native field set differs: {name}')
            invalid_fields.update(FIELDS)
            continue
        for field in FIELDS:
            a, b = left['fields'][field], right['fields'][field]
            if a.shape != b.shape or left['shapes'][field] != right['shapes'][field]:
                errors.append(f'Native field shape differs: {name}/{field}')
                invalid_fields.add(field)
                continue
            if any(value.size == 0 or value.dtype.kind != 'f' or value.dtype.itemsize != 8 or not np.isfinite(value).all()
                   for value in (a, b)):
                errors.append(f'Nonfinite or non-float64 field: {name}/{field}')
                invalid_fields.add(field)
                continue
            a, b = a.astype(np.longdouble), b.astype(np.longdouble)
            totals[field][0] = max(totals[field][0], np.max(np.abs(a - b)))
            totals[field][1] = max(totals[field][1], np.max(np.abs(a)), np.max(np.abs(b)))
    for field, (difference, scale) in totals.items():
        relative = difference / scale if scale else np.longdouble(0)
        finite = difference <= np.finfo(np.float64).max and np.isfinite(relative)
        passed = bool(finite and relative <= tolerance and field not in invalid_fields)
        report['fields'][field] = {'max_abs_difference': float(difference) if finite else None,
            'global_field_scale': float(scale), 'max_scaled_difference': float(relative), 'passed': passed}
        if not passed:
            errors.append(f'Field tolerance exceeded: {field}')
    report['passed'] = not errors
    return report


def compare_snapshots(original_header, replay_header, tolerance=DEFAULT_TOLERANCE):
    """Return a JSON-safe pass/fail report; invalid checkpoints fail closed."""
    try:
        return compare_loaded(load_snapshot(original_header), load_snapshot(replay_header), tolerance)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        return {'schema': 1, 'passed': False, 'original': str(original_header), 'replay': str(replay_header),
                'errors': [str(error)], 'fields': {}, 'coordinates_equal': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_header', type=Path)
    parser.add_argument('replay_header', type=Path)
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    report = compare_snapshots(args.original_header, args.replay_header, args.tolerance)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
