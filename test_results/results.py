"""Strict readers shared by the run and plotting commands; units follow run metadata."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

FIELDS = ('er', 'fx', 'fy', 'fz')
NORMS = ('L1', 'L2', 'Linf')
CASES = ('streaming_wave', 'streaming_front', 'gaussian_pulse', 'equilibrium_sphere')
ALIASES = dict(wave=CASES[0], streaming=CASES[1], front=CASES[1],
               diffusion=CASES[2], gaussian=CASES[2], sphere=CASES[3])
CONSERVATION_FILE = 'radiation-conservation.csv'
CONSERVATION_COLUMNS = ('t', 'volume', *FIELDS,
                        *(f+'_boundary' for f in FIELDS),
                        *(f+'_source' for f in FIELDS))

def read_config(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line: continue
        key, sep, value = line.partition('=')
        if not sep: raise ValueError(f'Invalid INI row in {path}: {line}')
        key = key.strip()
        if key in values: raise ValueError(f'Duplicate INI option: {key}')
        values[key] = value.strip()
    return values

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def parse_log(text, case, time):
    matches = re.findall(r'^RADIATION_TEST_FINISHED\s+(\w+)\s+t=(\S+)\s*$', text, re.M)
    if len(matches) != 1 or matches[0][0].lower() != 'radiation_' + case:
        raise ValueError('Missing, duplicate, or wrong final comparison marker')
    actual = float(matches[0][1])
    if not math.isfinite(actual) or not math.isclose(actual, time, rel_tol=1e-11, abs_tol=0):
        raise ValueError(f'Wrong final time: {actual}, expected {time}')
    rows = {}
    for field, *values in re.findall(r'^\s*(er|fx|fy|fz)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', text, re.M):
        if field in rows: raise ValueError(f'Duplicate norm row for {field}')
        numbers = list(map(float, values))
        if any(not math.isfinite(v) or v < 0 for v in numbers):
            raise ValueError(f'Invalid norm for {field}: {values}')
        rows[field] = dict(zip(NORMS, numbers))
    if set(rows) != set(FIELDS): raise ValueError('Missing radiation error norms')
    return rows

def read_norms(folder, meta):
    rows = parse_log((folder/'run.log').read_text(errors='replace'), meta['case'], meta['time'])
    # These files contain dx, max_level, gas fields, then exactly four rad fields.
    for norm in NORMS:
        lines = [l.split() for l in (folder/(norm+'.dat')).read_text().splitlines() if l.strip()]
        if len(lines) != 1 or len(lines[0]) < 6:
            raise ValueError(f'{folder}/{norm}.dat must contain exactly one comparison')
        values = list(map(float, lines[0]))
        if any(not math.isfinite(v) for v in values): raise ValueError('Nonfinite norm-file value')
        if not math.isclose(values[0], meta['dx'], rel_tol=2e-6) or values[1] != meta['level']:
            raise ValueError('Norm-file resolution disagrees with run metadata')
        for field, value in zip(FIELDS, values[-4:]):
            if not math.isclose(value, rows[field][norm], rel_tol=2e-6, abs_tol=0):
                raise ValueError(f'Log and {norm}.dat disagree for {field}')
    return rows

def read_slice(folder, meta):
    import numpy as np
    files = sorted((folder/'radiation-slices').glob('slice-*.csv'))
    if not files: raise ValueError(f'No slice output in {folder}; install the hook and rebuild octotiger')
    columns = ('t','dx','x','y','z',*FIELDS,*(f+'_ref' for f in FIELDS))
    arrays = []
    for path in files:
        a = np.genfromtxt(path, delimiter=',', names=True, ndmin=1)
        if a.dtype.names != columns or not len(a): raise ValueError(f'Invalid slice file: {path}')
        arrays.append(a)
    a = np.concatenate(arrays); n = meta['cells']; dx = meta['dx']; length = meta['length']
    if len(a) != n*n: raise ValueError(f'Expected {n*n} slice cells; found {len(a)}')
    for name in columns:
        if not np.isfinite(a[name]).all(): raise ValueError(f'Nonfinite slice column: {name}')
    for name, expected in [('t', meta['time']), ('dx', dx), ('z', dx/2)]:
        if not np.allclose(a[name], expected, rtol=1e-11, atol=0):
            raise ValueError(f'Slice {name} differs from requested value {expected}')
    ii = np.rint((a['x']+length/2)/dx-.5).astype(int)
    jj = np.rint((a['y']+length/2)/dx-.5).astype(int)
    if np.any((ii<0)|(ii>=n)|(jj<0)|(jj>=n)) or len(np.unique(ii*n+jj)) != n*n:
        raise ValueError('Duplicate, missing, or out-of-domain slice cells')
    for name, index in [('x', ii), ('y', jj)]:
        if not np.allclose(a[name], -length/2+(index+.5)*dx, rtol=0, atol=1e-11*dx):
            raise ValueError('Slice coordinates are not aligned to the specified uniform grid')
    return a[np.argsort(ii*n+jj)].reshape(n, n)

def read_conservation(folder, meta):
    """Read full-domain budgets, or return None for runs predating diagnostics.

    Boundary values are cumulative outward transport. Source values are
    cumulative signed changes applied by the source update, including any
    externally imposed radiation source. Neither is a conservation error.
    A present but incomplete, stale, or malformed file is always an error.
    """
    import numpy as np
    path = Path(folder)/CONSERVATION_FILE
    if not path.exists(): return None
    def invalid(reason):
        raise ValueError(f'{path}: {reason}')
    with path.open(newline='') as stream:
        rows = csv.reader(stream)
        if tuple(next(rows, ())) != CONSERVATION_COLUMNS:
            invalid('invalid conservation columns')
        values = []
        for line, row in enumerate(rows, 2):
            if len(row) != len(CONSERVATION_COLUMNS):
                invalid(f'wrong column count on line {line}')
            try: numbers = tuple(map(float, row))
            except ValueError: invalid(f'invalid number on line {line}')
            if not all(math.isfinite(value) for value in numbers):
                invalid(f'nonfinite value on line {line}')
            values.append(numbers)
    if not values: invalid('no conservation samples')
    data = np.array(values, dtype=[(name, float) for name in CONSERVATION_COLUMNS])
    length = float(meta['length']); c = float(meta['c']); final_time = float(meta['time'])
    if not all(math.isfinite(v) for v in (length, c, final_time)) or length <= 0 or c <= 0:
        invalid('invalid length, light speed, or final time in metadata')
    expected_volume = length**3
    if not math.isfinite(expected_volume): invalid('nonfinite domain volume')
    if not np.allclose(data['volume'], expected_volume, rtol=1e-11, atol=0):
        invalid(f'volume differs from requested domain volume {expected_volume}')
    if data['t'][0] != 0: invalid('initial sample must be at t=0')
    if not np.all(np.diff(data['t']) > 0): invalid('sample times must strictly increase')
    if not math.isclose(data['t'][-1], final_time, rel_tol=1e-11, abs_tol=0):
        invalid(f'final sample time {data["t"][-1]} differs from requested time {final_time}')
    for field in FIELDS:
        for suffix in ('_boundary', '_source'):
            if data[field+suffix][0] != 0:
                invalid(f'initial {field+suffix} must be zero')

    energy_scale = np.maximum.reduce([np.full(len(data), abs(data['er'][0])),
        np.abs(data['er']), np.abs(data['er_boundary']), np.abs(data['er_source'])])
    residuals = {}; scales = {}; normalized = {}; summary = []
    cgs = meta.get('units', {}).get('system') == 'CGS'
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for field in FIELDS:
            q = data[field]; boundary = data[field+'_boundary']; source = data[field+'_source']
            residual = q-q[0]+boundary-source
            scale = energy_scale if field == 'er' else np.maximum.reduce([
                c*energy_scale, np.full(len(data), abs(q[0])), np.abs(q),
                np.abs(boundary), np.abs(source)])
            if not np.isfinite(residual).all() or not np.isfinite(scale).all():
                invalid(f'nonfinite derived budget for {field}')
            if np.any((scale == 0) & (residual != 0)):
                invalid(f'nonzero {field} residual with zero normalization scale')
            relative = np.divide(residual, scale, out=np.zeros_like(residual), where=scale != 0)
            residuals[field] = residual; scales[field] = scale; normalized[field] = relative
            summary.append(dict(field=field, initial=float(q[0]), final=float(q[-1]),
                raw_change=float(q[-1]-q[0]), boundary=float(boundary[-1]), source=float(source[-1]),
                residual=float(residual[-1]), normalization_scale=float(scale[-1]),
                normalized_error=abs(float(relative[-1])),
                max_abs_residual=float(np.max(np.abs(residual))),
                max_normalized_error=float(np.max(np.abs(relative))),
                integral_units=('erg' if field == 'er' else 'erg cm/s') if cgs else 'code units'))
    return dict(history=data, residuals=residuals, scales=scales,
                normalized_residuals=normalized, summary=summary)

def order(coarse_error, fine_error, coarse_dx, fine_dx):
    # A zero error is not evidence of infinite order; leave it unreported.
    if coarse_error <= 0 or fine_error <= 0: return None
    if coarse_dx <= fine_dx: raise ValueError('Resolutions must be distinct and ordered coarse to fine')
    return math.log(coarse_error/fine_error) / math.log(coarse_dx/fine_dx)
