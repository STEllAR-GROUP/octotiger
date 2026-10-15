#!/usr/bin/env python3
"""Small semantic fixtures; no Silo library, solver, network, or production data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

SPEC = importlib.util.spec_from_file_location('silo_compare', Path(__file__).with_name('silo_compare.py'))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def native_snapshot():
    leaves = {}
    for index, name in enumerate(('100', '101')):
        leaves[name] = {'coordinates': [np.arange(3, dtype=np.float64) + index for _ in range(3)],
            'fields': {field: np.arange(8, dtype=np.float64).reshape(2, 2, 2) + i + index
                       for i, field in enumerate(compare.FIELDS)},
            'shapes': {field: (2, 2, 2) for field in compare.FIELDS}}
    return {'header': 'original.silo', 'time': 21890., 'metadata': {'radiation_model': 0}, 'leaves': leaves}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.original = native_snapshot()
        self.replay = copy.deepcopy(self.original)
        self.replay['header'] = 'replay.silo'

    def test_exact_semantic_match(self):
        report = compare.compare_loaded(self.original, self.replay)
        self.assertTrue(report['passed'])
        self.assertEqual(set(report['fields']), set(compare.FIELDS))
        self.assertTrue(all(value['max_abs_difference'] == 0 for value in report['fields'].values()))
        json.dumps(report, allow_nan=False)

    def test_global_field_scale_across_leaves(self):
        self.original['leaves']['100']['fields']['er'].fill(1.e-20)
        self.original['leaves']['101']['fields']['er'].fill(1.e4)
        self.replay = copy.deepcopy(self.original)
        self.replay['leaves']['100']['fields']['er'][0, 0, 0] += 1.e-10
        report = compare.compare_loaded(self.original, self.replay)
        self.assertTrue(report['passed'])
        self.assertAlmostEqual(report['fields']['er']['global_field_scale'], 1.e4)
        self.assertAlmostEqual(report['fields']['er']['max_scaled_difference'], 1.e-14)

    def test_small_field_is_not_scaled_by_unity_or_another_field(self):
        for snapshot in (self.original, self.replay):
            for leaf in snapshot['leaves'].values():
                leaf['fields']['fx'].fill(1.e-20)
                leaf['fields']['egas'].fill(1.e20)
        self.replay['leaves']['100']['fields']['fx'][0, 0, 0] *= 1.00001
        report = compare.compare_loaded(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertFalse(report['fields']['fx']['passed'])

    def test_zero_fields_and_nonzero_replay(self):
        for snapshot in (self.original, self.replay):
            for leaf in snapshot['leaves'].values():
                leaf['fields']['pot'].fill(0)
        self.assertTrue(compare.compare_loaded(self.original, self.replay)['passed'])
        self.replay['leaves']['100']['fields']['pot'][0, 0, 0] = 1.e-100
        report = compare.compare_loaded(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertEqual(report['fields']['pot']['max_scaled_difference'], 1.)

    def test_field_difference_above_and_below_explicit_tolerance(self):
        self.replay['leaves']['101']['fields']['rho_1'][0, 0, 0] += 1.e-7
        self.assertFalse(compare.compare_loaded(self.original, self.replay, 1.e-10)['passed'])
        self.assertTrue(compare.compare_loaded(self.original, self.replay, 1.e-6)['passed'])

    def test_metadata_time_topology_and_coordinates_are_not_field_tolerance(self):
        for mutation, message in (
                (lambda d: d['metadata'].update(radiation_model=1), 'metadata'),
                (lambda d: d.update(time=22000.), 'times'),
                (lambda d: d['leaves'].pop('101'), 'topology'),
                (lambda d: d['leaves']['100']['coordinates'][0].__setitem__(0, 1.e-15), 'coordinates')):
            with self.subTest(message=message):
                other = copy.deepcopy(self.replay)
                mutation(other)
                report = compare.compare_loaded(self.original, other)
                self.assertFalse(report['passed'])
                self.assertIn(message, ' '.join(report['errors']))

    def test_field_identity_shape_precision_and_nonfinite_rejected(self):
        mutations = (
            lambda d: d['fields'].pop('tau'),
            lambda d: d['fields'].update(tau=d['fields']['tau'].reshape(-1)),
            lambda d: d['fields'].update(tau=d['fields']['tau'].astype(np.float32)),
            lambda d: d['fields']['tau'].__setitem__((0, 0, 0), np.nan),
            lambda d: d['fields']['tau'].__setitem__((0, 0, 0), np.inf))
        for mutation in mutations:
            other = copy.deepcopy(self.replay)
            mutation(other['leaves']['100'])
            report = compare.compare_loaded(self.original, other)
            self.assertFalse(report['passed'])
            self.assertFalse(report['fields']['tau']['passed'])

    def test_extreme_finite_difference_fails_without_invalid_json(self):
        self.original['leaves']['100']['fields']['er'].fill(1.e308)
        self.replay['leaves']['100']['fields']['er'].fill(-1.e308)
        report = compare.compare_loaded(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertIsNone(report['fields']['er']['max_abs_difference'])
        json.dumps(report, allow_nan=False)

    def test_invalid_tolerance_rejected(self):
        for tolerance in (-1., float('inf'), float('nan')):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                compare.compare_loaded(self.original, self.replay, tolerance)


class ReferenceGuardTests(unittest.TestCase):
    def test_physical_object_time_requires_float64_and_agrees_with_header(self):
        compare._object_time({'dtime': np.float64(21890.)}, 21890., 'leaf/er')
        for value in (np.float32(21890.), np.float64(22000.), np.float64(np.nan)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare._object_time({'dtime': value}, 21890., 'leaf/er')

    def test_reference_scope_and_symlink_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            header = Path(temporary) / 'final.silo'
            data = header.with_name('final.silo.data')
            data.mkdir()
            (data / '0.silo').write_bytes(b'fixture')
            path, leaf, field = compare._parse_reference(header, 'final.silo.data/0.silo:/100/er', 'er')
            self.assertEqual((path, leaf, field), (data / '0.silo', '100', 'er'))
            for ref in ('old.silo.data/0.silo:/100/er', '/final.silo.data/0.silo:/100/er',
                        '../final.silo.data/0.silo:/100/er', 'final.silo.data/sub/0.silo:/100/er',
                        'final.silo.data/0.silo:/108/er', 'final.silo.data/0.silo:/100/fx'):
                with self.subTest(reference=ref), self.assertRaises(ValueError):
                    compare._parse_reference(header, ref, 'er')
            (data / '1.silo').symlink_to(data / '0.silo')
            with self.assertRaisesRegex(ValueError, 'symlinked'):
                compare._parse_reference(header, 'final.silo.data/1.silo:/100/er')

    def test_actual_null_separated_uint8_reference_encoding(self):
        data = np.frombuffer(b'final.silo.data/0.silo:/100/er\0final.silo.data/0.silo:/101/er\0', dtype=np.uint8)
        self.assertEqual(compare._strings(data)[0].count('\0'), 2)
        with self.assertRaises(UnicodeDecodeError):
            compare._strings(np.array([255], dtype=np.uint8))


def silo_object(parent, name, attrs, silo_type):
    """Model actual Silo named datatype objects with compound 'silo' attrs."""
    tid = h5py.h5t.py_create(np.dtype('int32'))
    tid.commit(parent.id, name.encode())
    obj = parent[name]
    fields = []
    for key, value in attrs.items():
        if isinstance(value, str):
            fields.append((key, 'S256'))
        elif isinstance(value, (list, tuple)):
            fields.append((key, '<i8', (len(value),)))
        elif isinstance(value, float):
            fields.append((key, '<f8'))
        else:
            fields.append((key, '<i8'))
    compound = np.zeros((), dtype=np.dtype(fields))
    for key, value in attrs.items():
        compound[key] = value.encode() if isinstance(value, str) else value
    obj.attrs['silo'] = compound
    obj.attrs['silo_type'] = silo_type
    return obj


def hdf5_snapshot(header, renumber=False):
    data = header.with_name(header.name + '.data')
    data.mkdir()
    refs = {name: [] for name in ('quadmesh', *compare.FIELDS)}
    for index, name in enumerate(('100', '101')):
        shard_name = str(9 - index if renumber else index) + '.silo'
        with h5py.File(data / shard_name, 'w') as shard:
            shard['..'] = shard['/']
            group = shard.create_group(name)
            shard.create_group('.silo')
            counter = 1000 if renumber else 1
            def payload(values):
                nonlocal counter
                path = '/.silo/#' + str(counter).zfill(6)
                counter += 1
                shard.create_dataset(path, data=values)
                return path
            group['..'] = shard['/']
            mesh = {'ndims': 3, 'nspace': 3, 'coordtype': 130, 'dims': [9, 9, 9], 'dtime': 21890.}
            for axis in range(3):
                mesh[f'coord{axis}'] = payload(np.arange(9, dtype=np.float64) + index)
            silo_object(group, 'quadmesh', mesh, 130)
            for field_index, field in enumerate(compare.FIELDS):
                pointer = payload(np.arange(512, dtype=np.float64) + 100 * index + field_index)
                silo_object(group, field, {'meshid': 'quadmesh', 'value0': pointer,
                    'ndims': 3, 'nvals': 1, 'nels': 512, 'centering': 111, 'dims': [8, 8, 8], 'dtime': 21890.}, 501)
            for field in refs:
                refs[field].append(f'{header.name}.data/{shard_name}:/{name}/{field}')
    with h5py.File(header, 'w') as master:
        master['..'] = master['/']
        master.create_group('.silo')
        for index, (field, references) in enumerate(refs.items()):
            if renumber:
                references.reverse()
            path = '/.silo/#' + str(index + (500 if renumber else 1)).zfill(6)
            master.create_dataset(path, data=np.frombuffer(('\0'.join(references) + '\0').encode(), dtype=np.uint8))
            member = 'meshnames' if field == 'quadmesh' else 'varnames'
            silo_object(master, field, {member: path, 'nblocks' if field == 'quadmesh' else 'nvars': 2},
                        520 if field == 'quadmesh' else 521)
        metadata = {name: 1 for name in compare.REQUIRED_METADATA}
        metadata.update(version=112, radiation_model=0, radiation_integrator=0, radiation_flux_semantics=0,
                        radiation_limiter=0, radiation_options_version=3, omega=0.)
        for name, value in metadata.items():
            master.create_dataset(name, data=np.array([value]))
        for name in compare.COMPOSITION_METADATA:
            master.create_dataset(name, data=np.array([1.], dtype=np.float64))
        master.create_dataset('cgs_time', data=np.array([21890.], dtype=np.float64))
        master.create_dataset('leaf_count', data=np.array([2], dtype=np.int64))
        # Real hostname DB_CHAR buffers can contain unused non-UTF8 tail bytes.
        master.create_dataset('hostname', data=np.frombuffer(b'node\0\xff', dtype=np.uint8))


@unittest.skipIf(h5py is None, 'h5py is not installed; run these fixtures in the allocated h5py environment')
class HDF5Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original, self.replay = self.root / 'original.silo', self.root / 'replay.silo'
        hdf5_snapshot(self.original)
        hdf5_snapshot(self.replay, renumber=True)

    def edit_field(self, field='er'):
        return h5py.File(self.replay.with_name('replay.silo.data') / '9.silo', 'r+')

    def test_raw_dataset_and_shard_reordering_do_not_change_identity(self):
        report = compare.compare_snapshots(self.original, self.replay)
        self.assertTrue(report['passed'], report['errors'])
        self.assertEqual(report['leaf_count'], 2)

    def test_native_field_change_is_detected(self):
        with self.edit_field() as handle:
            pointer = compare._text(handle['100/er'].attrs['silo']['value0'])
            handle[pointer][0] += 1.
        report = compare.compare_snapshots(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertEqual(report['fields']['er']['max_abs_difference'], 1.)

    def test_foreign_same_looking_master_reference_rejected(self):
        with h5py.File(self.replay, 'r+') as handle:
            pointer = compare._text(handle['er'].attrs['silo']['varnames'])
            del handle[pointer]
            handle.create_dataset(pointer, data=np.frombuffer(b'original.silo.data/0.silo:/100/er\0', dtype=np.uint8))
        report = compare.compare_snapshots(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertIn('foreign', ' '.join(report['errors']))

    def test_external_hdf5_link_rejected(self):
        with h5py.File(self.replay, 'r+') as handle:
            handle['unrelated_link'] = h5py.ExternalLink(str(self.original), '/er')
        report = compare.compare_snapshots(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertIn('Nonlocal', ' '.join(report['errors']))

    def test_soft_coordinate_reference_rejected(self):
        with self.edit_field() as handle:
            pointer = compare._text(handle['100/quadmesh'].attrs['silo']['coord0'])
            handle.move(pointer, pointer + '_actual')
            handle[pointer] = h5py.SoftLink(pointer + '_actual')
        report = compare.compare_snapshots(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertIn('Nonlocal', ' '.join(report['errors']))

    def test_coordinate_mismatch_rejected(self):
        with self.edit_field() as handle:
            pointer = compare._text(handle['100/quadmesh'].attrs['silo']['coord0'])
            handle[pointer][0] -= 1.e-12
        report = compare.compare_snapshots(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertFalse(report['coordinates_equal'])

    def test_nonfinite_float32_and_wrong_shape_payload_rejected(self):
        for values in (np.full(512, np.nan), np.zeros(512, dtype=np.float32), np.zeros((16, 32))):
            with self.subTest(dtype=str(values.dtype), shape=values.shape):
                with self.edit_field() as handle:
                    pointer = compare._text(handle['100/er'].attrs['silo']['value0'])
                    del handle[pointer]
                    handle.create_dataset(pointer, data=values)
                self.assertFalse(compare.compare_snapshots(self.original, self.replay)['passed'])

    def test_missing_or_additional_native_field_rejected(self):
        with self.edit_field() as handle:
            handle['100/derived_rho'] = handle['100/rho_1']
        report = compare.compare_snapshots(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertIn('field set', ' '.join(report['errors']))
        with self.edit_field() as handle:
            del handle['100/derived_rho']
            del handle['100/tau']
        self.assertFalse(compare.compare_snapshots(self.original, self.replay)['passed'])

    def test_only_exact_hard_parent_backlinks_are_allowed(self):
        for soft in (True, False):
            with self.subTest(soft=soft):
                with self.edit_field() as handle:
                    del handle['100/..']
                    handle['100/..'] = h5py.SoftLink('/') if soft else handle['100']
                report = compare.compare_snapshots(self.original, self.replay)
                self.assertFalse(report['passed'])
                self.assertIn('parent backlink', ' '.join(report['errors']))

    def test_stale_shard_time_rejected_even_if_header_matches(self):
        with self.edit_field() as handle:
            obj = handle['100/er']
            value = obj.attrs['silo']
            value['dtime'] = 22000.
            obj.attrs.modify('silo', value)
        report = compare.compare_snapshots(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertIn('object time differs', ' '.join(report['errors']))

    def test_wrong_time_model_and_duplicate_leaf_rejected(self):
        with h5py.File(self.replay, 'r+') as handle:
            handle['cgs_time'][0] = 22000.
        self.assertFalse(compare.compare_snapshots(self.original, self.replay)['passed'])
        with h5py.File(self.replay, 'r+') as handle:
            handle['cgs_time'][0] = 21890.
            handle['radiation_model'][0] = 1
        self.assertFalse(compare.compare_snapshots(self.original, self.replay)['passed'])
        with h5py.File(self.replay, 'r+') as handle:
            handle['radiation_model'][0] = 0
            pointer = compare._text(handle['quadmesh'].attrs['silo']['meshnames'])
            del handle[pointer]
            text = 'replay.silo.data/9.silo:/100/quadmesh\0' * 2
            handle.create_dataset(pointer, data=np.frombuffer(text.encode(), dtype=np.uint8))
        report = compare.compare_snapshots(self.original, self.replay)
        self.assertFalse(report['passed'])
        self.assertIn('Duplicate octree leaf', ' '.join(report['errors']))


if __name__ == '__main__':
    unittest.main()
