"""Synthetic file fixtures only; never run a solver, Silo rewrite or network command."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from qb4 import supplemental_state as recovery


class SupplementalStateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='supplemental-state-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.original = self.root / 'original'
        self.replay = self.root / 'replay'
        self.export = self.root / 'export'
        self.original.mkdir()
        self.replay.mkdir()
        self.executable = self.root / 'frozen-solver'
        self.executable.write_bytes(b'fixture only, not an executable')
        self.times = {}
        self.config = {'problem': 'RADIATION_BLAST_WAVE', 'radiation_model': 'm1',
            'radiation_integrator': 'euler', 'radiation_backend': 'cpu', 'radiation': 'on',
            'hydro': 'on', 'gravity': 'off', 'rad_implicit': 'on', 'periodic': 'off',
            'reflect_bc': 'off', 'inflow_bc': 'off', 'eos': 'IDEAL', 'unigrid': 'on',
            'min_level': '2', 'max_level': '2', 'stop_time': '22000',
            'radiation_reduced_c_factor': '1', 'omega': '0', 'code_to_g': '1',
            'code_to_cm': '1', 'code_to_s': '1', 'cfl': '.4',
            'odt': '183.33333333333334', 'hard_dt': '36.666666666666671',
            'datadir': str(self.original) + '/'}
        self.metadata = {'status': 'complete', 'case': 'radiating_blast', 'level': 2,
            'radiation_model': 'm1', 'radiation_integrator': 'euler', 'radiation_algorithm_version': 1,
            'radiation_flux_semantics': 'evolved_m1_flux', 'time': 22000.,
            'physical_c': recovery.PHYSICAL_C, 'transport_c': recovery.PHYSICAL_C,
            'radiation_reduced_c_factor': 1., 'executable_sha256': recovery.file_hash(self.executable),
            'config': self.config, 'movie_capture': {'requested_snapshots': 121,
                'steps_per_output_check': 5, 'odt': 22000 / 120, 'hard_dt': 22000 / 120 / 5}}
        self.write_metadata()
        for name in ('run.log', 'radiation-conservation.csv', 'momentum-diagnostics.csv'):
            (self.original / name).write_text('original retained ' + name + '\n')
        (self.original / '.hidden-lock').touch()
        # One earlier output check was delayed; the last checkpoint is the
        # actual observed step595 time, not index118 multiplied by the cadence.
        for index in range(119):
            time = index * (22000 / 119) if index < 118 else 21816.666666666737
            self.snapshot(self.original, f'X.{index}.silo', time)
        self.snapshot(self.original, 'final.silo', 22000.)
        self.snapshot(self.replay, 'final.silo', 21890.000000000044)
        self.baseline = recovery.inventory_original(self.original)
        self.job = {'job_id': '1102400', 'node': 'qbd173', 'nodes': 1, 'tasks': 1, 'threads': 64}

    def write_metadata(self):
        (self.original / 'run.ini').write_text(''.join(f'{key}={value}\n' for key, value in self.config.items()))
        (self.original / 'run.json').write_text(json.dumps(self.metadata))

    def snapshot(self, folder, name, time):
        folder.mkdir(exist_ok=True)
        (folder / name).write_text('synthetic master for ' + name)
        shards = folder / (name + '.data')
        shards.mkdir()
        (shards / '0.silo').write_text('synthetic state ' + str(time))
        self.times[folder / name] = time

    def read_time(self, path):
        return self.times[Path(path)]

    def validate(self):
        return recovery.validate_original(self.original, self.baseline, self.read_time)

    def provenance(self):
        return recovery.recovery_provenance(self.original, self.baseline, self.executable,
            self.replay, self.export, recovery.ORIGINAL_JOB, self.job, self.read_time)

    def test_original_and_native_export_command_are_read_only(self):
        context = self.validate()
        self.assertEqual(context['source_time'], 21816.666666666737)
        command = recovery.native_export_command(self.original, self.baseline, self.executable,
                                                  self.replay, self.export)
        self.assertEqual(command['cwd'], str(self.replay))
        self.assertIn('--config_file=' + str(self.original / 'run.ini'), command['argv'])
        self.assertIn('--restart_filename=' + str(self.replay / 'final.silo'), command['argv'])
        self.assertIn('--output=X.119', command['argv'])
        self.assertIn('--datadir=' + str(self.export) + '/', command['argv'])
        self.assertFalse(any('--output_filename' in value for value in command['argv']))
        self.assertFalse(self.export.exists())
        recovery.validate_unchanged(self.original, self.baseline)

    def test_provenance_ties_actual_state_hashes_jobs_and_authoritative_ledgers(self):
        self.snapshot(self.export, 'X.119.silo', self.times[self.replay / 'final.silo'])
        doc = self.provenance()
        self.assertEqual(doc['status'], 'files-and-times-verified')
        self.assertEqual(doc['original_job'], recovery.ORIGINAL_JOB)
        self.assertEqual(doc['recovery_job'], self.job)
        self.assertEqual(doc['original_files'], self.baseline)
        self.assertEqual(len(doc['completed_snapshot_times']), 121)
        self.assertEqual(doc['completed_snapshot_times'][-1], 22000)
        self.assertLess(doc['completed_snapshot_times'][-2], 22000)
        self.assertEqual(doc['native_export']['files']['X.119.silo']['sha256'],
                         hashlib.sha256((self.export / 'X.119.silo').read_bytes()).hexdigest())
        self.assertEqual(set(doc['authoritative_ledgers']),
                         {'radiation-conservation.csv', 'momentum-diagnostics.csv'})
        self.assertIn('no SSP', doc['scientific_caveat'])
        self.assertIn('before installation', doc['required_external_validation'])
        self.assertIn('actual replay command', doc['required_external_validation'])
        self.assertNotIn('native_steps', doc['replay'])
        recovery.validate_unchanged(self.original, self.baseline)

    def test_original_rewrite_removal_addition_and_mode_changes_are_rejected(self):
        path = self.original / 'momentum-diagnostics.csv'
        original = path.read_bytes()
        path.write_bytes(b'changed original ledger')
        with self.assertRaisesRegex(ValueError, 'original files changed'):
            self.validate()
        path.write_bytes(original)
        self.baseline = recovery.inventory_original(self.original)
        added = self.original / 'unrecorded'
        added.touch()
        with self.assertRaisesRegex(ValueError, 'original files changed'):
            self.validate()
        added.unlink()
        path.chmod(0o400)
        with self.assertRaisesRegex(ValueError, 'original files changed'):
            self.validate()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'original files changed'):
            self.validate()

    def test_symlink_and_external_tree_are_rejected(self):
        (self.original / 'alias').symlink_to(self.executable)
        with self.assertRaisesRegex(ValueError, 'symlink or special'):
            recovery.inventory_original(self.original)
        (self.original / 'alias').unlink()
        alias = self.root / 'original-alias'
        alias.symlink_to(self.original, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'real directory'):
            recovery.inventory_original(alias)

    def test_missing_gap_extra_number_and_partial_supplement_are_rejected(self):
        original = self.baseline
        for name in ('X.50.silo', 'X.119.silo', 'X.119.silo.data/0.silo'):
            with self.subTest(name=name):
                path = self.original / name
                if name == 'X.50.silo':
                    saved = path.read_bytes(); path.unlink()
                else:
                    path.parent.mkdir(exist_ok=True); path.write_bytes(b'unexpected')
                self.baseline = recovery.inventory_original(self.original)
                with self.assertRaisesRegex(ValueError, 'contiguous|partial supplemental'):
                    self.validate()
                if name == 'X.50.silo':path.write_bytes(saved)
                else:path.unlink()
        self.baseline = original

    def test_model_speed_duration_and_capture_changes_are_rejected(self):
        original = copy.deepcopy(self.metadata)
        for key, value in (('radiation_model', 'fld'), ('physical_c', recovery.PHYSICAL_C / 10),
                           ('transport_c', recovery.PHYSICAL_C / 10), ('time', 23000),
                           ('status', 'running'), ('radiation_reduced_c_factor', .1)):
            with self.subTest(key=key):
                self.metadata = copy.deepcopy(original); self.metadata[key] = value
                self.write_metadata(); self.baseline = recovery.inventory_original(self.original)
                with self.assertRaises(ValueError):self.validate()
        self.metadata = original
        self.metadata['movie_capture']['requested_snapshots'] = 120
        self.write_metadata(); self.baseline = recovery.inventory_original(self.original)
        with self.assertRaisesRegex(ValueError, 'capture metadata'):
            self.validate()

    def test_ini_mismatch_and_changed_timestep_are_rejected(self):
        self.config['hard_dt'] = '110'
        self.write_metadata(); self.baseline = recovery.inventory_original(self.original)
        with self.assertRaisesRegex(ValueError, 'timestep/capture'):
            self.validate()
        self.config['hard_dt'] = '36.666666666666671'
        self.write_metadata()
        with (self.original / 'run.ini').open('a') as f:f.write('unrecorded=1\n')
        self.baseline = recovery.inventory_original(self.original)
        with self.assertRaisesRegex(ValueError, 'INI and run metadata'):
            self.validate()

    def test_supplement_requires_actual_distinct_interior_time(self):
        times = self.validate()['snapshot_times']
        for value in (times[-2], 22000, 22001, times[30], float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                recovery.validate_supplemental_time(times, value)
        self.assertEqual(recovery.validate_supplemental_time(times, 21890), 21890)

    def test_original_nonfinite_duplicate_and_wrong_endpoint_times_are_rejected(self):
        path = self.original / 'final.silo'
        for value in (float('nan'), float('inf'), 21999, 23000):
            self.times[path] = value
            with self.subTest(value=value), self.assertRaises(ValueError):self.validate()
        self.times[path] = 22000
        self.times[self.original / 'X.50.silo'] = self.times[self.original / 'X.49.silo']
        with self.assertRaisesRegex(ValueError, 'strict order'):self.validate()

    def test_empty_export_shards_and_changed_export_time_are_rejected(self):
        self.snapshot(self.export, 'X.119.silo', 21890.000000000044)
        shard = self.export / 'X.119.silo.data/0.silo'
        shard.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'snapshot shards'):self.provenance()
        shard.write_bytes(b'state')
        self.times[self.export / 'X.119.silo'] = 21900
        with self.assertRaisesRegex(ValueError, 'changed the recovered physical time'):self.provenance()

    def test_wrong_original_or_recovery_job_is_rejected(self):
        self.snapshot(self.export, 'X.119.silo', 21890.000000000044)
        for job in ({**self.job, 'job_id': '1102388'}, {**self.job, 'nodes': 2},
                    {**self.job, 'node': '../other'}, {**self.job, 'threads': 32}):
            with self.subTest(job=job), self.assertRaisesRegex(ValueError, 'recovery allocation'):
                recovery.recovery_provenance(self.original, self.baseline, self.executable,
                    self.replay, self.export, recovery.ORIGINAL_JOB, job, self.read_time)
        with self.assertRaisesRegex(ValueError, 'original solve allocation'):
            recovery.recovery_provenance(self.original, self.baseline, self.executable,
                self.replay, self.export, {**recovery.ORIGINAL_JOB, 'node': 'foreign'}, self.job, self.read_time)

    def test_changed_executable_or_existing_or_nested_export_is_rejected(self):
        self.executable.write_bytes(b'foreign solver')
        with self.assertRaisesRegex(ValueError, 'frozen executable changed'):
            recovery.native_export_command(self.original, self.baseline, self.executable, self.replay, self.export)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            recovery.native_export_command(self.original, self.baseline, self.executable, self.replay,
                                           self.original / 'nested')
        self.export.mkdir(); (self.export / 'already-there').touch()
        with self.assertRaisesRegex(ValueError, 'fresh and empty'):
            recovery.native_export_command(self.original, self.baseline, self.executable, self.replay, self.export)


if __name__ == '__main__':
    unittest.main()
