#!/usr/bin/env python3
"""Offline import/provenance regressions; no solver, network, or rendering."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_campaign as comparison
import run_campaign as campaign


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


class RemoteProvenance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        campaign.write_json(self.root / '.campaign/plan.json', {'frozen': 'synthetic offline plan'})
        sources = {'src/physics.cpp': hashed('remote source')}
        self.runtime = {'schema': 1, 'owner': campaign.REMOTE_OWNER, 'nodes': 1, 'tasks': 1,
            'executable_sha256': hashed('remote executable'), 'source_commit': 'a' * 40,
            'source_sha256': sources,
            'source_tree_sha256': hashed(json.dumps(sources, sort_keys=True, separators=(',', ':'))),
            'solver_library_sha256': {'libsolver.so': hashed('remote library')},
            'build': {'job_id': '1001', 'node': 'qb001'}}
        self.manifest = {'remote_runtime': self.runtime, 'executable_sha256': hashed('local executable'),
            'snapshot_sha256': {'.campaign/source/src/physics.cpp': hashed('local source')}}

    def add_run(self, model='fld'):
        run = {'id': f'l1-gaussian_pulse-{model}', 'directory': f'runs/l1/gaussian_pulse/{model}',
            'case': 'gaussian_pulse', 'level': 1, 'model': model, 'method': model,
            'integrator': campaign.INTEGRATORS[model], 'status': 'completed', 'stop_time': .25, 'hard_dt': .03125}
        batch = self.root / run['directory']
        leaf = batch / 'gaussian_pulse/l1'
        config = {'radiation_model': model, 'radiation_integrator': run['integrator'],
            'max_level': '1', 'min_level': '1', 'stop_time': '.25', 'hard_dt': '.0125',
            'problem': 'RADIATION_GAUSSIAN_PULSE', 'radiation_backend': 'cpu', 'radiation_reduced_c_factor': '1'}
        metadata = {'status': 'complete', 'case': run['case'], 'level': 1, 'time': .25,
            'radiation_model': model, 'radiation_integrator': run['integrator'], 'radiation_algorithm_version': 1,
            'radiation_flux_semantics': campaign.FLUX[model], 'reference_kind': 'numerical-only',
            'config': config, 'executable_sha256': self.runtime['executable_sha256'],
            'radiation_reduced_c_factor': 1., 'length': 2.,
            'comparison_signature': {'radiation_model': model, 'radiation_integrator': run['integrator'],
                                     'hard_dt': '.0125'}}
        campaign.write_json(leaf / 'run.json', metadata)
        campaign.atomic_text(leaf / 'run.ini', ''.join(f'{key}={value}\n' for key, value in config.items()))
        campaign.atomic_text(leaf / 'run.log', 'synthetic complete solver log\n')
        campaign.atomic_text(leaf / 'radiation-conservation.csv', 'synthetic energy ledger\n')
        campaign.write_json(batch / 'batch.json', {'source_sha256': self.runtime['source_sha256']})
        plot = batch / 'plots/gaussian_pulse/l1'
        fields = ['t', 'dx', 'x', 'y', 'z', *comparison.RAD_FIELDS]
        campaign.atomic_text(plot / 'slice.csv', ','.join(fields) + '\n' + ''.join(
            f'.25,1,{x},{y},.5,1,0,0,0\n' for x in (-.5, .5) for y in (-.5, .5)))
        campaign.write_json(plot / 'slice.units.json', {name: 'test units' for name in fields})
        receipt = {key: copy.deepcopy(value) for key, value in self.runtime.items() if key != 'build'}
        receipt.update(status='complete', run_id=run['id'], case=run['case'], level=run['level'],
            model=model, integrator=run['integrator'], job_id='1002', node='qb002',
            config_sha256=campaign.file_hash(leaf / 'run.ini'), run_metadata_sha256=campaign.file_hash(leaf / 'run.json'),
            files={str(path.relative_to(batch)): {'sha256': campaign.file_hash(path), 'bytes': path.stat().st_size}
                   for path in batch.rglob('*') if path.is_file()})
        self.pin(run, receipt)
        return run, receipt

    def pin(self, run, receipt):
        relative = f'.imports/{run["id"]}.json'
        wrapper = {'schema': 1, 'status': 'verified', 'owner': campaign.REMOTE_OWNER, 'run_id': run['id'],
            'receipt': receipt, 'imported_utc': '2026-10-14T00:00:00Z', 'directory': run['directory'],
            'plan_sha256': campaign.file_hash(self.root / '.campaign/plan.json')}
        campaign.write_json(self.root / relative, wrapper)
        run['remote_import'] = {'path': relative, 'sha256': campaign.file_hash(self.root / relative)}

    def test_receipt_backed_remote_executable_and_local_default(self):
        run, receipt = self.add_run()
        self.assertEqual(campaign.expected_executable(self.root, self.manifest, run), self.runtime['executable_sha256'])
        metadata = campaign.validate_completed(self.root, self.manifest, run)
        self.assertEqual(metadata['executable_sha256'], receipt['executable_sha256'])
        local = copy.deepcopy(run)
        del local['remote_import']
        self.assertEqual(campaign.expected_executable(self.root, self.manifest, local), self.manifest['executable_sha256'])
        with self.assertRaisesRegex(ValueError, 'identity/provenance'):
            campaign.validate_completed(self.root, self.manifest, local)

    def test_receipt_requires_exact_runtime_run_and_allocation_identity(self):
        run, receipt = self.add_run()
        mutations = {'owner': 'other', 'nodes': 2, 'tasks': 2, 'model': 'm1', 'run_id': 'other',
            'case': 'marshak', 'level': 2, 'integrator': 'euler', 'job_id': '', 'node': '',
            'source_commit': 'b' * 40, 'source_tree_sha256': 'b' * 64,
            'executable_sha256': 'b' * 64, 'solver_library_sha256': {'libsolver.so': 'b' * 64}}
        for key, value in mutations.items():
            bad = copy.deepcopy(receipt)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                campaign.validate_remote_receipt(self.manifest, run, bad)
        bad = copy.deepcopy(self.manifest)
        bad['remote_runtime']['build']['node'] = ''
        with self.assertRaisesRegex(ValueError, 'build job/node'):
            campaign.validate_remote_receipt(bad, run, receipt)

    def test_inventory_rejects_changes_missing_files_symlinks_and_escapes(self):
        run, receipt = self.add_run()
        batch = self.root / run['directory']
        log = batch / 'gaussian_pulse/l1/run.log'
        saved = log.read_text()
        log.write_text(saved + 'changed')
        with self.assertRaisesRegex(ValueError, 'output missing or changed'):
            campaign.validate_remote_import(self.root, self.manifest, run)
        log.unlink()
        with self.assertRaises(ValueError):
            campaign.validate_remote_import(self.root, self.manifest, run)
        outside = self.root / 'other-log'
        outside.write_text(saved)
        log.symlink_to(outside)
        with self.assertRaises(ValueError):
            campaign.validate_remote_import(self.root, self.manifest, run)
        for name in ('../escape', '/absolute', 'a/../b', 'a//b'):
            bad = copy.deepcopy(receipt)
            bad['files'][name] = {'sha256': hashed('x'), 'bytes': 1}
            with self.subTest(path=name), self.assertRaises(ValueError):
                campaign.validate_remote_receipt(self.manifest, run, bad)

    def test_import_pointer_plan_and_config_are_bound(self):
        run, receipt = self.add_run()
        wrapper = self.root / run['remote_import']['path']
        wrapper.write_text(wrapper.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'receipt hash'):
            campaign.validate_remote_import(self.root, self.manifest, run)
        self.pin(run, receipt)
        campaign.write_json(self.root / '.campaign/plan.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'different run or plan'):
            campaign.validate_remote_import(self.root, self.manifest, run)
        self.pin(run, receipt)
        batch = self.root / run['directory']
        ini = batch / 'gaussian_pulse/l1/run.ini'
        ini.write_text(ini.read_text().replace('hard_dt=.0125', 'hard_dt=.1'))
        receipt['config_sha256'] = campaign.file_hash(ini)
        receipt['files']['gaussian_pulse/l1/run.ini'] = {'sha256': campaign.file_hash(ini), 'bytes': ini.stat().st_size}
        self.pin(run, receipt)
        with self.assertRaisesRegex(ValueError, 'metadata and imported configuration'):
            campaign.validate_remote_import(self.root, self.manifest, run)
        meta_path = batch / 'gaussian_pulse/l1/run.json'
        meta = json.loads(meta_path.read_text())
        meta['config']['hard_dt'] = '.1'
        campaign.write_json(meta_path, meta)
        receipt['run_metadata_sha256'] = campaign.file_hash(meta_path)
        receipt['files']['gaussian_pulse/l1/run.json'] = {'sha256': campaign.file_hash(meta_path), 'bytes': meta_path.stat().st_size}
        self.pin(run, receipt)
        with self.assertRaisesRegex(ValueError, 'timestep exceeds'):
            campaign.validate_remote_import(self.root, self.manifest, run)

    def test_remote_comparison_requires_two_receipt_backed_matching_models(self):
        left, _ = self.add_run('m1')
        right, _ = self.add_run('fld')
        a = comparison.load_run(self.root, left, self.manifest)
        b = comparison.load_run(self.root, right, self.manifest)
        comparison.compatible(a, b, self.root, self.manifest)
        record = comparison.record_run(self.root, b)
        self.assertEqual(record['execution_provenance']['host'], 'QueenBee')
        self.assertEqual(record['execution_provenance']['job_id'], '1002')
        with self.assertRaisesRegex(ValueError, 'campaign runtime'):
            comparison.load_run(self.root, right)
        b['remote_receipt'] = None
        with self.assertRaisesRegex(ValueError, 'local/remote'):
            comparison.compatible(a, b, self.root, self.manifest)

    def test_missing_inventory_requires_exact_validated_retention_record(self):
        run, receipt = self.add_run()
        relative = 'gaussian_pulse/l1/X.1.silo'
        record = {'sha256': hashed('raw'), 'bytes': 3}
        receipt['files'][relative] = record
        self.pin(run, receipt)
        run['retention'] = {'path': 'unused-fixture', 'sha256': 'a' * 64}
        retained = {'removed_raw': [{'path': 'X.1.silo', **record}]}
        with patch('campaign_retention.validate_retention', return_value=retained):
            campaign.validate_remote_import(self.root, self.manifest, run)
        retained['removed_raw'][0]['sha256'] = 'b' * 64
        with patch('campaign_retention.validate_retention', return_value=retained), \
                self.assertRaisesRegex(ValueError, 'removed raw artifact'):
            campaign.validate_remote_import(self.root, self.manifest, run)


if __name__ == '__main__':
    unittest.main()
