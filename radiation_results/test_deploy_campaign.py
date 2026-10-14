#!/usr/bin/env python3
"""Deployment state-machine tests use local directories, never SSH or a solver."""
import copy
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import deploy_campaign as deploy


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'webroot'
        self.root.mkdir()
        (self.root / 'index.html').write_text('<html><head><base href="/campaigns/local-20261011/"></head>Historical</html>')
        (self.root / 'campaigns/.generations/old').mkdir(parents=True)
        (self.root / 'campaigns/.generations/old/index.html').write_text('Historical campaign')
        (self.root / 'campaigns/local-20261011').symlink_to('.generations/old')
        self.original = deploy.file_hash(self.root / 'index.html')
        self.generation = deploy.SLUG + '-20261013T230000Z-0123abcd'
        self.stage = self.base / 'site'
        self.stage.mkdir()
        (self.stage / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><main><h1>M1 RK1 and FLD</h1></main></html>')
        (self.stage / 'profiles.csv').write_text('m1_er,fld_er\n1,1\n')
        deploy.write_json(self.stage / 'site-manifest.json', {'schema': 2, 'workflow': 'm1-fld-radiation',
            'status': 'staged_locally', 'completed_runs': 2, 'local_links_verified': True,
            'source': str(self.base / 'campaign'), 'file_sha256': {p.name: deploy.file_hash(p) for p in self.stage.iterdir()}})
        namespace = {'__name__': 'synthetic_remote'}
        exec(deploy.REMOTE, namespace)
        self.remote_run = namespace['run']

    def remote(self, action, **values):
        return self.remote_run({'slug': deploy.SLUG, 'generation': self.generation,
            'action': action, 'original_homepage_sha256': self.original, **values}, self.root)

    def upload(self):
        manifest = deploy.prepare_staged(self.stage, self.generation)
        prepared = self.remote('prepare', bytes=0, reserve_bytes=0)
        shutil.copytree(self.stage, prepared['incoming'], dirs_exist_ok=True)
        return manifest, prepared

    def activate(self, prepared):
        return self.remote('activate', previous_generation=prepared['previous_generation'],
            manifest_sha256=deploy.file_hash(self.stage / 'publication-manifest.json'))

    def verified_pilot(self):
        self.generation = deploy.PILOT_GENERATION
        path = self.stage / 'site-manifest.json'
        receipt = json.loads(path.read_text())
        receipt['completed_runs'] = 8
        deploy.write_json(path, receipt)
        _, prepared = self.upload()
        self.activate(prepared)
        return self.remote('verified', verified_utc='fixture')

    def archive_pilot(self):
        return self.remote('archive-pilot', pilot_generation=deploy.PILOT_GENERATION)

    def test_activation_preserves_old_campaign_and_archives_previous_homepage(self):
        manifest, prepared = self.upload()
        result = self.activate(prepared)
        self.assertEqual(result['status'], 'activated')
        self.assertEqual(deploy.file_hash(Path(result['archive']) / 'index.html'), self.original)
        self.assertEqual((self.root / 'campaigns/local-20261011/index.html').read_text(), 'Historical campaign')
        self.assertTrue((self.root / 'campaigns' / deploy.SLUG).is_symlink())
        homepage = (self.root / 'index.html').read_text()
        self.assertIn('<base href="/campaigns/m1-fld-20261013/">', homepage)
        self.assertIn('Historical Euler/ADER2 campaign', homepage)
        self.assertEqual(deploy.file_hash(self.root / 'index.html'), result['homepage_sha256'])
        self.assertEqual(self.remote('verified', verified_utc='fixture')['status'], 'verified')

    def test_read_only_inspection_does_not_create_new_paths(self):
        before = set(self.root.rglob('*'))
        result = self.remote('inspect')
        self.assertEqual(result['homepage_sha256'], self.original)
        self.assertEqual(set(self.root.rglob('*')), before)
        self.assertFalse(result['campaign_exists'])

    def test_changed_homepage_is_never_overwritten(self):
        manifest, prepared = self.upload()
        (self.root / 'index.html').write_text('Unrelated user edit')
        with self.assertRaisesRegex(ValueError, 'Homepage changed'):
            self.activate(prepared)
        self.assertEqual((self.root / 'index.html').read_text(), 'Unrelated user edit')
        self.assertFalse((self.root / 'campaigns' / deploy.SLUG).exists())

    def test_changed_uploaded_file_prevents_activation(self):
        manifest, prepared = self.upload()
        (Path(prepared['incoming']) / 'profiles.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Uploaded file changed'):
            self.activate(prepared)
        self.assertEqual(deploy.file_hash(self.root / 'index.html'), self.original)

    def test_remote_lock_prevents_concurrent_activation(self):
        manifest, prepared = self.upload()
        with (self.root / 'campaigns/.publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.activate(prepared)

    def test_staged_changes_or_extra_files_are_rejected(self):
        (self.stage / 'profiles.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Staged file changed'):
            deploy.prepare_staged(self.stage, self.generation)
        (self.stage / 'extra.bin').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            deploy.prepare_staged(self.stage, self.generation)

    def test_failed_https_verification_records_that_activation_succeeded(self):
        campaign = self.base / 'campaign'
        campaign.mkdir()
        manifest = deploy.prepare_staged(self.stage, self.generation)
        responses = [{'incoming': '/fixture', 'previous_generation': None},
            {'status': 'activated', 'homepage_sha256': 'sha', 'manifest_sha256': 'sha'}]
        with patch.object(deploy, 'remote', side_effect=responses), patch.object(deploy.subprocess, 'run'), \
                patch.object(deploy, 'verify_https', side_effect=ValueError('HTTP mismatch')):
            with self.assertRaisesRegex(ValueError, 'HTTP mismatch'):
                deploy.deploy(campaign, self.stage, self.generation, manifest)
        receipt = json.loads((campaign / '.deployment' / (self.generation + '.json')).read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['remote_activation'], 'activated')

    def test_local_preparation_has_no_network_calls(self):
        with patch.object(deploy, 'remote') as remote, patch.object(deploy.subprocess, 'run') as command:
            manifest = deploy.prepare_staged(self.stage, self.generation)
        remote.assert_not_called(); command.assert_not_called()
        self.assertEqual(manifest['completed_runs'], 2)

    def test_pilot_archive_is_exact_idempotent_and_does_not_change_current_homepage(self):
        pilot = self.verified_pilot()
        current = (self.root / 'index.html').read_bytes()
        receipt_path = self.root / 'campaigns' / ('.' + deploy.SLUG + '-receipt.json')
        saved_receipt = receipt_path.read_bytes()
        first = self.archive_pilot()
        second = self.archive_pilot()
        self.assertEqual(first['status'], 'archived')
        self.assertEqual(second['status'], 'already_archived')
        self.assertEqual((self.root / 'campaigns' / deploy.PILOT_SLUG).resolve(), Path(pilot['generation_path']))
        self.assertEqual((self.root / 'index.html').read_bytes(), current)
        self.assertEqual(receipt_path.read_bytes(), saved_receipt)
        self.assertEqual(deploy.file_hash(Path(first['archive_path']) / 'index.html'), first['index_sha256'])
        self.assertIn('/campaigns/' + deploy.PILOT_SLUG + '/', (self.stage / 'index.html').read_text())

    def test_pilot_archive_never_overwrites_foreign_directory_or_symlink(self):
        self.verified_pilot()
        alias = self.root / 'campaigns' / deploy.PILOT_SLUG
        alias.mkdir(); (alias / 'keep.txt').write_text('Unrelated user data')
        with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
            self.archive_pilot()
        self.assertEqual((alias / 'keep.txt').read_text(), 'Unrelated user data')
        (alias / 'keep.txt').unlink(); alias.rmdir()
        alias.symlink_to('.generations/old')
        with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
            self.archive_pilot()
        self.assertEqual((alias / 'index.html').read_text(), 'Historical campaign')

    def test_pilot_archive_rejects_changed_content_homepage_and_unexpected_generation(self):
        self.verified_pilot()
        with self.assertRaisesRegex(ValueError, 'Unexpected pilot generation'):
            self.remote('archive-pilot', pilot_generation=deploy.SLUG + '-unexpected')
        homepage = (self.root / 'index.html').read_bytes()
        (self.root / 'index.html').write_text('User homepage change')
        with self.assertRaisesRegex(ValueError, 'no longer matches'):
            self.archive_pilot()
        (self.root / 'index.html').write_bytes(homepage)
        (self.root / 'campaigns' / deploy.SLUG / 'profiles.csv').write_text('changed output')
        with self.assertRaisesRegex(ValueError, 'Pilot file changed'):
            self.archive_pilot()
        self.assertFalse((self.root / 'campaigns' / deploy.PILOT_SLUG).exists())

    def test_pilot_archive_respects_shared_publication_lock(self):
        self.verified_pilot()
        with (self.root / 'campaigns/.publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.archive_pilot()
        self.assertFalse((self.root / 'campaigns' / deploy.PILOT_SLUG).exists())

    def test_new_campaign_replacement_keeps_pilot_alias_and_original_receipt(self):
        pilot = self.verified_pilot()
        self.archive_pilot()
        pilot_receipt_path = Path(pilot['generation_path']) / 'publication-receipt.json'
        pilot_receipt = pilot_receipt_path.read_bytes()
        pilot_index = deploy.file_hash(Path(pilot['generation_path']) / 'index.html')
        self.stage = self.base / 'longer-campaign-site'
        self.stage.mkdir()
        (self.stage / 'index.html').write_text('<html><head></head><main><h1>Longer evolution</h1></main></html>')
        deploy.write_json(self.stage / 'site-manifest.json', {'schema': 2, 'workflow': 'm1-fld-radiation',
            'status': 'staged_locally', 'completed_runs': 2, 'local_links_verified': True,
            'source': str(self.base / 'new-campaign'), 'file_sha256': {'index.html': deploy.file_hash(self.stage / 'index.html')}})
        self.generation = deploy.SLUG + '-20261014T080000Z-abcddcba'
        _, prepared = self.upload()
        self.assertEqual(prepared['previous_generation'], pilot['generation_path'])
        next_receipt = self.activate(prepared)
        self.assertEqual(next_receipt['previous_generation_path'], pilot['generation_path'])
        self.assertIn('Longer evolution', (self.root / 'index.html').read_text())
        self.assertEqual(deploy.file_hash(self.root / 'campaigns' / deploy.PILOT_SLUG / 'index.html'), pilot_index)
        self.assertEqual(pilot_receipt_path.read_bytes(), pilot_receipt)
        self.assertEqual(self.archive_pilot()['status'], 'already_archived')


if __name__ == '__main__':
    unittest.main()
