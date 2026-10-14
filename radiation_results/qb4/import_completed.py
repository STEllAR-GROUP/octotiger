#!/usr/bin/env python3
"""Verify a locally transferred QueenBee result and explicitly import it.

This command has no network or solver capability. The default is a read-only
plan. --import accepts one verified completed batch under the campaign lock;
the original transferred batch and completed local results remain unchanged.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_campaign as runner
import campaign_retention as retention

OWNER = 'octotiger-m1-fld-qb4-20261014'


def read_json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular JSON receipt: ' + str(path))
    return json.loads(path.read_text())


def check_idle(root, manifest):
    if manifest.get('active') or any(run.get('status') == 'running' for run in manifest['runs']):
        raise ValueError('Campaign is active; import only at a completed local phase boundary')
    state_path = root / '.publication/state.json'
    if state_path.exists() and read_json(state_path).get('active'):
        raise ValueError('Publication is active; import only at a completed publication boundary')
    if (root / 'POSTPROCESS_STOP').exists():
        raise ValueError('POSTPROCESS_STOP is present')
    # Solver STOP is intentionally preserved: accepting existing results never
    # authorizes a local solver or a remote scheduler submission.


def inspect_source(batch, receipt, run, snapshots):
    if batch.is_symlink() or not batch.is_dir():
        raise ValueError('Transferred batch must be a regular directory')
    files = receipt.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Completion receipt requires a complete file inventory')
    actual = set()
    for parent, directories, names in os.walk(batch, followlinks=False):
        for name in directories + names:
            path = Path(parent) / name
            relative = str(path.relative_to(batch))
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError('Transferred output contains a symlink or special file: ' + relative)
            retention.safe_path(batch, relative)
            if stat.S_ISREG(mode):
                actual.add(relative)
    if actual != set(files):
        raise ValueError('Transferred file inventory differs from completion receipt')
    total = 0
    for relative, record in files.items():
        path = retention.safe_path(batch, relative)
        if (not isinstance(record, dict) or type(record.get('bytes')) is not int or record['bytes'] < 0 or
                not re.fullmatch(r'[0-9a-f]{64}', str(record.get('sha256', ''))) or
                path.stat().st_size != record['bytes'] or runner.file_hash(path) != record['sha256']):
            raise ValueError('Transferred file hash/size differs: ' + relative)
        total += record['bytes']
    prefix = run['case'] + '/l' + str(run['level']) + '/'
    numbered = sorted(int(match[1]) for name in files
        if (match := re.fullmatch(re.escape(prefix) + r'X\.(\d+)\.silo', name)))
    if (not numbered or numbered != list(range(numbered[-1] + 1)) or
            len(numbered) + 1 < snapshots or prefix + 'final.silo' not in files):
        raise ValueError('Completed import must include every numbered snapshot and both endpoints')
    for header in [*(prefix + f'X.{number}.silo' for number in numbered), prefix + 'final.silo']:
        if not files[header]['bytes'] or not any(name.startswith(header + '.data/') and record['bytes'] > 0
                                                for name, record in files.items()):
            raise ValueError('Completed import is missing raw Silo header or shards: ' + header)
    if prefix + 'retention.json' in files:
        raise ValueError('Import requires complete raw outputs; remote retention receipts are not accepted')
    return total


def load_inputs(root, batch, receipt_path):
    manifest = runner.load(root)
    runner.verify_snapshot(root, manifest, artifact_only=True)
    check_idle(root, manifest)
    receipt = read_json(receipt_path)
    run_id = receipt.get('run_id')
    if not isinstance(run_id, str) or not re.fullmatch(r'l[0-9]+-[a-z_]+-(m1|fld)', run_id):
        raise ValueError('Invalid remote run identity')
    matches = [run for run in manifest['runs'] if run['id'] == run_id]
    if len(matches) != 1:
        raise ValueError('Remote run is absent or duplicated in the frozen plan')
    run = matches[0]
    runner.validate_remote_receipt(manifest, run, receipt)
    total = inspect_source(batch, receipt, run, manifest['snapshots'])
    runner.validate_remote_outputs(batch, manifest, run, receipt)
    return manifest, run, receipt, total


def import_record(root, run, receipt):
    return {'schema': 1, 'status': 'verified', 'owner': OWNER, 'run_id': run['id'],
            'directory': run['directory'], 'plan_sha256': runner.file_hash(root / '.campaign/plan.json'),
            'imported_utc': runner.now(), 'receipt': receipt}


def matching_record(root, run, receipt, path):
    saved = read_json(path)
    expected = import_record(root, run, receipt)
    if not saved.get('imported_utc') or any(saved.get(key) != value for key, value in expected.items()
                                         if key != 'imported_utc'):
        raise ValueError('Existing import receipt conflicts with incoming result')
    return saved


def resource_check(root, manifest, needed=0):
    if (root / 'POSTPROCESS_STOP').exists():
        raise ValueError('POSTPROCESS_STOP is present')
    if shutil.disk_usage(root).free < needed + manifest['minimum_free_gib'] * 2**30:
        raise ValueError('Insufficient storage for import plus the campaign reserve')


def inspect(root, batch, receipt_path):
    manifest, run, receipt, total = load_inputs(root, batch, receipt_path)
    return {'read_only': True, 'explicit_import_required': True, 'run_id': run['id'],
            'status': run['status'], 'source': str(batch), 'destination': str(root / run['directory']),
            'files': len(receipt['files']), 'bytes': total, 'job_id': receipt['job_id'], 'node': receipt['node'],
            'already_imported': bool(run.get('remote_import')), 'solver_stop_preserved': (root / 'STOP').exists()}


def accept(root, batch, receipt_path):
    # Use the publisher's lock ordering. An import cannot race staging or make a
    # publisher's result inventory change between its checks and deployment.
    with ExitStack() as stack:
        for name in ('.publication.lock', '.campaign.lock'):
            handle = stack.enter_context((root / name).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, run, receipt, total = load_inputs(root, batch, receipt_path)
        destination = retention.safe_path(root, run['directory'])
        record_path = retention.safe_path(root, '.imports/' + run['id'] + '.json')
        if run['status'] == 'completed':
            if not run.get('remote_import'):
                raise ValueError('Preserving existing completed local result; it cannot be replaced by an import')
            matching_record(root, run, receipt, record_path)
            runner.validate_completed(root, manifest, run)
            return {'status': 'already-imported', 'run_id': run['id'], 'remote_import': run['remote_import']}
        if run['status'] != 'pending' or run.get('attempts'):
            raise ValueError('Import requires a pending run with no local attempts')
        if destination.exists() and not record_path.exists():
            raise ValueError('Existing output directory has no matching import receipt; refusing to adopt or overwrite it')
        if record_path.exists():
            record = matching_record(root, run, receipt, record_path)
        else:
            record = import_record(root, run, receipt)
        if destination.exists():
            # Crash recovery after rename and before campaign.json commit. The
            # immutable receipt was written first and must match every byte.
            inspect_source(destination, receipt, run, manifest['snapshots'])
            runner.validate_remote_outputs(destination, manifest, run, receipt)
        else:
            resource_check(root, manifest, total)
            staging = retention.safe_path(root, '.imports/staging')
            staging.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=run['id'] + '-', dir=staging))
            try:
                payload = temporary / 'batch'
                payload.mkdir()
                for relative in receipt['files']:
                    resource_check(root, manifest)
                    source = retention.safe_path(batch, relative)
                    target = retention.safe_path(payload, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                inspect_source(payload, receipt, run, manifest['snapshots'])
                runner.validate_remote_outputs(payload, manifest, run, receipt)
                resource_check(root, manifest)
                if not record_path.exists():
                    runner.write_json(record_path, record)
                destination.parent.mkdir(parents=True, exist_ok=True)
                payload.rename(destination)
            finally:
                # This directory was created in this invocation. Renamed final
                # outputs and interrupted committed receipts are never removed.
                shutil.rmtree(temporary)
        run.update(status='completed', completed_utc=record['imported_utc'],
            run_metadata_sha256=receipt['run_metadata_sha256'],
            remote_import={'path': str(record_path.relative_to(root)), 'sha256': runner.file_hash(record_path)})
        runner.validate_completed(root, manifest, run)
        manifest['status'] = 'completed' if all(item['status'] == 'completed' for item in manifest['runs']) else 'ready'
        runner.save(root, manifest)
        return {'status': 'imported', 'run_id': run['id'], 'remote_import': run['remote_import'],
                'files': len(receipt['files']), 'bytes': total}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True, help='Local transferred batch directory, including batch.json')
    parser.add_argument('--receipt', type=Path, required=True, help='Local remote-completion receipt outside the batch')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--import', dest='accept', action='store_true', help='Explicitly commit this verified completed result')
    action.add_argument('--plan', action='store_true', help='Validate and describe without changing files (default)')
    args = parser.parse_args(argv)
    root = args.campaign.resolve()
    if args.batch.is_symlink() or args.receipt.is_symlink():
        raise ValueError('Transferred batch and receipt cannot be symlinks')
    batch, receipt = args.batch.absolute(), args.receipt.absolute()
    result = accept(root, batch, receipt) if args.accept else inspect(root, batch, receipt)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
