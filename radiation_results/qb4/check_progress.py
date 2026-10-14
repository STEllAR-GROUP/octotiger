#!/usr/bin/env python3
"""One bounded progress/import/publication check through an existing SSH master.

This program cannot authenticate a new QueenBee connection. --advance may
import verified completed output, publish it, and submit one dependent one-node
job. A scheduler invokes checks; there is no Slurm polling or retry loop here.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

OWNER = 'octotiger-m1-fld-qb4-20261014'
HOST = 'dmarce1@qbd.loni.org'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def ssh_arguments(socket):
    # ControlMaster=no alone silently creates a new connection on socket loss.
    return ['ssh', '-S', str(socket), '-o', 'ControlMaster=no', '-o', 'BatchMode=yes',
            '-o', 'ProxyCommand=/bin/false']


def connection(migration):
    ledger = read(Path.home() / '.codex/qb4-access-state.json')
    active, task = ledger['active_connection'], ledger['current_task']
    if (task.get('owner') != OWNER or active.get('owner') != OWNER or
            active.get('status') != 'connected' or
            active.get('control_path') != migration['control_socket'] or
            task['task_attempts_reserved'] > task['user_authorized_task_maximum_logins']):
        raise ValueError('The authorized persistent QueenBee session is unavailable')
    argv = ssh_arguments(migration['control_socket'])
    subprocess.run([*argv, '-O', 'check', HOST], check=True, capture_output=True, timeout=10)
    return argv


def remote_python(ssh, script):
    command = '/usr/local/packages/python/3.11.5-anaconda/bin/python3.11 -c ' + shlex.quote(script)
    result = subprocess.run([*ssh, HOST, command], check=True, capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def capture(ssh, migration):
    script = '''from pathlib import Path
import json
root=Path(ROOT)
def read(name):
 p=root/name
 return json.loads(p.read_text()) if p.is_file() else None
worker=read('campaign/worker-state.json')
receipts={}
if worker:
 for run_id,record in worker.get('runs',{}).items():
  if record.get('status')=='complete':
   receipts[run_id]=read('campaign/receipts/'+run_id+'.json')
print(json.dumps({'build':read('build-progress.json'),'worker':worker,'receipts':receipts}))
'''.replace('ROOT', repr(migration['remote_root']))
    return remote_python(ssh, script)


def summary(snapshot):
    worker = snapshot.get('worker') or {}
    build = snapshot.get('build') or {}
    return {'build_status': build.get('status'), 'build_job_id': build.get('job_id'),
            'worker_status': worker.get('status'), 'allocation': worker.get('allocation'),
            'active_runs': {name: {key: record.get(key) for key in ('status', 'progress')}
                            for name, record in worker.get('runs', {}).items() if record.get('status') == 'running'},
            'completed_remote': len(snapshot['receipts']), 'checked_utc': snapshot['checked_utc']}


def pin_runtime(root, migration, runtime, runner):
    if runtime.get('source_commit') != migration['source_commit'] or runtime.get('worker_sha256') != migration['worker_sha256']:
        raise ValueError('QueenBee source or worker differs from the staged revision')
    # Check every available frozen application source against the compiled tree.
    source = root / '.campaign/source'
    for name, digest in runtime['source_sha256'].items():
        path = source / name
        if path.is_file() and runner.file_hash(path) != digest:
            raise ValueError('QueenBee application source differs: ' + name)
    with ExitStack() as stack:
        for name in ('.publication.lock', '.campaign.lock'):
            handle = stack.enter_context((root / name).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = runner.load(root)
        runner.verify_snapshot(root, manifest, artifact_only=True)
        if manifest.get('active') or any(run['status'] == 'running' for run in manifest['runs']):
            raise ValueError('A local solver is active during remote collection')
        previous = manifest.get('remote_runtime')
        if previous is not None and previous != runtime:
            raise ValueError('Previously pinned QueenBee runtime changed')
        if previous is None:
            manifest['remote_runtime'] = runtime
            runner.save(root, manifest)


def publish(root):
    subprocess.run([sys.executable, str(root / '.campaign/run_published_campaign.py'),
                    '--campaign', str(root), '--postprocess-only', '--start'], check=True)


def collect(root, migration, ssh, receipt, runner):
    run_id = receipt['run_id']
    if re.fullmatch(r'l[234]-[a-z_]+-(m1|fld)', run_id) is None:
        raise ValueError('Invalid remote run identifier')
    import campaign_retention as retention
    incoming = retention.safe_path(root, '.remote-incoming/' + run_id)
    incoming.mkdir(parents=True, exist_ok=True)
    receipt_path = incoming / 'completion.json'
    if receipt_path.exists() and read(receipt_path) != receipt:
        raise ValueError('Incoming completion receipt changed')
    write(receipt_path, receipt)
    total = sum(item['bytes'] for item in receipt['files'].values())
    manifest = runner.load(root)
    if shutil.disk_usage(root).free < 2 * total + manifest['minimum_free_gib'] * 2**30:
        raise ValueError('Insufficient storage for transfer, verified import and reserve')
    batch = incoming / 'batch'
    batch.mkdir(exist_ok=True)
    remote = HOST + ':' + migration['remote_root'] + '/campaign/runs/' + run_id + '/'
    subprocess.run(['rsync', '-a', '-z', '--checksum', '-e', shlex.join(ssh), remote, str(batch) + '/'], check=True)
    subprocess.run([sys.executable, str(root / '.campaign/qb4/import_completed.py'),
                    '--campaign', str(root), '--batch', str(batch), '--receipt', str(receipt_path), '--import'], check=True)
    return incoming


def submit_next(ssh, migration, worker):
    predecessor = str(worker['allocation']['job_id'])
    if re.fullmatch(r'[1-9][0-9]*', predecessor) is None:
        raise ValueError('Invalid predecessor job identity')
    script = '''from pathlib import Path
import json,subprocess,os
root=Path(ROOT)
record=root/'submissions'/('after-'+PREVIOUS+'.json')
record.parent.mkdir(exist_ok=True)
if record.exists():
 saved=json.loads(record.read_text())
 if saved.get('status')!='submitted' or saved.get('dependency')!=PREVIOUS or not str(saved.get('job_id','')).isdigit() or int(saved['job_id'])<1 or saved.get('nodes')!=1 or saved.get('tasks')!=1 or saved.get('returncode')!=0:
  raise RuntimeError('Unknown previous submission outcome; inspect before retrying')
 print(json.dumps(saved));raise SystemExit(0)
accounting=subprocess.run(['sacct','--noheader','--parsable2','--allocations','--jobs',PREVIOUS,'--format=JobIDRaw,State,ExitCode,AllocNodes'],text=True,capture_output=True,check=True)
rows=[line.split('|') for line in accounting.stdout.splitlines() if line.split('|')[0]==PREVIOUS]
if len(rows)!=1 or rows[0][1]!='COMPLETED' or rows[0][2]!='0:0' or rows[0][3]!='1':
 raise RuntimeError('Previous single-node job is not confirmed completed successfully; do not allocate another node')
fd=os.open(record,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:
 json.dump({'status':'reserved','dependency':PREVIOUS,'nodes':1},f);f.flush();os.fsync(f.fileno())
p=subprocess.run(['sbatch','--parsable',str(root/'run.sbatch')],cwd=root,text=True,capture_output=True)
job=p.stdout.strip().split(';')[0]
valid=job.isdigit() and int(job)>0
r={'status':('submitted' if valid else 'unknown') if p.returncode==0 else 'rejected','returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'dependency':PREVIOUS,'nodes':1,'tasks':1,'cpus':64,'previous_job_accounting':rows[0]}
if p.returncode==0 and valid:r['job_id']=job
temporary=record.with_suffix('.tmp');temporary.write_text(json.dumps(r,indent=2)+'\\n');temporary.replace(record)
if p.returncode==0 and not valid:raise RuntimeError('Unknown sbatch outcome; inspect before retrying')
print(json.dumps(r))
'''.replace('ROOT', repr(migration['remote_root'])).replace('PREVIOUS', repr(predecessor))
    return remote_python(ssh, script)


def check(directory, advance=False):
    directory = Path(directory).resolve()
    with (directory / '.check.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        migration = read(directory / 'migration.json')
        root = Path(migration['local_campaign']).resolve()
        sys.path.insert(0, str(root / '.campaign'))
        import run_campaign as runner
        import campaign_retention as retention
        if migration.get('owner') != OWNER or migration.get('nodes') != 1 or not (root / 'STOP').exists():
            raise ValueError('Invalid migration or the local solver STOP was removed')
        ssh = connection(migration)
        snapshot = capture(ssh, migration)
        snapshot['checked_utc'] = datetime.now(timezone.utc).isoformat()
        write(directory / 'progress.json', snapshot)
        worker = snapshot.get('worker')
        publication = read(root / '.publication/state.json')
        if not advance or publication.get('active'):
            return {'status': 'publication-active' if publication.get('active') else 'checked', **summary(snapshot)}
        if publication.get('status') in ('paused', 'failed'):
            raise ValueError('Publication failed; inspect and repair before an explicit retry')
        local_manifest = runner.load(root)
        if local_manifest.get('active') or any(run['status'] == 'running' for run in local_manifest['runs']):
            raise ValueError('Local solver state is still active or running')
        if worker and worker.get('status') == 'waiting_for_publication' and not snapshot['receipts']:
            raise ValueError('Publication-ready worker has no completion receipts')
        if worker:
            pin_runtime(root, migration, worker['remote_runtime'], runner)
        for run_id, receipt in snapshot['receipts'].items():
            if receipt.get('run_id') != run_id:
                raise ValueError('Receipt mapping key and run identity differ')
            manifest = runner.load(root)
            run = next(item for item in manifest['runs'] if item['id'] == run_id)
            if run['status'] == 'completed':
                if not run.get('remote_import'):
                    raise ValueError('QueenBee receipt cannot replace a completed local result')
                committed = read(root / run['remote_import']['path'])['receipt']
                if committed != receipt:
                    raise ValueError('Previously imported QueenBee completion receipt changed')
                runner.validate_completed(root, manifest, run)
            else:
                collect(root, migration, ssh, receipt, runner)
        manifest = runner.load(root)
        if any(run['status'] == 'completed' and run['id'] not in publication['published'] for run in manifest['runs']):
            publish(root)
        publication = read(root / '.publication/state.json')
        manifest = runner.load(root)
        incoming_parent = retention.safe_path(root, '.remote-incoming')
        pending_cleanup = list(incoming_parent.iterdir()) if incoming_parent.is_dir() else []
        for incoming in pending_cleanup:
            if incoming.is_symlink() or not incoming.is_dir():
                raise ValueError('Unexpected incoming transfer artifact')
            run = next(item for item in manifest['runs'] if item['id'] == incoming.name)
            if run['id'] not in publication['published'] or not run.get('remote_import'):
                continue
            runner.validate_completed(root, manifest, run)
            retention.validate_retention(root, run)
            committed = read(root / run['remote_import']['path'])['receipt']
            if read(incoming / 'completion.json') != committed:
                raise ValueError('Transfer receipt differs from the committed imported result')
            # This is only the verified duplicate transfer. Endpoints and all
            # extracted samples are retained in the committed imported result.
            shutil.rmtree(incoming)
        if worker and worker.get('status') == 'waiting_for_publication':
            job = str(worker['allocation']['job_id'])
            if all(run_id in publication['published'] for run_id in snapshot['receipts']) and migration.get('advanced_after_job') != job:
                submission = submit_next(ssh, migration, worker)
                migration['last_submission'] = submission
                if submission['returncode']:
                    write(directory / 'migration.json', migration)
                    raise ValueError('Next one-node job was rejected: ' + submission['stderr'])
                migration.update(advanced_after_job=job, current_run_job_id=submission['job_id'])
                write(directory / 'migration.json', migration)
        return {'status': 'checked-and-advanced', 'completed_local': sum(r['status'] == 'completed' for r in manifest['runs']),
                'published': len(publication['published']), 'current_run_job_id': migration.get('current_run_job_id'),
                **summary(snapshot)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--advance', action='store_true', help='Import, publish and submit the next single-node job when ready')
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.advance), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
