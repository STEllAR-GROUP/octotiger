#!/usr/bin/env python3
"""Inspect a frozen campaign, or explicitly run and publish it one result at a time.

The default and --plan/--status are read-only. Only --start can run a solver,
render movies, or deploy a website. Completed unpublished results are handled
before another solver starts. A failed phase stops the sequence for review.
"""
import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

import run_campaign as runner
import campaign_retention as retention

POLICY_NAME = 'publication-workflow.json'
SCRIPTS = ('run_published_campaign.py', 'run_campaign.py', 'campaign_movies.py',
           'publish_campaign.py', 'deploy_campaign.py', 'campaign_retention.py')


def read_json(path):
    return json.loads(path.read_text())


def load_policy(root):
    policy = read_json(root / '.campaign' / POLICY_NAME)
    if (policy.get('schema') != 1 or policy.get('workflow') != runner.WORKFLOW or
            policy.get('stages') != 'keep-latest-verified' or
            policy.get('sequence') != 'one-run-render-stage-deploy' or
            policy.get('deployment_script') != 'deploy_campaign.py' or
            policy.get('retention') != 'verified-samples-and-endpoints' or
            policy.get('raw_run_budget_gib') != {'2': 1, '3': 7, '4': 48}):
        raise ValueError('Unsupported frozen publication workflow')
    movie = policy.get('movie', {})
    if (set(movie) != {'seconds', 'fps', 'hold', 'width', 'height', 'vectors', 'full_domain'} or
            any(type(movie.get(key)) not in (int, float) or not math.isfinite(movie[key])
                for key in ('seconds', 'hold')) or
            any(type(movie.get(key)) is not int or movie[key] < 1
                for key in ('fps', 'width', 'height', 'vectors')) or
            movie['hold'] < 0 or movie['seconds'] <= 2 * movie['hold'] or
            min(movie['width'], movie['height']) < 256 or movie['width'] % 2 or movie['height'] % 2 or
            type(movie['full_domain']) is not bool):
        raise ValueError('Invalid frozen movie settings')
    return policy


def verify_inputs(root, manifest, postprocess_only=False):
    runner.verify_snapshot(root, manifest, artifact_only=postprocess_only)
    for name in (*SCRIPTS, POLICY_NAME):
        relative = '.campaign/' + name
        if relative not in manifest['snapshot_sha256']:
            raise ValueError('Publication input was not frozen: ' + name)
    for source in (Path(__file__), Path(runner.__file__), Path(retention.__file__)):
        if runner.file_hash(source) != runner.file_hash(root / '.campaign' / source.name):
            raise ValueError('Use the unchanged frozen publication runner and its frozen dependencies')


def state_path(root):
    return root / '.publication/state.json'


def load_state(root):
    identity = {'schema': 1, 'campaign': str(root),
                'plan_sha256': runner.file_hash(root / '.campaign/plan.json'),
                'policy_sha256': runner.file_hash(root / '.campaign' / POLICY_NAME)}
    if not state_path(root).exists():
        return {**identity, 'status': 'ready', 'active': False, 'published': {}, 'attempts': [], 'owned_runs': {}}
    state = read_json(state_path(root))
    if any(state.get(key) != value for key, value in identity.items()):
        raise ValueError('Publication state belongs to different frozen inputs')
    if not isinstance(state.get('published'), dict) or not isinstance(state.get('attempts'), list):
        raise ValueError('Invalid publication state')
    return state


def save_state(root, state):
    state['updated_utc'] = runner.now()
    runner.write_json(state_path(root), state)


def blockers(root, manifest, state, postprocess_only=False):
    result = []
    if (root / 'STOP').exists() and not postprocess_only:
        result.append('STOP is present; remove it only after explicit resume authorization')
    if (root / 'POSTPROCESS_STOP').exists():
        result.append('POSTPROCESS_STOP is present; explicit postprocessing resume is required')
    if manifest.get('active') or any(run['status'] == 'running' for run in manifest['runs']):
        result.append('Campaign active/running: verify its host PID/start identity before recovery')
    if state.get('active'):
        result.append('Publication wrapper active: verify its owner and child PID/start identity before recovery')
    checks = manifest.get('preparation_checks', {})
    if (checks.get('config_dry_runs') != len(manifest['runs']) or checks.get('solver_runs') != 0 or
            not checks.get('runtime_help_passed')):
        result.append('Campaign preparation has not passed')
    return result


def completed(root, manifest, state):
    result = []
    for run in manifest['runs']:
        if run['status'] != 'completed':
            if run['id'] in state['published']:
                raise ValueError('A previously published run is no longer completed')
            continue
        runner.validate_completed(root, manifest, run)
        digest = run['run_metadata_sha256']
        saved = state['published'].get(run['id'])
        if saved is not None and saved['run_metadata_sha256'] != digest:
            raise ValueError('Published run metadata changed: ' + run['id'])
        if saved is not None:
            receipt = saved['receipt']
            path = Path(receipt['path'])
            if (path.parent != root / '.deployment' or path.name != receipt['generation'] + '.json' or
                    not path.is_file() or path.is_symlink() or runner.file_hash(path) != receipt['sha256'] or
                    read_json(path).get('status') != 'verified'):
                raise ValueError('Durable publication receipt changed: ' + run['id'])
        result.append(run)
    return result


def script_command(root, name, *arguments):
    return [sys.executable, str(root / '.campaign' / name), *map(str, arguments)]


def run_command(root):
    return script_command(root, 'run_campaign.py', '--resume', root, '--start', '--run-limit', '1')


def movie_command(root, run, policy, postprocess_only=False):
    command = script_command(root, 'campaign_movies.py', '--campaign', root, '--run-id', run['id'])
    if postprocess_only:
        command.append('--postprocess-only')
    for key, value in policy['movie'].items():
        if key == 'full_domain':
            if value:
                command.append('--full-domain')
        else:
            command.extend(('--' + key.replace('_', '-'), str(value)))
    return command


def stage_path(root, attempt):
    if not re.fullmatch(r'[0-9a-f]{32}', attempt['id']):
        raise ValueError('Invalid generated stage identity')
    parent = root.parent / (root.name + '-publication-stages')
    if parent.is_symlink():
        raise ValueError('Stage parent must not be a symlink')
    return parent / attempt['id']


def inspect(root, postprocess_only=False):
    manifest = runner.load(root)
    verify_inputs(root, manifest, postprocess_only)
    policy, state = load_policy(root), load_state(root)
    done = completed(root, manifest, state)
    unpublished = [run for run in done if run['id'] not in state['published']]
    pending = [run['id'] for run in manifest['runs'] if run['status'] != 'completed']
    next_run = next((run for run in manifest['runs'] if run['status'] != 'completed'), None)
    raw_budget = policy['raw_run_budget_gib'].get(str(next_run['level'])) if next_run else 0
    storage = {'available_gib': shutil.disk_usage(root).free / 2**30,
               'reserve_gib': manifest['minimum_free_gib'], 'next_raw_run_budget_gib': raw_budget,
               'next_run_required_free_gib': None if raw_budget is None else raw_budget + manifest['minimum_free_gib'],
               'policy': 'Retain verified profiles, frames, full-volume integrals, initial/final 3D states and all checkpoints'}
    return {'campaign': str(root), 'read_only': True, 'explicit_start_required': True,
            'status': state['status'], 'postprocess_only': postprocess_only,
            'blocked_by': blockers(root, manifest, state, postprocess_only),
            'completed_unpublished': [run['id'] for run in unpublished], 'pending_runs': pending,
            'published_runs': list(state['published']), 'policy': policy,
            'storage': storage,
            'next_action': 'publish' if unpublished else ('waiting-for-remote' if postprocess_only else 'run') if pending else 'complete',
            'next_solver_command': run_command(root) if pending and not postprocess_only else None,
            'movie_commands': [movie_command(root, run, policy, postprocess_only) for run in unpublished],
            'sequence': 'Render and publish all completed unpublished results before the next single-run solver command'}


def process_table():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            result[int(path.parent.name)] = {'state': fields[0], 'parent': int(fields[1]),
                'session': int(fields[3]), 'start_ticks': fields[19],
                'rss': max(0, int(fields[21])) * os.sysconf('SC_PAGE_SIZE')}
        except (OSError, ValueError, IndexError):
            continue
    return result


def owned_processes(child_pid, sessions):
    # Both the campaign runner and movie tool create additional sessions for
    # their expensive children. Follow descendants, retaining discovered SIDs
    # so orphaned grandchildren remain supervised after a wrapper exits.
    table = process_table()
    members = {pid for pid, data in table.items() if data['session'] in sessions}
    # Linux subreaper adoption closes the race where an inner wrapper creates
    # a new session and exits before the first polling snapshot can see it.
    parent = getattr(sessions, 'adopted_parent', None)
    baseline = getattr(sessions, 'baseline', set())
    members.update(pid for pid, data in table.items() if data['parent'] == parent and
                   (pid, data.get('start_ticks')) not in baseline)
    if child_pid in table and table[child_pid]['session'] in sessions:
        members.add(child_pid)
    while True:
        added = {pid for pid, data in table.items() if data['parent'] in members} - members
        if not added:
            break
        members.update(added)
    for pid in members & table.keys():
        session = table[pid]['session']
        if session > 1 and session != os.getsid(0):
            sessions.add(session)
    return {pid: data for pid, data in table.items() if data['session'] in sessions}


class OwnedSessions(set):
    def __init__(self):
        super().__init__()
        self.adopted_parent = os.getpid()
        self.baseline = {(pid, data.get('start_ticks')) for pid, data in process_table().items()}


@contextmanager
def adopt_orphans():
    # This supervisor is Linux-specific already (/proc). Fail before launching
    # if the kernel cannot provide ownership of otherwise-unobservable orphans.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot read child-subreaper state')
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot supervise orphaned nested processes')
    try:
        yield OwnedSessions()
    finally:
        if libc.prctl(36, previous.value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot restore child-subreaper state')


def resource_check(root, manifest, extra_path=None, postprocess_only=False):
    paths = (root,) if extra_path is None else (root, extra_path)
    for path in paths:
        while not path.exists():
            path = path.parent
        if shutil.disk_usage(path).free < manifest['minimum_free_gib'] * 2**30:
            raise RuntimeError('Free storage fell below the saved campaign reserve')
    if (root / 'STOP').exists() and not postprocess_only:
        raise RuntimeError('Campaign STOP marker present')
    if (root / 'POSTPROCESS_STOP').exists():
        raise RuntimeError('Campaign POSTPROCESS_STOP marker present')


def drain_child(child, sessions):
    # Give Python wrappers time to cancel/reap their separately-sessioned work.
    if child.poll() is None:
        child.send_signal(signal.SIGINT)
    for signum, seconds in ((None, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        members = owned_processes(child.pid, sessions)
        if signum is not None:
            for session in sorted(sessions, key=lambda value: value == child.pid):
                runner.signal_session(session, signum)
        deadline = time.monotonic() + seconds
        while True:
            child.poll()
            members = owned_processes(child.pid, sessions)
            if child.returncode is not None and not any(p['state'] != 'Z' for p in members.values()):
                child.wait()
                for pid, process in members.items():
                    if pid != child.pid and process.get('parent') == os.getpid():
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(.05)
    raise RuntimeError('Owned child processes did not exit; verify host identities before recovery')


def phase(root, manifest, state, name, command, stage=None):
    postprocess_only = state.get('postprocess_only', False)
    if postprocess_only and name == 'solver':
        raise ValueError('Postprocess-only mode cannot launch a local solver')
    resource_check(root, manifest, stage, postprocess_only)
    verify_inputs(root, manifest, postprocess_only)
    state.update(phase=name, phase_started_utc=runner.now())
    state['phase_command'] = command
    save_state(root, state)
    log = root / '.publication/logs' / (uuid.uuid4().hex + '-' + name + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    environment.pop('LD_PRELOAD', None)
    # The frozen runner sets the solver's library path itself. VisIt and local
    # publication tools must use their own host runtime rather than HPX's libs.
    environment.pop('LD_LIBRARY_PATH', None)
    with log.open('w') as stream, adopt_orphans() as sessions:
        child = subprocess.Popen(command, cwd=root, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        sessions.add(child.pid)
        state['child'] = {'pid': child.pid, 'phase': name, 'log': str(log)}
        try:
            state['child']['start_ticks'] = process_table().get(child.pid, {}).get('start_ticks')
            save_state(root, state)
            while True:
                members = owned_processes(child.pid, sessions)
                if name == 'solver':
                    prove_run_ownership(root, state, members)
                if state['child'].get('sessions') != sorted(sessions):
                    state['child']['sessions'] = sorted(sessions)
                    save_state(root, state)
                resource_check(root, manifest, stage, postprocess_only)
                if sum(p['rss'] for p in members.values()) > manifest['memory_limit_gib'] * 2**30:
                    raise RuntimeError('Owned phase processes exceeded the saved memory budget')
                if child.poll() is not None:
                    break
                time.sleep(.25)
            if child.returncode:
                raise RuntimeError(f'{name} failed with exit status {child.returncode}; see {log}')
        finally:
            # A failed drain leaves child/active state intact; a later invocation
            # refuses to guess whether surviving work is safe to overlap.
            drain_child(child, sessions)
            state.pop('child', None)
            save_state(root, state)


def prove_run_ownership(root, state, members):
    for run in runner.load(root)['runs']:
        owner = state.get('owned_runs', {}).get(run['id'])
        if owner is None or not run.get('attempts'):
            continue
        attempt = run['attempts'][-1]
        pid, ticks = attempt.get('pid'), attempt.get('process_start_ticks')
        process = members.get(pid)
        if process is not None and ticks is not None and process.get('start_ticks') == ticks:
            identity = {'pid': pid, 'process_start_ticks': ticks}
            if owner.get('solver_identity') != identity:
                owner['solver_identity'] = identity
                save_state(root, state)


@contextmanager
def campaign_guard(root):
    # Exclude standalone campaign/movie edits while postprocessing. The stage
    # publisher acquires this lock itself, so its phase must not nest this guard.
    with (root / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = runner.load(root)
        if manifest.get('active') or any(run['status'] == 'running' for run in manifest['runs']):
            raise ValueError('Campaign became active; publication cannot overlap a solver')
        yield manifest


def verified_receipt(root, attempt):
    stage = stage_path(root, attempt)
    path = root / '.deployment/latest-receipt.json'
    if not path.is_file() or not (stage / 'publication-manifest.json').is_file():
        return None
    receipt = read_json(path)
    if (receipt.get('status') != 'verified' or Path(receipt.get('stage', '')).resolve() != stage or
            receipt.get('manifest_sha256') != runner.file_hash(stage / 'publication-manifest.json') or
            receipt.get('completed_runs') != len(attempt['runs'])):
        return None
    generation = receipt.get('generation', '')
    if not re.fullmatch(r'm1-fld-20261013-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}', generation):
        return None
    durable = root / '.deployment' / (generation + '.json')
    if not durable.is_file() or read_json(durable) != receipt:
        return None
    return {'generation': generation, 'path': str(durable), 'sha256': runner.file_hash(durable),
            'stage': str(stage), 'url': receipt.get('url')}


def prune_stages(root, state, keep):
    for attempt in state['attempts']:
        if attempt['id'] == keep:
            continue
        path = stage_path(root, attempt)
        if path.is_symlink():
            raise ValueError('Refusing to remove a symlinked generated stage')
        if path.exists():
            # Only exact UUID directories registered by this wrapper are removed.
            shutil.rmtree(path)
            attempt['stage_removed_utc'] = runner.now()


def retain_run(root, manifest, state, run):
    owner = state.get('owned_runs', {}).get(run['id'])
    if run.get('remote_import'):
        remote_owner = retention.remote_owner(root, run)
        if owner is None:
            owner = remote_owner
            state.setdefault('owned_runs', {})[run['id']] = owner
            save_state(root, state)
        elif any(owner.get(key) != value for key, value in remote_owner.items()):
            raise ValueError('Remote import ownership changed before retention')
    if owner is None:
        return  # Preexisting campaign outputs are never deletion candidates.
    attempt = run.get('attempts', [{}])[-1] if run.get('attempts') else {}
    identity = {'pid': attempt.get('pid'), 'process_start_ticks': attempt.get('process_start_ticks')}
    if not run.get('remote_import') and owner.get('solver_identity') != identity:
        raise ValueError('The completed attempt was not observed in this wrapper\'s owned process tree; raw cleanup requires review')
    if run.get('retention') and owner.get('retention_receipt') == run['retention'] and owner.get('retention_verified_utc'):
        return
    guard = lambda: resource_check(root, manifest, postprocess_only=state.get('postprocess_only', False))
    if not run.get('retention'):
        receipt = retention.make_receipt(root, manifest, run, owner, guard)
        path = retention.leaf(root, run) / 'retention.json'
        runner.write_json(path, receipt)
        run['retention'] = {'path': str(path.relative_to(root)), 'sha256': runner.file_hash(path)}
        # Commit the immutable receipt reference before the first unlink. A
        # failed deletion can then be resumed without inventing missing inputs.
        runner.save(root, manifest)
    reclaimed = retention.prune(root, run, guard)
    owner.update(retention_receipt=run['retention'], authorized_removal_bytes=reclaimed,
                 retention_verified_utc=runner.now())
    save_state(root, state)


def claim_next_run(root, manifest, state, policy):
    run = next(run for run in manifest['runs'] if run['status'] != 'completed')
    budget = policy['raw_run_budget_gib'].get(str(run['level']))
    if budget is None:
        raise ValueError('No frozen raw-storage estimate for this refinement level')
    if shutil.disk_usage(root).free < (manifest['minimum_free_gib'] + budget) * 2**30:
        raise RuntimeError('Insufficient free storage for the next raw run plus the campaign reserve')
    owned = state.setdefault('owned_runs', {})
    if run['id'] not in owned:
        folder = retention.leaf(root, run)
        raw = list(folder.glob('*.silo*')) if folder.exists() else []
        # A previously attempted run may contain outputs even without Silo;
        # do not adopt it as newly owned for deletion merely to resume it.
        if not raw and run['status'] == 'pending' and not run.get('attempts'):
            owned[run['id']] = {'directory': run['directory'], 'claimed_utc': runner.now()}
            save_state(root, state)


def publish_completed(root, manifest, state, policy, done):
    identities = {run['id']: run['run_metadata_sha256'] for run in done}
    # A deploy can finish immediately before a wrapper interruption. Its durable
    # verified receipt is sufficient to finish bookkeeping on an explicit resume.
    if state['attempts'] and state['attempts'][-1]['runs'] == identities:
        attempt = state['attempts'][-1]
        receipt = verified_receipt(root, attempt)
    else:
        receipt = None
    if receipt is None:
        for run in done:
            if run['id'] not in state['published']:
                with campaign_guard(root) as current:
                    phase(root, current, state, 'movie', movie_command(root, run, policy, state.get('postprocess_only', False)))
                    current_run = next(item for item in current['runs'] if item['id'] == run['id'])
                    retain_run(root, current, state, current_run)
        # Retain the last verified stage, reclaim failed/private attempts before
        # retrying so repeated publication failures cannot consume unbounded disk.
        keep = state.get('latest_verified_attempt')
        prune_stages(root, state, keep)
        attempt = {'id': uuid.uuid4().hex, 'runs': identities, 'created_utc': runner.now()}
        state['attempts'].append(attempt)
        save_state(root, state)
        stage = stage_path(root, attempt)
        phase(root, manifest, state, 'stage',
            script_command(root, 'publish_campaign.py', root, '--output', stage), stage)
        with campaign_guard(root) as current:
            # Check the result inventory again after staging acquired/released its
            # own campaign lock. Never deploy a stage while a solver is running.
            current_done = completed(root, current, state)
            if {run['id']: run['run_metadata_sha256'] for run in current_done} != identities:
                raise ValueError('Completed results changed during website staging')
            phase(root, current, state, 'deploy',
                script_command(root, 'deploy_campaign.py', root, '--staged-site', stage, '--deploy'), stage)
        receipt = verified_receipt(root, attempt)
        if receipt is None:
            raise RuntimeError('Deployment did not leave a matching durable verified receipt')
    attempt.update(status='verified', receipt=receipt, verified_utc=runner.now())
    for run in done:
        state['published'][run['id']] = {'run_metadata_sha256': run['run_metadata_sha256'], 'receipt': receipt}
    state['latest_verified_attempt'] = attempt['id']
    save_state(root, state)
    prune_stages(root, state, attempt['id'])
    save_state(root, state)


def start(root, postprocess_only=False):
    # Separate lock spans solver, renderer, staging and upload. Individual tools
    # retain their existing campaign lock; no nested acquisition around them.
    with (root / '.publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = runner.load(root)
        verify_inputs(root, manifest, postprocess_only)
        policy, state = load_policy(root), load_state(root)
        reasons = blockers(root, manifest, state, postprocess_only)
        if reasons:
            raise ValueError('; '.join(reasons))
        resource_check(root, manifest, postprocess_only=postprocess_only)
        state.update(active=True, status='running', postprocess_only=postprocess_only, owner={'pid': os.getpid(),
            'start_ticks': process_table().get(os.getpid(), {}).get('start_ticks')})
        save_state(root, state)
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        for sig in handlers:
            signal.signal(sig, interrupted)
        try:
            while True:
                manifest = runner.load(root)
                verify_inputs(root, manifest, postprocess_only)
                resource_check(root, manifest, postprocess_only=postprocess_only)
                if manifest.get('active') or any(run['status'] == 'running' for run in manifest['runs']):
                    raise ValueError('Campaign became active; verify its host identity before recovery')
                done = completed(root, manifest, state)
                for run in done:
                    if run.get('retention') and run['id'] in state.get('owned_runs', {}):
                        with campaign_guard(root) as current:
                            retain_run(root, current, state,
                                next(item for item in current['runs'] if item['id'] == run['id']))
                if any(run['id'] not in state['published'] for run in done):
                    publish_completed(root, manifest, state, policy, done)
                    continue
                if len(done) == len(manifest['runs']):
                    state['status'] = 'completed'
                    return 0
                if postprocess_only:
                    state['status'] = 'waiting-for-remote'
                    return 0
                before = {run['id'] for run in done}
                claim_next_run(root, manifest, state, policy)
                phase(root, manifest, state, 'solver', run_command(root))
                current = runner.load(root)
                after = {run['id'] for run in completed(root, current, state)}
                if not before < after or len(after - before) != 1:
                    raise RuntimeError('Single-run solver did not complete exactly one new result')
        except KeyboardInterrupt:
            marker = 'POSTPROCESS_STOP' if postprocess_only else 'STOP'
            runner.atomic_text(root / marker, 'Publication workflow interrupted; requires explicit resume.\n')
            state.update(status='stopped', error='Interrupted; no additional run started')
            return 1
        except Exception as error:
            state.update(status='paused', error=str(error))
            raise
        finally:
            if 'child' not in state:
                state['active'] = False
            save_state(root, state)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--postprocess-only', action='store_true',
        help='Publish completed results only; retain solver STOP and use POSTPROCESS_STOP for reporting cancellation')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--start', action='store_true', help='Explicitly authorize solver, movie and website publication phases')
    action.add_argument('--plan', action='store_true', help='Read-only next actions and frozen policy (default)')
    action.add_argument('--status', action='store_true', help='Read-only durable progress and blockers')
    args = parser.parse_args(argv)
    root = args.campaign.resolve()
    if args.start:
        return start(root, args.postprocess_only)
    print(json.dumps(inspect(root, args.postprocess_only), indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
