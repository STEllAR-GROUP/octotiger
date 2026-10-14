#!/usr/bin/env python3
"""Prepare or explicitly deploy the current M1/FLD website, retaining history.

--prepare stages locally without a network call. --inspect is remote read-only.
--deploy is the explicit upload/activation action and requires completed output.
--archive-pilot preserves the verified initial pilot at a separate public URL.
No mode starts a solver or movie renderer.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from urllib.request import urlopen

from publish_campaign import build_site
from run_campaign import file_hash, write_json

HOST = 'dmarce1@129.212.177.35'
WEBROOT = '/var/www/octotiger'
SITE = 'https://www.octotiger.org'
SLUG = 'm1-fld-20261013'
PILOT_SLUG = 'm1-fld-pilot-20261013'
PILOT_GENERATION = 'm1-fld-20261013-20261014T032248Z-3877e90e'
ORIGINAL_HOMEPAGE_SHA256 = 'a9bd1d8e0dacc787938e6e15416bdb4873cf6538079570c7dc028d2cb9543aad'
RESERVE_BYTES = 2 * 1024**3

# The same remote lock is used by the historical publisher. Its receipt,
# campaign pointer, generations, and all existing result files remain intact.
REMOTE = r'''
import fcntl, hashlib, json, os, pathlib, re, shutil, sys

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)

def run(p, root):
    slug = 'm1-fld-20261013'
    require(p.get('slug') == slug, 'Unexpected campaign slug')
    base = root / 'campaigns'
    stable = base / slug
    receipt_path = base / ('.' + slug + '-receipt.json')
    if p['action'] == 'inspect':
        return {'homepage_sha256': sha(root / 'index.html'),
            'receipt': json.loads(receipt_path.read_text()) if receipt_path.exists() else None,
            'stable': str(stable.resolve()) if stable.is_symlink() else None,
            'pilot_archive': str((base / 'm1-fld-pilot-20261013').resolve())
                if (base / 'm1-fld-pilot-20261013').is_symlink() else None,
            'campaign_exists': stable.exists() or stable.is_symlink(),
            'free_bytes': shutil.disk_usage(root).free}
    if p['action'] == 'archive-pilot':
        pilot_generation = 'm1-fld-20261013-20261014T032248Z-3877e90e'
        require(p.get('pilot_generation') == pilot_generation, 'Unexpected pilot generation')
        target = base / '.generations' / pilot_generation
        alias = base / 'm1-fld-pilot-20261013'
        with (base / '.publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(target.is_dir() and not target.is_symlink(), 'Verified pilot generation is missing')
            pilot = json.loads((target / 'publication-receipt.json').read_text())
            require(pilot.get('status') == 'verified' and pilot.get('generation') == pilot_generation and
                    pilot.get('generation_path') == str(target) and pilot.get('completed_runs') == 8,
                    'Pilot receipt does not identify the verified eight-run generation')
            require(sha(target / 'publication-manifest.json') == pilot['manifest_sha256'] and
                    sha(target / '.homepage.html') == pilot['homepage_sha256'], 'Pilot publication changed')
            manifest = json.loads((target / 'publication-manifest.json').read_text())
            require(manifest.get('generation') == pilot_generation and manifest.get('campaign') == slug,
                    'Pilot publication identity changed')
            for relative, info in manifest['files'].items():
                path = target / relative
                require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(target.resolve()) and
                        path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                        'Pilot file changed: ' + relative)
            exists = alias.exists() or alias.is_symlink()
            if exists:
                require(alias.is_symlink() and alias.resolve() == target, 'Refusing to overwrite an existing pilot archive')
            else:
                current = json.loads(receipt_path.read_text())
                require(current.get('status') == 'verified' and current.get('generation') == pilot_generation and
                        stable.is_symlink() and stable.resolve() == target and
                        sha(root / 'index.html') == pilot['homepage_sha256'],
                        'Current website no longer matches the verified pilot; archive was not created')
                # symlink creation is atomic and fails if another writer created
                # this name; it never replaces a directory, file, or foreign link.
                os.symlink(os.path.relpath(target, base), alias)
            return {'status': 'already_archived' if exists else 'archived',
                    'generation': pilot_generation, 'generation_path': str(target), 'archive_path': str(alias),
                    'index_sha256': manifest['files']['index.html']['sha256'],
                    'manifest_sha256': pilot['manifest_sha256'],
                    'url': 'https://www.octotiger.org/campaigns/m1-fld-pilot-20261013/'}
    generation = p['generation']
    require(re.fullmatch(slug + r'-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}', generation), 'Invalid generation')
    incoming = base / '.incoming' / generation
    target = base / '.generations' / generation
    archive = root / 'archive' / ('pre-' + slug)

    def finish(receipt):
        previous = receipt['previous_generation_path']
        allowed = [pathlib.Path(receipt['generation_path'])]
        if previous:
            allowed.append(pathlib.Path(previous))
        require(sha(root / 'index.html') in (receipt['previous_homepage_sha256'], receipt['homepage_sha256']),
            'Homepage changed during activation; retaining it')
        require((not stable.exists() and not stable.is_symlink()) or
            (stable.is_symlink() and stable.resolve() in allowed), 'Campaign pointer changed during activation')
        destination = pathlib.Path(receipt['generation_path'])
        page = root / ('.index-' + receipt['generation'] + '.html')
        shutil.copyfile(destination / '.homepage.html', page)
        page.chmod(0o644)
        require(sha(page) == receipt['homepage_sha256'], 'Saved homepage changed')
        link = base / ('.link-' + receipt['generation'])
        if link.is_symlink():
            link.unlink()
        os.symlink(os.path.relpath(destination, base), link)
        os.replace(link, stable)
        os.replace(page, root / 'index.html')
        receipt['status'] = 'activated'
        write(receipt_path, receipt)
        write(destination / 'publication-receipt.json', receipt)
        return receipt

    def state():
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if receipt and receipt['status'] == 'activating':
            receipt = finish(receipt)
        expected = receipt['homepage_sha256'] if receipt else p['original_homepage_sha256']
        require(sha(root / 'index.html') == expected, 'Homepage changed outside this publisher; retaining it')
        if receipt:
            require(stable.is_symlink() and stable.resolve() == pathlib.Path(receipt['generation_path']),
                'Campaign pointer changed outside this publisher')
        else:
            require(not stable.exists() and not stable.is_symlink(), 'Unexpected existing campaign path')
        return receipt

    with (base / '.publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = state()
        if p['action'] == 'prepare':
            require(shutil.disk_usage(root).free >= p['bytes'] + p['reserve_bytes'], 'Insufficient remote storage')
            require(not incoming.exists() and not target.exists(), 'Generation already exists')
            incoming.mkdir(parents=True)
            return {'incoming': str(incoming), 'previous_generation': receipt['generation_path'] if receipt else None}
        if p['action'] == 'activate':
            require((receipt['generation_path'] if receipt else None) == p['previous_generation'],
                'Another publisher advanced the campaign during upload')
            require(sha(incoming / 'publication-manifest.json') == p['manifest_sha256'], 'Manifest changed')
            manifest = json.loads((incoming / 'publication-manifest.json').read_text())
            require(manifest['schema'] == 2 and manifest['campaign'] == slug and manifest['generation'] == generation,
                'Manifest identity mismatch')
            require(manifest['completed_runs'] >= 1, 'No completed solver output')
            for relative, info in manifest['files'].items():
                path = incoming / relative
                require(not path.is_symlink() and path.resolve().is_relative_to(incoming.resolve()) and path.is_file(),
                    'Invalid public file path: ' + relative)
                require(path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                    'Uploaded file changed: ' + relative)
            require('index.html' in manifest['files'] and '.homepage.html' in manifest['files'], 'Missing public entry points')
            if not receipt:
                archive.mkdir(parents=True, exist_ok=True)
                if not (archive / 'index.html').exists():
                    shutil.copy2(root / 'index.html', archive / 'index.html')
                require(sha(archive / 'index.html') == p['original_homepage_sha256'], 'Homepage archive differs')
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), 'Immutable generation already exists')
            os.rename(incoming, target)
            result = {'status': 'activating', 'generation': generation, 'generation_path': str(target),
                'completed_runs': manifest['completed_runs'], 'homepage_sha256': sha(target / '.homepage.html'),
                'previous_homepage_sha256': sha(root / 'index.html'),
                'previous_generation_path': receipt['generation_path'] if receipt else None,
                'manifest_sha256': sha(target / 'publication-manifest.json'), 'archive': str(archive),
                'url': 'https://www.octotiger.org/campaigns/' + slug + '/'}
            write(receipt_path, result)
            return finish(result)
        if p['action'] == 'verified':
            require(receipt and receipt['generation'] == generation, 'Verification refers to an old generation')
            receipt.update(status='verified', verified_utc=p['verified_utc'])
            write(receipt_path, receipt)
            write(pathlib.Path(receipt['generation_path']) / 'publication-receipt.json', receipt)
            return receipt
        raise ValueError('Unknown publication action')

if __name__ == '__main__':
    print(json.dumps(run(json.load(sys.stdin), pathlib.Path('/var/www/octotiger'))))
'''


def remote(action, generation=None, **values):
    request = {'action': action, 'slug': SLUG, 'generation': generation,
               'original_homepage_sha256': ORIGINAL_HOMEPAGE_SHA256, **values}
    result = subprocess.run(['ssh', '-oBatchMode=yes', '-oConnectTimeout=15', HOST,
        'python3 -c ' + shlex.quote(REMOTE)], input=json.dumps(request), text=True,
        capture_output=True, check=True, timeout=120)
    return json.loads(result.stdout)


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, stage, generation):
    build_site(root, stage)
    return prepare_staged(stage, generation)


def prepare_staged(stage, generation):
    receipt = json.loads((stage / 'site-manifest.json').read_text())
    if (receipt.get('schema') != 2 or receipt.get('workflow') != 'm1-fld-radiation' or
            receipt.get('status') != 'staged_locally' or not receipt.get('local_links_verified') or
            receipt.get('completed_runs', 0) < 1):
        raise ValueError('A verified local M1/FLD site with completed results is required')
    actual = {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
    if actual != set(receipt['file_sha256']) | {'site-manifest.json'}:
        raise ValueError('Staged file inventory changed')
    for relative, digest in receipt['file_sha256'].items():
        path = stage / relative
        if path.is_symlink() or not path.resolve().is_relative_to(stage) or file_hash(path) != digest:
            raise ValueError('Staged file changed: ' + relative)
    history = ('<aside class="note"><strong>Current M1 RK1 and FLD results</strong><p>'
        'Results appear as runs complete; pending cases are not claimed as validated. '
        f'<a href="{SITE}/campaigns/{PILOT_SLUG}/">Initial M1/FLD pilot (three stored states per run)</a> · '
        f'<a href="{SITE}/campaigns/local-20261011/">Historical Euler/ADER2 campaign</a> · '
        f'<a href="{SITE}/archive/pre-{SLUG}/">Previous homepage</a></p></aside>')
    index = (stage / 'index.html').read_text()
    if '<main>' not in index or re.search(r'<base\b', index, re.I):
        raise ValueError('Unexpected campaign homepage structure')
    index = index.replace('<main>', '<main>' + history, 1)
    (stage / 'index.html').write_text(index)
    base = f'<base href="/campaigns/{SLUG}/">'
    opening = r'(<head\b[^>]*>)' if re.search(r'<head\b', index, re.I) else r'(<html\b[^>]*>)'
    homepage, replaced = re.subn(opening, lambda match: match.group(1) + base, index, count=1, flags=re.I)
    if replaced != 1:
        raise ValueError('Missing campaign homepage root element')
    (stage / '.homepage.html').write_text(homepage)
    receipt['file_sha256'] = {str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name != 'site-manifest.json'}
    receipt.update(campaign=SLUG, generation=generation)
    write_json(stage / 'site-manifest.json', receipt)
    files = {str(p.relative_to(stage)): {'sha256': file_hash(p), 'bytes': p.stat().st_size}
             for p in sorted(stage.rglob('*')) if p.is_file()}
    manifest = {'schema': 2, 'workflow': 'm1-fld-radiation', 'campaign': SLUG, 'generation': generation,
                'completed_runs': receipt['completed_runs'], 'created_utc': now(), 'files': files}
    write_json(stage / 'publication-manifest.json', manifest)
    return manifest


def verify_https(path, expected, generation):
    address = SITE + path + '?generation=' + generation
    with urlopen(address, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('HTTPS content hash mismatch: ' + address)
    return {'url': address, 'sha256': expected, 'bytes': len(content)}


def deploy(root, stage, generation, manifest):
    private = root / '.deployment'
    private.mkdir(exist_ok=True)
    receipt_path = private / (generation + '.json')
    result = {'status': 'staged', 'generation': generation, 'stage': str(stage),
              'completed_runs': manifest['completed_runs'], 'url': SITE + '/campaigns/' + SLUG + '/'}
    write_json(receipt_path, result)
    try:
        prepared = remote('prepare', generation, bytes=sum(item['bytes'] for item in manifest['files'].values()),
                          reserve_bytes=RESERVE_BYTES)
        subprocess.run(['rsync', '-a', '--checksum', '--no-owner', '--no-group', '--chmod=D755,F644',
            '-e', 'ssh -oBatchMode=yes -oConnectTimeout=15', str(stage) + '/',
            HOST + ':' + prepared['incoming'] + '/'], check=True, timeout=900)
        result.update(remote('activate', generation, previous_generation=prepared['previous_generation'],
                             manifest_sha256=file_hash(stage / 'publication-manifest.json')))
        result['remote_activation'] = 'activated'
        write_json(receipt_path, result)
        result['https'] = [verify_https('/', result['homepage_sha256'], generation),
            verify_https('/campaigns/' + SLUG + '/', manifest['files']['index.html']['sha256'], generation),
            verify_https('/campaigns/' + SLUG + '/publication-manifest.json', result['manifest_sha256'], generation)]
        result.update(remote('verified', generation, verified_utc=now()))
        write_json(private / 'latest-receipt.json', result)
    except Exception as error:
        result.update(status='failed', error=str(error))
        write_json(receipt_path, result)
        raise
    write_json(receipt_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', nargs='?', type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--deploy', action='store_true')
    action.add_argument('--inspect', action='store_true')
    action.add_argument('--archive-pilot', action='store_true',
        help='Create or verify the public archive alias for the exact initial eight-run pilot generation')
    staging = parser.add_mutually_exclusive_group()
    staging.add_argument('--stage', type=Path)
    staging.add_argument('--staged-site', type=Path, help='Use an already checked local website without accessing live campaign results')
    args = parser.parse_args(argv)
    if args.inspect:
        print(json.dumps(remote('inspect'), indent=2)); return 0
    if args.archive_pilot:
        result = remote('archive-pilot', pilot_generation=PILOT_GENERATION)
        result['https'] = [verify_https('/campaigns/' + PILOT_SLUG + '/', result['index_sha256'], PILOT_GENERATION),
            verify_https('/campaigns/' + PILOT_SLUG + '/publication-manifest.json', result['manifest_sha256'], PILOT_GENERATION)]
        print(json.dumps(result, indent=2)); return 0
    if args.campaign is None:
        parser.error('A completed campaign is required for preparation/deployment')
    root = args.campaign.resolve()
    generation = SLUG + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + os.urandom(4).hex()
    stage = args.staged_site.resolve() if args.staged_site else args.stage.resolve() if args.stage else root.parent / (root.name + '-website') / generation
    if args.staged_site:
        receipt = json.loads((stage / 'site-manifest.json').read_text())
        if Path(receipt['source']).resolve() != root:
            raise ValueError('Staged website belongs to a different campaign')
        manifest = prepare_staged(stage, generation)
    else:
        manifest = prepare(root, stage, generation)
    result = deploy(root, stage, generation, manifest) if args.deploy else {
        'status': 'prepared_locally', 'generation': generation, 'stage': str(stage),
        'completed_runs': manifest['completed_runs'], 'remote_deployment': False}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
