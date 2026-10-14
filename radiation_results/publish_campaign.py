#!/usr/bin/env python3
"""Stage a verified M1 RK1 / FLD website in a new local directory.

This command does not start simulations or deploy to a server. It excludes
checkpoints, runtime binaries and private campaign scratch. Remote publication
is a separate explicitly authorized action. Historical ADER sites are not
modified or relabeled.
"""
import argparse
from datetime import datetime, timezone
import fcntl
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import unquote, urlsplit

import compare_campaign
from run_campaign import file_hash, load, validate_completed, verify_snapshot, write_json

PUBLIC_EXTENSIONS = {'.html', '.css', '.js', '.png', '.jpg', '.svg', '.pdf', '.csv', '.json', '.txt', '.log', '.ini', '.mp4', '.dat'}


def local_only_path(path):
    return any(part.lower().endswith(('.silo', '.silo.data')) or
               part.lower().startswith('checkpoint') for part in Path(path).parts)


class PublicHTML(HTMLParser):
    """Preserve report markup, replace excluded downloads, and collect links."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.output, self.links, self.anchors = [], [], []
        self.identifiers = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        href = urlsplit(values.get('href') or '')
        excluded = tag == 'a' and not href.scheme and not href.netloc and local_only_path(unquote(href.path))
        if tag == 'a':
            self.anchors.append(excluded)
        if excluded:
            self.output.append('<span class="local-only" title="Retained in the local campaign; excluded from the staged site">')
        else:
            self.output.append(self.get_starttag_text())
            self.links.extend(value for name, value in attrs
                              if name in ('href', 'src', 'poster', 'xlink:href') and value)
        if values.get('id'):
            self.identifiers.add(values['id'])
        if tag == 'a' and values.get('name'):
            self.identifiers.add(values['name'])

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag == 'a':
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == 'a' and self.anchors:
            excluded = self.anchors.pop()
            self.output.append(' (local file; excluded from staged site)</span>' if excluded else '</a>')
        else:
            self.output.append(f'</{tag}>')

    def handle_data(self, value):
        self.output.append(value)

    def handle_entityref(self, value):
        self.output.append('&' + value + ';')

    def handle_charref(self, value):
        self.output.append('&#' + value + ';')

    def handle_comment(self, value):
        self.output.append('<!--' + value + '-->')

    def handle_decl(self, value):
        self.output.append('<!' + value + '>')


def prepare_html_and_validate_links(site):
    pages = {}
    for path in site.rglob('*.html'):
        parser = PublicHTML()
        parser.feed(path.read_text()); parser.close()
        path.write_text(''.join(parser.output))
        pages[path.resolve()] = parser
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                if url.scheme not in ('', 'http', 'https', 'mailto', 'tel', 'data'):
                    raise ValueError(f'Unsupported public link in {path.relative_to(site)}: {link}')
                continue
            decoded = unquote(url.path)
            target = ((site / decoded.lstrip('/')) if decoded.startswith('/') else
                      (path.parent / decoded) if decoded else path).resolve()
            if not target.is_relative_to(site):
                raise ValueError(f'Public link escapes the staged site: {link}')
            if target.is_dir():
                target /= 'index.html'
            if not target.is_file():
                raise ValueError(f'Broken public link in {path.relative_to(site)}: {link}')
            if url.fragment and target in pages and unquote(url.fragment) not in pages[target].identifiers:
                raise ValueError(f'Broken public anchor in {path.relative_to(site)}: {link}')


def build_site(root, destination):
    if Path(destination).is_symlink():
        raise ValueError('The staged site destination must not be a symlink')
    root, destination = Path(root).resolve(), Path(destination).resolve()
    with (root / '.campaign.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Campaign is locked by another operation; stage after it has stopped') from error
        return build_site_locked(root, destination)


def build_site_locked(root, destination):
    manifest = load(root)
    verify_snapshot(root, manifest)
    compare_campaign.validate_manifest(manifest)
    if manifest.get('active'):
        raise ValueError('Stage a website only after the current run has stopped')
    completed = [r for r in manifest['runs'] if r['status'] == 'completed']
    if not completed:
        raise ValueError('No completed runs to stage; the prepared index remains a local preview')
    for run in completed:
        meta = validate_completed(root, manifest, run)
        if 'fixture' in meta.get('origin', '').lower() or meta.get('synthetic', False):
            raise ValueError('Synthetic test fixtures must not be published as solver evidence')
    summary = compare_campaign.refresh(root, manifest)
    if any(r['status'] in ('incompatible', 'invalid_output')
           for records in summary['cases'].values() for r in records):
        raise ValueError('Invalid or incompatible comparison output prevents website staging')
    if destination == root or destination.is_relative_to(root):
        raise ValueError('The staged site must be separate from the campaign')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use a new or empty local website directory')
    def copy(source, target):
        if source.is_symlink():
            raise ValueError('Unexpected symlink in public results')
        if local_only_path(source.relative_to(root)):
            return
        if source.is_dir():
            for child in source.iterdir():
                if not child.name.startswith('.'):
                    copy(child, target / child.name)
        elif source.suffix.lower() in PUBLIC_EXTENSIONS or source.name == 'reference.bin':
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + destination.name + '-staging-', dir=destination.parent) as temporary:
        stage = Path(temporary)
        for name in ('index.html', 'campaign.json', 'commands.txt', 'comparisons'):
            source = root / name
            if source.exists():
                copy(source, stage / name)
        for run in completed:
            copy(root / run['directory'], stage / run['directory'])
        prepare_html_and_validate_links(stage)
        files = {str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        receipt = {'schema': 2, 'workflow': 'm1-fld-radiation', 'status': 'staged_locally',
                   'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(root),
                   'destination': str(destination), 'completed_runs': len(completed),
                   'remote_deployment': False, 'file_sha256': files, 'local_links_verified': True}
        write_json(stage / 'site-manifest.json', receipt)
        # Publish the complete checked directory at once. A concurrently added
        # destination file makes rmdir fail instead of overwriting user content.
        if destination.exists():
            destination.rmdir()
        os.rename(stage, destination)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    print(json.dumps(build_site(args.campaign, args.output), indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
