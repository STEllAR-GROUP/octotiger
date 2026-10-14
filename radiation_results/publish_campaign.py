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
from html import escape
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import unquote, urlsplit

import compare_campaign
from campaign_movies import MOVIE_SUBDIR, model_contract
from run_campaign import file_hash, load, validate_completed, verify_snapshot, write_json

PUBLIC_EXTENSIONS = {'.html', '.css', '.js', '.png', '.jpg', '.svg', '.pdf', '.csv', '.json', '.txt', '.log', '.ini', '.mp4', '.dat'}


def completed_movies(root, runs):
    """Only publish complete movie assets tied to the immutable saved run."""
    result = []
    for run in runs:
        folder = Path(run['directory']) / run['case'] / f"l{run['level']}"
        relative = folder / MOVIE_SUBDIR
        source = root / relative
        for path in (source, *source.parents):
            if path == root:
                break
            if path.is_symlink():
                raise ValueError('Unexpected symlink in movie results')
        manifest_path = source / 'campaign-movie.json'
        if not manifest_path.exists():
            if manifest_path.is_symlink():
                raise ValueError('Unexpected symlink in movie results')
            continue
        if manifest_path.is_symlink():
            raise ValueError('Unexpected symlink in movie results')
        movie = json.loads(manifest_path.read_text())
        metadata_path = root / folder / 'run.json'
        metadata = json.loads(metadata_path.read_text())
        contract = model_contract(metadata, run)
        digest = file_hash(metadata_path)
        if (movie.get('schema') != 2 or movie.get('status') != 'complete' or
                movie.get('run_id') != run['id'] or movie.get('case') != run['case'] or
                movie.get('level') != run['level'] or movie.get('method') != run['method'] or
                movie.get('run_metadata_sha256') != digest or run.get('run_metadata_sha256') != digest):
            raise ValueError(f"Movie does not match the completed run: {run['id']}")
        for key in ('radiation_model', 'radiation_integrator', 'radiation_algorithm_version',
                    'radiation_flux_semantics', 'reference_kind', 'reference_model', 'reference_exact_for_solver'):
            if movie.get(key) != contract.get(key):
                raise ValueError(f"Movie model/reference identity changed: {run['id']}: {key}")
        times = movie.get('snapshot_times')
        count, shown = movie.get('source_snapshots'), movie.get('displayed_snapshots')
        if (type(count) is not int or count < 1 or movie.get('frames') != count or
                type(shown) is not int or not 1 <= shown <= count or
                not isinstance(times, list) or len(times) != count or
                any(type(t) not in (int, float) or not math.isfinite(t) for t in times) or
                any(b <= a for a, b in zip(times, times[1:])) or
                'stored states held without solution interpolation' not in movie.get('timing', '')):
            raise ValueError(f"Invalid stored-state movie timing: {run['id']}")
        for name in ('movie.mp4', 'poster.png'):
            path, asset = source / name, movie.get('assets', {}).get(name, {})
            if (path.is_symlink() or not path.is_file() or
                    type(asset.get('bytes')) is not int or asset['bytes'] <= 0 or
                    path.stat().st_size != asset['bytes'] or file_hash(path) != asset.get('sha256')):
                raise ValueError(f"Movie asset is missing or changed: {run['id']}: {name}")
        result.append({'run': run, 'relative': relative, 'metadata': movie, 'contract': contract})
    return result


def relative_url(target, page):
    return Path(os.path.relpath(target, page.parent)).as_posix()


def movie_card(movie, page):
    run, metadata, contract = movie['run'], movie['metadata'], movie['contract']
    url = lambda name: escape(relative_url(movie['relative'] / name, page), quote=True)
    report = escape(relative_url(Path(run['directory']) / (run['case'] + '.html'), page), quote=True)
    title = f"{run['case'].replace('_', ' ').title()} · L{run['level']} · {contract['model_label']}"
    times = ', '.join(f'{value:.9g}' for value in metadata['snapshot_times'])
    description = str(metadata.get('description', ''))
    if metadata.get('reference_description'):
        description = description.replace(metadata['reference_description'], contract['reference_description'])
    return (f'<article class="campaign-movie" id="movie-{escape(run["id"], quote=True)}">'
        f'<h3>{escape(title)}</h3><video controls preload="none" playsinline poster="{url("poster.png")}" '
        f'style="display:block;width:100%;max-height:720px;background:#0c1820">'
        f'<source src="{url("movie.mp4")}" type="video/mp4">'
        f'<a href="{url("movie.mp4")}">Download MP4</a></video>'
        f'<p><strong>{metadata["source_snapshots"]} stored states; {metadata["displayed_snapshots"]} shown. '
        'Zero-order hold; no solution interpolation.</strong> Playback holds each saved numerical state '
        'over its time interval and includes endpoint holds.</p>'
        f'<p>Snapshot times: {escape(times)} {escape(str(metadata.get("time_units", "saved simulation units")))}.</p>'
        f'<p>{escape(contract["flux_description"])} Arrows have equal length and show direction only.</p>'
        f'<p>{escape(contract["reference_description"])}</p>'
        f'<p>{escape(description)}</p>'
        f'<p><a href="{url("movie.mp4")}">Download MP4</a> · '
        f'<a href="{url("campaign-movie.json")}">Rendering and timing metadata</a> · '
        f'<a href="{report}">Full run report</a></p></article>')


def insert_content(path, content):
    text = path.read_text()
    if '<main>' in text:
        start = text.index('<main>') + len('<main>')
        title = text.find('</h1>', start)
        position = title + len('</h1>') if title != -1 else start
        text = text[:position] + content + text[position:]
    elif '</body>' in text:
        text = text.replace('</body>', content + '</body>', 1)
    else:
        text += content
    path.write_text(text)


def add_movie_pages(root, stage, movies, completed):
    """Add report-only views in the stage, never rewrite solver metadata."""
    if not movies:
        return
    gallery = Path('movies/index.html')
    cards, entries = [], []
    for movie in movies:
        run, relative = movie['run'], movie['relative']
        destination = stage / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('movie.mp4', 'poster.png', 'campaign-movie.json'):
            shutil.copy2(root / relative / name, destination / name)
            if name != 'campaign-movie.json' and file_hash(destination / name) != movie['metadata']['assets'][name]['sha256']:
                raise ValueError(f'Movie asset changed during staging: {run["id"]}: {name}')
            if name == 'campaign-movie.json' and json.loads((destination / name).read_text()) != movie['metadata']:
                raise ValueError(f'Movie manifest changed during staging: {run["id"]}')
        cards.append(movie_card(movie, gallery))
        title = f"{run['case'].replace('_', ' ').title()} · L{run['level']} · {movie['contract']['model_label']}"
        anchor = 'movie-' + run['id']
        entries.append(f'<li><a href="#{escape(anchor, quote=True)}">{escape(title)}</a></li>')
        for filename in ('index.html', run['case'] + '.html'):
            page = Path(run['directory']) / filename
            target = stage / page
            if not target.is_file():
                continue
            gallery_link = escape(relative_url(gallery, page) + '#' + anchor, quote=True)
            content = ('<section class="section"><h2>Saved-state movie</h2>' + movie_card(movie, page) +
                       f'<p><a href="{gallery_link}">All campaign movies</a></p></section>')
            placeholder = '<h4>Movies</h4><p class="muted">No completed movie is available for this resolution.</p>'
            text = target.read_text()
            if filename != 'index.html' and placeholder in text:
                target.write_text(text.replace(placeholder, content, 1))
            else:
                insert_content(target, content)
    pending = len(completed) - len(movies)
    status = f'{len(movies)} movies available from {len(completed)} completed runs.'
    if pending:
        status += f' {pending} completed runs do not yet have a published movie.'
    sparse = (' With only three stored states per pilot run, playback is a sparse record of the evolution.'
              if all(movie['metadata']['source_snapshots'] == 3 for movie in movies) else '')
    (stage / gallery).parent.mkdir(parents=True, exist_ok=True)
    (stage / gallery).write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 RK1 and FLD movies</title>'
        '<style>body{font:16px/1.6 system-ui;color:#16303b;background:#eff4f5;margin:0}'
        'main{max-width:1100px;margin:2em auto;background:white;padding:2em}a{color:#067078}'
        'article{padding:1em 0 2em;border-top:1px solid #ccd;scroll-margin-top:1em}'
        'nav ul{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.5em 2em}'
        '</style></head><body><main><p><a href="../index.html">Campaign results</a> · '
        '<a href="../comparisons/index.html">Model comparisons</a></p>'
        f'<h1>M1 RK1 and FLD movies</h1><p>{status}</p>'
        f'<p>These movies show saved numerical states.{sparse} Intermediate solution states are not reconstructed.</p>'
        '<nav aria-label="Movies"><ul>' + ''.join(entries) + '</ul></nav>' + ''.join(cards) +
        '</main></body></html>')
    previews = []
    for movie in movies:
        run = movie['run']
        title = f"{run['case'].replace('_', ' ').title()} · L{run['level']} · {movie['contract']['model_label']}"
        previews.append(f'<a href="movies/index.html#movie-{escape(run["id"], quote=True)}" '
            'style="display:block;color:inherit;text-decoration:none">'
            f'<img src="{escape((movie["relative"] / "poster.png").as_posix(), quote=True)}" '
            f'alt="{escape(title, quote=True)} movie poster" loading="lazy" '
            'style="display:block;width:100%;height:auto;border:1px solid #ccd">'
            f'<span>{escape(title)}</span></a>')
    insert_content(stage / 'index.html', '<section aria-label="Campaign movies"><h2>Movies</h2>'
        f'<p>{status} <a href="movies/index.html">Open the movie gallery</a>.</p>'
        '<p>Saved states are shown with zero-order hold and no solution interpolation.</p>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1em">' +
        ''.join(previews) + '</div></section>')
    for case in {movie['run']['case'] for movie in movies}:
        page = Path('comparisons') / (case + '.html')
        if (stage / page).is_file():
            links = [f'<li><a href="../movies/index.html#movie-{escape(movie["run"]["id"], quote=True)}">'
                     f'L{movie["run"]["level"]} · {escape(movie["contract"]["model_label"])}</a></li>'
                     for movie in movies if movie['run']['case'] == case]
            insert_content(stage / page, '<section><h2>Saved-state movies</h2><ul>' + ''.join(links) +
                '</ul><p>Movies use stored numerical states, held without solution interpolation.</p></section>')


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
    movies = completed_movies(root, completed)
    def copy(source, target):
        if source.is_symlink():
            raise ValueError('Unexpected symlink in public results')
        if local_only_path(source.relative_to(root)):
            return
        # Rendering caches contain many duplicate frames and private logs.
        # Movie assets enter the site only through their verified manifest.
        if 'movies' in source.relative_to(root).parts:
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
        add_movie_pages(root, stage, movies, completed)
        prepare_html_and_validate_links(stage)
        files = {str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        receipt = {'schema': 2, 'workflow': 'm1-fld-radiation', 'status': 'staged_locally',
                   'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(root),
                   'destination': str(destination), 'completed_runs': len(completed),
                   'completed_movies': len(movies), 'movie_run_ids': [movie['run']['id'] for movie in movies],
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
