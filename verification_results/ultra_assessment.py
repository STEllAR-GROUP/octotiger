#!/usr/bin/env python3
"""Package a transparent ULTRA follow-up; never replace historical Step 07 data."""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from verification_results.audit_artifacts import audit
from verification_results.adapters import native_suite

modes = ('Debug', 'Release', 'RelWithDebInfo')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=root)


def jsonFile(path):
    return json.loads(path.read_text())


def test_log(path):
    text = path.read_text()
    count = re.findall(r'^Ran (\d+) tests? in ', text, re.M)
    return {'path': str(path), 'sha256': sha(path),
            'tests': int(count[-1]) if count else None,
            'passed': bool(re.search(r'^OK\s*$', text, re.M)),
            'skipped': 'skipped=' in text}


def makePatch():
    patch = git('diff', '--binary', 'HEAD', '--')
    for name in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0'):
        if not name:
            continue
        process = subprocess.run(['git', 'diff', '--no-index', '--binary', '--', '/dev/null', name],
                                 cwd=root, capture_output=True)
        if process.returncode not in (0, 1):
            raise RuntimeError(process.stderr.decode())
        patch += process.stdout
    return patch


def assess(evidence, destination, decode):
    destination.mkdir(parents=True, exist_ok=True)
    prefix = destination / 'octotiger-ultra-followup'
    patchPath = prefix.with_suffix('.patch')
    native_suite.atomicWrite(patchPath, makePatch())
    summary = {'decision': 'NO-GO for complete physics acceptance',
               'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'scope': 'Follow-up fixes, not a replacement Step 07 commit/archive',
               'base_commit': git('rev-parse', 'HEAD').decode().strip(),
               'working_tree_dirty': bool(git('status', '--porcelain').strip()),
               'patch': {'file': patchPath.name, 'sha256': sha(patchPath)},
               'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
               'dependencies': {
                   'boost_headers': {'version': '1.83.0',
                       'source': 'https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.bz2',
                       'archive_sha256': sha(evidence/'dependencies/boost_1_83_0.tar.bz2')},
                   'cmake_ctest': '3.31.6 (task-local)',
                   'fftw_headers': '3.3.10; official source and digests in regression/README.md'},
               'production_physics_diff': git('diff', 'HEAD', '--', 'src', 'octotiger').decode(),
               'matrix': {}, 'unit_tests': {}, 'parser': {}, 'limitations': [
                   'Full application hydro/gravity and pre-migration equivalence not run',
                   'Published Ensman cases remain conditional; sphere reference unverified',
                   'HPX/MPI distributed boundaries, full oblique transport and Silo restart unrun',
                   'Historical M1 suite is incompatible; current successor is not equivalence',
                   'Four original-profile fixture cases produce metrics, not inherited accuracy passes',
                   'Historical artifact-loss cause unproven; integrity checked at observation time',
                   'LeakSanitizer not validated; sanitizer runs explicitly disable leak checks']}
    if summary['production_physics_diff']:
        raise ValueError('Unexpected production physics change')
    total = {'native_runs': 0, 'passed_cases': 0, 'failed_cases': 0, 'conditional_cases': 0,
             'comparison_plots': 0, 'convergence_plots': 0, 'movies': 0}
    for mode in modes:
        folder = evidence/'matrix'/mode
        radiation = jsonFile(folder/'radiation/summary.json')
        hydroGravity = [item for family in ('hydro', 'gravity')
                         for item in jsonFile(folder/family/'verification.json')['tests']]
        cases = radiation + hydroGravity
        summary['matrix'][mode] = {'aggregate': jsonFile(folder/'summary.json'),
                                  'cases': cases}
        for case in cases:
            total[case['status']+'_cases'] += 1
            total['native_runs'] += len(case.get('runs', []))
        for key, pattern in [('comparison_plots', '**/comparison.png'),
                             ('convergence_plots', '**/convergence.png'), ('movies', '**/movie.mp4')]:
            total[key] += len(list((folder/'radiation').glob(pattern)))
        xml = evidence/'standalone'/f'{mode}.xml'
        root = ET.parse(xml).getroot()
        summary['unit_tests'][mode] = {
            'junit': str(xml), 'sha256': sha(xml),
            'cases': [{'name': case.attrib['name'],
                       'status': case.get('status'),
                       'failed': case.find('failure') is not None,
                       'skipped': case.find('skipped') is not None} for case in root.iter('testcase')]}
        executable = evidence/'parser'/('options-'+mode)
        process = subprocess.run([str(executable)], text=True, capture_output=True)
        summary['parser'][mode] = {'executable': str(executable), 'sha256': sha(executable),
                                   'returncode': process.returncode,
                                   'stdout': process.stdout, 'stderr': process.stderr,
                                   'scope': 'Real Boost compatibility helper parser, not full application'}
    summary['totals'] = total
    summary['harness_tests'] = test_log(evidence/'harness-tests-final.log')
    summary['options_source_tests'] = test_log(evidence/'options-contract.log')
    summary['artifact_audit'] = audit(evidence/'matrix', decode)
    summary['sample_reproducibility'] = {'cross_build': [], 'previous_step07': []}
    for reference in sorted((evidence/'matrix/Release/radiation').glob('*/l*/samples.csv')):
        relative = reference.relative_to(evidence/'matrix/Release/radiation')
        current = native_suite.readCsvStrict(reference, native_suite.rawSchemas['samples.csv'])
        for mode in ('Debug', 'RelWithDebInfo'):
            other = evidence/'matrix'/mode/'radiation'/relative
            data = native_suite.readCsvStrict(other, native_suite.rawSchemas['samples.csv'])
            summary['sample_reproducibility']['cross_build'].append({
                'mode': mode, 'path': str(relative), 'bitwise_equal': reference.read_bytes() == other.read_bytes(),
                'numeric_equal': bool(np.array_equal(current, data))})
        previous = evidence.parent/'step07-results/Release/radiation'/relative
        record = {'path': str(relative), 'scope': 'sample values only; original artifact failures remain failures'}
        try:
            data = native_suite.readCsvStrict(previous, native_suite.rawSchemas['samples.csv'])
            record['numeric_equal'] = bool(np.array_equal(current, data))
        except (OSError, ValueError) as error:
            record['unavailable'] = str(error)
        summary['sample_reproducibility']['previous_step07'].append(record)
    summary['source_difference_explanation'] = (
        'Matrix source snapshots predate final CTest archival durability, provenance '
        'and cleanup-containment hardening, plus smoke CLI compatibility fixes '
        '(--exe= routing and unused --ctest forwarding). Runtime identity records '
        'are preserved, not relabeled as exact final-tree runs. Any source mismatch '
        'remains a strict audit failure and is listed with hashes; it does not imply a '
        'numerical or artifact failure. Targeted wrapper tests run after the correction. '
        'No native pipeline or production physics changed during the matrix.')
    summary['evidence_notes'] = [str(path.relative_to(evidence)) for path in evidence.rglob('*.md')
                                 if 'dependencies' not in path.parts]
    native_suite.dump(prefix.with_suffix('.json'), summary)
    notes = (root/'doc/validation-ultra-followup.md').read_text()
    parts = ['<!doctype html><meta charset="utf-8"><title>ULTRA validation follow-up</title>',
             '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}iframe{width:100%;height:850px;border:1px solid #aaa}summary{cursor:pointer;margin:12px 0}</style>',
             '<h1>ULTRA follow-up: repaired harness, physics acceptance still NO-GO</h1>',
             '<pre>'+html.escape(notes)+'</pre>',
             '<details><summary>Machine-readable assessment</summary><pre>'+html.escape(json.dumps(summary, indent=2))+'</pre></details>']
    for note in summary['evidence_notes']:
        parts.append('<details><summary>'+html.escape(note)+'</summary><pre>'+html.escape((evidence/note).read_text())+'</pre></details>')
    for mode in modes:
        encoded = base64.b64encode((evidence/'matrix'/mode/'radiation/report.html').read_bytes()).decode()
        parts.append(f'<details><summary>{mode}: plots, movies, raw data and metadata</summary><iframe title="{mode}" src="data:text/html;base64,{encoded}"></iframe></details>')
    native_suite.atomicWrite(prefix.with_suffix('.html'), ''.join(parts))
    # Keep compact raw evidence, scripts and reports; avoid dependencies, binaries,
    # compiler trees, frames already present in decoded movies,
    # and duplicate embedded HTML. Nothing in the original evidence is deleted.
    bundle = prefix.with_suffix('.zip')
    allowed = {'.json', '.csv', '.log', '.txt', '.xml', '.md', '.py', '.patch', '.png', '.mp4',
               '.cpp', '.inc', '.hpp', '.h'}
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(patchPath, patchPath.name)
        archive.write(prefix.with_suffix('.json'), prefix.with_suffix('.json').name)
        archive.write(root/'doc/validation-ultra-followup.md', 'validation-ultra-followup.md')
        for path in sorted(evidence.rglob('*')):
            if not path.is_file() or (path.suffix not in allowed and path.name not in {'COPYING', 'LICENSE'}):
                continue
            relative = path.relative_to(evidence)
            if any(part in {'dependencies', 'CMakeFiles', 'frames', '__pycache__'} for part in relative.parts):
                continue
            if any(part.startswith('repro-') for part in relative.parts):
                continue
            archive.write(path, 'evidence/'+relative.as_posix())
    outputs = [patchPath, prefix.with_suffix('.json'), prefix.with_suffix('.html'), bundle]
    native_suite.atomicWrite(prefix.with_suffix('.sha256'),
                              ''.join(sha(path)+'  '+path.name+'\n' for path in outputs))
    print(json.dumps({'outputs': [str(path) for path in outputs], 'totals': total,
                      'artifact_audit_status': summary['artifact_audit']['status'],
                      'decision': summary['decision']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--decode', action='store_true')
    args = parser.parse_args()
    assess(args.evidence.resolve(), args.destination.resolve(), args.decode)
