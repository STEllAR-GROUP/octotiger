#!/usr/bin/env python3
"""Audit completed Step 07 artifacts and produce a portable NO-GO/PASS report.

This audits recorded data; it never promotes conditional or unrun tests to passes.
"""
import argparse
import base64
import collections
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from verification_results import runner


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New output prefix, outside source')
    args = parser.parse_args()
    prefix = runner.safeOutput(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = {'decision': 'NO-GO', 'commit': runner.gitValue('rev-parse', 'HEAD'),
               'dirty': bool(runner.gitValue('status', '--porcelain')), 'matrix': {}, 'audit_failures': [],
               'baseline_samples': {'identical': [], 'different': [], 'unavailable': []},
               'products': collections.Counter(), 'damping': [], 'source_hashes_match': True,
               'coverage': 'Serial production-method validation plus conditional application descriptors; not complete physics acceptance'}
    descriptions = runner.descriptors()
    previous = None
    pages = []
    for mode in ('Release', 'Debug', 'RelWithDebInfo'):
        folder = args.results/mode
        rad = json.loads((folder/'radiation/summary.json').read_text())
        cases = rad + [dict(item, id=item['identifier']) for family in ('hydro', 'gravity')
                       for item in json.loads((folder/family/'verification.json').read_text())['tests']]
        assert {r['id'] for r in cases} == set(descriptions), 'Missing registered descriptors'
        summary['matrix'][mode] = {'counts': dict(collections.Counter(r['status'] for r in cases)),
                                 'cases': [{k: r[k] for k in ('id', 'status', 'orders', 'reason') if k in r} for r in cases]}
        numerical = {}
        for r in rad:
            for run in r.get('runs', []):
                case = folder/'radiation'/r['id']/f"l{run['level']}"
                meta = json.loads((case/'run.json').read_text()); p = meta['options']
                assert meta['thread_count'] == 1 and meta['build']['build_type'].lower() == mode.lower()
                assert meta['descriptor'] == descriptions[r['id']][1], 'Descriptor changed after run'
                for filename, expected in meta['source']['files'].items():
                    if digest(root/filename) != expected: summary['source_hashes_match'] = False
                summary['products']['movies_generated'] += 1; summary['products']['comparison_plots'] += 1
                try:
                    with Image.open(case/'comparison.png') as picture: picture.verify()
                    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(case/'movie.mp4'), '-f', 'null', '-'], check=True, capture_output=True)
                    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                       'stream=nb_frames,width,height', '-of', 'json', str(case/'movie.mp4')], text=True))
                    assert int(probe['streams'][0]['nb_frames']) == p['frames']
                    summary['products']['movies_decoded'] += 1
                except (OSError, ValueError, AssertionError, subprocess.CalledProcessError) as error:
                    summary['audit_failures'].append(f'{mode}/{r["id"]}/l{run["level"]}: invalid visual product: {error}')
                try:
                    samples = np.genfromtxt(case/'samples.csv', delimiter=',', names=True)
                    recordedHistory = np.genfromtxt(case/'history.csv', delimiter=',', names=True)
                    assert len(samples) and len(recordedHistory)
                    assert (case/'run.log').stat().st_size > 0
                except (OSError, ValueError, IndexError, AssertionError) as error:
                    summary['audit_failures'].append(f'{mode}/{r["id"]}/l{run["level"]}: incomplete raw artifacts: {error}')
                    continue
                frames = np.unique(samples['frame']).astype(int)
                assert list(frames) == list(range(p['frames']))
                assert len(samples) == meta['cells']*p['frames'] and np.all(np.isfinite(samples['E']))
                times = np.array([samples['t'][samples['frame'] == f][0] for f in frames])
                np.testing.assert_allclose(times, np.linspace(0, p['final_time_s'], p['frames']), rtol=1e-14, atol=1e-15)
                key = r['id']+f"/l{run['level']}"
                numerical[key] = (case/'samples.csv').read_bytes()
                if mode == 'Release' and args.baseline:
                    baseline = args.baseline/key/'samples.csv'
                    try:
                        old = np.genfromtxt(baseline, delimiter=',', names=True)
                        equal = old.shape == samples.shape and np.array_equal(old, samples)
                        summary['baseline_samples']['identical' if equal else 'different'].append(key)
                    except (ValueError, OSError): summary['baseline_samples']['unavailable'].append(key)
                if r['id'].endswith('.damped_wave'):
                    history = np.genfromtxt(case/'history.csv', delimiter=',', names=True)
                    predicted = history['mean_E'][0]*np.prod(1/(1+p['c_cm_s']*p['chi_cm_inverse']*history['rad_dt'][1:]))
                    residual = float(history['mean_E'][-1]-predicted)
                    assert abs(residual) < 5e-14, 'Damped mean does not match independent BE amplification'
                    summary['damping'].append({'mode': mode, 'level': run['level'], 'BE_mean_residual': residual})
        summary['products']['convergence_plots'] += len(list((folder/'radiation').glob('*/convergence.png')))
        if previous is not None:
            shared = numerical.keys() & previous.keys()
            summary['matrix'][mode]['comparable_samples'] = len(shared)
            summary['matrix'][mode]['comparable_samples_bitwise_equal_to_Release'] = all(numerical[k] == previous[k] for k in shared)
        else: previous = numerical
        pages.append((mode, folder/'radiation/report.html'))
    if not summary['source_hashes_match']: summary['audit_failures'].append('Recorded runtime source hashes differ from current files')
    if summary['baseline_samples']['different']: summary['audit_failures'].append('Step 05 numerical sample differences')
    summary['thermal_repeats'] = {}
    for name in ('thermal-repeat', 'thermal-repeat-release'):
        folder = args.results/name
        results = json.loads((folder/'summary.json').read_text())
        assert all(r['status'] == 'passed' for r in results)
        for metaPath in folder.glob('*/l*/run.json'):
            meta = json.loads(metaPath.read_text()); case = metaPath.parent
            data = np.genfromtxt(case/'samples.csv', delimiter=',', names=True)
            history = np.genfromtxt(case/'history.csv', delimiter=',', names=True)
            assert len(data) == 9*meta['cells'] and len(history) == 8*meta['cells']+1
            np.testing.assert_allclose(np.unique(data['t']), np.linspace(0, 1, 9), atol=1e-15)
        summary['thermal_repeats'][name] = {'status': 'passed', 'levels': [0, 1, 2],
                                         'interpretation': 'Clean separate rerun; original incomplete artifacts retained'}
    # Full parser, distributed runtime, restart, Ensman definitions and legacy migration
    # equivalence cannot be certified by this artifact audit even if all serial cases pass.
    summary['unvalidated'] = ['Full Boost/application option parser', 'Hydro/gravity CTest and before/after numerical equivalence',
                              'HPX/MPI multi-node boundary exchange', 'Silo application restart',
                              'Published Ensman sphere/shock', 'Full oblique radiation transport regression',
                              'CLI override radiation levels 3..5 and multithreaded application runs']
    summary['gates'] = {}
    signatures = {'harness-final.log': 'Ran 27 tests', 'options.log': 'Ran 7 tests',
                  'so-opacity-before.log': 'All S&O serial production-method checks passed.',
                  'so-opacity-release.log': 'All S&O serial production-method checks passed.',
                  'so-opacity-address-undefined.log': 'All S&O serial production-method checks passed.',
                  'conservation.log': 'Actual collector and sampling code passed'}
    for mode in ('Release', 'Debug', 'RelWithDebInfo'):
        for name, marker in [('radiationM1', '175509 checks passed'), ('greyOpacity', 'checkpoint compatibility passed'),
                             ('fpeGuard', 'FpeGuard tests passed'), ('subcycle-reference', 'ratio=0.03 chi_scattering=0.7')]:
            signatures[f'{name}-{mode}.log'] = marker
    for name, marker in signatures.items():
        log = args.results/'gates'/name
        text = log.read_text() if log.exists() else ''
        passed = marker in text and 'Traceback' not in text and 'FAILED (' not in text
        summary['gates'][name] = {'status': 'passed' if passed else 'failed', 'evidence': marker}
        if not passed: summary['audit_failures'].append('Gate log missing expected success: '+name)
    for name in ('grid.log', 'grid-storage.log', 'regression-kernels-correct-invocation.log', 'comparator.log'):
        summary['gates'][name] = {'status': 'failed', 'reason': 'Stale legacy driver; see retained traceback'}
    summary['gates']['options-parser-build.log'] = {'status': 'conditional', 'reason': 'Missing Boost headers; actual parser not run'}
    summary['gates']['so-opacity-sanitize.log'] = {'status': 'conditional', 'reason': 'LeakSanitizer unsupported under tracing; address/UB rerun disables leak detection explicitly'}
    report = (root/'doc/validation-step-07.md').read_text()
    content = '<!doctype html><meta charset="utf-8"><title>Step 07 physics validation: NO-GO</title>'
    content += '<style>body{max-width:1100px;margin:30px auto;font:16px system-ui}pre{white-space:pre-wrap}iframe{width:100%;height:850px;border:1px solid #aaa}</style>'
    content += '<h1>Step 07 — NO-GO</h1><p>Handoff commit: <code>'+html.escape(summary['commit'])+'</code></p>'
    content += '<pre>'+html.escape(report)+'</pre><h2>Machine-readable assessment</h2><pre>'+html.escape(json.dumps(summary, indent=2))+'</pre>'
    for mode, page in pages:
        encoded = base64.b64encode(page.read_bytes()).decode()
        content += '<details><summary>'+mode+' radiation plots, movies, data and run metadata</summary><iframe src="data:text/html;base64,'+encoded+'"></iframe></details>'
        for family in ('hydro', 'gravity'):
            raw = (args.results/mode/family/'verification.json').read_text()
            content += '<details><summary>'+mode+' '+family+' conditional evidence</summary><pre>'+html.escape(raw)+'</pre></details>'
    for log in sorted((args.results/'gates').glob('*.log')):
        content += '<details><summary>'+html.escape(log.name)+'</summary><pre>'+html.escape(log.read_text(errors='replace'))+'</pre></details>'
    for name in summary['thermal_repeats']:
        encoded = base64.b64encode((args.results/name/'report.html').read_bytes()).decode()
        content += '<details><summary>'+name+' clean rerun artifacts</summary><iframe src="data:text/html;base64,'+encoded+'"></iframe></details>'
    for suffix, data in (('.json', json.dumps(summary, indent=2)+'\n'), ('.html', content)):
        target = prefix.with_suffix(suffix)
        if target.exists(): raise ValueError('Assessment output already exists: '+str(target))
        target.write_text(data)
    print(json.dumps({'decision': summary['decision'], 'commit': summary['commit'], 'products': summary['products'],
                      'counts': {k: v['counts'] for k, v in summary['matrix'].items()}}, indent=2))
    return 1


if __name__ == '__main__': raise SystemExit(main())
