#!/usr/bin/env python3
"""Run Octo-TIGER radiation tests and publish each result immediately.

Install beside test_results/run.py, then run:
    python3 test_results/run_live.py

Defaults: all four cases, levels 2 3 4, debug, 12 threads, t=0..4 seconds,
each coordinate -3e10..3e10 cm, with CGS used internally and in all output,
approximately 61 saved states, and 20-second movies. Coarse levels run first.
Each simulation is followed by its plots and movie before the next starts.
The local index.html opens immediately and refreshes every 10 seconds, except
during video playback. It also works after this script exits; no server needed.

Examples:
    python3 test_results/run_live.py all 2 3 4 debug
    python3 test_results/run_live.py wave 2 --no-build
    python3 test_results/run_live.py --dry-run
    python3 test_results/run_live.py --resume test_results/results/live-...

Install the accompanying CGS-aware plot.py, movies.py, and visit_movie.py.
This uses the existing results.py, run.execute, and movie_support.py. It sets
output cadence directly in run.ini; run.py does not need additional flags.
An older physcon.cpp that forces c=1 is backed up and corrected before the
default build. The running binary's unit factors and light speed are checked.
"""

import argparse
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from urllib.parse import quote
import webbrowser


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent if (HERE / 'run.py').is_file() else Path.home() / 'workspace/octotiger'
DEFAULT_VISIT = Path.home() / 'visit3_4_2.linux-x86_64/bin/frontendlauncher'
C_CGS = 2.99792458e10  # cm/s, matching src/physcon.cpp
HALF_WIDTH_CM = 3.e10
CGS_UNITS = dict(system='CGS', length='cm', time='s', mass='g',
                 energy_density='erg/cm^3', flux='erg/(cm^2 s)',
                 chi='1/cm', luminosity='erg/s', source='erg/(cm^3 s)')


def cgs_config(template):
    """Preserve optical depth, relative widths, E, and F/(c E).

    Templates specify the existing c=1 regression problems. Stretch lengths by
    S, divide extinction chi by S, and multiply luminosity by c*S**2 while
    retaining energy densities. The new physical stop time is set separately.
    Config templates are read only; run.ini contains the actual CGS values.
    """
    config = dict(template)
    old_half = float(config['xscale'])
    if not math.isfinite(old_half) or old_half <= 0:
        raise ValueError('Template xscale must be positive and finite')
    unit_keys = {'code_to_cm', 'code_to_s', 'code_to_g'}
    declared_c = C_CGS * float(config.get('code_to_s', 1)) / float(config.get('code_to_cm', 1))
    if unit_keys.intersection(config) and not math.isclose(declared_c, 1., rel_tol=1e-12):
        raise ValueError('Expected the original radiation-test templates; CGS runs use generated run.ini files')
    stretch = HALF_WIDTH_CM / old_half
    chi = float(config['rad_test_chi']) / stretch
    width = float(config['rad_test_width']) * stretch
    luminosity = float(config['rad_test_luminosity']) * C_CGS * stretch**2
    background = float(config['rad_test_background'])
    amplitude = float(config['rad_test_amplitude'])
    if not all(math.isfinite(v) for v in (chi, width, luminosity, background, amplitude)):
        raise ValueError('Nonfinite radiation-test parameter')
    if not (chi >= 0 and 0 < width <= HALF_WIDTH_CM/2 and min(luminosity, background, amplitude) > 0):
        raise ValueError('Invalid radiation-test parameter')
    config.update(xscale=str(HALF_WIDTH_CM), code_to_cm='1', code_to_s='1', code_to_g='1',
                  rad_test_chi=str(chi), rad_test_width=str(width), rad_test_luminosity=str(luminosity))
    # hard_dt in the templates is in their original c=1 time unit.
    if float(config.get('hard_dt', 0)) > 0:
        config['hard_dt'] = str(float(config['hard_dt']) * stretch / C_CGS)
    return config


def ensure_cgs_source(root, no_build=False, dry_run=False):
    """Correct only the known legacy time-unit override, preserving a backup."""
    path = root / 'src/physcon.cpp'
    original = path.read_text()
    branch = re.compile(r'(else if\s*\(opts\(\)\.radiation\)\s*\{[^{}]*?)(\bt\s*=\s*opts\(\)\.code_to_cm\s*/\s*2\.99792458e\+?10\s*;)')
    changed, count = branch.subn(r'\1t = opts().code_to_s;', original)
    if not count:
        return None  # The runtime check also covers newer/differently formatted code.
    if count != 1:
        raise ValueError(f'Ambiguous radiation unit initialization in {path}')
    if dry_run:
        print(f'Build preparation: correct the legacy c=1 override in {path} (with backup)', flush=True)
        return None
    if no_build:
        raise ValueError('The solver source forces c=1. Run again without --no-build to correct it and rebuild.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    backup = path.with_name(path.name + '.before-cgs-' + stamp)
    shutil.copy2(path, backup)
    atomic_text(path, changed)
    print(f'Updated radiation units to honor code_to_s; original saved at {backup}', flush=True)
    return str(backup)


class CGSLogCheck:
    """Validate numeric normalization blocks; diagnostic suffixes are harmless."""

    def __init__(self):
        self.pending = False
        self.factors_seen = False
        self.blocks = 0

    def feed(self, line):
        if re.match(r'^normalized\s+constants\b', line.strip(), re.I):
            if self.pending:
                raise ValueError('Incomplete solver unit block before the next normalization heading')
            self.pending = True
            self.factors_seen = False
            return
        if self.pending and not self.factors_seen:
            try:
                factors = [float(v) for v in line.split()]
            except ValueError:
                factors = []
            if len(factors) == 4:
                if not all(math.isfinite(v) and math.isclose(v, 1., rel_tol=1e-12) for v in factors):
                    raise ValueError(f'Solver is not using CGS unit factors: {factors}; rebuild octotiger')
                self.factors_seen = True
        match = re.search(r'\|\s*c\s*=\s*(\S+)\s*\|', line)
        if match:
            # physcon.cpp prints only six digits after the decimal.
            try:
                c = float(match[1])
            except ValueError:
                c = math.nan
            if not math.isfinite(c) or not math.isclose(c, C_CGS, rel_tol=5e-7):
                raise ValueError(f'Solver light speed is {match[1]}, expected {C_CGS} cm/s; rebuild octotiger')
            if self.pending and self.factors_seen:
                self.blocks += 1
                self.pending = False

    def finish(self, log):
        if not self.blocks or self.pending:
            raise ValueError(f'Could not verify the running solver\'s CGS units and light speed; see {log}')


def verify_cgs_log(log):
    check = CGSLogCheck()
    with Path(log).open(errors='replace') as stream:
        for line in stream:
            check.feed(line)
    check.finish(log)


def recover_run(folder, plan, binary_hash):
    """Reuse only verified finished output; never overwrite a partial run."""
    from results import digest, read_config, read_conservation, read_norms, read_slice
    from movie_support import numerical_silos
    meta = json.loads((folder / 'run.json').read_text())
    eligible = meta.get('status') == 'complete' or (
        meta.get('status') == 'failed' and
        str(meta.get('error', '')).startswith('Could not verify the running solver\'s CGS units and light speed;'))
    if not eligible:
        raise ValueError(f'{folder} is not a verified finished simulation; its existing files are retained. '
                         'Use a new batch for incomplete simulations.')
    expected = dict(case=plan['case'], level=plan['level'], cells=plan['cells'],
                    inx=plan['cells']//(2**plan['level']), origin='Octo-TIGER application',
                    length=2*HALF_WIDTH_CM, dx=2*HALF_WIDTH_CM/plan['cells'],
                    time=float(plan['config']['stop_time']), c=C_CGS, units=CGS_UNITS,
                    background=float(plan['config']['rad_test_background']),
                    executable_sha256=binary_hash, movie_capture=plan['capture'])
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'Saved {key} differs from the resumed batch in {folder}')
    recorded = meta['config']
    if read_config(folder / 'run.ini') != recorded:
        raise ValueError(f'Saved run.ini differs from run.json in {folder}')
    relevant = lambda config: {k: v for k, v in config.items() if k not in {'datadir', 'rad_reference'}}
    if relevant(recorded) != relevant(plan['config']):
        raise ValueError(f'Saved configuration differs from current template in {folder}')
    verify_cgs_log(folder / 'run.log')
    meta['norms'] = read_norms(folder, meta)
    read_slice(folder, meta)
    conservation = read_conservation(folder, meta)
    if conservation is not None:
        meta['conservation'] = conservation['summary']
    else:
        meta.pop('conservation', None)
    numerical_silos(folder)
    if meta['case'] == 'gaussian_pulse':
        if digest(Path(recorded['rad_reference'])) != meta.get('reference_sha256'):
            raise ValueError(f'Gaussian reference changed in {folder}')
    if meta.get('status') != 'complete':
        meta['recovered_error'] = meta.pop('error')
        meta['recovered_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    meta['status'] = 'complete'
    return meta


def execute_cgs(command, folder, log):
    """Stream the solver log and reject the wrong unit system at startup."""
    print('+', shlex.join(map(str, command)), flush=True)
    check = CGSLogCheck()
    with Path(log).open('w') as out:
        process = subprocess.Popen(list(map(str, command)), cwd=folder, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                out.write(line); out.flush()
                check.feed(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f'Exit code {code}; see {log}')
    check.finish(log)


def atomic_text(path, content):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def publish(output, state):
    """Replace the page atomically so a browser never reads half-written HTML."""
    state['updated_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    atomic_text(output / 'live.json', json.dumps(state, indent=2, allow_nan=False) + '\n')
    esc = html.escape

    def link(relative, label):
        if not (output / relative).is_file():
            return ''
        return f'<a href="{quote(relative, safe="/")}" target="_blank" rel="noopener">{esc(label)}</a>'

    rows, previews = [], []
    for run in state['runs']:
        case, level = run['case'], run['level']
        title = case.replace('_', ' ').capitalize()
        plots = f'plots/{case}/l{level}'
        profile = f'{plots}/profiles.png'
        has_run_plots = run['stage'] != 'Waiting' and (output / profile).is_file()
        movie = f'{case}/l{level}/movies/er-slice-z/movie.mp4'
        links = [link(f'{plots}/profiles.png', 'Profiles'),
                 link(f'{plots}/slice_er.png', 'Energy slice'),
                 link(f'{plots}/conservation.png', 'Conservation'),
                 link(f'{plots}/conservation.csv', 'Conservation CSV'),
                 link(f'plots/{case}/convergence.png', 'Convergence') if has_run_plots else '',
                 link(movie, 'MP4')]
        links = ' · '.join(value for value in links if value) or 'Waiting for output'
        count = run.get('source_snapshots')
        status = run['stage'] + (f' ({count} snapshots)' if count is not None else '')
        rows.append(f'<tr><th>{esc(title)}</th><td>{level}</td><td>{run["cells"]}³</td>'
                    f'<td>{esc(status)}</td><td>{links}</td></tr>')
        if has_run_plots:
            video = (f'<video controls preload="metadata" src="{quote(movie, safe="/")}"></video>'
                     if (output / movie).is_file() else '<p>Movie rendering is pending or in progress.</p>')
            conservation = f'{plots}/conservation.png'
            budget = (f'<details><summary>Radiation conservation</summary>'
                      f'<img loading="lazy" src="{quote(conservation, safe="/")}" '
                      f'alt="Radiation totals and balance errors for {esc(title)}"></details>'
                      if (output / conservation).is_file() else
                      '<p>Conservation data unavailable; rerun with the updated solver.</p>')
            previews.append(f'<section><h2>{esc(title)} · level {level} · {run["cells"]}³</h2>'
                            f'<p>{links}</p><div class="visuals"><a href="{quote(profile, safe="/")}" '
                            f'target="_blank" rel="noopener"><img loading="lazy" src="{quote(profile, safe="/")}" '
                            f'alt="Radiation profiles and errors for {esc(title)}"></a>{video}</div>{budget}</section>')
    reports = ' · '.join(value for value in [link('plots/index.html', 'Full plot report'),
                                            link('movies.html', 'Movie gallery')] if value)
    error = f'<p class="error">{esc(state["error"])}</p>' if state.get('error') else ''
    refresh = '''
<script>
// file:// needs no HTTP server or fetch. Do not interrupt a playing movie.
try { window.scrollTo(0, Number(sessionStorage.getItem('scroll:' + location.pathname) || 0)); } catch (_) {}
setInterval(() => {
    if (!document.getElementById('refresh').checked) return;
    if (Array.from(document.querySelectorAll('video')).some(v => !v.paused && !v.ended)) return;
    try { sessionStorage.setItem('scroll:' + location.pathname, String(window.scrollY)); } catch (_) {}
    window.location.reload();
}, 10000);
</script>''' if state['active'] else ''
    atomic_text(output / 'index.html', f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Octo-TIGER radiation results</title><style>
body{{font:18px/1.5 system-ui,sans-serif;max-width:1600px;margin:32px auto;padding:0 24px;color:#172335;background:#fafbfc}}
h1,h2{{line-height:1.2}}a{{color:#0755a2}}.status{{padding:16px;background:#e8eff6;border-left:4px solid #326494}}
.error{{padding:16px;background:#ffe5e5;color:#8b1414;white-space:pre-wrap}}
table{{border-collapse:collapse;width:100%;white-space:nowrap}}td,th{{padding:10px 14px;border-bottom:1px solid #bbc;text-align:left}}
.scroll{{overflow-x:auto}}section{{margin:32px 0;padding-top:20px;border-top:1px solid #bbc}}
.visuals{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,550px),1fr));gap:20px;align-items:start}}
img,video{{width:100%;height:auto;background:white}}label{{display:block;margin:12px 0}}input{{width:20px;height:20px}}
</style></head><body><h1>Octo-TIGER radiation results</h1>
<p>Each axis: −3 × 10<sup>10</sup> to +3 × 10<sup>10</sup> cm. Time: 0–{state['time']:g} s.<br>
CGS throughout: E in erg/cm³; F in erg/(cm² s); c = {C_CGS:.8e} cm/s.<br>
Target: approximately {state['snapshots']} saved states per run ·
{state['threads']} simulation threads. Lower resolutions run first.</p>
<div class="status"><strong>{esc(state['message'])}</strong><br>Updated {esc(state['updated_utc'])}</div>{error}
{('<label><input id="refresh" type="checkbox" checked> Refresh every 10 seconds; paused during video playback</label>') if state['active'] else '<p>Automatic refresh stopped. These saved results remain available.</p>'}
<p>{reports}</p><p>Use Ctrl+C in the run terminal to stop the batch. Completed results are retained.</p>
<div class="scroll"><table><thead><tr><th>Model</th><th>Level</th><th>Cells</th><th>Status</th><th>Results</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>{''.join(previews)}{refresh}</body></html>''')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case', nargs='?', default='all')
    parser.add_argument('levels_and_build', nargs='*', help='default: 2 3 4 debug')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--build', help='absolute build path or a path relative to the checkout')
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--generator', type=Path)
    parser.add_argument('--output', type=Path, help='new, empty batch directory')
    parser.add_argument('--resume', type=Path, help='resume a saved batch without rebuilding or repeating finished simulations')
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 2))
    parser.add_argument('--time', type=float, default=4., help='final physical time in seconds (initial time is zero)')
    parser.add_argument('--snapshots', type=int, default=61)
    parser.add_argument('--seconds', type=float, default=20., help='duration of each MP4')
    parser.add_argument('--visit', default=str(DEFAULT_VISIT) if DEFAULT_VISIT.is_file() else 'visit')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--no-open', action='store_true', help='print the page path without opening a browser')
    parser.add_argument('--dry-run', action='store_true', help='show settings without building or writing results')
    args = parser.parse_args()
    if args.threads < 1 or args.jobs < 1 or args.snapshots < 3:
        parser.error('threads/jobs must be positive; snapshots must be at least 3')
    if not math.isfinite(args.time) or args.time <= 0:
        parser.error('--time must be positive and finite')
    if not math.isfinite(args.seconds) or args.seconds <= 2.1:
        parser.error('--seconds must exceed 2.1 (movies include two 1-second endpoint holds)')
    parts = list(args.levels_and_build)
    args.build_name = args.build or 'debug'
    if parts and not parts[-1].isdigit():
        if args.build:
            parser.error('Specify the build only once')
        args.build_name = parts.pop()
    try:
        args.levels = sorted(set(map(int, parts or ['2', '3', '4'])))
    except ValueError:
        parser.error('Levels must be integers')
    if min(args.levels) < 0 or max(args.levels) > 9:
        parser.error('Levels must be between 0 and 9')
    return args


def main():
    args = arguments()
    resumed = None
    if args.resume:
        if args.output or args.levels_and_build or args.case != 'all' or args.build or args.exe:
            raise ValueError('--resume uses the saved cases, levels, build, executable and output directory')
        if args.time != 4. or args.snapshots != 61:
            raise ValueError('--resume uses the saved time and snapshot count; omit --time and --snapshots')
        args.resume = args.resume.expanduser().resolve()
        resumed = json.loads((args.resume / 'batch.json').read_text())
        if resumed.get('workflow') != 'run_live.py' or resumed.get('units') != CGS_UNITS:
            raise ValueError('--resume requires a CGS batch created by run_live.py')
        args.root = Path(resumed['root'])
        args.build_name = resumed['build']
        args.exe = Path(resumed['executable'])
        args.levels = resumed['levels']
        args.time = resumed['time']
        args.snapshots = resumed['snapshots']
        args.no_build = True
    root = args.root.expanduser().resolve()
    scripts = root / 'test_results'
    required = ('run.py', 'results.py', 'plot.py', 'movies.py', 'movie_support.py', 'visit_movie.py')
    for name in required:
        if not (scripts / name).is_file():
            raise ValueError(f'Missing {scripts / name}; use --root to select the Octo-TIGER checkout')
    sys.path.insert(0, str(scripts))
    from results import ALIASES, CASES, digest, read_config, read_conservation, read_norms, read_slice, write_json
    from movie_support import cadence, check_dependencies
    from run import execute

    case = ALIASES.get(args.case, args.case)
    if case != 'all' and case not in CASES:
        raise ValueError('Unknown case; choose all, ' + ', '.join(CASES))
    cases = tuple(resumed['cases']) if resumed else CASES if case == 'all' else (case,)
    if any(case not in CASES for case in cases):
        raise ValueError('Saved batch contains an unknown case')
    build = (root / Path(args.build_name).expanduser()).resolve()
    exe = (args.exe.expanduser() if args.exe else build / 'octotiger').resolve()
    generator = args.generator.expanduser().resolve() if args.generator else None
    cache = (build / 'CMakeCache.txt').read_text()
    match = re.search(r'^OCTOTIGER_WITH_GRIDDIM:[^=]+=([0-9]+)\s*$', cache, re.M)
    if not match or int(match[1]) < 1:
        raise ValueError(f'Cannot determine INX from {build / "CMakeCache.txt"}')
    inx = int(match[1])
    sources = [root / name for name in ('src/grid.cpp', 'src/physcon.cpp', 'src/radiation/rad_grid.cpp',
               'octotiger/test_problems/radiation/profiles.hpp', 'octotiger/test_problems/radiation/plot_output.hpp',
               'octotiger/radiation/conservation.hpp', 'octotiger/radiation/rad_grid.hpp',
               'src/node_server_actions_3.cpp')]
    if 'RADIATION_PLOT_EXPORT_BEGIN' not in sources[0].read_text():
        raise ValueError(f'First install the slice-export hook: python3 {scripts / "install.py"} {root}')

    print(f'CGS: each axis [-{HALF_WIDTH_CM:g}, +{HALF_WIDTH_CM:g}] cm; '
          f'0..{args.time:g} s; c={C_CGS:g} cm/s; '
          f'{args.time*C_CGS/(2*HALF_WIDTH_CM):.6f} full-box light crossings', flush=True)
    plans = []
    for level in args.levels:
        for case in cases:
            config = cgs_config(read_config(scripts / 'configs' / f'{case}.ini'))
            capture = cadence(args.time, args.snapshots, float(config['cfl']))
            previous_cap = float(config.get('hard_dt', 0))
            if not math.isfinite(previous_cap) or previous_cap < 0:
                raise ValueError(f'Invalid configured hard_dt for {case}')
            if previous_cap > 0:
                capture['hard_dt'] = min(previous_cap, capture['hard_dt'])
            cells = inx * 2 ** level
            if case == 'gaussian_pulse' and (cells < 4 or cells > 512 or cells % 2):
                raise ValueError('Gaussian reference generator requires even N from 4 through 512')
            config.update(max_level=str(level), min_level=str(level), stop_time=str(args.time),
                          odt=str(capture['odt']), hard_dt=str(capture['hard_dt']),
                          disable_output='off', disable_diagnostics='on', disable_analytic='off',
                          n_species='1', atomic_mass='1', atomic_number='1')
            plans.append(dict(case=case, level=level, cells=cells, config=config, capture=capture))
            print(f'{case:20s} level={level}  N={cells:<4d}  t={args.time:.6e}  '
                  f's  odt={capture["odt"]:.6e} s  hard_dt<={capture["hard_dt"]:.6e} s\n'
                  f'  chi={float(config["rad_test_chi"]):.9e} cm^-1  '
                  f'width={float(config["rad_test_width"]):.9e} cm  '
                  f'luminosity={float(config["rad_test_luminosity"]):.9e} erg/s', flush=True)
    if args.dry_run:
        ensure_cgs_source(root, dry_run=True)
        return

    import plot, movies
    if (not all(getattr(module, 'SUPPORTS_CGS', False) for module in (plot, movies)) or
            'SUPPORTS_CGS = True' not in (scripts / 'visit_movie.py').read_text()):
        raise ValueError('Install all files from radiation_live_cgs.zip so plots and movies carry CGS units')
    from plot import render
    from movies import make_movie, movie_index
    visit, ffmpeg = check_dependencies(args.visit, args.ffmpeg)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output = (args.resume if resumed else args.output.expanduser() if args.output else scripts / 'results' / f'live-{stamp}').resolve()
    if not resumed and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Output must be a new, empty directory: {output}')
    if resumed:
        if digest(exe) != resumed['executable_sha256']:
            raise ValueError('The simulation executable changed; use a new batch to keep results comparable')
        for name, expected_hash in resumed['source_sha256'].items():
            if digest(root / name) != expected_hash:
                raise ValueError(f'{name} changed since this batch began; use a new batch')
        old_state = json.loads((output / 'live.json').read_text())
        if old_state.get('active'):
            raise ValueError('This batch is marked active; stop its other runner before resuming')
        saved_runs = [(r['case'], r['level'], r['cells']) for r in old_state['runs']]
        if saved_runs != [(p['case'], p['level'], p['cells']) for p in plans]:
            raise ValueError('Saved run order or resolutions differ from this batch')
    output.mkdir(parents=True, exist_ok=True)
    state = dict(active=True, time=args.time, snapshots=args.snapshots, threads=args.threads,
                 units=CGS_UNITS, length=2*HALF_WIDTH_CM, c=C_CGS,
                 message='Preparing the batch', runs=[dict(case=p['case'], level=p['level'],
                 cells=p['cells'], stage='Waiting') for p in plans])
    if resumed:
        for row, old_row in zip(state['runs'], old_state['runs']):
            if old_row.get('stage') == 'Ready':
                row.update(old_row)
        state['message'] = 'Resuming the saved batch; checking finished simulations'
    page = output / 'index.html'
    publish(output, state)
    print(f'Live results page: {page}\n{page.as_uri()}', flush=True)
    current = None
    meta = None
    folder = None
    try:
        if not args.no_open:
            try:
                webbrowser.open(page.as_uri())
            except webbrowser.Error:
                print(f'Open the results page manually: {page}', flush=True)
        unit_source_backup = ensure_cgs_source(root, no_build=args.no_build)
        if not args.no_build:
            state['message'] = 'Building the simulation and reference generator'
            publish(output, state)
            targets = ['octotiger']
            if 'gaussian_pulse' in cases and generator is None:
                targets.append('gen_radiation_reference')
            execute(['cmake', '--build', build, '--target', *targets, '-j', args.jobs],
                    root, output / 'build.log')
        if not exe.is_file() or not os.access(exe, os.X_OK):
            raise ValueError(f'No executable: {exe}')
        if 'gaussian_pulse' in cases:
            if generator is None:
                candidates = [p for p in build.rglob('gen_radiation_reference')
                              if p.is_file() and os.access(p, os.X_OK)]
                if len(candidates) != 1:
                    raise ValueError('Use --generator to select gen_radiation_reference')
                generator = candidates[0].resolve()
            if not generator.is_file() or not os.access(generator, os.X_OK):
                raise ValueError(f'No reference generator executable: {generator}')
        binary_hash = digest(exe)
        if not resumed:
            write_json(output / 'batch.json', dict(created_utc=stamp, root=str(root), build=str(build),
                       executable=str(exe), executable_sha256=binary_hash,
                       source_sha256={str(p.relative_to(root)): digest(p) for p in sources},
                       origin='Octo-TIGER application', cases=cases, levels=args.levels,
                       time=args.time, snapshots=args.snapshots, workflow='run_live.py',
                       units=CGS_UNITS, length=2*HALF_WIDTH_CM, c=C_CGS,
                       unit_source_backup=unit_source_backup))

        for plan, current in zip(plans, state['runs']):
            case, level, cells = plan['case'], plan['level'], plan['cells']
            config = dict(plan['config'])
            folder = output / case / f'l{level}'
            if resumed and folder.exists():
                current['stage'] = 'Checking saved run'
                state['message'] = f'{case}, level {level}: checking existing simulation output'
                publish(output, state)
                meta = recover_run(folder, plan, binary_hash)
                write_json(folder / 'run.json', meta)
                print(f'Reusing finished simulation: {case}, level {level}', flush=True)
            else:
                folder.mkdir(parents=True)
                (folder / 'radiation-slices').mkdir()
                config['datadir'] = str(folder) + '/'
                length = 2 * float(config['xscale'])
                meta = dict(case=case, level=level, inx=inx, cells=cells, length=length, dx=length/cells,
                            time=args.time, c=C_CGS, units=CGS_UNITS,
                            origin='Octo-TIGER application', status='running',
                            background=float(config['rad_test_background']), executable_sha256=binary_hash,
                            movie_capture=plan['capture'])
                write_json(folder / 'run.json', meta)
                if case == 'gaussian_pulse':
                    current['stage'] = 'Generating reference'
                    state['message'] = f'{case}, level {level}: generating reference data'
                    publish(output, state)
                    reference = folder / 'reference.bin'
                    execute([generator, '--output', reference, '--cells', cells, '--length', length,
                             '--c', C_CGS, '--chi', config['rad_test_chi'], '--width', config['rad_test_width'],
                             '--background', config['rad_test_background'], '--amplitude', config['rad_test_amplitude'],
                             '--time', args.time], folder, folder / 'reference.log')
                    config['rad_reference'] = str(reference)
                    meta['reference_sha256'] = digest(reference)
                cfg = folder / 'run.ini'
                cfg.write_text('\n'.join(f'{k}={v}' for k, v in config.items()) + '\n')
                meta['config'] = config
                meta['comparison_signature'] = {k: v for k, v in config.items() if k not in
                    {'max_level', 'min_level', 'datadir', 'rad_reference', 'disable_output', 'odt'}}
                command = [exe, '--config_file=' + str(cfg), '--hpx:threads=' + str(args.threads)]
                meta['command'] = list(map(str, command))
                write_json(folder / 'run.json', meta)
                current['stage'] = 'Simulating'
                state['message'] = f'{case}, level {level}: simulation running'
                publish(output, state)
                execute_cgs(command, folder, folder / 'run.log')
                meta['norms'] = read_norms(folder, meta)
                read_slice(folder, meta)
                conservation = read_conservation(folder, meta)
                if conservation is None:
                    raise ValueError('Missing radiation-conservation.csv; rebuild octotiger with the conservation diagnostics')
                meta['conservation'] = conservation['summary']
                meta['status'] = 'complete'
                write_json(folder / 'run.json', meta)
            current['stage'] = 'Making plots'
            state['message'] = f'{case}, level {level}: simulation complete; generating plots'
            publish(output, state)
            render(output)
            current['stage'] = 'Rendering movie'
            state['message'] = f'{case}, level {level}: plots ready; rendering movie'
            publish(output, state)
            movie = make_movie(folder, seconds=args.seconds, visit=visit, ffmpeg=ffmpeg,
                               reuse_frames=bool(resumed and (folder / 'movies/er-slice-z/render.json').is_file()))
            movie_index(output)
            current.update(stage='Ready', source_snapshots=movie['source_snapshots'])
            state['message'] = f'{case}, level {level}: plots and movie ready'
            publish(output, state)
            print(f'Ready: {case}, level {level} — {page}', flush=True)
            current = None
            meta = None

        state.update(active=False, message='All simulations, plots, and movies are complete')
        publish(output, state)
        print(f'Finished. Open {page}', flush=True)
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        if meta is not None and meta['status'] != 'complete':
            meta.update(status='interrupted' if interrupted else 'failed', error=str(error))
            write_json(folder / 'run.json', meta)
        if current is not None:
            current['stage'] = ('Interrupted during ' if interrupted else 'Failed during ') + current['stage'].lower()
        message = 'Batch stopped; completed results are retained'
        if meta is not None and meta['status'] == 'complete':
            message += '. The current simulation is complete; postprocessing did not finish'
        state.update(active=False, message=message, error=str(error) or 'Interrupted with Ctrl+C')
        publish(output, state)
        raise


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
