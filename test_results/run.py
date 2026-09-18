#!/usr/bin/env python3
"""Run new radiation tests at multiple uniform resolutions and plot the results.

Examples: ./run.sh wave 2 3 4 release; ./run.sh all 2 3 --no-silo
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from results import ALIASES, CASES, digest, read_config, read_conservation, read_norms, read_slice, write_json

HERE = Path(__file__).resolve().parent

def execute(command, folder, log):
    print('+', shlex.join(map(str, command)), flush=True)
    with Path(log).open('w') as out:
        process = subprocess.Popen(list(map(str, command)), cwd=folder, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); out.write(line); out.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if code: raise RuntimeError(f'Exit code {code}; see {log}')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', choices=(*CASES, *ALIASES, 'all'))
    p.add_argument('levels_and_build', nargs='*', help='levels then optional release/debug; default: 2 3 release')
    p.add_argument('--root', type=Path, default=HERE.parent, help='Octo-TIGER checkout')
    p.add_argument('--build', default=None, help='build directory, absolute or relative to checkout')
    p.add_argument('--exe', type=Path, help='use a specific octotiger binary')
    p.add_argument('--generator', type=Path, help='gen_radiation_reference binary')
    p.add_argument('--output', type=Path, help='new, empty output directory')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--time', type=float, default=.2)
    p.add_argument('--odt', type=float, default=.05, help='Silo snapshot interval in code time')
    p.add_argument('--hard-dt', type=float, help='maximum timestep in code time')
    p.add_argument('--no-build', action='store_true')
    p.add_argument('--no-silo', action='store_true', help='keep CSV slices and norms, skip Silo files')
    p.add_argument('--dry-run', action='store_true', help='print planned resolutions; write/build/run nothing')
    a = p.parse_args()
    parts = list(a.levels_and_build); build_name = a.build or 'release'
    if parts and not parts[-1].isdigit():
        if a.build: p.error('Specify build only once')
        build_name = parts.pop()
    try: levels = sorted(set(map(int, parts or ['2','3'])))
    except ValueError: p.error('Levels must be integers')
    if not levels or min(levels)<0 or max(levels)>9: p.error('Levels must be between 0 and 9')
    import math
    if a.threads<1 or a.jobs<1 or not math.isfinite(a.time) or a.time<=0 or not math.isfinite(a.odt) or a.odt<=0:
        p.error('threads, jobs, time, and odt must be positive and finite')
    if a.hard_dt is not None and (not math.isfinite(a.hard_dt) or a.hard_dt<=0):
        p.error('--hard-dt must be positive and finite')
    root=a.root.resolve(); build=(root/build_name).resolve()
    exe=(a.exe or build/'octotiger').resolve()
    cases=CASES if a.case=='all' else (ALIASES.get(a.case,a.case),)
    cache=(build/'CMakeCache.txt').read_text()
    match=re.search(r'^OCTOTIGER_WITH_GRIDDIM:[^=]+=([0-9]+)\s*$', cache, re.M)
    if not match: raise ValueError(f'Cannot determine INX from {build}/CMakeCache.txt')
    inx=int(match[1])
    grid_source=root/'src/grid.cpp'
    if 'RADIATION_PLOT_EXPORT_BEGIN' not in grid_source.read_text():
        raise ValueError(f'First run: python3 {HERE}/install.py {root}')
    for level in levels:
        for case in cases:
            n=inx*2**level
            print(f'{case:20s} level={level}  N={n}  cells={n**3:,}  t={a.time:g}')
            if case=='gaussian_pulse' and (n>512 or n<4 or n%2):
                raise ValueError('Gaussian generator supports even N from 4 through 512')
    if a.dry_run: return
    # Check plot dependencies before spending time on simulations.
    import numpy, matplotlib
    if not a.no_build:
        targets=['octotiger']
        if 'gaussian_pulse' in cases and a.generator is None: targets+=['gen_radiation_reference']
        subprocess.run(['cmake','--build',str(build),'--target',*targets,'-j',str(a.jobs)],check=True)
    if not exe.is_file(): raise ValueError(f'No executable: {exe}')
    generator=a.generator.resolve() if a.generator else None
    if 'gaussian_pulse' in cases and generator is None:
        candidates=list(build.rglob('gen_radiation_reference'))
        candidates=[f for f in candidates if f.is_file() and os.access(f,os.X_OK)]
        if len(candidates)!=1: raise ValueError('Use --generator to select gen_radiation_reference')
        generator=candidates[0].resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output=(a.output or HERE/'results'/stamp).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError(f'Output directory is not empty: {output}')
    output.mkdir(parents=True,exist_ok=True)
    binary_hash=digest(exe)
    source_hashes={str(f.relative_to(root)):digest(f) for f in [grid_source,
        root/'src/radiation/rad_grid.cpp', root/'octotiger/test_problems/radiation/profiles.hpp',
        root/'octotiger/test_problems/radiation/plot_output.hpp',
        root/'octotiger/radiation/conservation.hpp', root/'octotiger/radiation/rad_grid.hpp',
        root/'src/node_server_actions_3.cpp']}
    write_json(output/'batch.json',dict(created_utc=stamp,root=str(root),build=str(build),
        executable=str(exe),executable_sha256=binary_hash,source_sha256=source_hashes,
        origin='Octo-TIGER application',cases=cases,levels=levels))
    print(f'Results: {output}',flush=True)
    for level in levels:
        for case in cases:
            folder=output/case/f'l{level}'; folder.mkdir(parents=True)
            (folder/'radiation-slices').mkdir()
            config=read_config(HERE/'configs'/f'{case}.ini')
            config.update(max_level=str(level),min_level=str(level),stop_time=str(a.time),
                odt=str(a.odt),disable_output='on' if a.no_silo else 'off',
                disable_diagnostics='on', disable_analytic='off', datadir=str(folder)+'/',
                # Avoid spatially uniform species proliferation in plots/diagnostics.
                n_species='1',atomic_mass='1',atomic_number='1')
            if a.hard_dt is not None:
                cap=float(config.get('hard_dt',0))
                config['hard_dt']=str(min(a.hard_dt,cap) if cap>0 else a.hard_dt)
            n=inx*2**level; length=2*float(config['xscale'])
            meta=dict(case=case,level=level,inx=inx,cells=n,length=length,dx=length/n,
                time=a.time,c=1.,origin='Octo-TIGER application',status='running',
                background=float(config['rad_test_background']),executable_sha256=binary_hash)
            write_json(folder/'run.json',meta)
            try:
                if case=='gaussian_pulse':
                    ref=folder/'reference.bin'
                    cmd=[generator,'--output',ref,'--cells',str(n),'--length',str(length),
                        '--c','1','--chi',config['rad_test_chi'],'--width',config['rad_test_width'],
                        '--background',config['rad_test_background'],'--amplitude',config['rad_test_amplitude'],
                        '--time',str(a.time)]
                    execute(cmd,folder,folder/'reference.log')
                    config['rad_reference']=str(ref)
                    meta['reference_sha256']=digest(ref)
                cfg=folder/'run.ini'
                cfg.write_text('\n'.join(f'{k}={v}' for k,v in config.items())+'\n')
                meta['config']=config
                meta['comparison_signature']={k:v for k,v in config.items() if k not in
                    {'max_level','min_level','datadir','rad_reference','disable_output','odt'}}
                cmd=[exe,'--config_file='+str(cfg),'--hpx:threads='+str(a.threads)]
                meta['command']=list(map(str,cmd)); write_json(folder/'run.json',meta)
                execute(cmd,folder,folder/'run.log')
                meta['norms']=read_norms(folder,meta)
                read_slice(folder,meta)
                conservation=read_conservation(folder,meta)
                if conservation is None:
                    raise ValueError('Missing radiation-conservation.csv; rebuild octotiger with the conservation diagnostics')
                meta['conservation']=conservation['summary']
                # Passing means complete/finite data, not a substituted CTest tolerance check.
                meta['status']='complete'; write_json(folder/'run.json',meta)
            except BaseException as e:
                meta.update(status='failed',error=str(e));write_json(folder/'run.json',meta)
                raise
    from plot import render
    report=render(output)
    print(f'Open {report}')

if __name__=='__main__':
    try: main()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as e: sys.exit(str(e))
