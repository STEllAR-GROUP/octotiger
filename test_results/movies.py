#!/usr/bin/env python3
"""Render numerical Silo snapshots with VisIt and encode inspection-length MP4s."""
import argparse
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from movie_support import check_dependencies, encode, executable, frame_schedule, numerical_silos
from results import CASES, read_norms, write_json

HERE = Path(__file__).resolve().parent
SUPPORTS_CGS = True


def render_signature(files, field, view, axis, position, width, height, color, minimum, maximum):
    return dict(silos=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files],
                field=field,view=view,axis=axis,slice_position=position,width=width,height=height,
                color_table=color,minimum=minimum,maximum=maximum)


def make_movie(folder, seconds=20., fps=30, hold=1., visit='visit', ffmpeg='ffmpeg',
               field='er', view='slice', axis='z', position=0., width=1280, height=960,
               color='hot', minimum=None, maximum=None, reuse_frames=False, allow_sparse=False):
    folder=Path(folder).resolve()
    meta=json.loads((folder/'run.json').read_text())
    if meta.get('status')!='complete': raise ValueError('Simulation is not complete: '+str(folder))
    read_norms(folder,meta)
    frame_schedule([0.,1.],seconds,fps,hold)  # Validate playback before rendering.
    if width<128 or height<128 or not math.isfinite(position): raise ValueError('Invalid movie size or slice position')
    if any(v is not None and not math.isfinite(v) for v in (minimum,maximum)):
        raise ValueError('Color limits must be finite')
    if minimum is not None and maximum is not None and minimum>=maximum:
        raise ValueError('Color minimum must be less than maximum')
    files=numerical_silos(folder)
    signature=render_signature(files,field,view,axis,position,width,height,color,minimum,maximum)
    cgs=meta.get('units',{}).get('system')=='CGS'
    if cgs:
        signature.update(units=meta['units'],expected_final_time=meta['time'])
    key=field+'-'+view+('-'+axis if view=='slice' else '')
    destination=folder/'movies'/key
    destination.mkdir(parents=True,exist_ok=True)
    current=destination/'render.json'
    if reuse_frames:
        saved=json.loads(current.read_text())
        if saved['signature']!=signature:
            raise ValueError('Saved frames have different inputs/style; rerun without --reuse-frames')
        frames_path=Path(saved['frames_json'])
    else:
        visit,ffmpeg=check_dependencies(visit,ffmpeg)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
        frames_folder=destination/'renders'/stamp
        frames_folder.mkdir(parents=True)
        requested=meta.get('movie_capture',{}).get('requested_snapshots',3)
        cfg=dict(silos=list(map(str,files)),render_directory=str(frames_folder),field=field,
                 title='%s | %d^3 cells' % (meta['case'].replace('_',' '),meta['cells']),
                 view=view,axis=axis,slice_position=position,width=width,height=height,
                 color_table=color,minimum=minimum,maximum=maximum,
                 minimum_snapshots=2 if allow_sparse else max(2,requested-1))
        if cgs:
            cfg.update(units=meta['units'],expected_final_time=meta['time'])
        settings=frames_folder/'settings.json';write_json(settings,cfg)
        command=[visit,'-nowin','-cli','-s',str(HERE/'visit_movie.py'),str(settings)]
        # The parent runner already handles cleanup and live log forwarding.
        from run import execute
        frames_path=frames_folder/'frames.json'
        try:
            execute(command,folder,frames_folder/'visit.log')
        except RuntimeError as error:
            if not str(error).startswith('Exit code 250;'): raise
            if not frames_path.is_file(): raise
            completed=json.loads(frames_path.read_text())
            if completed.get('status')!='complete': raise
            print('VisIt exited with 250 after saving frames; continuing.',flush=True)
        if not frames_path.is_file():
            raise RuntimeError('VisIt produced no completed frames.json; see '+str(frames_folder/'visit.log'))
        rendered=json.loads(frames_path.read_text())
        if rendered.get('status')!='complete': raise ValueError('VisIt rendering did not complete')
        write_json(current,dict(signature=signature,frames_json=str(frames_path)))
    rendered=json.loads(frames_path.read_text())
    actual=len(rendered['frames'])
    if actual<20:
        print('Only %d distinct snapshots: playback will be slowed, but motion will be coarse.' % actual,flush=True)
    result=encode(rendered,destination/'movie.mp4',seconds,fps,hold,ffmpeg)
    result.update(status='complete',case=meta['case'],cells=meta['cells'],field=field,
                  view=view,axis=axis,frames_json=str(frames_path),simulation_directory=str(folder),
                  color_limits=rendered['color_limits'],time_units=rendered['time_units'],
                  field_units=rendered.get('field_units','as stored in Silo'),
                  units=meta.get('units',{}))
    write_json(destination/'movie.json',result)
    print('Movie: %s (%g seconds, %d stored snapshots)' % (result['path'],result['seconds'],actual),flush=True)
    if result['displayed_snapshots']<actual:
        print('Some closely spaced snapshots were skipped at this playback rate; increase --seconds or --fps.',flush=True)
    return result


def movie_index(batch):
    batch=Path(batch).resolve();entries=[]
    for path in batch.glob('**/movies/*/movie.json'):
        row=json.loads(path.read_text())
        if row.get('status')=='complete' and Path(row['path']).is_file(): entries.append(row)
    if not entries:return None
    entries.sort(key=lambda r:(r['cells'],r['case'],r['field'],r['view']))
    parts=[]
    for row in entries:
        relative=html.escape(os.path.relpath(row['path'],batch),quote=True)
        label=html.escape('%s | %d^3 | %s | %s' % (row['case'],row['cells'],row['field'],row['view']))
        if row.get('units',{}).get('system')=='CGS':
            label += ' | ' + html.escape(row['field_units'])
        parts.append('<section><h2>'+label+'</h2><p>%g seconds; %d stored snapshots; %d fps MP4. '
                     '<a href="%s">Download MP4</a></p><video controls preload="metadata" src="%s"></video></section>'
                     % (row['seconds'],row['source_snapshots'],row['fps'],relative,relative))
    path=batch/'movies.html'
    path.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Octo-TIGER radiation movies</title><style>body{font:18px/1.5 system-ui;margin:32px auto;padding:0 24px;max-width:1400px;color:#172335;background:#fafbfc}video{width:100%;background:#fff}section{margin:40px 0;border-top:1px solid #bbc;padding-top:20px}h2{font-size:24px}a{color:#0755a2}</style>
<h1>Octo-TIGER radiation movies</h1><p>Numerical Silo states with fixed camera and color scale within each movie.
Physical time spacing is retained. Repeated video frames slow playback; no intermediate solution is synthesized.
CGS runs show time in seconds, positions in cm, energy density in erg/cm³ and flux in erg/(cm² s).
Other runs use their saved Silo output units. The reference file analytic.silo is excluded.</p>'''+''.join(parts)+'</html>')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='a batch directory or one case/level directory')
    p.add_argument('--seconds',type=float,default=20.,help='total MP4 duration, including endpoint holds')
    p.add_argument('--fps',type=int,default=30,help='encoded video frame rate')
    p.add_argument('--hold',type=float,default=1.,help='seconds held at each endpoint')
    p.add_argument('--visit',default='visit',help='VisIt executable, not a Python package')
    p.add_argument('--ffmpeg',default='ffmpeg')
    p.add_argument('--field',choices=('er','fx','fy','fz','fluxmag'),default='er')
    p.add_argument('--view',choices=('slice','3d'),default='slice')
    p.add_argument('--axis',choices=('x','y','z'),default='z')
    p.add_argument('--position',type=float,default=0.,help='slice position in Silo length units')
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=960)
    p.add_argument('--color',default='hot',help='VisIt color table name')
    p.add_argument('--minimum',type=float);p.add_argument('--maximum',type=float)
    p.add_argument('--reuse-frames',action='store_true',help='change playback without running VisIt again')
    p.add_argument('--allow-sparse',action='store_true',help='render fewer snapshots than movie mode requested')
    a=p.parse_args(); root=a.input.resolve()
    files=[root/'run.json'] if (root/'run.json').is_file() else list(root.glob('*/l*/run.json'))
    runs=[]
    for path in files:
        m=json.loads(path.read_text())
        if m.get('status')=='complete' and m.get('case') in CASES:
            runs.append((m['cells'],m['case'],path.parent))
    if not runs:raise ValueError('No completed runs found at '+str(root))
    for _,_,folder in sorted(runs):
        make_movie(folder,seconds=a.seconds,fps=a.fps,hold=a.hold,visit=a.visit,ffmpeg=a.ffmpeg,
                   field=a.field,view=a.view,axis=a.axis,position=a.position,width=a.width,height=a.height,
                   color=a.color,minimum=a.minimum,maximum=a.maximum,reuse_frames=a.reuse_frames,
                   allow_sparse=a.allow_sparse)
        movie_index(root)
    print('Open '+str(root/'movies.html'))


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:sys.exit('Movie generation interrupted; completed simulations and frames are retained.')
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as e:sys.exit(str(e))
