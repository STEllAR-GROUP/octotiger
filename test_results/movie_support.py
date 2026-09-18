"""Snapshot scheduling and time-faithful MP4 encoding (no VisIt imports)."""
import bisect
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def cadence(final_time, snapshots, cfl, odt=None):
    if snapshots < 3 or not math.isfinite(final_time) or final_time <= 0:
        raise ValueError('Movies require positive final time and at least 3 requested snapshots')
    if not math.isfinite(cfl) or cfl <= 0:
        raise ValueError('Invalid CFL for movie scheduling')
    interval = final_time / (snapshots - 1) if odt is None else odt
    if not math.isfinite(interval) or not 0 < interval < final_time:
        raise ValueError('Movie odt must be positive and smaller than the final time')
    # octotiger/util.hpp: refinement_freq() = integer(2.0 / opts().cfl + 0.5).
    # execute_solver checks Silo output once per refinement_freq() timesteps.
    batch_steps = max(1, int(2.0 / cfl + .5))
    return dict(odt=interval, hard_dt=interval / batch_steps,
                steps_per_output_check=batch_steps,
                requested_snapshots=int(math.ceil(final_time / interval - 1e-10)) + 1)


def numerical_silos(folder):
    folder = Path(folder)
    snapshots = []
    for file in folder.glob('X.*.silo'):
        m = re.fullmatch(r'X\.(\d+)\.silo', file.name)
        if m: snapshots.append((int(m[1]), file.resolve()))
    files = [f for _, f in sorted(snapshots)]
    final = folder / 'final.silo'
    if not files or not final.is_file():
        raise ValueError('Need X.*.silo snapshots and final.silo; run with Silo output enabled')
    # analytic.silo is a reference, never the last frame of the numerical movie.
    files.append(final.resolve())
    for file in files:
        if not file.stat().st_size or not file.with_name(file.name + '.data').is_dir():
            raise ValueError('Missing or empty Silo root/data directory: ' + str(file))
    return files


def executable(name):
    found = shutil.which(str(name))
    if not found:
        raise ValueError('Executable not found: %s. Use its full path.' % name)
    return str(Path(found).resolve())


def check_dependencies(visit='visit', ffmpeg='ffmpeg'):
    visit = executable(visit)
    ffmpeg = executable(ffmpeg)
    result = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], check=True,
                            capture_output=True, text=True)
    if 'libx264' not in result.stdout:
        raise ValueError('FFmpeg needs the libx264 encoder for browser-compatible MP4')
    return visit, ffmpeg


def frame_schedule(times, seconds=20., fps=30, hold=1.):
    """Map output frames to stored states, retaining actual relative time gaps.

    Hold the last available solution until the next stored state. No synthetic
    intermediate solution, blending, or optical-flow interpolation is generated.
    """
    if len(times) < 2 or any(not math.isfinite(t) for t in times):
        raise ValueError('Need at least two finite snapshot times')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Snapshot times must be strictly increasing')
    if not math.isfinite(seconds) or not math.isfinite(hold) or fps < 1 or seconds <= 0 or hold < 0:
        raise ValueError('Invalid movie duration, frame rate, or endpoint hold')
    total = round(seconds * fps); pause = round(hold * fps)
    active = total - 2 * pause
    if active < 2: raise ValueError('Movie duration must exceed both endpoint holds')
    result = [0] * pause
    for i in range(active):
        t = times[0] + (times[-1] - times[0]) * i / (active - 1)
        result.append(min(len(times)-1, max(0, bisect.bisect_right(times, t)-1)))
    # Protect the final state against rounding in the time interpolation above.
    result[-1] = len(times)-1
    result.extend([len(times)-1] * pause)
    return result


def encode(render, output, seconds=20., fps=30, hold=1., ffmpeg='ffmpeg'):
    output = Path(output).resolve()
    states = render['frames']
    sequence = frame_schedule([f['time'] for f in states], seconds, fps, hold)
    sources = [Path(f['image']).resolve() for f in states]
    for path in sources:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing rendered frame: ' + str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Temporary links repeat recorded frames without duplicating image bytes.
    # A numeric sequence also avoids shell/ffconcat quoting of user paths.
    with tempfile.TemporaryDirectory(prefix='encode-', dir=output.parent) as temp:
        temp = Path(temp)
        for i, state in enumerate(sequence):
            dest = temp / ('frame_%06d.png' % i)
            try: os.link(sources[state], dest)
            except OSError: dest.symlink_to(sources[state])
        staged = temp / 'movie.mp4'
        command = [executable(ffmpeg), '-hide_banner', '-loglevel', 'warning', '-y',
                   '-framerate', str(fps), '-start_number', '0', '-i', str(temp/'frame_%06d.png'),
                   '-frames:v', str(len(sequence)), '-an', '-c:v', 'libx264', '-crf', '18',
                   '-preset', 'medium', '-pix_fmt', 'yuv420p',
                   '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart', str(staged)]
        subprocess.run(command, check=True)
        if not staged.is_file() or not staged.stat().st_size:
            raise ValueError('FFmpeg did not produce an MP4')
        staged.replace(output)
    return dict(path=str(output), seconds=len(sequence)/fps, fps=fps, video_frames=len(sequence),
                source_snapshots=len(states), displayed_snapshots=len(set(sequence)),
                endpoint_hold_seconds=round(hold*fps)/fps,
                timing='Actual Silo time gaps; recorded states held without interpolation')
