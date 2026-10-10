#!/usr/bin/env python3
"""Ensure CTest fails on invalid, absent, stale and inaccurate results."""
import contextlib, importlib.util, io, pathlib
path=pathlib.Path(__file__).resolve().parents[2]/'test_problems/radiation/check.py'
spec=importlib.util.spec_from_file_location('radiation_check',path)
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
valid='RADIATION_TEST_FINISHED RADIATION_GAUSSIAN_PULSE t=0.2\n'+''.join(f'{f} 0 0 0\n' for f in ('er','fx','fy','fz'))
with contextlib.redirect_stdout(io.StringIO()): check.check(valid,'gaussian_pulse',.2)
for text in ['',valid.replace('t=0.2','t=0.1'),valid.replace('er 0 0 0','er nan 0 0'),
             valid.replace('fx 0 0 0','fx 0 inf 0'), valid.replace('fy 0 0 0\n',''),
             valid.replace('fz 0 0 0','fz -1 0 0'),valid.replace('er 0 0 0','er 1 1 1'),
             valid+valid,valid.replace('GAUSSIAN_PULSE','STREAMING_FRONT')]:
    try:
        with contextlib.redirect_stdout(io.StringIO()): check.check(text,'gaussian_pulse',.2)
    except ValueError: continue
    raise RuntimeError('Comparator accepted invalid output: '+repr(text))
print('Comparator rejection checks passed')
