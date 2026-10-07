#!/usr/bin/env python3
"""Preserve quoted definitions through an unchanged Kokkos nvcc_wrapper --host-only.

The upstream wrapper shell-escapes -D arguments for NVCC, then reuses them
without shell parsing for its direct host compiler invocation. GCC sees literal
backslashes instead of string-literal quotes. Keep Kokkos untouched and pass
host definitions in a GCC response file through the wrapper's -Xcompiler path.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

command = sys.argv[1:]
if not command:
    raise SystemExit('usage: nvccHostDefinitions.py COMPILER [ARG ...]')
if '--host-only' not in command:
    os.execvp(command[0], command)

definitions = [arg for arg in command[1:] if arg.startswith(('-D', '-U'))]
if not definitions:
    os.execvp(command[0], command)
command = [command[0]] + [arg for arg in command[1:] if not arg.startswith(('-D', '-U'))]
with tempfile.TemporaryDirectory(prefix='octotiger-host-definitions-') as directory:
    response = Path(directory) / 'definitions.args'
    response.write_text('\n'.join('"' + arg.replace('\\', '\\\\').replace('"', '\\"') + '"'
                                  for arg in definitions) + '\n')
    result = subprocess.run(command + ['-Xcompiler', '@' + str(response)])
raise SystemExit(result.returncode)
