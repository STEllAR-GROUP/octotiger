#!/usr/bin/env python3
"""Compose a CMake compiler launcher with unchanged Kokkos CUDA launchers.

The upstream wrapper shell-escapes -D arguments for NVCC, then reuses them
without shell parsing for its direct host compiler invocation. GCC sees literal
backslashes instead of string-literal quotes. Keep Kokkos untouched and pass
host definitions in a GCC response file through the wrapper's -Xcompiler path.

Kokkos's outer launch rule does not redirect CMake's Python launcher to NVCC.
For Kokkos targets, re-enter its launcher with the expected compiler pair so
its installed host-compiler defaults and recursion checks still apply.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

command = sys.argv[1:]
kokkos_compiler = None
if command[:1] == ['--kokkos-compiler']:
    if len(command) < 3:
        raise SystemExit('usage: nvccHostDefinitions.py [--kokkos-compiler PATH] COMPILER [ARG ...]')
    kokkos_compiler = command[1]
    command = command[2:]
if not command:
    raise SystemExit('usage: nvccHostDefinitions.py [--kokkos-compiler PATH] COMPILER [ARG ...]')
kokkos_definition = '-DKOKKOS_DEPENDENCE'
if (kokkos_compiler and kokkos_definition in command[1:]
        and Path(command[0]).resolve() != Path(kokkos_compiler).resolve()):
    launcher = str(Path(kokkos_compiler).with_name('kokkos_launch_compiler'))
    command = [launcher, kokkos_compiler, command[0]] + command
if '--host-only' not in command:
    os.execvp(command[0], command)

# The inner Kokkos launcher must still see its routing marker on the command line.
definitions = [arg for arg in command[1:] if arg.startswith(('-D', '-U'))
               and arg != kokkos_definition]
if not definitions:
    os.execvp(command[0], command)
command = [command[0]] + [arg for arg in command[1:]
                          if not arg.startswith(('-D', '-U')) or arg == kokkos_definition]
with tempfile.TemporaryDirectory(prefix='octotiger-host-definitions-') as directory:
    response = Path(directory) / 'definitions.args'
    response.write_text('\n'.join('"' + arg.replace('\\', '\\\\').replace('"', '\\"') + '"'
                                  for arg in definitions) + '\n')
    result = subprocess.run(command + ['-Xcompiler', '@' + str(response)])
raise SystemExit(result.returncode)
