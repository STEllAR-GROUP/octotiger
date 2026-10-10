#!/usr/bin/env python3
"""Compatibility entry point for the maintained production-grid checks."""
from pathlib import Path
import subprocess
import sys

command = [sys.executable, str(Path(__file__).resolve().parents[1] / "validateRadiationGrid.py")]
for argument in sys.argv[1:]:
    if argument == "--sanitize":
        command.extend(["--mode", "ubsan"])
    elif argument == "--release":
        command.extend(["--mode", "release"])
    else:
        command.append(argument)
subprocess.run(command, check=True)
