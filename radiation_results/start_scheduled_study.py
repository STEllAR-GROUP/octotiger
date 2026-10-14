#!/usr/bin/env python3
"""Explicit local campaign entry point; no automatic schedule or legacy resume.

Use --resume PREPARED_DIRECTORY --start only after authorizing execution.
The old Euler/ADER2 study retains its saved scripts and STOP controls.
"""
import sys
from run_campaign import main

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
