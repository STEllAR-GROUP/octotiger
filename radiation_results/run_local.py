#!/usr/bin/env python3
"""Local model-aware radiation preparation and explicit execution.

Uses the same frozen campaign workflow as run_both. Select --models m1 or
--models fld for one model, or omit it for both. --prepare performs no solver
runs; --resume DIR --start is required for execution. No QB access is used.
"""
import sys
from run_campaign import main

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
