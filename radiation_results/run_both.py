#!/usr/bin/env python3
"""Prepare or explicitly start a paired M1 RK1 / FLD local campaign.

Example: run_both.py --output output/m1-fld-local --prepare --levels 0 1
Starting requires a later: run_both.py --resume output/m1-fld-local --start
All options are defined by run_campaign.py; legacy ADER studies keep their
own frozen scripts and are never reinterpreted as FLD.
"""
import sys
import run_campaign


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(a.split('=')[0] in ('--models', '--radiation-integrator', '--radiation-model') for a in arguments):
        raise ValueError('run_both always selects M1 RK1 and FLD; use run_local for one model')
    return run_campaign.main(arguments)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
