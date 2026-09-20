#!/bin/sh
# Historical standalone comparison. Run in a prepared output directory containing
# sod.ini and original.silo, or from its parent containing sod/.
set -eu
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /absolute/path/to/octotiger /absolute/path/to/silodiff" >&2
    exit 2
fi
OCTOTIGER=$1
SILODIFF=$2
if [ ! -f sod.ini ]; then
    cd sod
fi
"$OCTOTIGER" --config_file=sod.ini
"$SILODIFF" -A 1.0e-10 -R 1.0e-10 original.silo final.silo
