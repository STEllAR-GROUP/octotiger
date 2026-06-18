#!/usr/bin/env bash

ROOT="$HOME/workspace/octotiger"
WORK="$ROOT/radiation_results/diffusion/"

set -x
cd "$ROOT/release" || exit 1
make -j all || exit 1

cp octotiger $WORK || exit 1

cd $WORK

for level in 2 3 4; do
   echo "Running max_level=$level"
   DATA="$WORK/data/l$level/"
   rm -r $DATA/*
   stdbuf -oL -eL ./octotiger --config_file=./radiation_diffusion.ini --max_level="$level" --hpx:ini=hpx.stacks.small_size=0x2000000 --hpx:ini=hpx.stacks.medium_size=0x4000000 --hpx:ini=hpx.stacks.large_size=0x8000000 --datadir="$DATA/" 2>&1 | tee "output.$level.txt"
   mv $DATA/final.silo $DATA/X.9998.silo
   mv $DATA/analytic.silo $DATA/X.9999.silo
   echo "Exit code for level $level was ${PIPESTATUS[0]}"
done 
  
  
