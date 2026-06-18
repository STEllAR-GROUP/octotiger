#!/usr/bin/env bash

WORK="$HOME/workspace/octotiger"

set -x
cd "$WORK/debug" || exit 1
make -j all || exit 1

cp octotiger "$WORK/radiation_results/streaming/" || exit 1

cd $WORK/radiation_results/streaming/

for level in 2 ; do
    echo "Running max_level=$level"

    stdbuf -oL -eL ./octotiger --config_file=./streaming.ini --max_level="$level" --hpx:ini=hpx.stacks.small_size=0x2000000 --hpx:ini=hpx.stacks.medium_size=0x4000000 --hpx:ini=hpx.stacks.large_size=0x8000000 --datadir="$WORK/radiation_results/streaming/data/l$level/" 2>&1 | tee "output.$level.txt"
   mv $HOME/workspace/octotiger/radiation_results/streaming/data/l$level/final.silo $HOME/workspace/octotiger/radiation_results/streaming/data/l$level/X.9998.silo
   mv $HOME/workspace/octotiger/radiation_results/streaming/data/l$level/analytic.silo $HOME/workspace/octotiger/radiation_results/streaming/data/l$level/X.9999.silo
   echo "Exit code for level $level was ${PIPESTATUS[0]}"
done 
  
  
