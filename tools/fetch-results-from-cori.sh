#!/bin/bash

compiler=$1

if [[ ! ($compiler == "gnu" || $compiler == "intel") ]]; then
		echo "specify either \"intel\" or \"gnu\""
		exit 1
fi;

# fetch new results from cori
if [[ $compiler == "intel" ]]; then
ssh cori <<'ENDSSH'
cd $SCRATCH/cori
mv -f results results-intel
rm -f results-intel.tar.gz
tar zcf results-intel.tar.gz results-intel
ENDSSH

scp cori:/global/cscratch1/sd/pfandedd/cori/results-intel.tar.gz .

elif [[ $compiler == "gnu" ]]; then
ssh cori <<'ENDSSH'
cd $SCRATCH/cori/gcc
mv -f results results-gcc
rm -f results-gcc.tar.gz
tar zcf results-gcc.tar.gz results-gcc
ENDSSH

scp cori:/global/cscratch1/sd/pfandedd/cori/gcc/results-gcc.tar.gz .
fi

rm -Rf results-$compiler
tar xf results-$compiler.tar.gz

cp results-$compiler.tar.gz node-level-scaling-latest-$compiler.tar
mv results-$compiler.tar.gz node-level-scaling-`date +%F`-$compiler.tar

./node-level-scaling-graph.py -w -c $compiler

cp total-time-latest-$compiler.png total-time-second-$compiler.png
cp total-time-$compiler.png total-time-latest-$compiler.png
mv total-time-$compiler.png total-time-`date +%F`-$compiler.png

cp parallel-efficiency-latest-$compiler.png parallel_efficiency-second-$compiler.png
cp parallel-efficiency-$compiler.png parallel-efficiency-latest-$compiler.png
mv parallel-efficiency-$compiler.png parallel-efficiency-`date +%F`-$compiler.png
