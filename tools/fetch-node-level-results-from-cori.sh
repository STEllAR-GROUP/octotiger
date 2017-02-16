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
rm -f results-intel.tar.gz
tar zcf results-node-level-scaling-intel.tar.gz results/node_level_scaling*
ENDSSH

scp cori:/global/cscratch1/sd/pfandedd/cori/results-node-level-scaling-intel.tar.gz .

elif [[ $compiler == "gnu" ]]; then
ssh cori <<'ENDSSH'
cd $SCRATCH/cori/gcc
rm -f results-gnu.tar.gz
tar zcf results-node-level-scaling-gnu.tar.gz results/node_level_scaling*
ENDSSH

scp cori:/global/cscratch1/sd/pfandedd/cori/gcc/results-node-level-scaling-gnu.tar.gz .
fi

rm -Rf results-node-level-scaling-$compiler
tar xf results-node-level-scaling-$compiler.tar.gz

mv results results-node-level-scaling-$compiler

cp results-node-level-scaling-$compiler.tar.gz results-node-level-scaling-$compiler-latest.tar.gz
mv results-node-level-scaling-$compiler.tar.gz results-node-level-scaling-$compiler-`date +%F`.tar.gz

./node-level-scaling-graph.py -w -c $compiler

cp total-time-node-level-scaling-$compiler-latest.png total-time-node-level-scaling-$compiler-second.png
cp total-time-node-level-scaling-$compiler.png total-time-node-level-scaling-$compiler-latest.png
mv total-time-node-level-scaling-$compiler.png total-time-node-level-scaling-$compiler-`date +%F`.png

cp parallel-efficiency-node-level-scaling-$compiler-latest.png parallel_efficiency-node-level-scaling-$compiler-second.png
cp parallel-efficiency-node-level-scaling-$compiler.png parallel-efficiency-node-level-scaling-$compiler-latest.png
mv parallel-efficiency-node-level-scaling-$compiler.png parallel-efficiency-node-level-scaling-$compiler-`date +%F`.png
