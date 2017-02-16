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
tar zcf results-distributed-scaling-intel.tar.gz results/distributed_scaling*
ENDSSH

scp cori:/global/cscratch1/sd/pfandedd/cori/results-distributed-scaling-intel.tar.gz .

elif [[ $compiler == "gnu" ]]; then
ssh cori <<'ENDSSH'
cd $SCRATCH/cori/gcc
rm -f results-gnu.tar.gz
tar zcf results-distributed-scaling-gnu.tar.gz results/distributed_scaling*
ENDSSH

scp cori:/global/cscratch1/sd/pfandedd/cori/gcc/results-distributed-scaling-gnu.tar.gz .
fi

rm -Rf results-distributed-scaling-$compiler
tar xf results-distributed-scaling-$compiler.tar.gz

mv results results-distributed-scaling-$compiler

cp results-distributed-scaling-$compiler.tar.gz results-distributed-scaling-$compiler-latest.tar.gz
mv results-distributed-scaling-$compiler.tar.gz results-distributed-scaling-$compiler-`date +%F`.tar.gz

./distributed-scaling-graph.py -w -c $compiler

cp total-time-distributed-scaling-$compiler-latest.png total-time-distributed-scaling-$compiler-second.png
cp total-time-distributed-scaling-$compiler.png total-time-distributed-scaling-$compiler-latest.png
mv total-time-distributed-scaling-$compiler.png total-time-distributed-scaling-$compiler-`date +%F`.png

cp parallel-efficiency-distributed-scaling-$compiler-latest.png parallel_efficiency-distributed-scaling-$compiler-second.png
cp parallel-efficiency-distributed-scaling-$compiler.png parallel-efficiency-distributed-scaling-$compiler-latest.png
mv parallel-efficiency-distributed-scaling-$compiler.png parallel-efficiency-distributed-scaling-$compiler-`date +%F`.png
