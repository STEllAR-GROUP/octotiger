#!/bin/bash

compiler=$1

if [[ ! ($compiler == "gnu" || $compiler == "intel") ]]; then
		echo "specify either \"intel\" or \"gnu\""
		exit 1
fi;

if [[ $compiler == "intel" ]]; then

ssh cori <<'ENDSSH'
cd $SCRATCH/cori
./build-all.sh
./run-all-scaling-experiments-single-node.sh
ENDSSH
elif [[ $compiler == "gnu" ]]; then
ssh cori <<'ENDSSH'
cd $SCRATCH/cori/gcc
./build-all.sh
./run-all-scaling-experiments-single-node.sh
ENDSSH
fi

echo "Jobs submitted, now wait for them to finish"
