#!/bin/bash

# fetch new results from cori
ssh cori <<'ENDSSH'
cd $SCRATCH/cori
./build-all.sh
./run-all-scaling-experiments-single-node.sh
ENDSSH

echo "Jobs submitted, now wait for them to finish"
