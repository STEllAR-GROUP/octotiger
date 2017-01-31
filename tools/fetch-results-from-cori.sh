#!/bin/bash

# fetch new results from cori
ssh cori <<'ENDSSH'
cd $SCRATCH/cori
rm -f results.tar.gz
tar zcf results.tar.gz results
ENDSSH

scp cori:/global/cscratch1/sd/pfandedd/cori/results.tar.gz .
tar xf results.tar.gz

cp results.tar.gz node-level-scaling-latest.tar
mv results.tar.gz node-level-scaling-`date +%F`.tar

./node-level-scaling-graph.py -w

cp total-time-latest.png total-time-second.png
cp total-time.png total-time-latest.png
mv total-time.png total-time-`date +%F`.png

cp parallel-efficiency-latest.png parallel_efficiency-second.png
cp parallel-efficiency.png parallel-efficiency-latest.png
mv parallel-efficiency.png parallel-efficiency-`date +%F`.png
