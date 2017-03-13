#!/bin/bash

name=distributed_scaling
level=1
MAX_NODES=8

#run the application with datapar (DRAM/MCDRAM):
COUNTER=1
while [ $COUNTER -le $MAX_NODES ]; do
    sbatch -N $COUNTER -n $COUNTER run-single-distributed-experiment.sh ${name} ${level} 68
    let COUNTER=COUNTER*2
done
