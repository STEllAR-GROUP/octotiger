#!/bin/bash
#PBS -j oe
#PBS -l nodes=128:ppn=20 
#PBS -l walltime=06:00:00
#PBS -q checkpt

set -x

cd $WORK/octotiger/debug

make -k all -j21

uniq $PBS_NODEFILE >actual.nodes.$$

export HPX_NODEFILE=actual.nodes.$$

unset PBS_NODEFILE

mpirun -f $HPX_NODEFILE ./octotiger -t20
