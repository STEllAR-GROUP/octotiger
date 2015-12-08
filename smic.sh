#!/bin/bash
#PBS -j oe
#PBS -l nodes=64:ppn=20 
#PBS -l walltime=24:00:00
#PBS -q checkpt
set -x
cd /work/$USER/octotiger
make -j20
cat $PBS_NODEFILE | awk 'NR % 20 == 0' > node.list
export HPX_NODEFILE=node.list
unset PBS_NODEFILE
export I_MPI_FABRICS=shm:ofa
export I_MPI_DAPL_PROVIDER="ofa-v2-mlx4_0-1u"
export I_MPI_OFA_ADAPTER_NAME=mlx4_0
export NPROCS=`wc -l $HPX_NODEFILE |gawk '//{print $1}'`
rm data.bin
mpirun -np $NPROCS --machinefile $HPX_NODEFILE ./octotiger -t20  
