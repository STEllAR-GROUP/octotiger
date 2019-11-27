#!/bin/bash
#PBS -j oe
#PBS -A loni_lrn08
#PBS -l nodes=64:ppn=20
#PBS -l walltime=8:00:00
#PBS -q checkpt

BUILD_TYPE=relwithdebinfo
source ~/scripts/source_all.sh $BUILD_TYPE

set -x 


cd $WORK/rotating_star_paper/rs_4_w
cat $PBS_NODEFILE | awk 'NR % 20 == 0' > node.list
export HPX_NODEFILE=node.list
unset PBS_NODEFILE
export NPROCS=`wc -l $HPX_NODEFILE |gawk '//{print $1}'`

mpirun -np $NPROCS --machinefile $HPX_NODEFILE  /home/dmarce1/local/$BUILD_TYPE/octotiger/bin/octotiger --hpx:threads 20 --config_file=config.ini
 




