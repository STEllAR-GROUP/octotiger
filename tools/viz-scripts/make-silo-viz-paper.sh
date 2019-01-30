#!/bin/bash -l
#   #SBATCH -t 24:00:00
#SBATCH --nodes=72 
#SBATCH --sockets-per-node=2 
#SBATCH --cores-per-socket=14 
#SBATCH --partition=defq 
#SBATCH -o /home/khuck/src/operation-gordon-bell/viz-paper/slurm-%j.out
#SBATCH -x n074

source /home/khuck/src/operation-gordon-bell/scripts/gcc.sh
buildtype=Release
malloc=jemalloc
cd ${basedir}/viz-paper

export I_MPI_PMI_LIBRARY=/cm/shared/apps/slurm/16.05.8/lib64/libpmi.so

if [ -z ${basedir+x} ] ; then
    echo "basedir is not set. Please source sourceme.sh";
    kill -INT $$
fi

export APEX_DISABLE=1
#export APEX_SCREEN_OUTPUT=1
#export APEX_PROCESS_ASYNC_STATE=0
#export APEX_OTF2=1

set_args()
{
    app="${basedir}/${myarch}-build/octotiger-${malloc}-${buildtype}/octotiger"
    #args="-Problem=dwd -Max_level=${level} -Xscale=4.0 -Eos=wd -Angcon=1 -Restart=X.91.chk ${stopstep} --hpx:threads ${threads} -Datadir=${basedir}/silo-output-slices -SiloPlanesOnly"
    args="-Ngrids=100000 -Xscale=8.031372549 -Problem=dwd -Max_level=${level} -VariableOmega=0 -Restart=restart.chk ${stopstep} --hpx:threads ${threads} -Datadir=${basedir}/viz-paper -ParallelSilo"
}

parallel()
{
    threads=28
    level=11
    set_args
    srun -u -n ${SLURM_NNODES} -N ${SLURM_NNODES} -c ${threads} $app $args -Ihpx.parcel.bootstrap=mpi $binding
}

set -x
#stopstep="-Stopstep=0"
binding="--hpx:bind balanced \
-Ihpx.parcel.message_handlers=0 \
-Ihpx.max_background_threads=16 \
-Ihpx.max_busy_loop_count=500 \
-Ihpx.lcos.collectives.cut_off=200000 \
-Ihpx.parcel.max_connections=20000 \
-Ihpx.stacks.use_guard_pages=0"

parallel
