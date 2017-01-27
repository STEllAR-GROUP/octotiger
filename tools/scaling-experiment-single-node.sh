#!/bin/bash
#SBATCH -A xpress
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C knl,quad,flat
#SBATCH -c 68
#SBATCH -p regular
#SBATCH -J octotiger_node_level_experiments
#SBATCH --mail-user=David.Pfander@ipvs.uni-stuttgart.de
#SBATCH -t 10:00:00

#
# Scaling experiment on a single node
# Meant to be used to run different thread values within a single job
#

name=$1
level=$2
# set 1 (MCDRAM) or 0
memory_type=$3
threads=$4
threads_increment=$5

# Add these in case of crashes: --hpx:ini=hpx.stacks.small_size=0xC0000 -Ihpx.stacks.use_guard_pages=0

while [[ ${threads} -le 68 ]]; do
    echo "srun numactl -m ${memory_type} ./knl-build/octotiger-Release/octotiger -Disableoutput -Problem=dwd -Max_level=${level} -Xscale=36.0 -Eos=wd -Angcon=1 -Stopstep=30 --hpx:threads=${threads} -Restart=restart${level}.chk --hpx:print-bind --hpx:print-counter /threads{locality#*/total}/idle-rate  > results/${name}_N${SLURM_NNODES}_t${threads}_l${level}_m${memory_type} 2>&1"
    echo ${threads}
    let threads=threads+${threads_increment}
done
