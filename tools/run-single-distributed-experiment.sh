#!/bin/bash
#SBATCH -A xpress
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C knl,quad,flat
#SBATCH -c 68
#SBATCH -p regular
#SBATCH -J octotiger_node_level_experiments
#SBATCH --mail-user=David.Pfander@ipvs.uni-stuttgart.de
#SBATCH -t 03:00:00

name=$1
level=$2
threads=$3

srun numactl -m 1 ./knl-build/octotiger-Release/octotiger -Disableoutput -Problem=dwd -Max_level=${level} -Xscale=36.0 -Eos=wd -Angcon=1 -Stopstep=100 --hpx:threads=${threads} -Restart=restart${level}.chk --hpx:ini=hpx.stacks.small_size=0xC0000 -Ihpx.stacks.use_guard_pages=0 --hpx:print-bind --hpx:print-counter /threads{locality#*/total}/idle-rate  > results/${name}_N${SLURM_NNODES}_t${threads}_l${level}_m1 2>&1

srun numactl -m 0 ./knl-build/octotiger-Release/octotiger -Disableoutput -Problem=dwd -Max_level=${level} -Xscale=36.0 -Eos=wd -Angcon=1 -Stopstep=100 --hpx:threads=${threads} -Restart=restart${level}.chk --hpx:ini=hpx.stacks.small_size=0xC0000 -Ihpx.stacks.use_guard_pages=0 --hpx:print-bind --hpx:print-counter /threads{locality#*/total}/idle-rate  > results/${name}_N${SLURM_NNODES}_t${threads}_l${level}_m0 2>&1
