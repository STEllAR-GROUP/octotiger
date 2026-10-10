#!/usr/bin/env bash
# Only the results runner's solver invocation is distributed.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an allocated Slurm job}"
: "${SLURM_NNODES:?Missing allocated node count}"
: "${SLURM_NTASKS:?Missing allocated task count}"
: "${SLURM_CPUS_PER_TASK:?Missing allocated CPU count}"
: "${QB_MPI_TYPE:?Set the verified Slurm MPI plugin in the site configuration}"
: "${SLURMD_NODENAME:?Run on an allocated compute node}"
[[ "$(hostname -s)" == "${SLURMD_NODENAME%%.*}" ]]
[[ "$SLURM_NNODES" =~ ^[1-9][0-9]*$ && "$SLURM_NTASKS" == "$SLURM_NNODES" ]]
[[ "$SLURM_CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]]
[[ " $* " == *" --hpx:threads=$SLURM_CPUS_PER_TASK "* ]]
[[ $# -ge 2 && -x "$1" ]]
mkdir -p mpi-logs
printf 'Per-locality logs: %s/mpi-logs\n' "$PWD"
# HPX's ! suffix permits this key when TCP was compiled out, while replacing
# the existing entry with zero in builds that include the TCP parcelport.
set +e
srun --unbuffered --kill-on-bad-exit=1 --mpi="$QB_MPI_TYPE" \
  --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  --output="$PWD/mpi-logs/task-%t.out" --error="$PWD/mpi-logs/task-%t.err" \
  "$@" --hpx:ignore-batch-env --hpx:localities="$SLURM_NNODES" \
  --hpx:ini=hpx.parcel.bootstrap=mpi --hpx:ini=hpx.parcel.mpi.enable=1 \
  '--hpx:ini=hpx.parcel.tcp.enable!=0'
status=$?
set -e
# CGS/norm readers need unprefixed locality-zero output, not srun --label text.
if [[ -f mpi-logs/task-0.out ]]; then cat mpi-logs/task-0.out; fi
if [[ -f mpi-logs/task-0.err ]]; then cat mpi-logs/task-0.err >&2; fi
exit "$status"
