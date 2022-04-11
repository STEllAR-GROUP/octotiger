#!/bin/bash -l

set -eux

# ToDo move to sbatch...
srun -p mi100 -N 1 -n 1 -t 24:00:00 bash -c "module load gcc/9.4.0 rocm/4.3.1 hwloc && pip3 install --upgrade matplotlib pandas && ./build-all.sh Release with-CC-clang without-cuda without-mpi without-papi without-apex with-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && ./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/blast_aggregation_performance_hip.sh &&  python3 ./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/plot_blast_aggregation_performance_hip.py --filename=performance_results.log --gpu-name='AMD MI100'
  " 
