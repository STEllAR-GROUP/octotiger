#!/bin/bash -l

set -eux

# ToDo move to sbatch...
srun -p cuda-A100 -N 1 -n 1 -t 24:00:00 bash -c "module load llvm/12.0.1 rocm/4.3.1 hwloc && module unload boost && pip3 install --upgrade matplotlib pandas && ./build-all.sh Release with-CC-clang with-cuda without-mpi without-papi without-apex with-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && ./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/blast_aggregation_performance_cuda.sh &&  python3 ./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/plot_blast_aggregation_performance.py --filename=performance_results.log --gpu-name='NVIDIA A100'
  " 
