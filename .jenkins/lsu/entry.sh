#!/bin/bash -l

set -eux

pwd
srun -p QxV100 -N 1 -n 1 -t 00:20:00 bash -c 'module load hwloc cuda; ./build-all.sh ${build_type} ${compiler_config} with-cuda without-mpi without-papi without-apex with-kokkos with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger' 
srun -p QxV100 -N 1 -n 1 -t 00:20:00 bash -c 'module load hwloc cuda; cd build/octotiger/build;ctest'
