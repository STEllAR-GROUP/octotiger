#!/bin/bash -l

set -eux

pwd
srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load hwloc cuda && ./build-all.sh Release with-CC with-cuda without-mpi without-papi without-apex without-kokkos with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx cppuddle octotiger && cd build/octotiger/build && ctest -R gpu --verbose' 
