#!/bin/bash -l

set -eux

pwd
module load gcc/9.3.0 cuda/11.0 hwloc
# Tests with griddim = 8
srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load gcc/9.3.0 cuda/11.0 hwloc && ./build-all.sh Release with-CC with-cuda without-mpi without-papi without-apex with-kokkos with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest ' 
# Tests with griddim = 16
sed -i 's/GRIDDIM=8/GRIDDIM=16/' build-octotiger.sh
srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load gcc/9.3.0 cuda/11.0 hwloc && ./build-all.sh Release with-CC with-cuda without-mpi without-papi without-apex with-kokkos with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest ' 
# Reset griddim (in case of failure, the next job will reset it in the checkout step)
sed -i 's/GRIDDIM=16/GRIDDIM=8/' build-octotiger.sh
