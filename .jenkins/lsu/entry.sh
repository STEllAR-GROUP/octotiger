#!/bin/bash -l

set -eux

#default: Assume gcc 
compiler_module="gcc/9.3.0"

# if clang: change modules
if [ "${compiler_config}" = "with-CC-clang" ]; then
  compiler_module="clang/11.0.1"
fi

# Load everything
echo "Loading modules: "
module load "${compiler_module}" cuda/11.0 hwloc

# Tests with griddim = 8
echo "Running tests with griddim=8"
srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load ${compiler_module} cuda/11.0 hwloc && ./build-all.sh Release ${compiler_config} with-cuda without-mpi without-papi without-apex with-kokkos with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest ' 

# Tests with griddim = 16
sed -i 's/GRIDDIM=8/GRIDDIM=16/' build-octotiger.sh
echo "Running tests with griddim=16"
srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load ${compiler_module} cuda/11.0 hwloc && ./build-all.sh Release ${compiler_config} with-cuda without-mpi without-papi without-apex with-kokkos with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling octotiger && cd build/octotiger/build && ctest ' 

# Reset buildscripts (in case of failure, the next job will reset it in the checkout step)
sed -i 's/GRIDDIM=16/GRIDDIM=8/' build-octotiger.sh
