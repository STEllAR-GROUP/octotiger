#!/bin/bash -l

set -eux

#default: Assume gcc and cuda
module_list="gcc/9.3.0 cuda/11.0 hwloc"

# if clang: change modules
if [ "${compiler_config}" = "with-CC-clang" ]; then
  module_list="clang/9.0.1 cuda/11.0 hwloc"
fi

# Load everything
echo "Loading modules: ${module_list}"
module load "${module_list}"

# Tests with griddim = 8
echo "Running tests with griddim=8"
srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load ${module_list} && ./build-all.sh Release with-CC with-cuda without-mpi without-papi without-apex with-kokkos with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest ' 

# Tests with griddim = 16
sed -i 's/GRIDDIM=8/GRIDDIM=16/' build-octotiger.sh
echo "Running tests with griddim=16"
srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load ${module_list} && ./build-all.sh Release with-CC with-cuda without-mpi without-papi without-apex with-kokkos with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling octotiger && cd build/octotiger/build && ctest ' 

# Reset buildscripts (in case of failure, the next job will reset it in the checkout step)
sed -i 's/GRIDDIM=16/GRIDDIM=8/' build-octotiger.sh
