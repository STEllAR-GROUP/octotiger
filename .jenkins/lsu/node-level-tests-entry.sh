#!/bin/bash -l

set -eux

#default: Assume gcc 
compiler_module="gcc/10.3.0"
simd_config="with-simd"

# if clang: change modules and no blast test (no quadmath..)
if [ "${compiler_config}" = "with-CC-clang" ]; then
  compiler_module="llvm/12.0.1"
  sed -i 's/OCTOTIGER_WITH_BLAST_TEST=ON/OCTOTIGER_WITH_BLAST_TEST=OFF/' build-octotiger.sh
fi


# Load everything
echo "Loading modules: "
module load "${compiler_module}" cuda/11.5 hwloc

# Tests with griddim = 8
echo "Running tests with griddim=8"
srun --partition=jenkins-cuda -N 1 -n 1 -t 08:00:00 bash -c "module load ${compiler_module} cuda/11.5 hwloc && module list && ./build-all.sh Release ${compiler_config} ${cuda_config} without-mpi without-papi without-apex ${kokkos_config} ${simd_config} with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest --output-on-failure " 

# Tests with griddim = 16 - only test in full kokkos + cuda build
if [ "${cuda_config}" = "with-cuda" ] && [ "${kokkos_config}" = "with-kokkos" ]; then
	sed -i 's/GRIDDIM=8/GRIDDIM=16/' build-octotiger.sh
	echo "Running tests with griddim=16 on diablo"
	rm -rf build # in case we end up on a different cuda node we need to rebuild with its architecture
	srun --partition=jenkins-cuda -N 1 -n 1 -t 08:00:00 bash -c "module load ${compiler_module} cuda/11.5 hwloc && module list && ./build-all.sh Release ${compiler_config} ${cuda_config} without-mpi without-papi without-apex ${kokkos_config} ${simd_config} with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest --output-on-failure " 
	sed -i 's/GRIDDIM=16/GRIDDIM=8/' build-octotiger.sh
fi

# Reset buildscripts (in case of failure, the next job will reset it in the checkout step)
if [ "${compiler_config}" = "with-CC-clang" ]; then
  sed -i 's/OCTOTIGER_WITH_BLAST_TEST=OFF/OCTOTIGER_WITH_BLAST_TEST=ON/' build-octotiger.sh
fi
