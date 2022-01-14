#!/bin/bash -l

set -eux

#default: Assume gcc 
compiler_module="gcc/9.4.0"
simd_config="with-simd"

# if clang: change modules and no blast test (no quadmath..)
if [ "${compiler_config}" = "with-CC-clang" ]; then
  compiler_module="llvm/11.1.0"
  sed -i 's/OCTOTIGER_WITH_BLAST_TEST=ON/OCTOTIGER_WITH_BLAST_TEST=OFF/' build-octotiger.sh
fi
# Test gcc 9.4.0 and nvcc without kokkos simd types for now
if [ "${compiler_config}" = "with-CC" ] && [ "${kokkos_config}" = "with-kokkos" ] && [ "${kokkos_config}" = "with-kokkos" ]; then
  simd_config="without-simd"
fi


# Load everything
echo "Loading modules: "
module load "${compiler_module}" cuda/11.4 hwloc

# Tests with griddim = 8
if [ "${kokkos_config}" = "with-kokkos" ]; then
	echo "Running tests with griddim=8 on diablo"
	srun -p cuda -N 1 -n 1 -t 08:00:00 bash -c "module load ${compiler_module} cuda/11.4 hwloc && ./build-all.sh Release ${compiler_config} ${cuda_config} without-mpi without-papi without-apex ${kokkos_config} ${simd_config} with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest --output-on-failure " 

	# Tests with griddim = 16 - only test in full kokkos + cuda build
	if [ "${cuda_config}" = "with-cuda" ]; then
		sed -i 's/GRIDDIM=8/GRIDDIM=16/' build-octotiger.sh
		echo "Running tests with griddim=16 on diablo"
		rm -rf build/kokkos build/octotiger # in case we end up on a different cuda node we need to rebuild with its architecture
		srun -p cuda -N 1 -n 1 -t 08:00:00 bash -c "module load ${compiler_module} cuda/11.4 hwloc && ./build-all.sh Release ${compiler_config} ${cuda_config} without-mpi without-papi without-apex ${kokkos_config} ${simd_config} with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest --output-on-failure " 
		sed -i 's/GRIDDIM=16/GRIDDIM=8/' build-octotiger.sh
	fi
else
	# TODO Run this on a different server as intended to cut the test time?
	echo "Running tests with griddim=8 on diablo"
	srun -p cuda -N 1 -n 1 -t 08:00:00 bash -c "module load ${compiler_module} cuda/11.4 hwloc && ./build-all.sh Release ${compiler_config} ${cuda_config} without-mpi without-papi without-apex ${kokkos_config} ${simd_config} with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx cppuddle octotiger && cd build/octotiger/build && ctest --output-on-failure " 
fi

# Reset buildscripts (in case of failure, the next job will reset it in the checkout step)
if [ "${compiler_config}" = "with-CC-clang" ]; then
  sed -i 's/OCTOTIGER_WITH_BLAST_TEST=OFF/OCTOTIGER_WITH_BLAST_TEST=ON/' build-octotiger.sh
fi
