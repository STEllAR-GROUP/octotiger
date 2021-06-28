#!/bin/bash -l

set -eux

#default: Assume gcc 
compiler_module="gcc/9.3.0"

# if clang: change modules and no blast test
if [ "${compiler_config}" = "with-CC-clang" ]; then
  compiler_module="clang/11.0.1"
  sed -i 's/OCTOTIGER_WITH_BLAST_TEST=ON/OCTOTIGER_WITH_BLAST_TEST=OFF/' build-octotiger.sh
fi

# Load everything
echo "Loading modules: "
module load "${compiler_module}" cuda/11.0 hwloc

# Tests with griddim = 8
if [ "${kokkos_config}" = "with-kokkos" ]; then
	echo "Running tests with griddim=8 on diablo"
	srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load ${compiler_module} cuda/11.0 hwloc && ./build-all.sh Release ${compiler_config} ${cuda_config} without-mpi without-papi without-apex ${kokkos_config} with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest ' 

	# Tests with griddim = 16 - only test in full kokkos + cuda build
	if [ "${cuda_config}" = "with-cuda" ]; then
		sed -i 's/GRIDDIM=8/GRIDDIM=16/' build-octotiger.sh
		echo "Running tests with griddim=16 on diablo"
		srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load ${compiler_module} cuda/11.0 hwloc && ./build-all.sh Release ${compiler_config} ${cuda_config} without-mpi without-papi without-apex ${kokkos_config} with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest ' 
		sed -i 's/GRIDDIM=16/GRIDDIM=8/' build-octotiger.sh
	fi
else
	echo "Running tests with griddim=8 on diablo"
	srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load ${compiler_module} cuda/11.0 hwloc && ./build-all.sh Release ${compiler_config} ${cuda_config} without-mpi without-papi without-apex ${kokkos_config} with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx cppuddle octotiger && cd build/octotiger/build && ctest ' 

	# Tests with griddim = 16 - do not test this without kokkos for now (too time-consuming)
	# sed -i 's/GRIDDIM=8/GRIDDIM=16/' build-octotiger.sh
	# echo "Running tests with griddim=16 on diablo"
	# srun -p QxV100 -N 1 -n 1 -t 01:00:00 bash -c 'module load ${compiler_module} cuda/11.0 hwloc && ./build-all.sh Release ${compiler_config} ${cuda_config} without-mpi without-papi without-apex ${kokkos_config} with-simd with-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling boost jemalloc hdf5 silo vc hpx cppuddle octotiger && cd build/octotiger/build && ctest ' 
fi

# Reset buildscripts (in case of failure, the next job will reset it in the checkout step)
#sed -i 's/GRIDDIM=16/GRIDDIM=8/' build-octotiger.sh
if [ "${compiler_config}" = "with-CC-clang" ]; then
  sed -i 's/OCTOTIGER_WITH_BLAST_TEST=OFF/OCTOTIGER_WITH_BLAST_TEST=ON/' build-octotiger.sh
fi
