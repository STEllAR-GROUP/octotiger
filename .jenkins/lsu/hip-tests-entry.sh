#!/bin/bash -l

set -eux

# Tests with griddim = 8
srun -p jenkins-amdgpu -N 1 -n 1 -t 02:00:00 bash -c "module load gcc/9.4.0 rocm/4.3.1 hwloc && module list && hipcc --version && rocminfo && ./build-all.sh Release with-CC-clang without-cuda without-mpi without-papi without-apex with-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest --output-on-failure " 

# Tests with griddim = 16
sed -i 's/GRIDDIM=8/GRIDDIM=16/' build-octotiger.sh
srun -p jenkins-amdgpu -N 1 -n 1 -t 02:00:00 bash -c "module load gcc/9.4.0 rocm/4.3.1 hwloc && module list && hipcc --version && rocminfo && ./build-all.sh Release with-CC-clang without-cuda without-mpi without-papi without-apex with-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 octotiger && cd build/octotiger/build && ctest --output-on-failure " 
sed -i 's/GRIDDIM=16/GRIDDIM=8/' build-octotiger.sh

