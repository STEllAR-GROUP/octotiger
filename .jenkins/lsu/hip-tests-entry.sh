#!/bin/bash -l

#  Copyright (c) 2021-2022 Gregor Daiß
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#

set -eux

# Tests with griddim = 8
# we need the llvm module for the silo fortran compilation...
srun -p mi100 -N 1 -n 1 -t 04:00:00 bash -lc "module load rocm/5.4 hwloc llvm/14 && module list && hipcc --version && rocminfo && ./build-all.sh Release with-CC-clang without-cuda without-mpi without-papi without-apex with-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger && cd build/octotiger/build && ctest --output-on-failure " 

# Tests with griddim = 16
sed -i 's/GRIDDIM=8/GRIDDIM=16/' build-octotiger.sh
srun -p mi100 -N 1 -n 1 -t 04:00:00 bash -lc "module load rocm/5.4 hwloc llvm/14 && module list && hipcc --version && rocminfo && ./build-all.sh Release with-CC-clang without-cuda without-mpi without-papi without-apex with-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 octotiger && cd build/octotiger/build && ctest --output-on-failure -E legacy " 
sed -i 's/GRIDDIM=16/GRIDDIM=8/' build-octotiger.sh

