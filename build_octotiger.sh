

set -x

source ~/scripts/sourceme.sh gperftools
source ~/scripts/sourceme.sh hwloc
source ~/scripts/sourceme.sh vc
source ~/scripts/sourceme.sh silo
source ~/scripts/sourceme.sh $1/hpx

cd $1
rm CMakeCache.txt
rm -r CMakeFiles


cmake -DCMAKE_PREFIX_PATH="$HOME/local/$1/hpx" -DCMAKE_CXX_FLAGS="-DBOOST_USE_VALGRIND" \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_FLAGS="-DBOOST_USE_VALGRIND -L$HOME/local/boost/lib -march=native" \
      -DCMAKE_C_FLAGS="-L$HOME/local/boost/lib" \
      -DCMAKE_BUILD_TYPE=$1                                                                                                                            \
      -DCMAKE_INSTALL_PREFIX="$HOME/local/$1/octotiger"                                   \
      -DSilo_ROOT=$HOME/local/silo/ \
      -DHDF5_LIBRARY=$HOME/local/hdf5/lib/libhdf5.a \
      -DHDF5_INCLUDE_DIR=$HOME/local/hdf5/include \
      ..


make -j48


