

set -x

source ~/scripts/sourceme.sh gperftools
source ~/scripts/sourceme.sh hwloc
source ~/scripts/sourceme.sh vc
source ~/scripts/sourceme.sh silo
source ~/scripts/sourceme.sh $1/hpx

rm -rf $1
mkdir $1
cd $1
rm CMakeCache.txt
rm -r CMakeFiles


cmake -DCMAKE_PREFIX_PATH="$HOME/local/$1/hpx" -DCMAKE_CXX_FLAGS="-DBOOST_USE_VALGRIND" \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_FLAGS="-DBOOST_USE_VALGRIND -L$HOME/local/boost/lib -march=native" \
      -DCMAKE_C_FLAGS="-L$HOME/local/boost/lib" \
      -DCMAKE_BUILD_TYPE=$1                                                                                                                            \
      -DCMAKE_INSTALL_PREFIX="$HOME/local/$1/octotiger"                                   \
      -DHDF5_ROOT=$HOME/local/hdf5 \
      -DSilo_DIR=$HOME/local/silo \
      ..


make -j VERBOSE=1


