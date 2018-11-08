#!/bin/bash -e
set -x

#export LD_LIBRARY_PATH=/home/circleci/silo_install/lib64:/home/circleci/silo_install/lib:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/home/circleci/silo_install/lib64:/home/circleci/silo_install/lib:$LIBRARY_PATH
#export PATH=$HOME/home/circleci/silo_install/bin:$PATH
#export CPATH=/home/circleci/silo_install/include:$CPATH

mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="-march=native" -DBOOST_ROOT=/home/circleci/boost_1_63_0_install -DOCTOTIGER_WITH_SILO=ON -DCMAKE_PREFIX_PATH=/home/circleci/hpx_install -DCMAKE_BUILD_TYPE=release -DHPX_IGNORE_COMPILER_COMPATIBILITY=ON -DHDF5_INCLUDE_DIR=/home/circleci/hdf_install/include -DHDF5_LIBRARY=/home/circleci/hdf_install/hdf5/lib/libhdf5.a -DSilo_INCLUDE_DIR=/home/circleci/silo_install/include  -DSilo_LIBRARY=/home/circleci/silo_install/lib/libsiloh5.a ../
make -j2 VERBOSE=1
cd ..

