#!/bin/bash -e
set -x

mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="-march=native" -DBOOST_ROOT=/home/circleci/boost_1_63_0_install -DOCTOTIGER_WITH_SILO=OFF -DCMAKE_PREFIX_PATH=/home/circleci/hpx_install -DCMAKE_BUILD_TYPE=release -DHPX_IGNORE_COMPILER_COMPATIBILITY=ON -DSilo_INCLUDE_DIR=/home/circleci/silo_install/include  -DSilo_LIBRARY=$HOME/home/circleci/silo_install/lib/libsilo.a ../
make -j2 VERBOSE=1
cd ..

