#!/bin/bash -e
set -x

mkdir build
cd build
cmake -DBOOST_ROOT=/home/circleci/boost_1_63_0_install -DOCTOTIGER_WITH_SILO=OFF -DCMAKE_PREFIX_PATH=/home/circleci/hpx_install -DCMAKE_BUILD_TYPE=release -DHPX_IGNORE_COMPILER_COMPATIBILITY=ON ../
make -j2 VERBOSE=1
cd ..
