#!/bin/bash -e
set -x

mkdir build
cd build
cmake -DBOOST_ROOT="$Boost_ROOT" -DOCTOTIGER_WITH_SILO=OFF -DCMAKE_PREFIX_PATH="$HPX_ROOT" -DCMAKE_BUILD_TYPE=release ../
make -j2 VERBOSE=1
cd ..
