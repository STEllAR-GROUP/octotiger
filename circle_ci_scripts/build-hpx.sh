#!/bin/bash -e
set -x

if [ ! -d "hpx/" ]; then
    git clone https://github.com/STEllAR-GROUP/hpx.git
else
    cd hpx
    git pull
    cd ..
fi

mkdir -p hpx/build
cd hpx/build

# detection of Vc doesn't work with a relative path
cmake -DBOOST_ROOT=/home/circleci/boost_1_63_0_install -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_DATAPAR_VC=true -DVc_ROOT=/home/circleci/Vc_install -DCMAKE_INSTALL_PREFIX=/home/circleci/hpx_install -DCMAKE_BUILD_TYPE=release ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
make -j2 VERBOSE=1 install
cd ../..
