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
cmake -DBOOST_ROOT="$Boost_ROOT" -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_DATAPAR_VC=true -DVc_ROOT="$Vc_ROOT" -DCMAKE_INSTALL_PREFIX="$HPX_ROOT" -DCMAKE_BUILD_TYPE=release ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
make -j2 VERBOSE=1 install
cd ../..
