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

# workaround for hpx not detecting Vc at a relative path
Vc_ROOT=`realpath ../../Vc_install/`
cmake -DBOOST_ROOT="../../boost_1_63_0_install" -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_DATAPAR_VC=true -DVc_ROOT="$Vc_ROOT" -DCMAKE_INSTALL_PREFIX="../../hpx_install" ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
make -j2 install
cd ../..
