#!/bin/bash -e
set -x

if [ ! -d "Vc/" ]; then
    git clone https://github.com/VcDevel/Vc.git
else
    cd Vc
    git pull
    cd ..
fi

mkdir -p Vc/build
cd Vc/build
/home/circleci/cmake_install/bin/cmake -DCMAKE_INSTALL_PREFIX=/home/circleci/Vc_install -DCMAKE_BUILD_TYPE=release -DBUILD_TESTING=OFF ../
make -j2 VERBOSE=1 install
cd ../..
