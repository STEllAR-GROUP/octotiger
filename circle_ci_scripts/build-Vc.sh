#!/bin/bash -e
set -x

if [ ! -d "Vc/" ]; then
    https://github.com/VcDevel/Vc.git
else
    cd Vc
    git checkout tags/1.4.1
    cd ..
fi

mkdir -p Vc/build
cd Vc/build
/home/circleci/cmake_install/bin/cmake -DCMAKE_INSTALL_PREFIX=/home/circleci/Vc_install -DCMAKE_BUILD_TYPE=release -DBUILD_TESTING=OFF ../
make -j2 VERBOSE=1 install
cd ../..
