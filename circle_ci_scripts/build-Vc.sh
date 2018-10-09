#!/bin/bash -e
set -x

if [ ! -d "Vc/" ]; then
    git clone https://github.com/STEllAR-GROUP/Vc
else
    cd Vc
    git pull
    cd ..
fi

cd Vc
git checkout pfandedd_inlining_AVX512
cd ..

mkdir -p Vc/build
cd Vc/build
/home/circleci/cmake_install/bin/cmake -DCMAKE_INSTALL_PREFIX=/home/circleci/Vc_install -DCMAKE_BUILD_TYPE=release -DBUILD_TESTING=OFF ../
make -j2 VERBOSE=1 install
cd ../..
