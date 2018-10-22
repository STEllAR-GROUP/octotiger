#!/bin/bash -e
set -x

cd ..

if [ ! -d "cmake-3.8.0/" ]; then
    wget https://cmake.org/files/v3.8/cmake-3.8.0.tar.gz
	tar -xvzf cmake-3.8.0.tar.gz
	cd cmake-3.8.0/
	mkdir -p build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=/home/circleci/cmake_install ..
	make -j 4
    make install
	cd ../..
	
fi


if [ ! -d "cmake_install" ]; then
	cd cmake-3.8.0/build
	make install
	cd ../..
fi




