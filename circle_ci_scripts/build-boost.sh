#!/bin/bash -e
set -x

if [ ! -d "boost_1_63_0/" ]; then
    wget 'http://downloads.sourceforge.net/project/boost/boost/1.63.0/boost_1_63_0.tar.bz2'
    tar xf boost_1_63_0.tar.bz2
fi

if [ ! -d "boost_1_63_0_install/" ]; then
    cd boost_1_63_0
    ./bootstrap.sh --prefix="$Boost_ROOT"
    ./b2 -j4 install
    cd ..
fi



