#!/bin/bash -e
set -x

if [ ! -d "silo/" ]; then
    mkdir silo
    cd silo
    wget https://wci.llnl.gov/content/assets/docs/simulation/computer-codes/silo/silo-4.10.2/silo-4.10.2.tar.gz
    tar -xvf silo-4.10.2.tar.gz
    cd ..
else
    cd silo
    git pull
    cd ..
fi

cd silo
cd silo-4.10.2
./configure --prefix=/home/circleci/silo_install --with-hdf5=/home/circleci/hdf5_install/include,/home/circleci/hdf5_install/lib/ --enable-optimization
make -j2 VERBOSE=1 install


