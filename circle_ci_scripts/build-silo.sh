#!/bin/bash -e
set -x

if [ ! -d "silo/" ]; then
    mkdir silo
    cd silo
    if [ ! -d "silo-4.10.2" ]; then
       wget phys.lsu.edu/~dmarcel/silo-4.10.2.tar.gz
    fi
       tar -xvf silo-4.10.2.tar.gz
    cd ..
fi

cd silo
cd silo-4.10.2
cat configure | sed 's/-lhdf5/$hdf5_lib\/libhdf5.a -ldl/g' > tmp
mv tmp configure
./configure --prefix=/home/circleci/silo_install --with-hdf5=/home/circleci/hdf5_install/include,/home/circleci/hdf5_install/lib/ --enable-optimization
make -j2 VERBOSE=1 install


