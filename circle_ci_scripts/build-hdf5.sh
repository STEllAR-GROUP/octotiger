


if [ ! -d "hdf5/" ]; then
    git clone https://github.com/live-clones/hdf5
else
    cd hdf5
    git pull
    cd ..
fi

cd hdf5
git checkout hdf5_1_10_4 
cd ..

mkdir hdf5_build
cd hdf5_build
cmake \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DBUILD_TESTING=OFF \                                                                                                                               \
      -DCMAKE_BUILD_TYPE=release                                                                                                                               \
      -DCMAKE_INSTALL_PREFIX="/home/circleci/hdf5_install"                               \
       ../hdf5

/home/circleci/cmake_install/bin/cmake -j2 VERBOSE=1 install

cd ../..
