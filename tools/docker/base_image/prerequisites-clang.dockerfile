# Copyright (c) 2018-2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# SYNOPSIS
#
# docker build [-t tag] [-build-arg GCC_RELEASE=6|7|8]
#   [-build-arg BUILD_TYPE=Debug,Release,RelWithDebInfo]
#   [-build-arg CMAKE_VERSION=version] [...]
#
# DESCRIPTION
# This is a Docker file that is used for building Octo-Tiger on CircleCI. It can
# be configured to use Clang 7, 8, or 9, a desired CMake version, with desired
# build types for Boost, Vc, and HPX

ARG UBUNTU_RELEASE=18.04
FROM ubuntu:$UBUNTU_RELEASE as builder

ARG UBUNTU_RELEASE
ARG LLVM_RELEASE=7.0.1
ARG BUILD_TYPE=Release
ARG HPX_BRANCH=master
ARG CMAKE_VERSION=3.12.4

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        ca-certificates \
        curl \
        file \
        git \
        libgoogle-perftools-dev \
        ninja-build \
        xz-utils \
        vim \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -JL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz \
        | tar xz \
    && ( \
        cd cmake-${CMAKE_VERSION} \
        && ./bootstrap --parallel=22 -- -DCMAKE_BUILD_TYPE=Release \
        && make -j22 && make install \
    ) \
    && rm -rf ${CMAKE_VERSION}

#            https://download.open-mpi.org/release/hwloc/v1.11/hwloc-1.11.12.tar.gz
RUN curl -JL https://download.open-mpi.org/release/hwloc/v2.0/hwloc-2.0.4.tar.gz \
        | tar xz \
    && ( \
        cd hwloc-2.0.4 \
        && ./configure --prefix=/local/hwloc \
        && make -j22 && make install \
    ) \
    && rm -rf hwloc-2.0.4

RUN git clone https://github.com/live-clones/hdf5.git --depth=1 --branch=hdf5-1_10_4 \
    && cmake -Hhdf5 -Bhdf5/build \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_CPP_LIB=OFF \
        -DCMAKE_INSTALL_PREFIX=/local/hdf5 \
        -DCMAKE_BUILD_TYPE=Release \
        -GNinja \
    && cmake --build hdf5/build --target install \
    && rm -rf hdf5

RUN curl -JL https://wci.llnl.gov/content/assets/docs/simulation/computer-codes/silo/silo-4.10.2/silo-4.10.2.tar.gz \
        | tar xz \
    && ( \
        cd silo-4.10.2 \
        && ./configure --disable-fortran --prefix=/local/silo \
            --with-hdf5=/local/hdf5/include,/local/hdf5/lib --enable-optimization \
        && make -j22 install \
    ) \
    && rm -rf silo-4.10.2

RUN curl -JL https://releases.llvm.org/${LLVM_RELEASE}/clang+llvm-${LLVM_RELEASE}-x86_64-linux-gnu-ubuntu-${UBUNTU_RELEASE}.tar.xz | \
        tar xJ --transform='s,clang+llvm-[^/]*,llvm,'

ENV PATH=/local/silo/bin:/local/hdf5/bin:/local/hpx/bin:/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/local/silo/lib:/local/hdf5/lib:/local/boost/lib:/local/vc/lib:/local/hpx/lib:/llvm/lib

# update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 \
# && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100
RUN curl -JL 'http://downloads.sourceforge.net/project/boost/boost/1.63.0/boost_1_63_0.tar.gz' \
        | tar xz \
    && ( \
        cd boost_1_63_0 \
        && ./bootstrap.sh --prefix=/local/boost --with-toolset=clang \
        && ./b2 -j22 --with-atomic --with-filesystem --with-program_options \
            --with-regex --with-system --with-chrono --with-date_time \
            --with-thread $(echo ${BUILD_TYPE} | tr '[:upper:]' '[:lower:]' ) \
            toolset=clang cxxflags="-stdlib=libc++" linkflags="-stdlib=libc++" \
            install \
    ) \
    && rm -rf boost_1_63_0

RUN git clone https://github.com/VcDevel/Vc.git --depth=1 --branch=1.4.1 \
    && cmake -HVc -BVc/build \
        -DBUILD_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX=/local/vc \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
        -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++" \
        -GNinja \
    && cmake --build Vc/build --target install \
    && rm -rf Vc

RUN git clone https://github.com/STEllAR-GROUP/hpx.git --depth=1 --branch=${HPX_BRANCH} \
    && cmake -Hhpx -Bhpx/build \
        -DBOOST_ROOT=/local/boost \
        -DHPX_WITH_EXAMPLES=OFF \
        -DHWLOC_ROOT=/local/hwloc \
        -DCMAKE_INSTALL_PREFIX=/local/hpx \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
        -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++" \
        -GNinja \
    && cmake --build hpx/build --target install \
    && rm -rf hpx
