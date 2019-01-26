# Copyright (c) 2018 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

FROM gcc:7

RUN apt update && apt install -y libhwloc-dev libgoogle-perftools-dev ninja-build vim

RUN CMAKE_VERSION=3.10.0 && \
	curl -JL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz | tar xz && \
	(cd cmake-${CMAKE_VERSION} && ./bootstrap --parallel=22 && make -j22 && make install) && \
	rm -rf ${CMAKE_VERSION}

RUN curl -JL 'http://downloads.sourceforge.net/project/boost/boost/1.63.0/boost_1_63_0.tar.gz' | tar xz && \
	cd boost_1_63_0 && \
	./bootstrap.sh --prefix=/local/boost && \
	./b2 -j22 install --with-atomic --with-filesystem --with-program_options --with-regex --with-system --with-chrono --with-date_time --with-thread && \
	rm -rf boost_1_63_0

RUN git clone https://github.com/VcDevel/Vc.git --depth=1 --branch=1.4.1 && \
	cmake -HVc -BVc/build -DCMAKE_INSTALL_PREFIX=/local/vc -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=Off -GNinja && \
	cmake --build Vc/build --target install && \
	rm -rf Vc

RUN git clone https://github.com/live-clones/hdf5.git --depth=1 --branch=hdf5_1_10_4 && \
	cmake -Hhdf5 -Bhdf5/build -DBUILD_TESTING=Off -DCMAKE_BUILD_TYPE=Release -GNinja -DCMAKE_INSTALL_PREFIX=/local/hdf5 && \
	cmake --build hdf5/build --target install && \
	rm -rf hdf5

RUN curl -JL https://wci.llnl.gov/content/assets/docs/simulation/computer-codes/silo/silo-4.10.2/silo-4.10.2.tar.gz | tar xz && \
	cd silo-4.10.2 && \
	./configure --disable-fortran --prefix=/local/silo --with-hdf5=/local/hdf5/include,/local/hdf5/lib --enable-optimization && \
	make -j22 install && \
	rm -rf silo-4.10.2

RUN git clone https://github.com/STEllAR-GROUP/hpx.git --depth=1 && \
	cmake -Hhpx -Bhpx/build -DBOOST_ROOT=/local/boost -DHPX_WITH_EXAMPLES=Off -DHPX_WITH_DATAPAR_VC=On -DVc_DIR=/local/vc/lib/cmake/Vc -DCMAKE_INSTALL_PREFIX=/local/hpx -DCMAKE_BUILD_TYPE=Release -GNinja && \
	cmake --build hpx/build --target install && \
	rm -rf hpx

ENV PATH=/local/silo/bin:/local/hdf5/bin:/local/hpx/bin:$PATH \
	LD_LIBRARY_PATH=/local/silo/lib:/local/hdf5/lib:/local/boost/lib:/local/vc/lib:/local/hpx/lib
