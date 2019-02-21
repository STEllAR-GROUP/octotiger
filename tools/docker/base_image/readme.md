# Supported tags and respective `Dockerfile` links
* `prerequisites-clang7-debug`, LLVM 7.0.1, HPX, Vc, and Boost built in Debug mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-clang.dockerfile)
	* build arguments: `UBUNTU_RELEASE=16.04` `LLVM_RELEASE=7.0.1` `BUILD_TYPE=Debug`
* `prerequisites-clang6-debug`, LLVM 6.0.1, HPX, Vc, and Boost built in Debug mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-clang.dockerfile)
	* build arguments: `UBUNTU_RELEASE=18.04` `LLVM_RELEASE=6.0.1` `BUILD_TYPE=Debug`
* `prerequisites-gcc8-relwithdebinfo`, GCC 8.2.0, HPX and Vc built in RelWithDebInfo mode, Boost built in release mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-gcc.dockerfile)
	* build arguments: `GCC_RELEASE=8` `BUILD_TYPE=RelWithDebInfo`
* `prerequisites-gcc7-relwithdebinfo`, GCC 7.4.0, HPX and Vc built in RelWithDebInfo mode, Boost built in release mode
	[Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-gcc.dockerfile)
	* build arguments: `GCC_RELEASE=8` `BUILD_TYPE=RelWithDebInfo`
* `prerequisites-gcc6-relwithdebinfo`, GCC 6.5.0, HPX and Vc built in RelWithDebInfo mode, Boost built in release mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-gcc.dockerfile)
	* build arguments: `GCC_RELEASE=8` `BUILD_TYPE=RelWithDebInfo`
* `prerequisites-gcc8-debug`, GCC 8.2.0, HPX, Vc, and Boost built in Debug mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-gcc.dockerfile)
	* build arguments: `GCC_RELEASE=8` `BUILD_TYPE=Debug`
* `prerequisites-gcc7-debug`, GCC 7.4.0, HPX, Vc, and Boost built in Debug mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-gcc.dockerfile)
	* build arguments: `GCC_RELEASE=7` `BUILD_TYPE=Debug`
* `prerequisites-gcc6-debug`, GCC 6.5.0, HPX, Vc, and Boost built in Debug mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-gcc.dockerfile)
	* build arguments: `GCC_RELEASE=6` `BUILD_TYPE=Debug`
* `prerequisites-gcc8`, GCC 8.2.0, HPX, Vc, and Boost built in Release mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-gcc.dockerfile)
	* build arguments: `GCC_RELEASE=8` `BUILD_TYPE=Release`
* `prerequisites-gcc7`, GCC 7.4.0, HPX, Vc, and Boost built in Release mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-gcc.dockerfile)
	* build arguments: `GCC_RELEASE=7` `BUILD_TYPE=Release`
* `prerequisites-gcc6`, GCC 6.5.0, HPX, Vc, and Boost built in Release mode
	* [Dockerfile](https://github.com/STEllAR-GROUP/octotiger/blob/master/tools/docker/base_image/prerequisites-gcc.dockerfile)
	* build arguments: `GCC_RELEASE=6` `BUILD_TYPE=Release`

# Quick Reference

* **Where to get help**:

	IRC Channel `ste||ar` on [freenode.net](https://freenode.net/)

* **Where to file issues**:

	[Octo-tiger Issue Tracker](https://github.com/STEllAR-GROUP/octotiger/issues)

* **Supported architectures**:

	`amd64`

# What is Octo-tiger?
From <https://doi.org/10.1145/3204919.3204938>:
> Octo-Tiger is an astrophysics program simulating the evolution of star systems based
on the fast multipole method on adaptive Octrees. It was implemented using
high-level C++ libraries, specifically HPX and Vc, which allows its use on
different hardware platforms

# How to use this image
This image exists for sake of facilitating testing non-CUDA Octo-tiger builds
with different versions of GCC and Clang. It contains Boost, HPX, Vc, HDF5,
Silo, and build tools needed to build Octo-tiger. The most straightforward way
to use this image is creating a container with `docker run -it --rm
stellargroup/octotiger:prerequisites-gcc6` and building Octo-tiger in the
container.

## Build Octo-tiger inside the Docker container
If you would like to experiment with Octo-tiger inside a container, you can try
a set of commands like this:
```console
$ docker run -it stellargroup/octotiger:prerequisites-gcc6-debug
$ git clone https://github.com/STEllAR-GROUP/octotiger.git --depth=1
$ cmake -H/octotiger -B/octotiger/build -DCMAKE_BUILD_TYPE=Debug -DHPX_DIR=/local/hpx/lib/cmake/HPX -DVc_DIR=/local/vc/lib/cmake/Vc -DSilo_DIR=/local/silo -DHDF5_ROOT=/local/hdf5 -DBOOST_ROOT=/local/boost -GNinja
$ cmake --build /octotiger/build
$ cd /octotiger/build
$ ctest --output-on-failure
```
This will clone the current Octo-tiger HEAD on master and run all the tests.

# License
Distributed under the Boost Software License, Version 1.0. (See 
<http://www.boost.org/LICENSE_1_0.txt>)
