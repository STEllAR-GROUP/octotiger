:: Copyright (c) 2018 Parsa Amini
::
:: Distributed under the Boost Software License, Version 1.0. (See accompanying
:: file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

:: Description
:: This Batch script is used to build the Docker images used to build and test
:: Octotiger on CircleCI

:: Useful docker build diagnostics argument(s):
::     --no-cache    Re-run every stage

docker build -t stellargroup/octotiger:prerequisites-gcc6 -f prerequisites-gcc.dockerfile --build-arg GCC_RELEASE=6 .
docker build -t stellargroup/octotiger:prerequisites-gcc7 -f prerequisites-gcc.dockerfile --build-arg GCC_RELEASE=7 .
docker build -t stellargroup/octotiger:prerequisites-gcc8 -f prerequisites-gcc.dockerfile --build-arg GCC_RELEASE=8 .

docker build -t stellargroup/octotiger:prerequisites-gcc6-debug -f prerequisites-gcc.dockerfile --build-arg GCC_RELEASE=6 --build-arg BUILD_TYPE=Debug .
docker build -t stellargroup/octotiger:prerequisites-gcc7-debug -f prerequisites-gcc.dockerfile --build-arg GCC_RELEASE=7 --build-arg BUILD_TYPE=Debug .
docker build -t stellargroup/octotiger:prerequisites-gcc8-debug -f prerequisites-gcc.dockerfile --build-arg GCC_RELEASE=8 --build-arg BUILD_TYPE=Debug .

docker push stellargroup/octotiger:prerequisites-gcc6
docker push stellargroup/octotiger:prerequisites-gcc7
docker push stellargroup/octotiger:prerequisites-gcc8

docker push stellargroup/octotiger:prerequisites-gcc6-debug
docker push stellargroup/octotiger:prerequisites-gcc7-debug
docker push stellargroup/octotiger:prerequisites-gcc8-debug
pause