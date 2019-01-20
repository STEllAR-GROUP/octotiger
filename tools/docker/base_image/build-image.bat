:: Copyright (c) 2018 Parsa Amini
::
:: Distributed under the Boost Software License, Version 1.0. (See accompanying
:: file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

docker build -t stellargroup/octotiger:prerequisites-gcc6 -f prerequisites-gcc6.dockerfile .
docker build -t stellargroup/octotiger:prerequisites-gcc7 -f prerequisites-gcc7.dockerfile .
docker build -t stellargroup/octotiger:prerequisites-gcc8 -f prerequisites-gcc8.dockerfile .
docker push stellargroup/octotiger:prerequisites-gcc6
docker push stellargroup/octotiger:prerequisites-gcc7
docker push stellargroup/octotiger:prerequisites-gcc8
