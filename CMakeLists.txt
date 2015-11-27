# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

project(octotiger CXX)

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)

################################################################################
# enable solution folders for MSVC
if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

################################################################################
find_package(HPX REQUIRED)
find_package(MPI REQUIRED)

################################################################################
include_directories(${CMAKE_SOURCE_DIR}/src)

set( CMAKE_CXX_FLAGS "-mt_mpi -xHOST -std=c++14" )

add_hpx_executable(
  octotiger
  SOURCES src/grid.cpp
          src/grid_fmm.cpp
          src/grid_output.cpp
          src/lane_emden.cpp
          src/geometry.cpp
          src/node_octstruct.cpp
          src/grid_scf.cpp
          src/main.cpp
          src/node_client.cpp
          src/node_location.cpp
          src/node_server.cpp
          src/node_server_decomp.cpp
          src/node_server_output.cpp
          src/problem.cpp
          src/roe.cpp
          src/taylor.cpp
  HEADERS src/channel.hpp
          src/defs.hpp
          src/future.hpp
          src/grid.hpp
          src/lane_emden.hpp
          src/node_client.hpp
          src/node_location.hpp
          src/node_server.hpp
          src/problem.hpp
          src/roe.hpp
          src/geometry.hpp
          src/simd.hpp
          src/space_vector.hpp
          src/taylor.hpp
  FOLDER  "Octotiger"
)


add_executable(
  octotiger_lite
          src/hpx_lite/hpx_lite.cpp
          src/grid.cpp
          src/grid_fmm.cpp
          src/grid_output.cpp
          src/lane_emden.cpp
          src/geometry.cpp
          src/node_octstruct.cpp
          src/grid_scf.cpp
          src/main.cpp
          src/node_client.cpp
          src/node_location.cpp
          src/node_server.cpp
          src/node_server_decomp.cpp
          src/node_server_output.cpp
          src/problem.cpp
          src/roe.cpp
          src/taylor.cpp
 )

target_compile_definitions(octotiger_lite
                           PUBLIC MINI_HPX
)
