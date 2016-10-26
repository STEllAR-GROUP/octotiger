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
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(MPI REQUIRED)

################################################################################
include_directories(${CMAKE_SOURCE_DIR}/src ~/include)

set( CMAKE_CXX_FLAGS "-I/usr/include/mpi -std=c++14 -g3 -O3 -march=native -Wno-attributes -Wno-deprecated-declarations" )
set( CMAKE_C_FLAGS "-I/usr/include/mpi -g3 -O3 -march=native -Wno-attributes -Wno-deprecated-declarations" )
set( CMAKE_F_FLAGS "-g3 -O3 -march=native -Wno-attributes -Wno-deprecated-declarations" )


enable_language(Fortran)

add_hpx_executable(
  octotiger
  DEPENDENCIES
        silo
  SOURCES 
        src/units.cpp
      	src/helmholtz.cpp
      	src/eos.cpp
        src/sedov3.f
        src/node_server_actions_1.cpp
        src/node_server_actions_2.cpp
        src/scf_data.cpp
        src/exact_sod.cpp
        src/sedov.c
        src/profiler.cpp	
	src/geometry.cpp
	src/grid.cpp
	src/grid_fmm.cpp
	src/grid_output.cpp
	src/grid_scf.cpp
	src/lane_emden.cpp
	src/main.cpp
	src/new.cpp
	src/node_client.cpp
	src/node_location.cpp
	src/node_server.cpp
        src/options.cpp
	src/problem.cpp
	src/real.cpp
	src/roe.cpp
	src/stack_trace.cpp
	src/taylor.cpp
     	src/util.cpp
   HEADERS 
        src/diagnostics.hpp
		src/channel.hpp
		src/defs.hpp
		src/eos.hpp
		src/future.hpp
		src/geometry.hpp
		src/grid.hpp
		src/lane_emden.hpp
		src/node_client.hpp
		src/node_location.hpp
		src/node_server.hpp
		src/options.hpp
		src/problem.hpp
		src/real.hpp
		src/util.hpp
		src/rad_grid.hpp
		src/roe.hpp
		src/simd.hpp
		src/space_vector.hpp
		src/taylor.hpp
  FOLDER  "Octotiger"
)
