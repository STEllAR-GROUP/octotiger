# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

project(octotiger CXX Fortran)

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

################################################################################
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

include_directories(${HPX_INCLUDE_DIRS})
link_directories(${HPX_LIBRARY_DIR})

if(NOT MSVC)
    find_package(Silo REQUIRED)

    include_directories(${CMAKE_SOURCE_DIR}/src ~/include ${Silo_INCLUDE_DIR})

    set( CMAKE_CXX_FLAGS "-std=c++14 -g3 -O3 -march=native -Wno-attributes -Wno-deprecated-declarations" )
    set( CMAKE_C_FLAGS "-g3 -O3 -march=native -Wno-attributes -Wno-deprecated-declarations" )
    set( CMAKE_F_FLAGS "-g3 -O3 -march=native -Wno-attributes -Wno-deprecated-declarations" )
else()
    # enable solution folders for MSVC
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
    include_directories(${CMAKE_SOURCE_DIR}/src)

    hpx_add_compile_flag(-wd4244)
    hpx_add_compile_flag(-wd4267)
    hpx_add_compile_flag(-wd4800)
endif()

set(source_files)
set(dependencies)

enable_language(Fortran)

# We have to compile the Fortran sources into a separate static library as
# cmake does not support generating mixed C++/Fortran modules for MSVC.
if(MSVC)
    set(CMAKE_Fortran_FLAGS "-DDOUBLE_PRECISION ${CMAKE_Fortran_FLAGS}")

    add_hpx_library(octotiger_fortran
        SHARED
        SOURCES src/sedov3.f src/sedov3.def
        FOLDER "Octotiger/Dependencies")
    set_target_properties(octotiger_fortran_lib
        PROPERTIES LINKER_LANGUAGE Fortran)

    set(dependencies octotiger_fortran_lib)

    # Special hackery to make everything work for Windows Intel compiler
    if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
        # Make sure all Fortran symbols are lower case for Windows Intel ifort
        # compiler.
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} /names:lowercase /assume:underscore")

        # Make sure the C++ linker will find the Fortran runtime libraries.
        link_directories("$ENV{IFORT_COMPILER16}/compiler/lib/intel64_win")
    endif()
else()
    set(source_files ${source_files} src/sedov3.f)
    set(dependencies ${Silo_LIBRARY})
endif()

# Octotiger executable
set(source_files ${source_files}
    src/eos.cpp
    src/node_server_actions_1.cpp
    src/node_server_actions_2.cpp
    src/node_server_actions_3.cpp
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
)

set(header_files
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
    src/roe.hpp
    src/simd.hpp
    src/space_vector.hpp
    src/taylor.hpp
)

add_hpx_executable(
  octotiger
  DEPENDENCIES
    ${dependencies}
  SOURCES
    ${source_files}
  HEADERS
    ${header_files}
  FOLDER  "Octotiger"
)


