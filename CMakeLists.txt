# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

project(octotiger C CXX)

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

################################################################################
# options
################################################################################
option(OCTOTIGER_WITH_SILO "Enable support for Silo output" ON)
option(OCTOTIGER_WITH_GRAV_PAR "Enable parallelism in gravitational solver" OFF)
option(OCTOTIGER_WITH_RADIATION "Enable radiation transport solver" OFF)
option(OCTOTIGER_WITH_CUDA "Enable cuda fmm kernels" OFF)

################################################################################
# Find required packages
################################################################################
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
if (OCTOTIGER_WITH_SILO)
  find_package(Silo REQUIRED)
endif()

if (OCTOTIGER_WITH_CUDA)
  add_definitions(-DOCTOTIGER_CUDA_ENABLED)
  find_package(CUDA  REQUIRED)
  include_directories("${CUDA_INCLUDE_DIRS}")
  link_directories(${CUDA_CUBLAS_LIBRARIES})
  string(APPEND HPX_CUDA_CLANG_FLAGS " -fno-fast-math -Xcuda-ptxas -v")
endif()

include_directories(${HPX_INCLUDE_DIRS})
link_directories(${HPX_LIBRARY_DIR})

################################################################################
# Silo support
################################################################################
if (OCTOTIGER_WITH_SILO)
  add_definitions(-DOCTOTIGER_HAVE_SILO)
else()
  set(Silo_LIBRARY "" CACHE STRING "" FORCE)
  set(Silo_INCLUDE_DIR "" CACHE STRING "" FORCE)
endif()

if(NOT MSVC)
    include_directories(${CMAKE_SOURCE_DIR}/src ~/include ${Silo_INCLUDE_DIR})

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set( COMMON_FLAGS "-Wno-ignored-attributes -Wno-attributes -Wno-deprecated-declarations -ffast-math")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        #set( CMAKE_CXX_FLAGS_RELEASE "-Ofast -ipo" )
        #set( CMAKE_C_FLAGS_RELEASE "-Ofast -ipo" )
        #set( CMAKE_Fortran_FLAGS_RELEASE "-Ofast -ipo" )
        #set( CMAKE_EXE_LINKER_FLAGS_RELEASE "-Ofast -ipo" )
        set( COMMON_FLAGS "-Wno-attributes -Wno-deprecated")
    endif()
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${COMMON_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_F_FLAGS "${COMMON_FLAGS} ${CMAKE_F_FLAGS}")
else()
    # enable solution folders for MSVC
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
    include_directories(${CMAKE_SOURCE_DIR}/src)

    hpx_add_compile_flag(-wd4068)   # unknown pragma
    hpx_add_compile_flag(-wd4244)
    hpx_add_compile_flag(-wd4267)
    hpx_add_compile_flag(-wd4800)
endif()

# handle other options
if (OCTOTIGER_WITH_GRAV_PAR)
  add_definitions(-DOCTOTIGER_HAVE_GRAV_PAR)
endif()

if (OCTOTIGER_WITH_RADIATION)
  add_definitions(-DOCTOTIGER_HAVE_RADIATION)
endif()

set(source_files)
set(dependencies)

set(dependencies ${Silo_LIBRARY})

# Octotiger executable
set(source_files ${source_files}
    src/test_problems/rotating_star/rotating_star.cpp
    src/eos.cpp
    src/scf_data.cpp
    src/exact_sod.cpp
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
    src/node_server_actions_1.cpp
    src/node_server_actions_2.cpp
    src/node_server_actions_3.cpp
    src/options.cpp
    src/physcon.cpp
    src/problem.cpp
    src/profiler.cpp
    src/radiation/rad_grid.cpp
    src/radiation/implicit.cpp
    src/real.cpp
    src/roe.cpp
    src/scf_data.cpp
    src/stack_trace.cpp
    src/taylor.cpp
    src/util.cpp
    src/compute_factor.cpp
    src/multipole_interactions/multipole_interaction_interface.cpp
    src/multipole_interactions/cuda_multipole_interaction_interface.cpp
    src/multipole_interactions/multipole_cuda_kernel.cu
    src/multipole_interactions/calculate_stencil.cpp
    src/multipole_interactions/multipole_cpu_kernel.cpp
    src/monopole_interactions/cuda_p2p_interaction_interface.cpp
    src/monopole_interactions/p2p_cuda_kernel.cu
    src/monopole_interactions/p2p_cpu_kernel.cpp
    src/monopole_interactions/p2p_interaction_interface.cpp
    src/monopole_interactions/calculate_stencil.cpp
    src/monopole_interactions/p2m_interaction_interface.cpp
    src/monopole_interactions/p2m_kernel.cpp
    src/common_kernel/interactions_iterators.cpp
    src/cuda_util/cuda_scheduler.cpp
)

set(header_files
    src/mesa/mesa.hpp
    src/channel.hpp
    src/container_device.hpp
    src/defs.hpp
    src/diagnostics.hpp
    src/eos.hpp
    src/exact_sod.hpp
    src/future.hpp
    src/geometry.hpp
    src/grid.hpp
    src/lane_emden.hpp
    src/node_client.hpp
    src/node_location.hpp
    src/node_server.hpp
    src/options.hpp
    src/physcon.hpp
    src/problem.hpp
    src/profiler.hpp
    src/radiation/rad_grid.hpp
    src/real.hpp
    src/roe.hpp
    src/scf_data.hpp
    src/set_locality_data.hpp
    src/simd.hpp
    src/space_vector.hpp
    src/state.hpp
    src/struct_eos.hpp
    src/taylor.hpp
    src/util.hpp
    src/compute_factor.hpp
    src/common_kernel/interaction_constants.hpp
    src/common_kernel/multiindex.hpp
    src/multipole_interactions/multipole_interaction_interface.hpp
    src/multipole_interactions/compute_kernel_templates.hpp
    src/multipole_interactions/multipole_cuda_kernel.hpp
    src/multipole_interactions/cuda_multipole_interaction_interface.hpp
    src/multipole_interactions/calculate_stencil.hpp
    src/multipole_interactions/multipole_cpu_kernel.hpp
    src/monopole_interactions/cuda_p2p_interaction_interface.hpp
    src/monopole_interactions/p2p_interaction_interface.hpp
    src/monopole_interactions/p2p_cuda_kernel.hpp
    src/monopole_interactions/p2p_cpu_kernel.hpp
    src/monopole_interactions/calculate_stencil.hpp
    src/monopole_interactions/p2m_interaction_interface.hpp
    src/monopole_interactions/p2m_kernel.hpp
    src/monopole_interactions/compare_interaction_lists.hpp
    src/common_kernel/kernel_simd_types.hpp
    src/common_kernel/kernel_taylor_set_basis.hpp
    src/cuda_util/cuda_scheduler.hpp
)


set(silo_cmp_dependencies ${Silo_LIBRARY} )
set(silo_cmp_source_files
    test/silo_cmp.cpp
)
set(silo_cmp_header_files)


set(analysis_source_files
    analysis_src/analysis.cpp
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

if (OCTOTIGER_WITH_SILO)
add_hpx_executable(
  silocmp
  DEPENDENCIES
    ${silo_cmp_dependencies}
  SOURCES
    ${silo_cmp_source_files}
  HEADERS
    ${silo_cmp_header_files}
  FOLDER  "Octotiger"
)
endif()


add_executable( analysis ${analysis_source_files} )
