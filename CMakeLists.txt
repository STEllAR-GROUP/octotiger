# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
# Copyright (c) 2018-2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

project(octotiger CXX)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0003 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
################################################################################
# options
################################################################################
#option(OCTOTIGER_WITH_SILO "Enable support for Silo output" ON)
option(OCTOTIGER_WITH_GRAV_PAR "Enable parallelism in gravitational solver" OFF)
option(OCTOTIGER_WITH_RADIATION "Enable radiation transport solver" OFF)
option(OCTOTIGER_WITH_CUDA "Enable CUDA fmm kernels" OFF)

################################################################################
# Find required packages
################################################################################
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Silo REQUIRED)
find_package(HDF5 REQUIRED)
find_package(Vc REQUIRED)
find_package(Boost REQUIRED)

add_library(octotiger::hdf5 INTERFACE IMPORTED)
set_property(TARGET octotiger::hdf5
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS})
set_property(TARGET octotiger::hdf5
  PROPERTY INTERFACE_LINK_LIBRARIES ${HDF5_LIBRARIES})

if(OCTOTIGER_WITH_CUDA)
  add_definitions(-DOCTOTIGER_CUDA_ENABLED)
  find_package(CUDA REQUIRED)

  add_library(octotiger::cuda INTERFACE IMPORTED)
  set_property(TARGET octotiger::cuda
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
  set_property(TARGET octotiger::cuda
    PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
  string(APPEND HPX_CUDA_CLANG_FLAGS " -fno-fast-math -Xcuda-ptxas -v")
endif()

################################################################################
# Set up Octotiger target
################################################################################
set(dependencies silo::silo Vc::Vc Boost::boost)

if(NOT MSVC)
  list(APPEND dependencies octotiger::hdf5 dl z quadmath)
endif()

# Octotiger executable
set(source_files
    src/test_problems/marshak/marshak.cpp
    src/test_problems/blast/sedov.cpp
    src/test_problems/blast/sedovf_c.cpp
    src/test_problems/rotating_star/rotating_star.cpp
    src/test_problems/sod/sod.cpp
    src/silo.cpp
    src/node_registry.cpp
    src/eos.cpp
    src/scf_data.cpp
    src/test_problems/sod/exact_sod.cpp
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
    src/channel.hpp
    src/container_device.hpp
    src/defs.hpp
    src/diagnostics.hpp
    src/eos.hpp
    src/test_problems/sod/exact_sod.hpp
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
    src/common_kernel/kernel_simd_types.hpp
    src/common_kernel/kernel_taylor_set_basis.hpp
    src/cuda_util/cuda_scheduler.hpp
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

target_include_directories(octotiger PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_SILO)

if(NOT MSVC)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(octotiger PRIVATE
      -Wno-ignored-attributes -Wno-attributes -Wno-deprecated-declarations
      -Wno-unused-result -ffast-math)
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      #set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -ipo -Wunused-result")
      #set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Ofast -ipo")
      target_compile_options(octotiger -Wno-attributes -Wno-deprecated)
  endif()
else()
  # enable solution folders for MSVC
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  target_compile_definitions(octotiger PRIVATE
    _USE_MATH_DEFINES
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    __restrict__=__restrict)

  target_compile_options(octotiger PRIVATE
    /wd4018
    /wd4068 # unknown pragma
    /wd4146
    /wd4244
    /wd4267
    /wd4334
    /wd4521
    /wd4800)

  target_compile_options(octotiger PRIVATE /EHsc)
endif()

if(USE_AVX2)
  if(MSVC)
    target_compile_options(octotiger PRIVATE /arch:AVX2)
  else()
    target_compile_options(octotiger PRIVATE -mavx2)
  endif()
elseif(USE_AVX)
  if(MSVC)
    target_compile_options(octotiger PRIVATE /arch:AVX)
  else()
    target_compile_options(octotiger PRIVATE -mavx)
  endif()
endif()

# Handle CUDA
if(OCTOTIGER_WITH_CUDA)
  target_link_libraries(octotiger octotiger::cuda)
  set_property(TARGET octotiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Handle other options
if (OCTOTIGER_WITH_GRAV_PAR)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_GRAV_PAR)
endif()

if (OCTOTIGER_WITH_RADIATION)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_RADIATION)
endif()

################################################################################
# Set up silo_compare target
################################################################################
add_executable(silo_compare src/compare/compare.cpp)

target_link_libraries(silo_compare silo::silo)
if(NOT MSVC)
  target_link_libraries(silo_compare octotiger::hdf5 dl pthread dl z)
else()
  target_compile_options(silo_compare PRIVATE /EHsc)
endif()
