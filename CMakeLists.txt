# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
# Copyright (c) 2018-2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

project(octotiger CXX)

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0003 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
################################################################################
# options
################################################################################
#option(OCTOTIGER_WITH_SILO "Enable support for Silo output" ON)
option(OCTOTIGER_WITH_GRAV_PAR "Enable parallelism in gravitational solver" OFF)
option(OCTOTIGER_WITH_RADIATION "Enable radiation transport solver" OFF)
option(OCTOTIGER_WITH_CUDA "Enable CUDA fmm kernels" OFF)
option(OCTOTIGER_WITH_BOOST_MULTIPRECISION
  "Use Boost.Multiprecision Instead of GCC Quad-Precision Math Library" OFF)

################################################################################
# Find required packages
################################################################################
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Silo REQUIRED)
find_package(Vc REQUIRED)
find_package(Boost REQUIRED)

if(OCTOTIGER_WITH_CUDA)
  enable_language(CUDA)

  string(APPEND HPX_CUDA_CLANG_FLAGS " -fno-fast-math -Xcuda-ptxas -v")
endif()

################################################################################
# Set up Octotiger target
################################################################################
# Octotiger executable
set(source_files
    src/compute_factor.cpp
    src/eos.cpp
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
    src/node_registry.cpp
    src/node_server.cpp
    src/node_server_actions_1.cpp
    src/node_server_actions_2.cpp
    src/node_server_actions_3.cpp
    src/options.cpp
    src/physcon.cpp
    src/problem.cpp
    src/profiler.cpp
    src/real.cpp
    src/roe.cpp
    src/scf_data.cpp
    src/scf_data.cpp
    src/silo.cpp
    src/stack_trace.cpp
    src/taylor.cpp
    src/util.cpp
    src/common_kernel/interactions_iterators.cpp
    src/cuda_util/cuda_scheduler.cpp
    src/monopole_interactions/cuda_p2p_interaction_interface.cpp
    src/monopole_interactions/p2p_cuda_kernel.cu
    src/monopole_interactions/p2p_cpu_kernel.cpp
    src/monopole_interactions/p2p_interaction_interface.cpp
    src/monopole_interactions/calculate_stencil.cpp
    src/monopole_interactions/p2m_interaction_interface.cpp
    src/monopole_interactions/p2m_kernel.cpp
    src/multipole_interactions/multipole_interaction_interface.cpp
    src/multipole_interactions/cuda_multipole_interaction_interface.cpp
    src/multipole_interactions/multipole_cuda_kernel.cu
    src/multipole_interactions/calculate_stencil.cpp
    src/multipole_interactions/multipole_cpu_kernel.cpp
    src/radiation/rad_grid.cpp
    src/radiation/implicit.cpp
    src/test_problems/marshak/marshak.cpp
    src/test_problems/blast/sedov.cpp
    src/test_problems/blast/sedovf_c.cpp
    src/test_problems/rotating_star/rotating_star.cpp
    src/test_problems/sod/sod.cpp
    src/test_problems/sod/exact_sod.cpp
)

set(header_files
    octotiger/channel.hpp
    octotiger/compute_factor.hpp
    octotiger/const.hpp
    octotiger/container_device.hpp
    octotiger/defs.hpp
    octotiger/diagnostics.hpp
    octotiger/eos.hpp
    octotiger/future.hpp
    octotiger/geometry.hpp
    octotiger/grid.hpp
    octotiger/grid_flattened_indices.hpp
    octotiger/interaction_types.hpp
    octotiger/lane_emden.hpp
    octotiger/node_client.hpp
    octotiger/node_location.hpp
    octotiger/node_registry.hpp
    octotiger/node_server.hpp
    octotiger/options.hpp
    octotiger/options_enum.hpp
    octotiger/physcon.hpp
    octotiger/problem.hpp
    octotiger/profiler.hpp
    octotiger/real.hpp
    octotiger/roe.hpp
    octotiger/safe_math.hpp
    octotiger/scf_data.hpp
    octotiger/silo.hpp
    octotiger/simd.hpp
    octotiger/space_vector.hpp
    octotiger/state.hpp
    octotiger/struct_eos.hpp
    octotiger/taylor.hpp
    octotiger/util.hpp
    octotiger/common_kernel/helper.hpp
    octotiger/common_kernel/interaction_constants.hpp
    octotiger/common_kernel/interactions_iterators.hpp
    octotiger/common_kernel/kernel_simd_types.hpp
    octotiger/common_kernel/kernel_taylor_set_basis.hpp
    octotiger/common_kernel/kernel_taylor_set_basis.hpp
    octotiger/common_kernel/multiindex.hpp
    octotiger/common_kernel/struct_of_array_data.hpp
    octotiger/compute_factor.hpp
    octotiger/cuda_util/cuda_global_def.hpp
    octotiger/cuda_util/cuda_helper.hpp
    octotiger/cuda_util/cuda_scheduler.hpp
    octotiger/monopole_interactions/calculate_stencil.hpp
    octotiger/monopole_interactions/cuda_p2p_interaction_interface.hpp
    octotiger/monopole_interactions/monopole_kernel_templates.hpp
    octotiger/monopole_interactions/p2m_interaction_interface.hpp
    octotiger/monopole_interactions/p2m_kernel.hpp
    octotiger/monopole_interactions/p2p_cpu_kernel.hpp
    octotiger/monopole_interactions/p2p_cuda_kernel.hpp
    octotiger/monopole_interactions/p2p_interaction_interface.hpp
    octotiger/multipole_interactions/calculate_stencil.hpp
    octotiger/multipole_interactions/compute_kernel_templates.hpp
    octotiger/multipole_interactions/cuda_multipole_interaction_interface.hpp
    octotiger/multipole_interactions/multipole_cpu_kernel.hpp
    octotiger/multipole_interactions/multipole_cuda_kernel.hpp
    octotiger/multipole_interactions/multipole_interaction_interface.hpp
    octotiger/radiation/implicit.hpp
    octotiger/radiation/opacities.hpp
    octotiger/radiation/rad_grid.hpp
    octotiger/test_problems/rotating_star/rotating_star.hpp
    octotiger/test_problems/sod/sod.hpp
    octotiger/test_problems/sod/exact_sod.hpp
)

add_hpx_executable(
  octotiger
  DEPENDENCIES
    Silo::silo Vc::Vc Boost::boost
  SOURCES
    ${source_files}
  HEADERS
    ${header_files}
)

target_include_directories(octotiger PRIVATE ${PROJECT_SOURCE_DIR})

if(MSVC)
  # Enable solution folders for MSVC
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
    /wd4800
    /wd4996)

  target_compile_options(octotiger PRIVATE /EHsc)

  if(NOT OCTOTIGER_WITH_BOOST_MULTIPRECISION)
    message(WARNING "GCC Quad-Precision Math Library is not available with "
    "MSVC. Consider setting OCTOTIGER_WITH_BOOST_MULTIPRECISION to ON")
  endif()
else()
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    target_compile_options(octotiger PRIVATE
      -Wno-ignored-attributes -Wno-attributes -Wno-deprecated-declarations
      -Wno-unused-result -ffast-math)
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
      #set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -ipo -Wunused-result")
      #set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Ofast -ipo")
      target_compile_options(octotiger -Wno-attributes -Wno-deprecated)
  endif()
endif()

if(OCTOTIGER_WITH_BOOST_MULTIPRECISION)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_BOOST_MULTIPRECISION)
else()
  target_link_libraries(octotiger quadmath)
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
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_CUDA)
  set_property(TARGET octotiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Handle other options
if(OCTOTIGER_WITH_GRAV_PAR)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_GRAV_PAR)
endif()

if(OCTOTIGER_WITH_RADIATION)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_RADIATION)
endif()

################################################################################
# Set up silo_compare target
################################################################################
add_executable(silo_compare src/compare/compare.cpp)

target_link_libraries(silo_compare Silo::silo)
if(MSVC)
  target_compile_options(silo_compare PRIVATE /EHsc)
endif()
