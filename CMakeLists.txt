# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
# Copyright (c) 2018-2019 Parsa Amini
# Copyright (c) 2018-2020 Gregor Daiß
# Copyright (c) 2018-2020 Theresa Pollinger

#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)

# Libraries linked via full path no longer produce linker search paths.
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)

project(octotiger CXX C)

# Search path for CMake modules to be loaded by include() and find_package()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 14)
################################################################################
# Options
################################################################################
option(OCTOTIGER_WITH_GRAV_PAR "Enable parallelism in gravitational solver" OFF)
option(OCTOTIGER_WITH_RADIATION "Enable radiation transport solver" OFF)
option(OCTOTIGER_WITH_CUDA "Enable CUDA fmm kernels" OFF)
option(OCTOTIGER_WITH_KOKKOS "Enable the build with Kokkos" OFF)
option(OCTOTIGER_SPACK_BUILD "Project is build with the spack package" OFF)
option(OCTOTIGER_WITH_BLAST_TEST "Enable the Blast test" ON)
option(OCTOTIGER_WITH_BOOST_MULTIPRECISION
  "Use Boost.Multiprecision Instead of GCC Quad-Precision Math Library" OFF)
option(OCTOTIGER_WITH_QUADMATH "Enable sections using GCC Quad-Precision Math Library" ON)
option(OCTOTIGER_WITH_Vc "" ON)
# Re-enables Vc datatypes in the octotiger parts - use with caution
option(OCTOTIGER_WITH_LEGACY_Vc "" OFF)
option(OCTOTIGER_WITH_AVX "" OFF)
option(OCTOTIGER_WITH_AVX2 "" OFF)
option(OCTOTIGER_WITH_AVX512 "" OFF)
option(OCTOTIGER_WITH_TESTS "Enable tests" ON)
set(OCTOTIGER_WITH_GRIDDIM "8" CACHE STRING "Grid size")
set(OCTOTIGER_THETA_MINIMUM "0.34" CACHE STRING "Minimal allowed theta value - important for optimizations")
option(OCTOTIGER_WITH_DOCU "Enable the target to build the documentation" OFF)


# silence warnings for deprecated HPX includes
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cpp")

# add -fPIC option to avoid this error:
# https://cmake.org/pipermail/cmake/2015-January/059513.html 
# in case of relocatable device code
# (modifying CMAKE_CUDA_(CXX_)_FLAGS did not help, 
# so we keep this flag at the cost of duplicating it for host code)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

################################################################################
# Spack setup
################################################################################
# see https://spack.readthedocs.io/en/latest/workflows.html#write-the-cmake-build
if (OCTOTIGER_SPACK_BUILD)
  set(CMAKE_MACOSX_RPATH 1)
  set(CMAKE_SKIP_BUILD_RPATH  FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif("${isSystemDir}" STREQUAL "-1")
  include_directories($ENV{SPACK_TRANSITIVE_INCLUDE_PATH})
endif()
################################################################################
# Find required packages
################################################################################
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(CPPuddle REQUIRED)
find_package(Silo REQUIRED)
find_package(Boost REQUIRED)
if(NOT MSVC)
  find_package(Threads REQUIRED)  # Required by target gen_rotating_star_init
endif()

if(OCTOTIGER_WITH_CUDA)
  enable_language(CUDA)
  #set(MY_CUDA_FLAGS -Xptxas=-fmad=true,-dlcm=cg,--opt-level=4)
  #string(APPEND HPX_CUDA_CLANG_FLAGS " -fast-math -Xcuda-ptxas -v")
  string(APPEND CUDA_NVCC_FLAGS ";-Xptxas=-v,-fmad=true,--opt-level=4")

  if (OCTOTIGER_WITH_LEGACY_Vc)
    message(FATAL_ERROR " CUDA cannot be used together with OCTOTIGER_WITH_LEGACY_Vc!")
  endif()
endif()

if(OCTOTIGER_WITH_KOKKOS)
  message(INFO " KOKKOS support is turned on!")
  find_package(Kokkos REQUIRED)
  find_package(HPXKokkos REQUIRED)
  if (OCTOTIGER_WITH_LEGACY_Vc)
    message(FATAL_ERROR " Kokkos cannot be used together with OCTOTIGER_WITH_LEGACY_Vc!")
  endif()
endif()


################################################################################
# Set up Octo-Tiger target
################################################################################
# Octo-Tiger library sources

# Octo-Tiger library headers
set(source_files
    src/compute_factor.cpp
    src/eos.cpp
    src/geometry.cpp
    src/grid_amr.cpp
    src/grid.cpp
    src/grid_fmm.cpp
    src/grid_output.cpp
    src/grid_scf.cpp
    src/lane_emden.cpp
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
    src/io/silo.cpp
    src/io/silo_out.cpp
    src/io/silo_in.cpp
    src/stack_trace.cpp
    src/taylor.cpp
    src/util.cpp
    src/common_kernel/interactions_iterators.cpp
    src/cuda_util/cuda_scheduler.cpp
    src/monopole_interactions/legacy/cuda_p2p_interaction_interface.cpp
    src/monopole_interactions/legacy/p2p_cuda_kernel.cu
    src/monopole_interactions/legacy/p2p_cpu_kernel.cpp
    src/monopole_interactions/legacy/p2p_interaction_interface.cpp
    src/monopole_interactions/p2p_kernel_interface.cpp
    src/multipole_interactions/multipole_kernel_interface.cpp
    src/monopole_interactions/util/calculate_stencil.cpp
    src/monopole_interactions/legacy/p2m_interaction_interface.cpp
    src/monopole_interactions/legacy/p2m_kernel.cpp
    src/multipole_interactions/legacy/multipole_interaction_interface.cpp
    src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp
    src/multipole_interactions/legacy/multipole_cuda_kernel.cu
    src/multipole_interactions/util/calculate_stencil.cpp
    src/multipole_interactions/legacy/multipole_cpu_kernel.cpp
    src/radiation/cuda_kernel.cu
    src/radiation/rad_grid.cpp
    src/test_problems/marshak/marshak.cpp
    src/test_problems/rotating_star/rotating_star.cpp
    src/test_problems/sod/sod.cpp
    src/test_problems/sod/exact_sod.cpp
    src/test_problems/amr/amr.cpp

    src/unitiger/hydro_impl/hydro_boundary_exchange.cpp # TODO Move to different location - it's not unitiger
    src/unitiger/hydro_impl/hydro_boundary_exchange.cu # TODO Move to different location - it's not unitiger
)

# Octo-Tiger library headers
set(header_files
    octotiger/config/export_definitions.hpp
    octotiger/channel.hpp
    octotiger/compute_factor.hpp
    octotiger/config.hpp
    octotiger/const.hpp
    octotiger/container_device.hpp
    octotiger/defs.hpp
    octotiger/diagnostics.hpp
    octotiger/eos.hpp
    octotiger/future.hpp
    octotiger/geometry.hpp
    octotiger/grid.hpp
    octotiger/grid_flattened_indices.hpp
    octotiger/grid_fmm.hpp
    octotiger/grid_scf.hpp
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
    octotiger/io/silo.hpp
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
    octotiger/common_kernel/kokkos_util.hpp
    octotiger/compute_factor.hpp
    octotiger/cuda_util/cuda_global_def.hpp
    octotiger/cuda_util/cuda_helper.hpp
    octotiger/cuda_util/cuda_scheduler.hpp
    octotiger/monopole_interactions/util/calculate_stencil.hpp
    octotiger/monopole_interactions/legacy/cuda_p2p_interaction_interface.hpp
    octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp
    octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp
    octotiger/monopole_interactions/legacy/p2m_kernel.hpp
    octotiger/monopole_interactions/legacy/p2p_cpu_kernel.hpp
    octotiger/monopole_interactions/legacy/p2p_cuda_kernel.hpp
    octotiger/monopole_interactions/legacy/p2p_interaction_interface.hpp
    octotiger/multipole_interactions/util/calculate_stencil.hpp
    octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp
    octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp
    octotiger/multipole_interactions/legacy/multipole_cpu_kernel.hpp
    octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp
    octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp
    octotiger/radiation/cpu_kernel.hpp
    octotiger/radiation/cuda_kernel.hpp
    octotiger/radiation/implicit.hpp
    octotiger/radiation/kernel_interface.hpp
    octotiger/radiation/opacities.hpp
    octotiger/radiation/rad_grid.hpp
    octotiger/test_problems/rotating_star.hpp
)

list(APPEND octo_dependencies HPX::hpx Silo::silo Vc::Vc Boost::boost CPPuddle::buffer_manager CPPuddle::stream_manager)
if(OCTOTIGER_WITH_KOKKOS)
  list(APPEND octo_dependencies Kokkos::kokkos HPXKokkos::hpx_kokkos)
endif()

# Octo-Tiger library
add_hpx_library(
  octolib
  DEPENDENCIES
    ${octo_dependencies}
  SOURCES
    ${source_files}
  HEADERS
    ${header_files}
)
# causes internal compiler error on cuda 10.0: TODO try with later version
# set_source_files_properties(${source_files} ${header_files} PROPERTIES COMPILE_FLAGS " --expt-relaxed-constexpr")

set(hydro_header_files
    octotiger/unitiger/hydro.hpp
    octotiger/unitiger/physics.hpp
    octotiger/unitiger/physics_impl.hpp
    octotiger/unitiger/cell_geometry.hpp
    octotiger/unitiger/safe_real.hpp
    octotiger/unitiger/hydro_impl/boundaries.hpp
    octotiger/unitiger/hydro_impl/advance.hpp
    octotiger/unitiger/hydro_impl/output.hpp
    octotiger/unitiger/hydro_impl/hydro.hpp
    octotiger/unitiger/hydro_impl/reconstruct.hpp
    octotiger/unitiger/hydro_impl/flux.hpp
    octotiger/unitiger/radiation/radiation_physics.hpp
    octotiger/unitiger/radiation/radiation_physics_impl.hpp
    octotiger/test_problems/exact_sod.hpp
    octotiger/unitiger/hydro_impl/hydro_cuda/hydro_cuda_kernels.hpp)

set(hydro_source_files
   src/unitiger/hydro.cpp
   src/unitiger/hydro_impl/flux_kernel_interface.cpp
   src/unitiger/hydro_impl/flux_cpu_kernel.cpp
   src/unitiger/hydro_impl/flux_cuda_kernel.cu
   src/unitiger/hydro_impl/reconstruct_cuda_kernel.cu
   src/unitiger/hydro_impl/reconstruct_kernel_interface.cpp
   src/unitiger/hydro_impl/reconstruct_cpu_kernel.cpp
   src/test_problems/sod/exact_sod.cpp
   src/unitiger/hydro_impl/hydro_cuda/reconstruct_cuda_kernel.cu)

list(APPEND hydro_dependencies HPX::hpx Silo::silo Vc::Vc Boost::boost octolib)
if(OCTOTIGER_WITH_KOKKOS)
  list(APPEND hydro_dependencies Kokkos::kokkos HPXKokkos::hpx_kokkos)
endif()
if(OCTOTIGER_WITH_BLAST_TEST)
  list(APPEND hydro_source_files 
    src/test_problems/blast/sedov.cpp
    src/test_problems/blast/sedovf_c.cpp)
  list(APPEND hydro_dependencies quadmath)
endif()

# Hydro library
add_hpx_library(
  hydrolib
  DEPENDENCIES
    ${hydro_dependencies}
  SOURCES
    ${hydro_source_files}
  HEADERS
    ${hydro_header_files}
)

if(OCTOTIGER_WITH_KOKKOS)
  # Deal with the incompatibility of Kokkos+Cuda and Vc, HPX actions and general constexpr problems
  set(host_only_source_files
    # Source files that need Vc
    src/compute_factor.cpp
    src/roe.cpp
    src/common_kernel/interactions_iterators.cpp
    src/monopole_interactions/legacy/p2m_kernel.cpp
    src/monopole_interactions/legacy/p2m_interaction_interface.cpp
    src/monopole_interactions/legacy/p2p_cpu_kernel.cpp
    src/monopole_interactions/legacy/p2p_interaction_interface.cpp
    src/multipole_interactions/legacy/multipole_cpu_kernel.cpp
    src/multipole_interactions/legacy/multipole_interaction_interface.cpp
    src/unitiger/hydro_impl/reconstruct_cpu_kernel.cpp
    src/unitiger/hydro_impl/flux_cpu_kernel.cpp

    # Other reasons for host-only compilation (hpx actions)
    src/grid.cpp
    src/grid_fmm.cpp
    src/grid_output.cpp
    src/grid_scf.cpp 
    src/node_client.cpp
    src/node_location.cpp
    src/node_registry.cpp
    src/node_server.cpp
    src/node_server_actions_1.cpp
    src/node_server_actions_2.cpp
    src/node_server_actions_3.cpp
    src/physcon.cpp # TODO(daissgr) Check if actions can be made to work with Kokkos
    src/io/silo_out.cpp
    src/io/silo_in.cpp
    src/radiation/rad_grid.cpp
    frontend/frontend-helper.cpp

    #misc
    src/cuda_util/cuda_scheduler.cpp
    src/unitiger/hydro_impl/flux_kernel_interface.cpp
    src/unitiger/hydro_impl/reconstruct_kernel_interface.cpp
  )
  set(host_only_header_files
    octotiger/simd.hpp
    octotiger/common_kernel/kernel_simd_types.hpp
  )
  message(INFO " Host-only list set!")
  set_source_files_properties(${host_only_source_files} ${host_only_header_files} PROPERTIES COMPILE_FLAGS " -isystem ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} --host-only")

  # Add symbols
  target_compile_definitions(octolib PUBLIC OCTOTIGER_HAVE_KOKKOS)
  target_compile_definitions(octolib PUBLIC HPX_WITH_CUDA) # workaround for the kokkos hpx include
  target_compile_definitions(hydrolib PUBLIC OCTOTIGER_HAVE_KOKKOS)
  target_compile_definitions(hydrolib PUBLIC HPX_WITH_CUDA) # workaround for the kokkos hpx include
endif()

# Enable exported functions
target_compile_definitions(octolib PRIVATE OCTOTIGER_EXPORTS)
# Header files
set_property(TARGET octolib PROPERTY FOLDER "Octo-Tiger")

# add octotiger/* to compiler search path
target_include_directories(octolib PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(hydrolib PUBLIC ${PROJECT_SOURCE_DIR})

# Octo-Tiger executable
add_hpx_executable(
  octotiger
  DEPENDENCIES
    HPX::hpx
    octolib
    hydrolib
  SOURCES
    frontend/main.cpp
    frontend/frontend-helper.cpp
    frontend/init_methods.cpp
)
set_property(TARGET octotiger PROPERTY FOLDER "Octo-Tiger")

# Unitiger executable
add_hpx_executable(
  unitiger
  DEPENDENCIES
    Silo::silo hydrolib 
  SOURCES
    src/unitiger/main.cpp
  HEADERS
    ${hydro_header_files}
)
set_property(TARGET octotiger PROPERTY FOLDER "Octo-Tiger")

if(MSVC)
  # Enable solution folders for MSVC
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY
    VS_STARTUP_PROJECT octotiger)

  string(REPLACE "/" "\\" SILO_ROOTPATH ${Silo_DIR})
  string(REPLACE "/lib/cmake/HPX" "" HPX_ROOTPATH ${HPX_DIR})
  string(REPLACE "/" "\\" HPX_ROOTPATH ${HPX_ROOTPATH})
  add_custom_command(TARGET octotiger PRE_BUILD
    COMMAND xcopy /D /Y ${SILO_ROOTPATH}\\SiloWindows\\MSVC2012\\x64\\Release\\*.dll $(TargetDir)
    COMMAND xcopy /D /Y ${HPX_ROOTPATH}\\$(Configuration)\\bin\\*.dll $(TargetDir)
    COMMENT "Copying files from Silo and HPX for $(Configuration) configuration")

  target_compile_definitions(octolib PUBLIC
    _USE_MATH_DEFINES
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    __restrict__=__restrict
    _ENABLE_EXTENDED_ALIGNED_STORAGE)

  set(MSVC_WARNING_SILENCERS
    /wd4018
    /wd4068 # unknown pragma
    /wd4146
    /wd4244
    /wd4267
    /wd4334
    /wd4521
    /wd4800
    /wd4996
  )
  target_compile_options(octolib PUBLIC ${MSVC_WARNING_SILENCERS})
  # HACK: Silencing warnings in octolib does not propagate to Octo-Tiger. No clue why.
  target_compile_options(octotiger PRIVATE ${MSVC_WARNING_SILENCERS})

  target_compile_options(octolib PUBLIC /EHsc)

else()
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    target_compile_options(octolib PRIVATE
      -Wno-ignored-attributes -Wno-attributes -Wno-deprecated-declarations
      -Wno-unused-result -ffast-math -Wno-write-strings)
    target_compile_options(unitiger PRIVATE
      -Wno-ignored-attributes -Wno-attributes -Wno-deprecated-declarations
      -Wno-unused-result -ffast-math -Wno-write-strings)
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
      #set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -ipo -Wunused-result ")
      #set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Ofast -ipo")
      target_compile_options(octolib -Wno-attributes -Wno-deprecated)
  endif()
endif()

if(OCTOTIGER_WITH_Vc)
  find_package(Vc REQUIRED)

  target_compile_definitions(octolib PUBLIC
    OCTOTIGER_HAVE_VC
    # Need serialization support from HPX
    # This just spits out lots of warnings
  )
  if (OCTOTIGER_WITH_LEGACY_Vc)
     target_compile_definitions(octolib PUBLIC OCTOTIGER_LEGACY_VC
       # Need serialization support from HPX
       # This just spits out lots of warnings
       HPX_HAVE_DATAPAR
       HPX_HAVE_DATAPAR_VC 
     )
  endif()
endif()

if(OCTOTIGER_WITH_Vc AND USE_AVX512 AND OCTOTIGER_WITH_AVX512)
  if(MSVC)
    target_compile_options(octolib PRIVATE /arch:AVX512)
  else()
    target_compile_options(octolib PRIVATE -mavx512f)
  endif()
  message(STATUS "Enabled AVX512 compile options")
  set(OCTOTIGER_WITH_AVX OFF CACHE BOOL "" FORCE)
  set(OCTOTIGER_WITH_AVX2 OFF CACHE BOOL "" FORCE)
elseif(OCTOTIGER_WITH_Vc AND USE_AVX2 AND OCTOTIGER_WITH_AVX2)
  if(MSVC)
    target_compile_options(octolib PRIVATE /arch:AVX2)
  else()
    target_compile_options(octolib PRIVATE -mavx2)
  endif()
  message(STATUS "Enabled AVX2 compile options")
  set(OCTOTIGER_WITH_AVX OFF CACHE BOOL "" FORCE)
  set(OCTOTIGER_WITH_AVX512 OFF CACHE BOOL "" FORCE)
elseif(OCTOTIGER_WITH_Vc AND USE_AVX AND OCTOTIGER_WITH_AVX)
  if(MSVC)
    target_compile_options(octolib PRIVATE /arch:AVX)
  else()
    target_compile_options(octolib PRIVATE -mavx)
  endif()
  message(STATUS "Enabled AVX compile options")
  set(OCTOTIGER_WITH_AVX2 OFF CACHE BOOL "" FORCE)
  set(OCTOTIGER_WITH_AVX512 OFF CACHE BOOL "" FORCE)
endif()

# Handle CUDA
if(OCTOTIGER_WITH_CUDA)
  message(INFO " CUDA support is turned on!")
  target_compile_definitions(octolib PUBLIC OCTOTIGER_HAVE_CUDA)
  set_property(TARGET octolib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET octolib PROPERTY INTERFACE_COMPILE_OPTIONS "${MY_CUDA_FLAGS}")


endif()

# Handle other options
if(OCTOTIGER_WITH_GRAV_PAR)
  target_compile_definitions(octolib PUBLIC OCTOTIGER_HAVE_GRAV_PAR)
endif()

if(OCTOTIGER_WITH_RADIATION)
  target_compile_definitions(octolib PUBLIC OCTOTIGER_HAVE_RADIATION)
endif()

# Grid size
message(STATUS "Octo-Tiger grid size: ${OCTOTIGER_WITH_GRIDDIM}")
target_compile_definitions(octolib PUBLIC OCTOTIGER_GRIDDIM=${OCTOTIGER_WITH_GRIDDIM})

# Theta minimum
message(STATUS "Octo-Tiger minimal allowed theta: ${OCTOTIGER_THETA_MINIMUM}")
target_compile_definitions(octolib PUBLIC OCTOTIGER_THETA_MINIMUM=${OCTOTIGER_THETA_MINIMUM})

# Blast test problem quad-precision float support
if(OCTOTIGER_WITH_BLAST_TEST)
  target_compile_definitions(octolib PUBLIC OCTOTIGER_HAVE_BLAST_TEST)
  set(blast_sources
    src/test_problems/blast/sedov.cpp
    src/test_problems/blast/sedovf_c.cpp
    )
  target_sources(octolib PRIVATE ${blast_sources})
  source_group(TREE ${PROJECT_SOURCE_DIR}
    PREFIX "Source Files" FILES ${blast_sources})

  if(OCTOTIGER_WITH_BOOST_MULTIPRECISION)
    target_compile_definitions(octolib PUBLIC
      OCTOTIGER_HAVE_BOOST_MULTIPRECISION)
    set(OCTOTIGER_WITH_QUADMATH OFF CACHE BOOL "" FORCE)

  elseif(OCTOTIGER_WITH_QUADMATH)
    # Check if on PowerPC with GCC <8
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
      message(FATAL_ERROR "GCC Quad-Precision Math Library is only available "
        "with GCC 8 and above on PowerPC")
    elseif(MSVC)
      message(FATAL_ERROR "GCC Quad-Precision Math Library is not available with MSVC")
    endif()
    target_compile_definitions(octolib PUBLIC OCTOTIGER_HAVE_QUADMATH)
    target_link_libraries(octolib PUBLIC quadmath)

  else()
      message(FATAL_ERROR "Octo-Tiger with OCTOTIGER_WITH_BLAST_TEST=ON cannot "
        "be built without quadmath or Boost.Multiprecision")
  endif()
endif()

if(OCTOTIGER_WITH_DOCU)
  add_subdirectory(doc)    
endif()


################################################################################
# Tool targets
################################################################################
add_subdirectory(tools)

################################################################################
# Tests
################################################################################
if(OCTOTIGER_WITH_TESTS)
    enable_testing()

  add_subdirectory(test_problems)
endif()
