# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
# Copyright (c) 2018-2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.10)
cmake_policy(VERSION 3.10)

# Libraries linked via full path no longer produce linker search paths.
cmake_policy(SET CMP0003 NEW)

project(octotiger CXX)

# Search path for CMake modules to be loaded by include() and find_package()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

################################################################################
# Options
################################################################################
option(OCTOTIGER_WITH_GRAV_PAR "Enable parallelism in gravitational solver" OFF)
option(OCTOTIGER_WITH_RADIATION "Enable radiation transport solver" OFF)
option(OCTOTIGER_WITH_CUDA "Enable CUDA fmm kernels" OFF)
option(OCTOTIGER_WITH_BLAST_TEST "Enable the Blast test" ON)
option(OCTOTIGER_WITH_BOOST_MULTIPRECISION
  "Use Boost.Multiprecision Instead of GCC Quad-Precision Math Library" OFF)
option(OCTOTIGER_WITH_QUADMATH "Enable sections using GCC Quad-Precision Math Library" ON)
option(OCTOTIGER_WITH_Vc "" ON)
option(OCTOTIGER_WITH_AVX "" OFF)
option(OCTOTIGER_WITH_AVX2 "" OFF)
option(OCTOTIGER_WITH_AVX512 "" OFF)
option(OCTOTIGER_WITH_TESTS "Enable tests" ON)

set(OCTOTIGER_WITH_GRIDDIM "8" CACHE STRING "Grid size")

################################################################################
# Find required packages
################################################################################
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
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
    src/radiation/cuda_kernel.cu
    src/radiation/rad_grid.cpp
    src/test_problems/marshak/marshak.cpp
    src/test_problems/rotating_star/rotating_star.cpp
    src/test_problems/sod/sod.cpp
    src/test_problems/sod/exact_sod.cpp
)

set(header_files
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
    octotiger/radiation/cpu_kernel.hpp
    octotiger/radiation/cuda_kernel.hpp
    octotiger/radiation/implicit.hpp
    octotiger/radiation/kernel_interface.hpp
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

set_property(TARGET octotiger PROPERTY FOLDER "Octotiger")
target_include_directories(octotiger PRIVATE ${PROJECT_SOURCE_DIR})

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

if(OCTOTIGER_WITH_Vc)
  find_package(Vc REQUIRED)

  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_VC)
endif()

if(OCTOTIGER_WITH_Vc AND USE_AVX512 AND OCTOTIGER_WITH_AVX512)
  if(MSVC)
    target_compile_options(octotiger PRIVATE /arch:AVX512)
  else()
    target_compile_options(octotiger PRIVATE -mavx512f)
  endif()
  message(STATUS "Enabled AVX512 compile options")
  set(OCTOTIGER_WITH_AVX OFF CACHE BOOL "" FORCE)
  set(OCTOTIGER_WITH_AVX2 OFF CACHE BOOL "" FORCE)
elseif(OCTOTIGER_WITH_Vc AND USE_AVX2 AND OCTOTIGER_WITH_AVX2)
  if(MSVC)
    target_compile_options(octotiger PRIVATE /arch:AVX2)
  else()
    target_compile_options(octotiger PRIVATE -mavx2)
  endif()
  message(STATUS "Enabled AVX2 compile options")
  set(OCTOTIGER_WITH_AVX OFF CACHE BOOL "" FORCE)
  set(OCTOTIGER_WITH_AVX512 OFF CACHE BOOL "" FORCE)
elseif(OCTOTIGER_WITH_Vc AND USE_AVX AND OCTOTIGER_WITH_AVX)
  if(MSVC)
    target_compile_options(octotiger PRIVATE /arch:AVX)
  else()
    target_compile_options(octotiger PRIVATE -mavx)
  endif()
  message(STATUS "Enabled AVX compile options")
  set(OCTOTIGER_WITH_AVX2 OFF CACHE BOOL "" FORCE)
  set(OCTOTIGER_WITH_AVX512 OFF CACHE BOOL "" FORCE)
endif()

# Handle CUDA
if(OCTOTIGER_WITH_CUDA)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_CUDA)
  set_property(TARGET octotiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET octotiger PROPERTY INTERFACE_COMPILE_OPTIONS "${MY_CUDA_FLAGS}")
endif()

# Handle other options
if(OCTOTIGER_WITH_GRAV_PAR)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_GRAV_PAR)
endif()

if(OCTOTIGER_WITH_RADIATION)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_RADIATION)
endif()

# Grid size
message(STATUS "Octotiger grid size: ${OCTOTIGER_WITH_GRIDDIM}")
target_compile_definitions(octotiger PRIVATE OCTOTIGER_GRIDDIM=${OCTOTIGER_WITH_GRIDDIM})

# Tests
if(OCTOTIGER_WITH_BLAST_TEST)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_BLAST_TEST)
  set(blast_sources
    src/test_problems/blast/sedov.cpp
    src/test_problems/blast/sedovf_c.cpp)
  target_sources(octotiger PRIVATE ${blast_sources})
  source_group(TREE ${PROJECT_SOURCE_DIR}
    PREFIX "Source Files" FILES ${blast_sources})

  if(OCTOTIGER_WITH_BOOST_MULTIPRECISION)
    target_compile_definitions(octotiger PRIVATE
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
    target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_QUADMATH)
    target_link_libraries(octotiger quadmath)

  else()
      message(FATAL_ERROR "Octotiger with OCTOTIGER_WITH_BLAST_TEST=ON cannot "
        "be built without quadmath or Boost.Multiprecision")
  endif()
endif()

################################################################################
# Set up silo_compare target
################################################################################
add_executable(silo_compare src/compare/compare.cpp)
target_link_libraries(silo_compare Silo::silo)
set_property(TARGET silo_compare PROPERTY FOLDER "Tools")
if(MSVC)
  target_compile_options(silo_compare PRIVATE /EHsc)
endif()

################################################################################
# Set up gen_rotating_star_init target
################################################################################
add_executable(gen_rotating_star_init src/test_problems/rotating_star/make_bin.cpp)
set_property(TARGET gen_rotating_star_init PROPERTY FOLDER "Tools")
if(MSVC)
  target_compile_definitions(gen_rotating_star_init PRIVATE
    _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(gen_rotating_star_init PRIVATE
    -Wno-ignored-attributes -Wno-attributes -Wno-deprecated-declarations
    -Wno-unused-result -ffast-math -O3)
  target_link_libraries(gen_rotating_star_init Threads::Threads)
endif()

################################################################################
# Set up silo_counter target
################################################################################
add_executable(silo_counter src/silo_counter.cpp)

target_link_libraries(silo_counter Silo::silo)

set_property(TARGET silo_counter PROPERTY FOLDER "Tools")
if(MSVC)
  target_compile_options(silo_counter PRIVATE /EHsc)
endif()

################################################################################
# Tests
################################################################################
if(OCTOTIGER_WITH_TESTS)
  enable_testing()

  # *.diff tests fail when the output contains any .vals
  set(OCTOTIGER_SILODIFF_FAIL_PATTERN ".vals")

  # Marshak
  file(DOWNLOAD phys.lsu.edu/~dmarcel/marshak.2.silo ${PROJECT_BINARY_DIR}/marshak.silo)
  add_test(NAME test_problems.marshak
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/marshak/marshak.ini)
  add_test(NAME test_problems.marshak.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/marshak.silo ${PROJECT_BINARY_DIR}/final.${OCTOTIGER_WITH_GRIDDIM}.silo)

  set_tests_properties(test_problems.marshak PROPERTIES
    FIXTURES_SETUP test_problems.marshak)
  set_tests_properties(test_problems.marshak.diff PROPERTIES
    FIXTURES_REQUIRED test_problems.marshak
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

  # Blast
  if(OCTOTIGER_WITH_BLAST_TEST)
    file(DOWNLOAD phys.lsu.edu/~dmarcel/blast.4.silo ${PROJECT_BINARY_DIR}/blast.silo)
    add_test(NAME test_problems.blast
      COMMAND octotiger
        --config_file=${PROJECT_SOURCE_DIR}/test_problems/blast/blast.ini)
    add_test(NAME test_problems.blast.diff
      COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
        ${PROJECT_BINARY_DIR}/blast.silo ${PROJECT_BINARY_DIR}/final.${OCTOTIGER_WITH_GRIDDIM}.silo)

    set_tests_properties(test_problems.blast PROPERTIES
      FIXTURES_SETUP test_problems.blast)
    set_tests_properties(test_problems.blast.diff PROPERTIES
      FIXTURES_REQUIRED test_problems.blast
      FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
  endif()

  # Sod Shock Tube
  file(DOWNLOAD phys.lsu.edu/~dmarcel/sod.4.silo ${PROJECT_BINARY_DIR}/sod.silo)
  add_test(NAME test_problems.sod
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/sod/sod.ini)

  add_test(NAME test_problems.sod.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/sod.silo ${PROJECT_BINARY_DIR}/final.${OCTOTIGER_WITH_GRIDDIM}.silo)

  set_tests_properties(test_problems.sod PROPERTIES
    FIXTURES_SETUP test_problems.sod)
  set_tests_properties(test_problems.sod.diff PROPERTIES
    FIXTURES_REQUIRED test_problems.sod
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

  # Sphere
  file(DOWNLOAD phys.lsu.edu/~dmarcel/sphere.2.silo ${PROJECT_BINARY_DIR}/sphere.silo)
  add_test(NAME test_problems.sphere
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/sphere/sphere.ini)
  add_test(NAME test_problems.sphere.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/sphere.silo ${PROJECT_BINARY_DIR}/X.${OCTOTIGER_WITH_GRIDDIM}.silo)

  set_tests_properties(test_problems.sphere PROPERTIES
    FIXTURES_SETUP test_problems.sphere)
  set_tests_properties(test_problems.sphere.diff PROPERTIES
    FIXTURES_REQUIRED test_problems.sphere
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

  # Rotating Star
  file(DOWNLOAD phys.lsu.edu/~dmarcel/rotating_star.2.silo
    ${PROJECT_BINARY_DIR}/rotating_star.silo)
  add_test(NAME test_problems.rotating_star.init COMMAND gen_rotating_star_init)
  add_test(NAME test_problems.rotating_star
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini)
  add_test(NAME test_problems.rotating_star.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/final.${OCTOTIGER_WITH_GRIDDIM}.silo)

  set_tests_properties(test_problems.rotating_star.init PROPERTIES
    FIXTURES_SETUP test_problems.rotating_star.init)
  set_tests_properties(test_problems.rotating_star PROPERTIES
    FIXTURES_REQUIRED test_problems.rotating_star.init)

  set_tests_properties(test_problems.rotating_star PROPERTIES
    FIXTURES_SETUP test_problems.rotating_star
    FIXTURES_REQUIRED test_problems.rotating_star.init)
  set_tests_properties(test_problems.rotating_star.diff PROPERTIES
    FIXTURES_REQUIRED "test_problems.rotating_star.init;test_problems.rotating_star"
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

endif()
