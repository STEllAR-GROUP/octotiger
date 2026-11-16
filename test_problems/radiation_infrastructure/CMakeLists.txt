# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
cmake_minimum_required(VERSION 3.20)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(OctoTigerRadiationInfrastructureTests LANGUAGES CXX)
  enable_testing()
endif()
get_filename_component(octotiger_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(radiation_infrastructure_test infrastructure.cpp)
target_compile_features(radiation_infrastructure_test PRIVATE cxx_std_20)
target_include_directories(radiation_infrastructure_test PRIVATE "${octotiger_root}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|IntelLLVM")
  target_compile_options(radiation_infrastructure_test PRIVATE -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME radiation.infrastructure COMMAND radiation_infrastructure_test)
set_tests_properties(radiation.infrastructure PROPERTIES PROCESSORS 1 TIMEOUT 60 LABELS "radiation;unit")
if(TARGET octolib)
  add_hpx_executable(radiation_grid_transfers_test DEPENDENCIES octolib SOURCES grid_transfers.cpp)
  separate_arguments(radiation_arch_options NATIVE_COMMAND "${OCTOTIGER_ARCH_FLAG}")
  target_compile_options(radiation_grid_transfers_test PRIVATE ${radiation_arch_options})
  add_test(NAME radiation.grid_transfers COMMAND radiation_grid_transfers_test --hpx:threads=1 --hpx:bind=none)
  set_tests_properties(radiation.grid_transfers PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS "radiation;unit")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tree_smoke")
  add_test(NAME radiation.tree_smoke COMMAND octotiger
    --config_file=${CMAKE_CURRENT_SOURCE_DIR}/tree_smoke.ini
    --hpx:threads=1 --hpx:bind=none)
  set_tests_properties(radiation.tree_smoke PROPERTIES PROCESSORS 1 TIMEOUT 120
    LABELS "radiation;integration"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tree_smoke"
    PASS_REGULAR_EXPRESSION "TS 3:: t: 3.000000e-12")
endif()
