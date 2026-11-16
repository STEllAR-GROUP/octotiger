# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.

cmake_minimum_required(VERSION 3.20)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(OctoTigerEllipticTests LANGUAGES CXX)
  enable_testing()
endif()

# The FLD linear solver can be checked without HPX, Silo, or a radiation runtime.
get_filename_component(octotiger_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(elliptic_composite_test test_composite.cpp)
target_compile_features(elliptic_composite_test PRIVATE cxx_std_20)
target_include_directories(elliptic_composite_test PRIVATE "${octotiger_root}")
# Preserve finite checks and the tiny-residual/constant-state regression cases.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|IntelLLVM")
  target_compile_options(elliptic_composite_test PRIVATE -fno-fast-math -ffp-contract=off)
elseif(MSVC)
  target_compile_options(elliptic_composite_test PRIVATE /fp:strict)
endif()
add_test(NAME test_problems.elliptic.composite COMMAND elliptic_composite_test)
set_tests_properties(test_problems.elliptic.composite PROPERTIES
  LABELS "elliptic;multigrid;unit" TIMEOUT 120 PROCESSORS 1)
