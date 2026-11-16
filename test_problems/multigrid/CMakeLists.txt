# This numerical fixture also builds without HPX or the application dependencies.
cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(octotiger_multigrid_numerics LANGUAGES CXX)
  enable_testing()
  set(OCTOTIGER_WITH_GRIDDIM 8 CACHE STRING "Interior cells per block axis")
endif()

add_compile_definitions(OCTOTIGER_GRIDDIM=${OCTOTIGER_WITH_GRIDDIM})

get_filename_component(multigrid_source_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(multigrid_numerics numerical.cpp
  "${multigrid_source_root}/src/multigrid/block.cpp")
target_include_directories(multigrid_numerics PRIVATE "${multigrid_source_root}")
target_compile_features(multigrid_numerics PRIVATE cxx_std_17)
add_test(NAME multigrid_numerics COMMAND multigrid_numerics)
set_tests_properties(multigrid_numerics PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS multigrid)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_isolated_boundary.cpp")
  add_executable(multigrid_isolated_boundary test_isolated_boundary.cpp)
  target_include_directories(multigrid_isolated_boundary PRIVATE "${multigrid_source_root}")
  target_compile_features(multigrid_isolated_boundary PRIVATE cxx_std_17)
  add_test(NAME multigrid_isolated_boundary COMMAND multigrid_isolated_boundary)
  set_tests_properties(multigrid_isolated_boundary PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS multigrid)
endif()
