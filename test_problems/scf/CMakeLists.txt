# These numerical checks also build without HPX or reference test data.
cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(octotiger_scf_numerics LANGUAGES CXX)
  enable_testing()
endif()

get_filename_component(scf_source_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(scf_geometry_numerics test_geometry.cpp)
target_include_directories(scf_geometry_numerics PRIVATE "${scf_source_root}")
target_compile_features(scf_geometry_numerics PRIVATE cxx_std_17)
add_test(NAME scf_geometry_numerics COMMAND scf_geometry_numerics)
set_tests_properties(scf_geometry_numerics PROPERTIES
  PROCESSORS 1 TIMEOUT 120 LABELS scf)
