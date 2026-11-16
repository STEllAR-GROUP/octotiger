# The numerical fixtures also build without HPX or application dependencies.
cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(octotiger_fld_numerics LANGUAGES CXX)
  enable_testing()
  set(OCTOTIGER_WITH_GRIDDIM 8 CACHE STRING "Interior cells per block axis")
endif()

add_compile_definitions(OCTOTIGER_GRIDDIM=${OCTOTIGER_WITH_GRIDDIM})

get_filename_component(fld_source_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

foreach(fixture opacity)
  add_executable(fld_${fixture}_numerics test_${fixture}.cpp)
  target_include_directories(fld_${fixture}_numerics PRIVATE "${fld_source_root}")
  target_compile_features(fld_${fixture}_numerics PRIVATE cxx_std_17)
  add_test(NAME fld_${fixture}_numerics COMMAND fld_${fixture}_numerics)
  set_tests_properties(fld_${fixture}_numerics PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS fld)
endforeach()
