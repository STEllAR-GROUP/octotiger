cmake_minimum_required(VERSION 3.20)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(OctoTigerRadiationChecks LANGUAGES CXX)
  enable_testing()
  add_executable(radiationM1Tests test_m1.cpp)
  target_compile_features(radiationM1Tests PRIVATE cxx_std_20)
  target_include_directories(radiationM1Tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../..")
  add_test(NAME radiationM1 COMMAND radiationM1Tests)
endif()
get_filename_component(radiation_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
include("${radiation_root}/radiation_results/reference.cmake")
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(test_radiation_hydro_benchmarks test_hydro_benchmarks.cpp)
target_compile_features(test_radiation_hydro_benchmarks PRIVATE cxx_std_20)
target_include_directories(test_radiation_hydro_benchmarks PRIVATE "${radiation_root}")
add_test(NAME radiation.unit.hydro_benchmarks COMMAND test_radiation_hydro_benchmarks)
set_tests_properties(radiation.unit.hydro_benchmarks PROPERTIES
  LABELS "radiation;radiation-hydro" TIMEOUT 60)
add_test(NAME radiation.production.hydro_boundaries
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_hydro_boundaries.py"
    --cxx "${CMAKE_CXX_COMPILER}" --sanitize)
set_tests_properties(radiation.production.hydro_boundaries PROPERTIES
  LABELS "radiation;radiation-hydro" TIMEOUT 120 PROCESSORS 1 RUN_SERIAL TRUE)

add_executable(test_radiation_reference test_reference.cpp)
target_compile_features(test_radiation_reference PRIVATE cxx_std_20)
target_include_directories(test_radiation_reference PRIVATE "${radiation_root}")
foreach(_reference IN ITEMS unit units)
  set(_parameters)
  if(_reference STREQUAL "units")
    list(APPEND _parameters --c 7 --time .04)
  endif()
  add_test(NAME radiation.unit.reference.${_reference}.generate COMMAND gen_radiation_reference
    --output "${CMAKE_CURRENT_BINARY_DIR}/${_reference}-reference.bin" --cells 32 ${_parameters})
  add_test(NAME radiation.unit.reference.${_reference}.check COMMAND test_radiation_reference
    "${CMAKE_CURRENT_BINARY_DIR}/${_reference}-reference.bin")
  if(_reference STREQUAL "unit")
    set(_fixture radiation.unit.reference)
  else()
    set(_fixture radiation.unit.reference.units)
  endif()
  set_tests_properties(radiation.unit.reference.${_reference}.generate PROPERTIES
    FIXTURES_SETUP "${_fixture}" LABELS radiation TIMEOUT 60)
  set_tests_properties(radiation.unit.reference.${_reference}.check PROPERTIES
    FIXTURES_REQUIRED "${_fixture}" LABELS radiation TIMEOUT 60)
endforeach()
add_test(NAME radiation.unit.comparator
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_comparator.py")
set_tests_properties(radiation.unit.comparator PROPERTIES LABELS radiation TIMEOUT 30)
add_test(NAME radiation.production.regressions
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_regression_kernels.py"
    --reference "${CMAKE_CURRENT_BINARY_DIR}/unit-reference.bin" --cxx "${CMAKE_CXX_COMPILER}")
set_tests_properties(radiation.production.regressions PROPERTIES FIXTURES_REQUIRED radiation.unit.reference
  LABELS radiation TIMEOUT 300 PROCESSORS 1 RUN_SERIAL TRUE)
add_test(NAME radiation.production.rsla_regressions
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_regression_kernels.py"
    --reference "${CMAKE_CURRENT_BINARY_DIR}/unit-reference.bin" --cxx "${CMAKE_CXX_COMPILER}"
    --radiation-reduced-c-factor 0.1)
set_tests_properties(radiation.production.rsla_regressions PROPERTIES FIXTURES_REQUIRED radiation.unit.reference
  LABELS "radiation;rsla" TIMEOUT 300 PROCESSORS 1 RUN_SERIAL TRUE)
