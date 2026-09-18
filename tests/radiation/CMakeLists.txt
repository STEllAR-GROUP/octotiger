cmake_minimum_required(VERSION 3.18)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(octotiger_radiation_checks LANGUAGES CXX)
  enable_testing()
  add_subdirectory(../../tools/radiation "${CMAKE_CURRENT_BINARY_DIR}/reference-tool")
endif()
get_filename_component(radiation_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(test_radiation_m1 test_m1.cpp)
target_compile_features(test_radiation_m1 PRIVATE cxx_std_20)
target_include_directories(test_radiation_m1 PRIVATE "${radiation_root}")
add_test(NAME radiation.unit.m1 COMMAND test_radiation_m1)
set_tests_properties(radiation.unit.m1 PROPERTIES LABELS radiation TIMEOUT 60)
add_executable(test_radiation_reference test_reference.cpp)
target_compile_features(test_radiation_reference PRIVATE cxx_std_20)
target_include_directories(test_radiation_reference PRIVATE "${radiation_root}")
add_test(NAME radiation.unit.reference.generate COMMAND gen_radiation_reference
  --output "${CMAKE_CURRENT_BINARY_DIR}/unit-reference.bin" --cells 32)
set_tests_properties(radiation.unit.reference.generate PROPERTIES
  FIXTURES_SETUP radiation.unit.reference LABELS radiation TIMEOUT 60)
add_test(NAME radiation.unit.reference.check COMMAND test_radiation_reference
  "${CMAKE_CURRENT_BINARY_DIR}/unit-reference.bin")
set_tests_properties(radiation.unit.reference.check PROPERTIES
  FIXTURES_REQUIRED radiation.unit.reference LABELS radiation TIMEOUT 60)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME radiation.unit.comparator COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_comparator.py")
set_tests_properties(radiation.unit.comparator PROPERTIES LABELS radiation TIMEOUT 30)

add_test(NAME radiation.unit.reference.physical_units.generate COMMAND gen_radiation_reference
  --output "${CMAKE_CURRENT_BINARY_DIR}/units-reference.bin" --cells 32 --c 7 --time .04)
set_tests_properties(radiation.unit.reference.physical_units.generate PROPERTIES
  FIXTURES_SETUP radiation.unit.reference.units LABELS radiation TIMEOUT 60)
add_test(NAME radiation.unit.reference.physical_units.check COMMAND test_radiation_reference
  "${CMAKE_CURRENT_BINARY_DIR}/units-reference.bin")
set_tests_properties(radiation.unit.reference.physical_units.check PROPERTIES
  FIXTURES_REQUIRED radiation.unit.reference.units LABELS radiation TIMEOUT 60)

add_test(NAME radiation.unit.grid COMMAND "${CMAKE_COMMAND}" -E env "CXX=${CMAKE_CXX_COMPILER}"
  "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_grid.py")
set_tests_properties(radiation.unit.grid PROPERTIES LABELS radiation TIMEOUT 180)
