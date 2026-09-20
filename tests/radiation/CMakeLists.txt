cmake_minimum_required(VERSION 3.20)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(octotiger_radiation_checks LANGUAGES CXX)
  enable_testing()
endif()
get_filename_component(radiation_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
# The reference generator moved with the results harness. Build only its
# standalone FFTW target here; the report application also needs Silo/OpenSSL.
if(NOT TARGET gen_radiation_reference)
  find_path(FFTW3_INCLUDE_DIR fftw3.h)
  find_library(FFTW3_LIBRARY NAMES fftw3)
  if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)
    message(FATAL_ERROR
      "The Gaussian reference tests require FFTW3. Install its development package "
      "or set FFTW3_INCLUDE_DIR and FFTW3_LIBRARY.")
  endif()
  add_executable(gen_radiation_reference
    "${radiation_root}/verification_results/radiation/cpp/gen_radiation_reference.cpp")
  target_compile_features(gen_radiation_reference PRIVATE cxx_std_23)
  target_include_directories(gen_radiation_reference PRIVATE
    "${radiation_root}" "${FFTW3_INCLUDE_DIR}")
  target_link_libraries(gen_radiation_reference PRIVATE "${FFTW3_LIBRARY}")
endif()
# Preserve the historical Hanawa/minmod/couple assertions in test_m1.cpp.
# They are not an equivalent test of the current conserved-variable VL solver.
# Keep an explicit failing legacy target so old build commands cannot silently
# run a different suite under the historical target name.
message(STATUS
  "Historical test_radiation_m1 is unsupported (removed Hanawa/minmod/couple API). "
  "Registering distinct radiation.unit.m1.current from tests/radiationM1.cpp; "
  "this is a successor suite, not a historical-equivalence pass.")
add_custom_target(test_radiation_m1
  COMMAND "${CMAKE_COMMAND}" -E echo
    "Historical test_m1.cpp requires removed Hanawa/minmod/couple APIs and is not supported. Build test_radiation_m1_current for the distinct current solver suite; historical equivalence is not claimed."
  COMMAND "${CMAKE_COMMAND}" -E false
  VERBATIM)
add_executable(test_radiation_m1_current "${radiation_root}/tests/radiationM1.cpp")
target_compile_features(test_radiation_m1_current PRIVATE cxx_std_20)
target_include_directories(test_radiation_m1_current PRIVATE "${radiation_root}")
add_test(NAME radiation.unit.m1.current COMMAND test_radiation_m1_current)
set_tests_properties(radiation.unit.m1.current PROPERTIES LABELS radiation TIMEOUT 60)
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
