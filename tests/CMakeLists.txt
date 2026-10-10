cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(OctoTigerRadiationTests LANGUAGES CXX)
  enable_testing()
endif()

# The numerical suites compile without HPX, Silo, or the application build.
add_executable(radiationM1Tests radiationM1.cpp)
target_include_directories(radiationM1Tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(radiationM1Tests PRIVATE cxx_std_20)
add_test(NAME radiationM1 COMMAND radiationM1Tests)

add_executable(fpeGuardTests fpeGuard.cpp)
target_include_directories(fpeGuardTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(fpeGuardTests PRIVATE cxx_std_20)
add_test(NAME fpeGuard COMMAND fpeGuardTests)
set_tests_properties(radiationM1 fpeGuard PROPERTIES LABELS radiation TIMEOUT 120)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
foreach(_suite IN ITEMS Grid ConservationDriver)
  add_test(NAME radiation.production.${_suite}
    COMMAND "${CMAKE_COMMAND}" -E env "CXX=${CMAKE_CXX_COMPILER}"
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/validateRadiation${_suite}.py")
  set_tests_properties(radiation.production.${_suite} PROPERTIES LABELS radiation TIMEOUT 300 PROCESSORS 1 RUN_SERIAL TRUE)
endforeach()
add_test(NAME radiation.production.conservation
  COMMAND "${CMAKE_COMMAND}" -E env "CXX=${CMAKE_CXX_COMPILER}"
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../radiation_results/tests/test_conservation_solver.py")
set_tests_properties(radiation.production.conservation PROPERTIES LABELS radiation TIMEOUT 180 PROCESSORS 1 RUN_SERIAL TRUE)

add_subdirectory(radiation)

# These checks run the actual HPX application and retain their individual logs.
# Original error thresholds were calibrated at 32 cells per side, c=1 and t=.2.
if(TARGET octotiger AND OCTOTIGER_WITH_RADIATION)
  add_test(NAME radiation.application.restart
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/radiation/test_restart.py"
      --executable $<TARGET_FILE:octotiger> --silo-library "${Silo_LIBRARY}" --threads 2)
  set_tests_properties(radiation.application.restart PROPERTIES LABELS "radiation;application"
    PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 600)
  if(OCTOTIGER_WITH_GRIDDIM EQUAL 8)
    foreach(_case IN ITEMS streaming_wave streaming_front gaussian_pulse equilibrium_sphere)
      set(_reference_args)
      if(_case STREQUAL "gaussian_pulse")
        list(APPEND _reference_args --reference "${CMAKE_CURRENT_BINARY_DIR}/radiation/unit-reference.bin")
      endif()
      add_test(NAME radiation.application.${_case}
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../test_problems/radiation/check.py"
          run --case "${_case}" --directory "${CMAKE_CURRENT_BINARY_DIR}/application/${_case}"
          --exe $<TARGET_FILE:octotiger>
          --config "${CMAKE_CURRENT_SOURCE_DIR}/../radiation_results/configs/${_case}.ini"
          --level 2 --time .2 ${_reference_args})
      set_tests_properties(radiation.application.${_case} PROPERTIES LABELS "radiation;application"
        PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 900)
      if(_case STREQUAL "gaussian_pulse")
        set_tests_properties(radiation.application.${_case} PROPERTIES FIXTURES_REQUIRED radiation.unit.reference)
      endif()
    endforeach()
  else()
    message(STATUS "32-cell radiation application regressions require OCTOTIGER_WITH_GRIDDIM=8")
  endif()
endif()
