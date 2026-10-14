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
foreach(_radiation_test IN ITEMS implicit_source ap_flux marshak_boundary plot_output)
  add_executable(test_radiation_${_radiation_test} test_${_radiation_test}.cpp)
  target_compile_features(test_radiation_${_radiation_test} PRIVATE cxx_std_20)
  target_include_directories(test_radiation_${_radiation_test} PRIVATE "${radiation_root}")
  add_test(NAME radiation.unit.${_radiation_test} COMMAND test_radiation_${_radiation_test})
  set_tests_properties(radiation.unit.${_radiation_test} PROPERTIES LABELS "radiation;unit" TIMEOUT 120)
endforeach()
add_executable(test_fld test_fld.cpp)
target_compile_features(test_fld PRIVATE cxx_std_20)
target_include_directories(test_fld PRIVATE "${radiation_root}")
add_test(NAME radiation_fld COMMAND test_fld)
set_tests_properties(radiation_fld PROPERTIES LABELS "radiation;fld" TIMEOUT 120)
add_executable(radiation_elliptic_test test_elliptic.cpp)
target_compile_features(radiation_elliptic_test PRIVATE cxx_std_20)
target_include_directories(radiation_elliptic_test PRIVATE "${radiation_root}")
add_test(NAME radiation_elliptic COMMAND radiation_elliptic_test)
set_tests_properties(radiation_elliptic PROPERTIES LABELS "radiation;elliptic" TIMEOUT 120)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  include("${radiation_root}/cmake/RadiationAccelerators.cmake")
  octotiger_add_radiation_accelerators()
endif()
add_executable(test_positive_quartic test_positive_quartic.cpp)
target_compile_features(test_positive_quartic PRIVATE cxx_std_20)
target_include_directories(test_positive_quartic PRIVATE "${radiation_root}")
add_test(NAME radiation.unit.positive_quartic COMMAND test_positive_quartic)
set_tests_properties(radiation.unit.positive_quartic PROPERTIES LABELS radiation TIMEOUT 120)
if(TARGET radiation_accelerators)
  add_executable(test_radiation_accelerator test_accelerator.cpp)
  target_link_libraries(test_radiation_accelerator PRIVATE radiation_accelerators)
  foreach(_backend IN ITEMS cpu cuda kokkos)
    string(TOUPPER "${_backend}" _upper_backend)
    if(_backend STREQUAL "cpu" OR OCTOTIGER_WITH_RADIATION_${_upper_backend})
      add_test(NAME radiation.accelerator.${_backend} COMMAND test_radiation_accelerator ${_backend})
      set_tests_properties(radiation.accelerator.${_backend} PROPERTIES
        LABELS "radiation;accelerator" TIMEOUT 120 RUN_SERIAL TRUE)
    endif()
  endforeach()
endif()
include("${radiation_root}/radiation_results/reference.cmake")
find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(TARGET octotiger AND OCTOTIGER_WITH_RADIATION AND
    (OCTOTIGER_WITH_RADIATION_CUDA OR OCTOTIGER_WITH_RADIATION_KOKKOS))
  set(_radiation_device_backends)
  if(OCTOTIGER_WITH_RADIATION_CUDA)
    list(APPEND _radiation_device_backends cuda)
  endif()
  if(OCTOTIGER_WITH_RADIATION_KOKKOS)
    list(APPEND _radiation_device_backends kokkos)
  endif()
  add_test(NAME radiation.accelerator.application
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_accelerator_application.py"
      --executable $<TARGET_FILE:octotiger> --silo-library "${Silo_LIBRARY}"
      --backends ${_radiation_device_backends})
  set_tests_properties(radiation.accelerator.application PROPERTIES
    LABELS "radiation;accelerator;application" TIMEOUT 1200 PROCESSORS 2 RUN_SERIAL TRUE)
endif()

if(TARGET octotiger AND OCTOTIGER_WITH_RADIATION)
  if(OCTOTIGER_WITH_GRIDDIM EQUAL 8)
    add_test(NAME radiation.fld.application
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_fld_application.py"
        --executable $<TARGET_FILE:octotiger> --silo-library "${Silo_LIBRARY}" --threads 2)
    set_tests_properties(radiation.fld.application PROPERTIES
      LABELS "radiation;fld;application" TIMEOUT 1200 PROCESSORS 2 RUN_SERIAL TRUE)
  endif()
  add_test(NAME radiation.euler.subcycling
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_euler_subcycling.py"
      --executable $<TARGET_FILE:octotiger> --silo-library "${Silo_LIBRARY}" --threads 2)
  set_tests_properties(radiation.euler.subcycling PROPERTIES
    LABELS "radiation;euler;application" TIMEOUT 1200 PROCESSORS 2 RUN_SERIAL TRUE)
  add_test(NAME radiation.euler.restart
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_restart.py"
      --executable $<TARGET_FILE:octotiger> --radiation-integrator euler --factor 0.1
      --silo-library "${Silo_LIBRARY}")
  add_test(NAME radiation.euler.amr
    COMMAND "${Python3_EXECUTABLE}" "${radiation_root}/tests/test_gradient_radiation.py"
      --executable $<TARGET_FILE:octotiger> --radiation-integrator euler --finest-level 3
      --threads 2 --timeout 240 --silo-library "${Silo_LIBRARY}")
  set(_euler_application_tests radiation.euler.restart radiation.euler.amr)
  foreach(_case IN ITEMS subcritical_shock supercritical_shock radiating_blast)
    add_test(NAME radiation.euler.${_case}
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_hydro_application.py"
        --executable $<TARGET_FILE:octotiger> --radiation-integrator euler --case "${_case}"
        --config "${radiation_root}/radiation_results/configs/${_case}.ini"
        --silo-library "${Silo_LIBRARY}")
    list(APPEND _euler_application_tests radiation.euler.${_case})
  endforeach()
  set_tests_properties(${_euler_application_tests} PROPERTIES
    LABELS "radiation;euler;application" TIMEOUT 1200 PROCESSORS 2 RUN_SERIAL TRUE)
endif()

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
