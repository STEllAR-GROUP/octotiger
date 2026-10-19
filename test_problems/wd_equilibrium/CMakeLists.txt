# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.

# Exercise the same tabulated model used by the STAR + WD initializer.
add_executable(wd_equilibrium_model_test
  model.cpp
  ${PROJECT_SOURCE_DIR}/src/test_problems/wd_equilibrium.cpp)
target_include_directories(wd_equilibrium_model_test PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(wd_equilibrium_model_test PRIVATE cxx_std_17)
add_test(NAME test_problems.wd_equilibrium.model COMMAND wd_equilibrium_model_test)
set_tests_properties(test_problems.wd_equilibrium.model PROPERTIES
  PROCESSORS 1 TIMEOUT 30 LABELS "eos;wd_equilibrium")

# Block size changes the AMR hierarchy and SILO object inventory, even when
# the finest cell width agrees. Do not compare a grid16 run to a grid8 file.
if(NOT OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  message(STATUS "Skipping WD equilibrium application regression: reference requires grid8")
  return()
endif()

if(OCTOTIGER_WITH_CUDA OR OCTOTIGER_WITH_HIP OR
   (OCTOTIGER_WITH_KOKKOS AND
    (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)))
  message(STATUS "Skipping WD equilibrium application regression: reference uses a CPU-only build")
  return()
endif()

set(wd_equilibrium_reference
  "${OCTOTIGER_TESTDATA_DIR}/wd_equilibrium_1msun_grid8.silo")
if(NOT EXISTS "${wd_equilibrium_reference}")
  message(FATAL_ERROR
    "Missing WD equilibrium reference: ${wd_equilibrium_reference}. "
    "Update the testdata submodule or set OCTOTIGER_TESTDATA_DIR to the matching checkout.")
endif()

set(wd_equilibrium_run_directory "${CMAKE_CURRENT_BINARY_DIR}/run_grid8")
set(wd_equilibrium_runner "${CMAKE_CURRENT_BINARY_DIR}/run_equilibrium.cmake")

# A fresh directory prevents a stale final.silo from satisfying a failed run.
# Keep the latest outputs and logs available after CTest for inspection.
file(WRITE "${wd_equilibrium_runner}" [=[
if(NOT DEFINED OCTOTIGER_EXECUTABLE OR NOT DEFINED EQUILIBRIUM_CONFIG OR
   NOT DEFINED EQUILIBRIUM_RUN_DIRECTORY)
  message(FATAL_ERROR "WD equilibrium runner is missing required arguments")
endif()
file(REMOVE_RECURSE "${EQUILIBRIUM_RUN_DIRECTORY}")
file(MAKE_DIRECTORY "${EQUILIBRIUM_RUN_DIRECTORY}")
execute_process(
  COMMAND "${OCTOTIGER_EXECUTABLE}"
    --hpx:threads=2 --hpx:bind=none --hpx:localities=1
    "--config_file=${EQUILIBRIUM_CONFIG}"
  WORKING_DIRECTORY "${EQUILIBRIUM_RUN_DIRECTORY}"
  OUTPUT_FILE "${EQUILIBRIUM_RUN_DIRECTORY}/stdout.log"
  ERROR_FILE "${EQUILIBRIUM_RUN_DIRECTORY}/stderr.log"
  RESULT_VARIABLE equilibrium_result
  TIMEOUT 280)
if(NOT equilibrium_result STREQUAL "0")
  message(FATAL_ERROR
    "WD equilibrium simulation failed (${equilibrium_result}); see "
    "${EQUILIBRIUM_RUN_DIRECTORY}/stdout.log and stderr.log")
endif()
foreach(snapshot X.0.silo X.0.silo.data/0.silo final.silo final.silo.data/0.silo)
  if(NOT EXISTS "${EQUILIBRIUM_RUN_DIRECTORY}/${snapshot}")
    message(FATAL_ERROR "WD equilibrium simulation did not produce ${snapshot}")
  endif()
endforeach()
]=])

set(wd_equilibrium_test test_problems.cpu.wd_equilibrium.legacy)
add_test(NAME ${wd_equilibrium_test}
  COMMAND ${CMAKE_COMMAND}
    "-DOCTOTIGER_EXECUTABLE=$<TARGET_FILE:octotiger>"
    "-DEQUILIBRIUM_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/equilibrium.ini"
    "-DEQUILIBRIUM_RUN_DIRECTORY=${wd_equilibrium_run_directory}"
    -P "${wd_equilibrium_runner}")
set_tests_properties(${wd_equilibrium_test} PROPERTIES
  FIXTURES_SETUP ${wd_equilibrium_test}
  PROCESSORS 2 TIMEOUT 300 LABELS "eos;hydro;gravity;wd_equilibrium")

add_test(NAME ${wd_equilibrium_test}.diff
  COMMAND ${Silo_BROWSER} --proper-exit-code -e diff -q -x 1.0 -R 1.0e-12
    -E "idle_rate,locality,/_hdf5libinfo,/_silolibinfo"
    "${wd_equilibrium_reference}"
    "${wd_equilibrium_run_directory}/final.silo.data/0.silo")
set_tests_properties(${wd_equilibrium_test}.diff PROPERTIES
  FIXTURES_REQUIRED ${wd_equilibrium_test}
  FAIL_REGULAR_EXPRESSION "${OCTOTIGER_SILODIFF_FAIL_PATTERN}"
  PROCESSORS 1 TIMEOUT 120 LABELS "eos;hydro;gravity;wd_equilibrium")

# The SILO comparison remains available without Python. With NumPy, also check
# the physical mass, resolution, AMR hierarchy, and short-time equilibrium.
find_package(Python3 3.8 QUIET COMPONENTS Interpreter)
if(NOT Silo_LIBRARY MATCHES "[.](so([.][0-9]+)*|dylib|dll)$")
  message(STATUS "Shared Silo library unavailable; skipping WD equilibrium physical checks")
elseif(Python3_Interpreter_FOUND)
  execute_process(COMMAND ${CMAKE_COMMAND} -E env
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      "${Python3_EXECUTABLE}" -c "import numpy"
    RESULT_VARIABLE wd_equilibrium_numpy_status
    OUTPUT_QUIET ERROR_QUIET TIMEOUT 15)
  if(wd_equilibrium_numpy_status EQUAL 0)
    add_test(NAME ${wd_equilibrium_test}.physics
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/validate.py"
        --initial "${wd_equilibrium_run_directory}/X.0.silo"
        --final "${wd_equilibrium_run_directory}/final.silo"
        --output "${wd_equilibrium_run_directory}/metrics.json"
        --silo-library "${Silo_LIBRARY}")
    set_tests_properties(${wd_equilibrium_test}.physics PROPERTIES
      FIXTURES_REQUIRED ${wd_equilibrium_test}
      ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1;MKL_NUM_THREADS=1"
      PROCESSORS 1 TIMEOUT 120 LABELS "eos;hydro;gravity;wd_equilibrium")
  else()
    message(STATUS "NumPy unavailable; skipping WD equilibrium physical checks")
  endif()
else()
  message(STATUS "Python 3 unavailable; skipping WD equilibrium physical checks")
endif()
