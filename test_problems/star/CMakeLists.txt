# Copyright (c) 2022 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Star
##############################################################################

# Reference output to compare againsts:
# Collected with scenario:
#./build/octotiger/build/octotiger -t4 --config_file=src/octotiger/test_problems/star/star.ini --monopole_host_kernel_type=LEGACY --multipole_host_kernel_type=LEGACY --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY --amr_boundary_kernel_type=AMR_LEGACY
# Adapt in case of physic changes (note, replace special symbols like + with .)

function(test_star_scenario test_name test_log_file kernel_parameters)
  # Run test
  add_test(NAME ${test_name}
    COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/star/star.ini ${kernel_parameters} | tee ${test_log_file}")
  set_tests_properties(${test_name} PROPERTIES
    FIXTURES_SETUP ${test_name})

  # Analyze test log file: Timestep output needs to match
  add_test(NAME ${test_name}.TS11_dt_regex COMMAND cat ${test_log_file})
  set_tests_properties(${test_name}.TS11_dt_regex PROPERTIES
  FIXTURES_REQUIRED ${test_name}
  PASS_REGULAR_EXPRESSION "TS 1.:: t: 7.352407e.00, dt: 6.715325e-01")

  add_test(NAME ${test_name}.TS11_a_regex COMMAND cat ${test_log_file})
  set_tests_properties(${test_name}.TS11_a_regex PROPERTIES
  FIXTURES_REQUIRED ${test_name}
  PASS_REGULAR_EXPRESSION "a: 9.307070e-01")

  add_test(NAME ${test_name}.TS11_ur_regex COMMAND cat ${test_log_file})
  set_tests_properties(${test_name}.TS11_ur_regex PROPERTIES
  FIXTURES_REQUIRED ${test_name}
  PASS_REGULAR_EXPRESSION "ur: 1.680614e-02")

  add_test(NAME ${test_name}.TS11_ul_regex COMMAND cat ${test_log_file})
  set_tests_properties(${test_name}.TS11_ul_regex PROPERTIES
  FIXTURES_REQUIRED ${test_name}
  PASS_REGULAR_EXPRESSION "ul: 1.365091e-02")

  add_test(NAME ${test_name}.TS11_vr_regex COMMAND cat ${test_log_file})
  set_tests_properties(${test_name}.TS11_vr_regex PROPERTIES
  FIXTURES_REQUIRED ${test_name}
  PASS_REGULAR_EXPRESSION "vr: 6.075061e-02")

  add_test(NAME ${test_name}.TS11_vl_regex COMMAND cat ${test_log_file})
  set_tests_properties(${test_name}.TS11_vl_regex PROPERTIES
  FIXTURES_REQUIRED ${test_name}
  PASS_REGULAR_EXPRESSION "vl: 2.912973e-02")

  add_test(NAME ${test_name}.TS11_boundaries_regex COMMAND cat ${test_log_file})
  set_tests_properties(${test_name}.TS11_boundaries_regex PROPERTIES
  FIXTURES_REQUIRED ${test_name}
  PASS_REGULAR_EXPRESSION "amr_boundaries: 312")

  # Analyze test log file: silodiff 
  add_test(NAME ${test_name}.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-11
     ${PROJECT_SOURCE_DIR}/octotiger-testdata/star_with_eos_ipr.silo ${PROJECT_BINARY_DIR}/test_problems/star/final.silo.data/0.silo)
  set_tests_properties(${test_name}.diff PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    FAIL_REGULAR_EXPRESSION "(egas|gx|gy|gz|pot|rho_1|sx|sy|sy|tau).vals")

  # Cleanup
  add_test(${test_name}.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/star/final.silo ${PROJECT_BINARY_DIR}/test_problems/star/final.silo.data/0.silo ${test_log_file})
  set_tests_properties(${test_name}.fixture_cleanup PROPERTIES
      FIXTURES_CLEANUP ${test_name}
  )
endfunction()

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  # EOS ipr star - Basic CPU test
  test_star_scenario(test_problems.cpu.star.eos_ipr.legacy star_eos_ipr_legacy.txt "  --monopole_host_kernel_type=LEGACY --multipole_host_kernel_type=LEGACY --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY --amr_boundary_kernel_type=AMR_LEGACY")

# TODO Add CUDA/HIP/KOKKOS tests as the kernels get ported...
endif()


