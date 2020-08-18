# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Sod Shock Tube
##############################################################################
download_test_reference("Sod shock tube test"
  "phys.lsu.edu/~dmarcel/sod.13.silo"
  ${PROJECT_BINARY_DIR}/sod.silo)

# Sod Shock Tube - CPU
add_test(NAME test_problems.cpu.sod
  COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/sod/sod.ini > sod_log.txt")
set_tests_properties(test_problems.cpu.sod PROPERTIES
  FIXTURES_SETUP test_problems.cpu.sod)

add_test(NAME test_problems.cpu.sod.rho_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.egas_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.tau_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.pot_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.sx_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.sy_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.sz_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.zx_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.zy_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.zz_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.spc1_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.spc2_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.spc3_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.spc4_regex COMMAND cat sod_log.txt)
add_test(NAME test_problems.cpu.sod.spc5_regex COMMAND cat sod_log.txt)
set_tests_properties(test_problems.cpu.sod.rho_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "rho 8.616107e-03 1.774427e-02")
set_tests_properties(test_problems.cpu.sod.egas_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "egas 2.234463e-02 4.586487e-02")
set_tests_properties(test_problems.cpu.sod.tau_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "tau 1.460285e-02 3.493483e-02")
set_tests_properties(test_problems.cpu.sod.pot_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "pot 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sod.sx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "sx 8.155587e-03 1.650219e-02")
#sy sz zx are too close to 0 to be reasonably compared here as strings
set_tests_properties(test_problems.cpu.sod.zy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "zy 2.250000e-02 5.975532e-02")
set_tests_properties(test_problems.cpu.sod.zz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "zz 2.250000e-02 5.975532e-02")
set_tests_properties(test_problems.cpu.sod.spc1_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "spc_1 8.616107e-03 1.774427e-02")
set_tests_properties(test_problems.cpu.sod.spc2_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "spc_2 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sod.spc3_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "spc_3 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sod.spc4_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "spc_4 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sod.spc5_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  PASS_REGULAR_EXPRESSION "spc_5 0.000000e.00 0.000000e.00")

add_test(NAME test_problems.cpu.sod.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/sod.silo ${PROJECT_BINARY_DIR}/test_problems/sod/final.silo.data/0.silo)
set_tests_properties(test_problems.cpu.sod.diff PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

add_test(test_problems.cpu.sod.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/sod/final.silo ${PROJECT_BINARY_DIR}/test_problems/sod/final.silo.data/0.silo sod_log.txt)
set_tests_properties(test_problems.cpu.sod.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.cpu.sod
)

# Sod Shock Tube - GPU
if(OCTOTIGER_WITH_CUDA)
  add_test(NAME test_problems.gpu.sod
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/sod/sod.ini
      --cuda_number_gpus=1 --cuda_streams_per_gpu=8)
  add_test(NAME test_problems.gpu.sod.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/sod.silo ${PROJECT_BINARY_DIR}/final.silo)

  set_tests_properties(test_problems.gpu.sod PROPERTIES
    FIXTURES_SETUP test_problems.gpu.sod)
  set_tests_properties(test_problems.gpu.sod.diff PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.sod
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
endif()
