# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Rotating Star
##############################################################################
download_test_reference("Rotating star test"
  "phys.lsu.edu/~dmarcel/rotating_star.13.silo"
  ${PROJECT_BINARY_DIR}/rotating_star.silo)



# Rotating Star - CPU
add_test(NAME test_problems.cpu.rotating_star.init COMMAND gen_rotating_star_init)
add_test(NAME test_problems.cpu.rotating_star
  COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini --multipole_kernel_type=SOA_CPU --p2p_kernel_type=SOA_CPU --p2m_kernel_type=SOA_CPU > rotating_star_log.txt")
set_tests_properties(test_problems.cpu.rotating_star PROPERTIES
  FIXTURES_SETUP test_problems.cpu.rotating_star
  FIXTURES_REQUIRED test_problems.cpu.rotating_star.init)
set_tests_properties(test_problems.cpu.rotating_star.init PROPERTIES
  FIXTURES_SETUP test_problems.cpu.rotating_star.init)
set_tests_properties(test_problems.cpu.rotating_star PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star.init)

add_test(NAME test_problems.cpu.rotating_star.rho_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.egas_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.tau_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.pot_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.sx_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.sy_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.sz_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.zx_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.zy_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.zz_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc1_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc2_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc3_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc4_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc5_regex COMMAND cat rotating_star_log.txt)
set_tests_properties(test_problems.cpu.rotating_star.rho_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "rho 3.178289e-05 2.006156e-04")
set_tests_properties(test_problems.cpu.rotating_star.egas_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "egas 5.813054e-06 4.747507e-05")
set_tests_properties(test_problems.cpu.rotating_star.tau_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "tau 1.377203e-05 8.266311e-05")
set_tests_properties(test_problems.cpu.rotating_star.pot_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "pot 5.264186e-03 3.940757e-02")
set_tests_properties(test_problems.cpu.rotating_star.sx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "sx 7.353904e-06 6.084916e-05")
set_tests_properties(test_problems.cpu.rotating_star.sy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "sy 7.354391e-06 6.085113e-05")
set_tests_properties(test_problems.cpu.rotating_star.sz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "sz 5.123306e-06 6.002162e-05")
set_tests_properties(test_problems.cpu.rotating_star.zx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "zx 1.460722e-04 8.854924e-04")
set_tests_properties(test_problems.cpu.rotating_star.zy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "zy 1.460722e-04 8.854922e-04")
set_tests_properties(test_problems.cpu.rotating_star.zz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "zz 5.833309e-04 3.077133e-03")
set_tests_properties(test_problems.cpu.rotating_star.spc1_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "spc_1 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.rotating_star.spc2_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "spc_2 3.178289e-05 2.006156e-04")
set_tests_properties(test_problems.cpu.rotating_star.spc3_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "spc_3 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.rotating_star.spc4_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "spc_4 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.rotating_star.spc5_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION "spc_5 0.000000e.00 0.000000e.00")
   
add_test(NAME test_problems.cpu.rotating_star.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo)
set_tests_properties(test_problems.cpu.rotating_star.diff PROPERTIES
  FIXTURES_REQUIRED "test_problems.cpu.rotating_star.init;test_problems.cpu.rotating_star"
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

add_test(test_problems.cpu.rotating_star.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo rotating_star_log.txt)
set_tests_properties(test_problems.cpu.rotating_star.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.cpu.rotating_star
)

# Rotating Star - OLD CPU
add_test(NAME test_problems.cpu.rotating_star_old
  COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini --multipole_kernel_type=OLD --p2p_kernel_type=OLD --p2m_kernel_type=OLD > rotating_star_old_log.txt")
set_tests_properties(test_problems.cpu.rotating_star_old PROPERTIES
  FIXTURES_SETUP test_problems.cpu.rotating_star_old
  FIXTURES_REQUIRED test_problems.cpu.rotating_star.init)
add_test(NAME test_problems.cpu.rotating_star_old.rho_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.egas_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.tau_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.pot_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.sx_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.sy_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.sz_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.zx_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.zy_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.zz_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc1_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc2_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc3_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc4_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc5_regex COMMAND cat rotating_star_old_log.txt)
set_tests_properties(test_problems.cpu.rotating_star_old.rho_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "rho 3.178289e-05 2.006156e-04")
set_tests_properties(test_problems.cpu.rotating_star_old.egas_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "egas 5.813054e-06 4.747507e-05")
set_tests_properties(test_problems.cpu.rotating_star_old.tau_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "tau 1.377203e-05 8.266311e-05")
set_tests_properties(test_problems.cpu.rotating_star_old.pot_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "pot 5.264186e-03 3.940757e-02")
set_tests_properties(test_problems.cpu.rotating_star_old.sx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "sx 7.353904e-06 6.084916e-05")
set_tests_properties(test_problems.cpu.rotating_star_old.sy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "sy 7.354391e-06 6.085113e-05")
set_tests_properties(test_problems.cpu.rotating_star_old.sz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "sz 5.123306e-06 6.002162e-05")
set_tests_properties(test_problems.cpu.rotating_star_old.zx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "zx 1.460722e-04 8.854924e-04")
set_tests_properties(test_problems.cpu.rotating_star_old.zy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "zy 1.460722e-04 8.854922e-04")
set_tests_properties(test_problems.cpu.rotating_star_old.zz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "zz 5.833309e-04 3.077133e-03")
set_tests_properties(test_problems.cpu.rotating_star_old.spc1_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "spc_1 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.rotating_star_old.spc2_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "spc_2 3.178289e-05 2.006156e-04")
set_tests_properties(test_problems.cpu.rotating_star_old.spc3_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "spc_3 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.rotating_star_old.spc4_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "spc_4 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.rotating_star_old.spc5_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION "spc_5 0.000000e.00 0.000000e.00")

add_test(NAME test_problems.cpu.rotating_star_old.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo)
set_tests_properties(test_problems.cpu.rotating_star_old.diff PROPERTIES
  FIXTURES_REQUIRED "test_problems.cpu.rotating_star.init;test_problems.cpu.rotating_star_old"
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

add_test(test_problems.cpu.rotating_star_old.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/rotating_star_old/final.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star_old/final.silo.data/0.silo rotating_star_old_log.txt)
set_tests_properties(test_problems.cpu.rotating_star_old.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.cpu.rotating_star_old
)

# Rotating Star - GPU
if(OCTOTIGER_WITH_CUDA)
  add_test(NAME test_problems.gpu.rotating_star.init COMMAND gen_rotating_star_init)
  add_test(NAME test_problems.gpu.rotating_star
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini
      --cuda_number_gpus=1 --cuda_streams_per_gpu=8)
  add_test(NAME test_problems.gpu.rotating_star.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/final.silo)

  set_tests_properties(test_problems.gpu.rotating_star.init PROPERTIES
    FIXTURES_SETUP test_problems.gpu.rotating_star.init)
  set_tests_properties(test_problems.gpu.rotating_star PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star.init)

  set_tests_properties(test_problems.gpu.rotating_star PROPERTIES
    FIXTURES_SETUP test_problems.gpu.rotating_star
    FIXTURES_REQUIRED test_problems.gpu.rotating_star.init)
  set_tests_properties(test_problems.gpu.rotating_star.diff PROPERTIES
    FIXTURES_REQUIRED "test_problems.gpu.rotating_star.init;test_problems.gpu.rotating_star"
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
endif()
