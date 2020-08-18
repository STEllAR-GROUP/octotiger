# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Sphere
##############################################################################
download_test_reference("Solid sphere test"
  "phys.lsu.edu/~dmarcel/sphere.13.silo"
  ${PROJECT_BINARY_DIR}/sphere.silo)

# Sphere - CPU
add_test(NAME test_problems.cpu.sphere
  COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/sphere/sphere.ini --multipole_kernel_type=SOA_CPU --p2p_kernel_type=SOA_CPU --p2m_kernel_type=SOA_CPU > sphere_log.txt")
set_tests_properties(test_problems.cpu.sphere PROPERTIES
  FIXTURES_SETUP test_problems.cpu.sphere)

add_test(NAME test_problems.cpu.sphere.rho_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.egas_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.tau_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.pot_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.sx_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.sy_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.sz_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.zx_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.zy_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.zz_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.spc1_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.spc2_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.spc3_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.spc4_regex COMMAND cat sphere_log.txt)
add_test(NAME test_problems.cpu.sphere.spc5_regex COMMAND cat sphere_log.txt)
set_tests_properties(test_problems.cpu.sphere.rho_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "rho 1.048130e-03 1.989298e-02")
set_tests_properties(test_problems.cpu.sphere.egas_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "egas 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.tau_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "tau 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.pot_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "pot 4.470774e-01 3.176879e.00")
set_tests_properties(test_problems.cpu.sphere.sx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "sx 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.sy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "sy 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.sz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "sz 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.zx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "zx 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.zy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "zy 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.zz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "zz 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.spc1_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "spc_1 1.048130e-03 1.989298e-02")
set_tests_properties(test_problems.cpu.sphere.spc2_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "spc_2 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.spc3_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "spc_3 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.spc4_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "spc_4 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere.spc5_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  PASS_REGULAR_EXPRESSION "spc_5 0.000000e.00 0.000000e.00")


add_test(NAME test_problems.cpu.sphere.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/sphere.silo ${PROJECT_BINARY_DIR}/test_problems/sphere/X.${OCTOTIGER_WITH_GRIDDIM}.silo.data/0.silo)
set_tests_properties(test_problems.cpu.sphere.diff PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

add_test(test_problems.cpu.sphere.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/sphere/final.silo ${PROJECT_BINARY_DIR}/test_problems/sphere/final.silo.data/0.silo sphere_log.txt)
set_tests_properties(test_problems.cpu.sphere.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.cpu.sphere
)



# Sphere - OLD CPU
add_test(NAME test_problems.cpu.sphere_old
  COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/sphere/sphere.ini --multipole_kernel_type=OLD --p2p_kernel_type=OLD --p2m_kernel_type=OLD > sphere_old_log.txt")
set_tests_properties(test_problems.cpu.sphere_old PROPERTIES
  FIXTURES_SETUP test_problems.cpu.sphere_old)

add_test(NAME test_problems.cpu.sphere_old.rho_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.egas_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.tau_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.pot_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.sx_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.sy_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.sz_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.zx_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.zy_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.zz_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.spc1_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.spc2_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.spc3_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.spc4_regex COMMAND cat sphere_old_log.txt)
add_test(NAME test_problems.cpu.sphere_old.spc5_regex COMMAND cat sphere_old_log.txt)
set_tests_properties(test_problems.cpu.sphere_old.rho_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "rho 1.048130e-03 1.989298e-02")
set_tests_properties(test_problems.cpu.sphere_old.egas_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "egas 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.tau_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "tau 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.pot_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "pot 4.470774e-01 3.176879e.00")
set_tests_properties(test_problems.cpu.sphere_old.sx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "sx 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.sy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "sy 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.sz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "sz 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.zx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "zx 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.zy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "zy 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.zz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "zz 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.spc1_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "spc_1 1.048130e-03 1.989298e-02")
set_tests_properties(test_problems.cpu.sphere_old.spc2_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "spc_2 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.spc3_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "spc_3 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.spc4_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "spc_4 0.000000e.00 0.000000e.00")
set_tests_properties(test_problems.cpu.sphere_old.spc5_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  PASS_REGULAR_EXPRESSION "spc_5 0.000000e.00 0.000000e.00")


add_test(NAME test_problems.cpu.sphere_old.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/sphere_old.silo ${PROJECT_BINARY_DIR}/test_problems/sphere_old/X.${OCTOTIGER_WITH_GRIDDIM}.silo.data/0.silo)
set_tests_properties(test_problems.cpu.sphere_old.diff PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere_old
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

add_test(test_problems.cpu.sphere_old.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/sphere_old/final.silo ${PROJECT_BINARY_DIR}/test_problems/sphere_old/final.silo.data/0.silo sphere_old_log.txt)
set_tests_properties(test_problems.cpu.sphere_old.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.cpu.sphere_old
)


# Sphere - GPU
if(OCTOTIGER_WITH_CUDA)
  add_test(NAME test_problems.gpu.sphere
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/sphere/sphere.ini
      --cuda_number_gpus=1 --cuda_streams_per_gpu=8)
  add_test(NAME test_problems.gpu.sphere.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/sphere.silo ${PROJECT_BINARY_DIR}/X.${OCTOTIGER_WITH_GRIDDIM}.silo)

  set_tests_properties(test_problems.gpu.sphere PROPERTIES
    FIXTURES_SETUP test_problems.gpu.sphere)
  set_tests_properties(test_problems.gpu.sphere.diff PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.sphere
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
endif()
