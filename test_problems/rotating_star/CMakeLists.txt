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
  COMMAND octotiger
    --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini)

add_file_size_test(test_problems.cpu.rotating_star.filesize
  ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo)

add_test(NAME test_problems.cpu.rotating_star.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo)

set_tests_properties(test_problems.cpu.rotating_star.init PROPERTIES
  FIXTURES_SETUP test_problems.cpu.rotating_star.init)
set_tests_properties(test_problems.cpu.rotating_star PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star.init)

set_tests_properties(test_problems.cpu.rotating_star.filesize PROPERTIES
  FIXTURES_SETUP test_problems.cpu.rotating_star)
set_tests_properties(test_problems.cpu.rotating_star PROPERTIES
  FIXTURES_SETUP test_problems.cpu.rotating_star
  FIXTURES_REQUIRED test_problems.cpu.rotating_star.init)
set_tests_properties(test_problems.cpu.rotating_star.diff PROPERTIES
  FIXTURES_REQUIRED "test_problems.cpu.rotating_star.init;test_problems.cpu.rotating_star"
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

# Rotating Star - GPU
if(OCTOTIGER_WITH_CUDA)
  add_test(NAME test_problems.gpu.rotating_star.init COMMAND gen_rotating_star_init)
  add_test(NAME test_problems.gpu.rotating_star
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini
      --cuda_streams_per_locality=1 --cuda_streams_per_gpu=1)
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
