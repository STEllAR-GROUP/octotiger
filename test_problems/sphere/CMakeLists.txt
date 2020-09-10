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
  COMMAND octotiger
    --config_file=${PROJECT_SOURCE_DIR}/test_problems/sphere/sphere.ini)

add_file_size_test(test_problems.cpu.sphere.filesize
  ${PROJECT_BINARY_DIR}/sphere.silo ${PROJECT_BINARY_DIR}/test_problems/sphere/X.${OCTOTIGER_WITH_GRIDDIM}.silo)

add_test(NAME test_problems.cpu.sphere.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/sphere.silo ${PROJECT_BINARY_DIR}/test_problems/sphere/X.${OCTOTIGER_WITH_GRIDDIM}.silo.data/0.silo)

set_tests_properties(test_problems.cpu.sphere PROPERTIES
  FIXTURES_SETUP test_problems.cpu.sphere)
set_tests_properties(test_problems.cpu.sphere.filesize PROPERTIES
  FIXTURES_SETUP test_problems.cpu.sphere)
set_tests_properties(test_problems.cpu.sphere.diff PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sphere
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

# Sphere - GPU
if(OCTOTIGER_WITH_CUDA)
  add_test(NAME test_problems.gpu.sphere
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/sphere/sphere.ini
      --cuda_streams_per_locality=1 --cuda_streams_per_gpu=1)
  add_test(NAME test_problems.gpu.sphere.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/sphere.silo ${PROJECT_BINARY_DIR}/X.${OCTOTIGER_WITH_GRIDDIM}.silo)

  set_tests_properties(test_problems.gpu.sphere PROPERTIES
    FIXTURES_SETUP test_problems.gpu.sphere)
  set_tests_properties(test_problems.gpu.sphere.diff PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.sphere
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
endif()
