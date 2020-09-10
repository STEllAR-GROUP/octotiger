# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Marshak wave test
##############################################################################
download_test_reference("Marshak wave test"
  "phys.lsu.edu/~dmarcel/marshak.13.silo"
  ${PROJECT_BINARY_DIR}/marshak.silo)

# Marshak - CPU
add_test(NAME test_problems.cpu.marshak
  COMMAND octotiger
    --config_file=${PROJECT_SOURCE_DIR}/test_problems/marshak/marshak.ini)

add_file_size_test(test_problems.cpu.marshak.filesize
  ${PROJECT_BINARY_DIR}/marshak.silo ${PROJECT_BINARY_DIR}/test_problems/marshak/final.silo)

add_test(NAME test_problems.cpu.marshak.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/marshak.silo ${PROJECT_BINARY_DIR}/test_problems/marshak/final.silo.data/0.silo)

set_tests_properties(test_problems.cpu.marshak PROPERTIES
  FIXTURES_SETUP test_problems.cpu.marshak)
set_tests_properties(test_problems.cpu.marshak.filesize PROPERTIES
  FIXTURES_SETUP test_problems.cpu.marshak)
set_tests_properties(test_problems.cpu.marshak.diff PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.marshak
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
