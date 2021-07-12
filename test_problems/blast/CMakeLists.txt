# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Blast
##############################################################################
download_test_reference("Blast test"
  "phys.lsu.edu/~dmarcel/blast.13.silo"
  ${PROJECT_BINARY_DIR}/blast.silo)

# Blast - CPU
add_test(NAME test_problems.cpu.blast
  COMMAND octotiger
    --config_file=${PROJECT_SOURCE_DIR}/test_problems/blast/blast.ini)
add_test(NAME test_problems.cpu.blast.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/blast.silo ${PROJECT_BINARY_DIR}/test_problems/blast/final.silo.data/0.silo)

set_tests_properties(test_problems.cpu.blast PROPERTIES
  FIXTURES_SETUP test_problems.cpu.blast)
set_tests_properties(test_problems.cpu.blast.diff PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.blast
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
