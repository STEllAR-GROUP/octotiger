##############################################################################
# Blast
##############################################################################
message(STATUS "Downloading reference SILO file for Blast test")
if(NOT EXISTS ${PROJECT_BINARY_DIR}/blast.silo)
  file(DOWNLOAD phys.lsu.edu/~dmarcel/blast.5.silo ${PROJECT_BINARY_DIR}/blast.silo)
endif()
message(STATUS "Downloading reference SILO file for Blast test -- Success")

# Blast - CPU
add_test(NAME test_problems.cpu.blast
COMMAND octotiger
  --config_file=${PROJECT_SOURCE_DIR}/test_problems/blast/blast.ini)
add_test(NAME test_problems.cpu.blast.diff
COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
  ${PROJECT_BINARY_DIR}/blast.silo ${PROJECT_BINARY_DIR}/final.silo)

set_tests_properties(test_problems.cpu.blast PROPERTIES
FIXTURES_SETUP test_problems.cpu.blast)
set_tests_properties(test_problems.cpu.blast.diff PROPERTIES
FIXTURES_REQUIRED test_problems.cpu.blast
FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
