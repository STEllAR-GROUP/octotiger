##############################################################################
# Marshak wave test
##############################################################################
message(STATUS "Downloading reference SILO file for Marshak wave test")
if(NOT EXISTS ${PROJECT_BINARY_DIR}/marshak.silo)
  file(DOWNLOAD phys.lsu.edu/~dmarcel/marshak.3.silo ${PROJECT_BINARY_DIR}/marshak.silo)
endif()
message(STATUS "Downloading reference SILO file for Marshak wave test -- Success")

# Marshak - CPU
add_test(NAME test_problems.cpu.marshak
  COMMAND octotiger
    --config_file=${PROJECT_SOURCE_DIR}/test_problems/marshak/marshak.ini)
add_test(NAME test_problems.cpu.marshak.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/marshak.silo ${PROJECT_BINARY_DIR}/final.silo)

set_tests_properties(test_problems.cpu.marshak PROPERTIES
  FIXTURES_SETUP test_problems.cpu.marshak)
set_tests_properties(test_problems.cpu.marshak.diff PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.marshak
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
