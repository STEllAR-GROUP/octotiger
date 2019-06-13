##############################################################################
# Sod Shock Tube
##############################################################################
message(STATUS "Downloading reference SILO file for Sod shock tube test")
file(DOWNLOAD phys.lsu.edu/~dmarcel/sod.5.silo ${PROJECT_BINARY_DIR}/sod.silo)
message(STATUS "Downloading reference SILO file for Sod shock tube test -- Success")

# Sod Shock Tube - CPU
add_test(NAME test_problems.cpu.sod
  COMMAND octotiger
    --config_file=${PROJECT_SOURCE_DIR}/test_problems/sod/sod.ini)

add_test(NAME test_problems.cpu.sod.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/sod.silo ${PROJECT_BINARY_DIR}/final.silo)

set_tests_properties(test_problems.cpu.sod PROPERTIES
  FIXTURES_SETUP test_problems.cpu.sod)
set_tests_properties(test_problems.cpu.sod.diff PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.sod
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

# Sod Shock Tube - GPU
if(OCTOTIGER_WITH_CUDA)
  add_test(NAME test_problems.gpu.sod
    COMMAND octotiger
      --config_file=${PROJECT_SOURCE_DIR}/test_problems/sod/sod.ini
      --cuda_streams_per_locality=1 --cuda_streams_per_gpu=1)

  add_test(NAME test_problems.gpu.sod.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/sod.silo ${PROJECT_BINARY_DIR}/final.silo)

  set_tests_properties(test_problems.gpu.sod PROPERTIES
    FIXTURES_SETUP test_problems.gpu.sod)
  set_tests_properties(test_problems.gpu.sod.diff PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.sod
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
endif()
