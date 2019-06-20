##############################################################################
# Sphere
##############################################################################
message(STATUS "Downloading reference SILO file for Solid sphere test")
if(NOT EXISTS ${PROJECT_BINARY_DIR}/sphere.silo)
  file(DOWNLOAD phys.lsu.edu/~dmarcel/sphere.2.silo ${PROJECT_BINARY_DIR}/sphere.silo)
endif()
message(STATUS "Downloading reference SILO file for Solid sphere test -- Success")

# Sphere - CPU
add_test(NAME test_problems.cpu.sphere
  COMMAND octotiger
    --config_file=${PROJECT_SOURCE_DIR}/test_problems/sphere/sphere.ini)
add_test(NAME test_problems.cpu.sphere.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/sphere.silo ${PROJECT_BINARY_DIR}/X.${OCTOTIGER_WITH_GRIDDIM}.silo)

set_tests_properties(test_problems.cpu.sphere PROPERTIES
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
