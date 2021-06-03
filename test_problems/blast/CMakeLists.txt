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

# Reference output to compare againsts:
# Collected with the blast.ini scenario!
# Adapt in case of physic changes (note, replace special symbols like + with .)  

set(rho_regex "rho 1.015611e-03 1.918630e-02")
set(egas_regex "egas 3.813222e-02 7.135766e-01")
set(tau_regex "tau 5.179120e-03 1.096847e-01")
set(pot_regex "pot 0.000000e.00 0.000000e.00")
set(sx_regex "sx 2.945534e-03 6.536674e-02")
set(sy_regex "sy 2.945534e-03 6.536674e-02")
set(sz_regex "sz 2.945534e-03 6.536674e-02")
set(zx_regex "zx 4.706839e-05 1.144265e-03")
set(zy_regex "zy 4.706839e-05 1.144265e-03")
set(zz_regex "zz 4.706839e-05 1.144265e-03")
set(spc1_regex "spc_1 1.015611e-03 1.918630e-02")
set(spc2_regex "spc_3 0.000000e.00 0.000000e.00")
set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")

  
function(test_blast_scenario test_name test_log_file kernel_parameters)
  add_test(NAME ${test_name}
    COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/blast/blast.ini ${kernel_parameters} > ${test_log_file}")
  set_tests_properties(${test_name} PROPERTIES
    FIXTURES_SETUP ${test_name})

  add_test(NAME ${test_name}.rho_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.egas_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.tau_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.pot_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.sx_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.sy_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.sz_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.zx_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.zy_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.zz_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.spc1_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.spc2_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.spc3_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.spc4_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.spc5_regex COMMAND cat ${test_log_file})
  set_tests_properties(${test_name}.rho_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${rho_regex})
  set_tests_properties(${test_name}.egas_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${egas_regex})
  set_tests_properties(${test_name}.tau_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${tau_regex})
  set_tests_properties(${test_name}.pot_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${pot_regex})
  set_tests_properties(${test_name}.sx_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${sx_regex})
  set_tests_properties(${test_name}.sy_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${sy_regex})
  set_tests_properties(${test_name}.sz_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${sz_regex})
  set_tests_properties(${test_name}.zx_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${zx_regex})
  set_tests_properties(${test_name}.zy_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${zy_regex})
  set_tests_properties(${test_name}.zz_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${zz_regex})
  set_tests_properties(${test_name}.spc1_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${spc1_regex})
  set_tests_properties(${test_name}.spc2_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${spc2_regex})
  set_tests_properties(${test_name}.spc3_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${spc3_regex})
  set_tests_properties(${test_name}.spc4_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${spc4_regex})
  set_tests_properties(${test_name}.spc5_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${spc5_regex})


  
  # TODO replace with proper diff comparison - needs reference files
  add_test(NAME ${test_name}.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/blast.silo ${PROJECT_BINARY_DIR}/test_problems/blast/final.silo)
  set_tests_properties(${test_name}.diff PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

  add_test(${test_name}.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/sphere/final.silo ${PROJECT_BINARY_DIR}/test_problems/sphere/final.silo.data/0.silo ${test_log_file})
  set_tests_properties(${test_name}.fixture_cleanup PROPERTIES
      FIXTURES_CLEANUP ${test_name}
  )
endfunction()

test_blast_scenario(test_problems.cpu.blast_legacy blast_old_log.txt " --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
if(OCTOTIGER_WITH_CUDA)
  test_blast_scenario(test_problems.gpu.blast_cuda blast_cuda_log.txt " --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=LEGACY")
endif()

if(OCTOTIGER_WITH_KOKKOS)
  test_blast_scenario(test_problems.cpu.blast_kokkos blast_kokkos_log.txt " --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS")
  if(OCTOTIGER_WITH_CUDA)
    test_blast_scenario(test_problems.gpu.blast_kokkos blast_kokkos_cuda_log.txt " --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=KOKKOS")
  endif()
endif()