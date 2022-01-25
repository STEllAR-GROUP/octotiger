# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Sod Shock Tube
##############################################################################

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  set(rho_regex "rho 8.616107e-03 1.774427e-02")
  set(egas_regex "egas 2.234463e-02 4.586487e-02")
  set(tau_regex "tau 1.460285e-02 3.493483e-02")
  set(pot_regex "pot 0.000000e.00 0.000000e.00")
  set(sx_regex "sx 8.155587e-03 1.650219e-02")
  # sy sz zx are too close to 0 to be reasonably compared here as strings
  # meaning they're beyond machine precision so we won't get an exact string match
  set(sy_regex "sy .*")
  set(sz_regex "sz .*")
  set(zx_regex "zx .*")
  set(zy_regex "zy 2.250000e-02 5.975532e-02")
  set(zz_regex "zz 2.250000e-02 5.975532e-02")
  set(spc1_regex "spc_1 8.616107e-03 1.774427e-02")
  set(spc2_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(ini_scenario_filename "sod.ini")
  set(silo_reference_filename "sod_without_am.silo")
elseif (OCTOTIGER_WITH_GRIDDIM EQUAL 16)
  set(rho_regex "rho 4.603921e-03 1.217094e-02")
  set(egas_regex "egas 1.672459e-02 4.383427e-02")
  set(tau_regex "tau 1.207561e-02 3.052800e-02")
  set(pot_regex "pot 0.000000e.00 0.000000e.00")
  set(sx_regex "sx 7.426969e-03 1.964336e-02")
  # sy sz zx are too close to 0 to be reasonably compared here as strings
  # meaning they're beyond machine precision so we won't get an exact string match
  set(sy_regex "sy .*")
  set(sz_regex "sz .*")
  set(zx_regex "zx .*")
  set(zy_regex "zy 2.250000e-02 6.615994e-02")
  set(zz_regex "zz 2.250000e-02 6.615994e-02")
  set(spc1_regex "spc_1 4.603921e-03 1.217094e-02")
  set(spc2_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(ini_scenario_filename "sod.ini")
  set(silo_reference_filename "none")
else()
  message(FATAL_ERROR "Octo-Tiger Tests will not work with this OCTOTIGER_WTIH_GRIDDIM."
  " Compile Octotiger without tests (OCTOTIGER_WITH_TESTS=OFF) to proceed."
  " Your configuration of Octo-Tiger is not verified to work by any tests in this case, so take care!")
endif()

function(test_sod_scenario test_name test_log_file ini_filename reference_filename check_sz_zx kernel_parameters)
  add_test(NAME ${test_name}
    COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/sod/${ini_filename} ${kernel_parameters} > ${test_log_file}")
  set_tests_properties(${test_name} PROPERTIES
    FIXTURES_SETUP ${test_name})

  add_test(NAME ${test_name}.rho_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.egas_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.tau_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.pot_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.sx_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.sy_regex COMMAND cat ${test_log_file})
  # Those two tests will only succeed (be EXACTLY 0.0 instead of 1e-20) with the correct build flags being enforced
  # Considering the other cases are still within machine precision this should not be an issue
  # Still for builds with the correct flags it's prudent to check that it is exactly 0.0
  if (check_sz_zx OR (NOT OCTOTIGER_WITH_FAST_FP_CONTRACT))
    add_test(NAME ${test_name}.sz_regex COMMAND cat ${test_log_file})
    add_test(NAME ${test_name}.zx_regex COMMAND cat ${test_log_file})
  endif()
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
  # Those two tests will only succeed (be EXACTLY 0.0 instead of 1e-20) with the correct build flags being enforced
  # Considering the other cases are still within machine precision this should not be an issue
  # Still for builds with the correct flags it's prudent to check that it is exactly 0.0
  if (check_sz_zx OR (NOT OCTOTIGER_WITH_FAST_FP_CONTRACT))
    set_tests_properties(${test_name}.sz_regex PROPERTIES
      FIXTURES_REQUIRED ${test_name}
      PASS_REGULAR_EXPRESSION ${sz_regex})
    set_tests_properties(${test_name}.zx_regex PROPERTIES
      FIXTURES_REQUIRED ${test_name}
      PASS_REGULAR_EXPRESSION ${zx_regex})
  endif()
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

  if (NOT ${reference_filename} STREQUAL "none" )
    add_test(NAME ${test_name}.diff
      COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
       ${PROJECT_SOURCE_DIR}/octotiger-testdata/${reference_filename} ${PROJECT_BINARY_DIR}/test_problems/sod/final.silo.data/0.silo)
    set_tests_properties(${test_name}.diff PROPERTIES
      FIXTURES_REQUIRED ${test_name}
      FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
  endif()

  add_test(${test_name}.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/sphere/final.silo ${PROJECT_BINARY_DIR}/test_problems/sphere/final.silo.data/0.silo ${test_log_file})
  set_tests_properties(${test_name}.fixture_cleanup PROPERTIES
      FIXTURES_CLEANUP ${test_name}
  )
endfunction()

test_sod_scenario(test_problems.cpu.am_hydro_on.sod_legacy sod_old_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
"--correct_am_hydro=1 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
if(OCTOTIGER_WITH_CUDA)
  test_sod_scenario(test_problems.gpu.am_hydro_on.sod_cuda sod_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
  "--correct_am_hydro=1 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
endif()
if(OCTOTIGER_WITH_HIP)
	test_sod_scenario(test_problems.gpu.am_hydro_off.sod_hip sod_hip_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
    "--correct_am_hydro=off --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY")
endif()

if(OCTOTIGER_WITH_KOKKOS)
  test_sod_scenario(test_problems.cpu.am_hydro_on.sod_kokkos sod_kokkos_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
  "--correct_am_hydro=1 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS")
  if(OCTOTIGER_WITH_CUDA)
    test_sod_scenario(test_problems.gpu.am_hydro_on.sod_kokkos sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
    "--correct_am_hydro=1 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
  endif()
  if(OCTOTIGER_WITH_HIP)
	  test_sod_scenario(test_problems.gpu.am_hydro_off.sod_kokkos sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
      "--correct_am_hydro=off --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY")
  endif()
endif()


if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  set(rho_regex "rho 8.616107e-03 1.774427e-02")
  set(egas_regex "egas 2.234463e-02 4.586487e-02")
  set(tau_regex "tau 1.460285e-02 3.493483e-02")
  set(pot_regex "pot 0.000000e.00 0.000000e.00")
  set(sx_regex "sx 8.155587e-03 1.650219e-02")
  set(sy_regex "sy 0.000000e.00 0.000000e.00")
  set(sz_regex "sz 0.000000e.00 0.000000e.00")
  set(zx_regex "zx 0.000000e.00 0.000000e.00")
  set(zy_regex "zy 2.250000e-02 5.975532e-02")
  set(zz_regex "zz 2.250000e-02 5.975532e-02")
  set(spc1_regex "spc_1 8.616107e-03 1.774427e-02")
  set(spc2_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(ini_scenario_filename "sod.ini")
  set(silo_reference_filename "sod_without_am.silo")
elseif (OCTOTIGER_WITH_GRIDDIM EQUAL 16)
  set(rho_regex "rho 4.603921e-03 1.217094e-02")
  set(egas_regex "egas 1.672459e-02 4.383427e-02")
  set(tau_regex "tau 1.207561e-02 3.052800e-02")
  set(pot_regex "pot 0.000000e.00 0.000000e.00")
  set(sx_regex "sx 7.426969e-03 1.964336e-02")
  set(sy_regex "sy 0.000000e.00 0.000000e.00")
  set(sz_regex "sz 0.000000e.00 0.000000e.00")
  set(zx_regex "zx 0.000000e.00 0.000000e.00")
  set(zy_regex "zy 2.250000e-02 6.615994e-02")
  set(zz_regex "zz 2.250000e-02 6.615994e-02")
  set(spc1_regex "spc_1 4.603921e-03 1.217094e-02")
  set(spc2_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(ini_scenario_filename "sod.ini")
  set(silo_reference_filename "none")
else()
  message(FATAL_ERROR "Octo-Tiger Tests will not work with this OCTOTIGER_WTIH_GRIDDIM."
  " Compile Octotiger without tests (OCTOTIGER_WITH_TESTS=OFF) to proceed."
  " Your configuration of Octo-Tiger is not verified to work by any tests in this case, so take care!")
endif()


test_sod_scenario(test_problems.cpu.am_hydro_off.sod_legacy sod_old_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
"--correct_am_hydro=0 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
if(OCTOTIGER_WITH_CUDA)
  test_sod_scenario(test_problems.gpu.am_hydro_off.sod_cuda sod_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
  "--correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
endif()

if(OCTOTIGER_WITH_KOKKOS)
  test_sod_scenario(test_problems.cpu.am_hydro_off.sod_kokkos sod_kokkos_log.txt  ${ini_scenario_filename} ${silo_reference_filename} ON
  "--correct_am_hydro=0 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS")
  if(OCTOTIGER_WITH_CUDA)
    test_sod_scenario(test_problems.gpu.am_hydro_off.sod_kokkos sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
    "--correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
  endif()
endif()

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  set(rho_regex "rho 2.683250e-03 1.009641e-02")
  set(egas_regex "egas 5.643869e-03 2.257252e-02")
  set(tau_regex "tau 4.688523e-03 1.824658e-02")
  set(pot_regex "pot 0.000000e.00 0.000000e.00")
  set(sx_regex "sx 2.559705e-03 1.038547e-02")
  set(sy_regex "sy 0.000000e.00 0.000000e.00")
  set(sz_regex "sz 0.000000e.00 0.000000e.00")
  set(zx_regex "zx 0.000000e.00 0.000000e.00")
  set(zy_regex "zy 4.725000e-03 2.115041e-02")
  set(zz_regex "zz 4.725000e-03 2.115041e-02")
  set(spc1_regex "spc_1 2.683250e-03 1.009641e-02")
  set(spc2_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(ini_scenario_filename "sod_big.ini")
  set(silo_reference_filename "sod_without_am.silo")

  test_sod_scenario(test_problems.cpu.am_hydro_off.sod_big_legacy sod_old_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
  "--correct_am_hydro=0 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
  if(OCTOTIGER_WITH_CUDA)
    test_sod_scenario(test_problems.gpu.am_hydro_off.sod_big_cuda sod_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
    "--correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
      # Test combined cpu + gpu execution: Only makes sense when kernels use the same FP_Contract (otherwise there'll be an ~ 1e-20 error due to interactions between kernels executed on different devices)
    if(NOT OCTOTIGER_WITH_FAST_FP_CONTRACT)
      test_sod_scenario(test_problems.cpu_gpu.am_hydro_off.sod_big_cuda sod_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
      "--correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=4 --cuda_buffer_capacity=1 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=LEGACY")
    endif()
  endif()

  if(OCTOTIGER_WITH_KOKKOS)
    test_sod_scenario(test_problems.cpu.am_hydro_off.sod_big_kokkos sod_kokkos_log.txt  ${ini_scenario_filename} ${silo_reference_filename} ON
    "--correct_am_hydro=0 --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS")
    if(OCTOTIGER_WITH_CUDA)
      test_sod_scenario(test_problems.gpu.am_hydro_off.sod_big_kokkos sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
      "--correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
      # Test combined cpu + gpu execution: Only makes sense when kernels use the same FP_Contract (otherwise there'll be an ~ 1e-20 error due to interactions between kernels executed on different devices)
      if(NOT OCTOTIGER_WITH_FAST_FP_CONTRACT)
        test_sod_scenario(test_problems.cpu_gpu.am_hydro_off.sod_big_kokkos sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
        "--correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=4 --cuda_buffer_capacity=1 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=KOKKOS")
      endif()
    endif()
  endif()

endif()

