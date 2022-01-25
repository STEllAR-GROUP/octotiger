# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Rotating Star
##############################################################################

# Reference output to compare againsts:
# Collected with scenario:
#./build/octotiger/build/octotiger --config_file=src/octotiger/test_problems/rotating_star/rotating_star.ini --max_level=3 --stop_step=10 --theta=0.5 
# Adapt in case of physic changes (note, replace special symbols like + with .)

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  set(rho_regex "rho 3.178357e-05 2.006139e-04")
  set(egas_regex "egas 5.813066e-06 4.747447e-05")
  set(tau_regex "tau 1.377749e-05 8.266475e-05")
  set(pot_regex "pot 5.264231e-03 3.940782e-02")
  set(sx_regex "sx 7.353941e-06 6.084900e-05")
  set(sy_regex "sy 7.354428e-06 6.085096e-05")
  set(sz_regex "sz 5.123514e-06 6.002297e-05")
  set(zx_regex "zx 1.460736e-04 8.854996e-04")
  set(zy_regex "zy 1.460736e-04 8.854994e-04")
  set(zz_regex "zz 5.833372e-04 3.077163e-03")
  set(spc1_regex "spc_1 3.178357e-05 2.006139e-04")
  set(spc2_regex "spc_2 7.995711e-13 3.137956e-12")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(silo_scenario_filename "rotating_star_with_am.silo")
elseif (OCTOTIGER_WITH_GRIDDIM EQUAL 16)
  set(rho_regex "rho 8.312015e-06 5.528138e-05")
  set(egas_regex "egas 1.609556e-06 1.453207e-05")
  set(tau_regex "tau 3.647229e-06 2.379104e-05")
  set(pot_regex "pot 5.281480e-03 3.958284e-02")
  set(sx_regex "sx 2.216982e-06 1.738476e-05")
  set(sy_regex "sy 2.217014e-06 1.738469e-05")
  set(sz_regex "sz 1.782826e-06 1.727482e-05")
  set(zx_regex "zx 1.461287e-04 8.880189e-04")
  set(zy_regex "zy 1.461287e-04 8.880189e-04")
  set(zz_regex "zz 5.833354e-04 3.077143e-03")
  set(spc1_regex "spc_1 8.312014e-06 5.528138e-05")
  set(spc2_regex "spc_2 5.345237e-13 2.816412e-12")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(silo_scenario_filename "none")
else()
  message(FATAL_ERROR "Octo-Tiger Tests will not work with this OCTOTIGER_WTIH_GRIDDIM."
  " Compile Octotiger without tests (OCTOTIGER_WITH_TESTS=OFF) to proceed."
  " Your configuration of Octo-Tiger is not verified to work by any tests in this case, so take care!")
endif()

# Init inputfile (independent of kernel variants, it's sufficient to do this once)
add_test(NAME test_problems.rotating_star.init COMMAND gen_rotating_star_init)
set_tests_properties(test_problems.rotating_star.init PROPERTIES
  FIXTURES_SETUP test_problems.rotating_star.init)

function(test_rotating_star_scenario test_name test_log_file reference_filename kernel_parameters)
  add_test(NAME ${test_name}
    COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini ${kernel_parameters} > ${test_log_file}")
  set_tests_properties(${test_name} PROPERTIES
    FIXTURES_SETUP ${test_name}
    FIXTURES_REQUIRED test_problems.rotating_star.init)

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
    
  if (NOT ${reference_filename} STREQUAL "none" )
    add_test(NAME ${test_name}.diff
      COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
       ${PROJECT_SOURCE_DIR}/octotiger-testdata/${reference_filename} ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo)
    set_tests_properties(${test_name}.diff PROPERTIES
      FIXTURES_REQUIRED ${test_name}
      FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
  endif()

  add_test(${test_name}.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo ${test_log_file})
  set_tests_properties(${test_name}.fixture_cleanup PROPERTIES
      FIXTURES_CLEANUP ${test_name}
  )
endfunction()

# Rotating Star - CPU
test_rotating_star_scenario(test_problems.cpu.am_hydro_on.rotating_star_vc rotating_star_am_hydro_on_log.txt ${silo_scenario_filename} 
"  --correct_am_hydro=1 --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
# Rotating Star - OLD CPU
test_rotating_star_scenario(test_problems.cpu.am_hydro_on.rotating_star_legacy rotating_star_am_hydro_on_old_log.txt ${silo_scenario_filename} 
"  --correct_am_hydro=1 --monopole_host_kernel_type=LEGACY --multipole_host_kernel_type=LEGACY --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
if(OCTOTIGER_WITH_CUDA)
  test_rotating_star_scenario(test_problems.gpu.am_hydro_on.rotating_star_cuda rotating_star_am_hydro_on_cuda_log.txt ${silo_scenario_filename} 
  "  --correct_am_hydro=1 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
  if(NOT OCTOTIGER_WITH_FAST_FP_CONTRACT)
    # combined cpu+gpu
    test_rotating_star_scenario(test_problems.cpu_gpu.am_hydro_on.rotating_star_cuda rotating_star_am_hydro_on_cuda_log.txt ${silo_scenario_filename} 
    "  --correct_am_hydro=1 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1 --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=LEGACY")
  endif()
endif()
if(OCTOTIGER_WITH_HIP)
  test_rotating_star_scenario(test_problems.gpu.am_hydro_on.rotating_star_hip rotating_star_am_hydro_on_hip_log.txt ${silo_scenario_filename} 
    "  --correct_am_hydro=1 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=HIP --multipole_device_kernel_type=HIP --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY ")
endif()

if(OCTOTIGER_WITH_KOKKOS)
  test_rotating_star_scenario(test_problems.cpu.am_hydro_on.rotating_star_kokkos rotating_star_am_hydro_on_kokkos_log.txt ${silo_scenario_filename} 
  "  --correct_am_hydro=1 --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS")
  if(OCTOTIGER_WITH_CUDA)
    test_rotating_star_scenario(test_problems.gpu.am_hydro_on.rotating_star_kokkos rotating_star_am_hydro_on_kokkos_cuda_log.txt ${silo_scenario_filename} 
    "  --correct_am_hydro=1 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
    if(NOT OCTOTIGER_WITH_FAST_FP_CONTRACT)
      # combined cpu+gpu
      test_rotating_star_scenario(test_problems.cpu_gpu.am_hydro_on.rotating_star_kokkos rotating_star_am_hydro_on_kokkos_cuda_log.txt ${silo_scenario_filename} 
      "  --correct_am_hydro=1 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1 --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=KOKKOS")
    endif()
  endif()
  if(OCTOTIGER_WITH_HIP)
    test_rotating_star_scenario(test_problems.gpu.am_hydro_on.rotating_star_kokkos rotating_star_am_hydro_on_kokkos_hip_log.txt ${silo_scenario_filename} 
      "  --correct_am_hydro=1 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_HIP --multipole_device_kernel_type=KOKKOS_HIP --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY")
  endif()
endif()

## TESTS with correct_am_hydro=off

# Reference output to compare againsts:
# Collected with scenario:
#./build/octotiger/build/octotiger --config_file=src/octotiger/test_problems/rotating_star/rotating_star.ini --max_level=3 --stop_step=10 --theta=0.5  --correct_am_hydro=0
# Adapt in case of physic changes (note, replace special symbols like + with .)

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  set(rho_regex "rho 3.173643e-05 2.000538e-04")
  set(egas_regex "egas 5.812879e-06 4.740067e-05")
  set(tau_regex "tau 1.377465e-05 8.244716e-05")
  set(pot_regex "pot 5.264233e-03 3.940785e-02")
  set(sx_regex "sx 7.396100e-06 6.126571e-05")
  set(sy_regex "sy 7.396592e-06 6.126777e-05")
  set(sz_regex "sz 5.153324e-06 6.025281e-05")
  set(zx_regex "zx 1.460910e-04 8.855637e-04")
  set(zy_regex "zy 1.460910e-04 8.855636e-04")
  set(zz_regex "zz 5.833372e-04 3.077175e-03")
  set(spc1_regex "spc_1 3.173643e-05 2.000538e-04")
  set(spc2_regex "spc_2 9.203075e-13 3.307571e-12")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(silo_scenario_filename "rotating_star_without_am.silo")
elseif (OCTOTIGER_WITH_GRIDDIM EQUAL 16)
  set(rho_regex "rho 8.315855e-06 5.527682e-05")
  set(egas_regex "egas 1.613169e-06 1.458050e-05")
  set(tau_regex "tau 3.657545e-06 2.384480e-05")
  set(pot_regex "pot 5.281480e-03 3.958285e-02")
  set(sx_regex "sx 2.230570e-06 1.749989e-05")
  set(sy_regex "sy 2.230590e-06 1.749979e-05")
  set(sz_regex "sz 1.796048e-06 1.730076e-05")
  set(zx_regex "zx 1.461325e-04 8.880304e-04")
  set(zy_regex "zy 1.461325e-04 8.880304e-04")
  set(zz_regex "zz 5.833354e-04 3.077138e-03")
  set(spc1_regex "spc_1 8.315855e-06 5.527682e-05")
  set(spc2_regex "spc_2 5.472634e-13 2.749800e-12")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(silo_scenario_filename "none")
else()
  message(FATAL_ERROR "Octo-Tiger Tests will not work with this OCTOTIGER_WTIH_GRIDDIM."
  " Compile Octotiger without tests (OCTOTIGER_WITH_TESTS=OFF) to proceed."
  " Your configuration of Octo-Tiger is not verified to work by any tests in this case, so take care!")
endif()

# Rotating Star - CPU
test_rotating_star_scenario(test_problems.cpu.am_hydro_off.rotating_star_vc rotating_star_am_hydro_off_log.txt ${silo_scenario_filename} 
" --correct_am_hydro=0 --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
# Rotating Star - OLD CPU
test_rotating_star_scenario(test_problems.cpu.am_hydro_off.rotating_star_legacy rotating_star_am_hydro_off_old_log.txt ${silo_scenario_filename}
" --correct_am_hydro=0 --monopole_host_kernel_type=LEGACY --multipole_host_kernel_type=LEGACY --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
if(OCTOTIGER_WITH_CUDA)
  test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_cuda rotating_star_am_hydro_off_cuda_log.txt ${silo_scenario_filename}
  " --correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
  if(NOT OCTOTIGER_WITH_FAST_FP_CONTRACT)
  # combined cpu+gpu
    test_rotating_star_scenario(test_problems.cpu_gpu.am_hydro_off.rotating_star_cuda rotating_star_am_hydro_off_cuda_log.txt ${silo_scenario_filename} 
    "  --correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1 --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=LEGACY")
  endif()
endif()
if(OCTOTIGER_WITH_HIP)
  test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_hip rotating_star_am_hydro_off_hip_log.txt ${silo_scenario_filename}
    " --correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=HIP --multipole_device_kernel_type=HIP --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY")
endif()

if(OCTOTIGER_WITH_KOKKOS)
  test_rotating_star_scenario(test_problems.cpu.am_hydro_off.rotating_star_kokkos rotating_star_am_hydro_off_kokkos_log.txt ${silo_scenario_filename}
  " --correct_am_hydro=0 --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS")
  if(OCTOTIGER_WITH_CUDA)
    test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_kokkos rotating_star_am_hydro_off_kokkos_cuda_log.txt ${silo_scenario_filename}
    " --correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
    if(NOT OCTOTIGER_WITH_FAST_FP_CONTRACT)
      # combined cpu+gpu
      test_rotating_star_scenario(test_problems.cpu_gpu.am_hydro_off.rotating_star_kokkos rotating_star_am_hydro_off_kokkos_cuda_log.txt ${silo_scenario_filename} 
      "  --correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1 --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=KOKKOS")
    endif()
  endif()
  if(OCTOTIGER_WITH_HIP)
    test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_kokkos rotating_star_am_hydro_off_kokkos_hip_log.txt ${silo_scenario_filename}
      " --correct_am_hydro=0 --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_HIP --multipole_device_kernel_type=KOKKOS_HIP --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY")
  endif()
endif()

add_test(test_problems.rotating_star.init.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/rotating_star/rotating_star.bin)
set_tests_properties(test_problems.rotating_star.init.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.rotating_star.init
)


# Rotating star test with different equation of state (eos=wd)

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  set(rho_regex "rho 1.395105e-03 1.034051e-02")
  set(egas_regex "egas 1.282717e-03 1.179200e-02")
  set(tau_regex "tau 6.104387e-04 4.524548e-03")
  set(pot_regex "pot 4.917696e-03 3.234157e-02")
  set(sx_regex "sx 2.162120e-03 1.434366e-02")
  set(sy_regex "sy 2.162120e-03 1.434366e-02")
  set(sz_regex "sz 2.531845e-03 1.679645e-02")
  set(zx_regex "zx 2.003861e-04 1.158221e-03")
  set(zy_regex "zy 2.003861e-04 1.158221e-03")
  set(zz_regex "zz 5.833372e-04 2.900349e-03")
  set(spc1_regex "spc_1 1.395105e-03 1.034051e-02")
  # Around 1e-18 we neglet differences as we are hitting the limits of double accuracy 
  # see the wildcard . to check where we beginn to neglet differences
  set(spc2_regex "spc_2 8.6876..e-13 3.23906.e-12")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  # TODO add reference file
  set(silo_scenario_filename "none")

  # Rotating Star - CPU
  test_rotating_star_scenario(test_problems.cpu.eos_wd.rotating_star_vc rotating_star_eos_wd_log.txt ${silo_scenario_filename} 
    " --correct_am_hydro=0 --eos=WD --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY --amr_boundary_kernel_type=AMR_LEGACY")
  # Rotating Star - OLD CPU
  test_rotating_star_scenario(test_problems.cpu.eos_wd.rotating_star_legacy rotating_star_eos_wd_old_log.txt ${silo_scenario_filename}
    " --correct_am_hydro=0 --eos=WD --monopole_host_kernel_type=LEGACY --multipole_host_kernel_type=LEGACY --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
  if(OCTOTIGER_WITH_CUDA)
    test_rotating_star_scenario(test_problems.gpu.eos_wd.rotating_star_cuda rotating_star_eos_wd_cuda_log.txt ${silo_scenario_filename}
      " --correct_am_hydro=0 --eos=WD --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
  endif()
  if(OCTOTIGER_WITH_HIP)
    test_rotating_star_scenario(test_problems.gpu.eos_wd.rotating_star_hip rotating_star_eos_wd_hip.txt ${silo_scenario_filename}
      " --correct_am_hydro=0 --eos=WD --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=HIP --multipole_device_kernel_type=HIP --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY")
  endif()
  # TODO Add eos=wd CPU+GPU test

  if(OCTOTIGER_WITH_KOKKOS)
    test_rotating_star_scenario(test_problems.cpu.eos_wd.rotating_star_kokkos rotating_star_eos_wd_kokkos_log.txt ${silo_scenario_filename}
      " --correct_am_hydro=0 --eos=WD --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS")
    if(OCTOTIGER_WITH_CUDA)
      test_rotating_star_scenario(test_problems.gpu.eos_wd.rotating_star_kokkos_cuda rotating_star_eos_wd_kokkos_cuda_log.txt ${silo_scenario_filename}
        " --correct_am_hydro=0 --eos=WD --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
    endif()
    if(OCTOTIGER_WITH_HIP)
      test_rotating_star_scenario(test_problems.gpu.eos_wd.rotating_star_kokkos_hip rotating_star_eos_wd_kokkos_hip.txt ${silo_scenario_filename}
        " --correct_am_hydro=0 --eos=WD --cuda_number_gpus=1 --cuda_streams_per_gpu=32 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_HIP --multipole_device_kernel_type=KOKKOS_HIP --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY")
    endif()
  endif()
  # TODO Add eos=wd CPU+GPU kokkos test

endif()


add_test(test_problems.rotating_star.init.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/rotating_star/rotating_star.bin)
set_tests_properties(test_problems.rotating_star.init.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.rotating_star.init
)

