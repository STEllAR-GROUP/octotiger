# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Rotating Star
##############################################################################
download_test_reference("Rotating star test"
  "phys.lsu.edu/~dmarcel/rotating_star.13.silo"
  ${PROJECT_BINARY_DIR}/rotating_star.silo)

# Reference output to compare againsts:
# Collected with scenario:
#./build/octotiger/build/octotiger --config_file=src/octotiger/test_problems/rotating_star/rotating_star.ini --max_level=3 --stop_step=10 --theta=0.5 
# Adapt in case of physic changes (note, replace special symbols like + with .)

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  set(rho_regex "rho 3.178386e-05 2.006162e-04")
  set(egas_regex "egas 5.813086e-06 4.747489e-05")
  set(tau_regex "tau 1.377758e-05 8.266548e-05")
  set(pot_regex "pot 5.264186e-03 3.940757e-02")
  set(sx_regex "sx 7.354162e-06 6.084948e-05")
  set(sy_regex "sy 7.354649e-06 6.085143e-05")
  set(sz_regex "sz 5.123326e-06 6.002226e-05")
  set(zx_regex "zx 1.460722e-04 8.854924e-04")
  set(zy_regex "zy 1.460722e-04 8.854922e-04")
  set(zz_regex "zz 5.833309e-04 3.077133e-03")
  set(spc1_regex "spc_1 3.178386e-05 2.006162e-04")
  set(spc2_regex "spc_2 7.995822e-13 3.137961e-12")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
elseif (OCTOTIGER_WITH_GRIDDIM EQUAL 16)
  set(rho_regex "rho 8.312298e-06 5.528339e-05")
  set(egas_regex "egas 1.609607e-06 1.453264e-05")
  set(tau_regex "tau 3.647345e-06 2.379180e-05")
  set(pot_regex "pot 5.281435e-03 3.958259e-02")
  set(sx_regex "sx 2.217392e-06 1.738639e-05")
  set(sy_regex "sy 2.217442e-06 1.738633e-05")
  set(sz_regex "sz 1.783532e-06 1.727836e-05")
  set(zx_regex "zx 1.461274e-04 8.880116e-04")
  set(zy_regex "zy 1.461274e-04 8.880116e-04")
  set(zz_regex "zz 5.833291e-04 3.077112e-03")
  set(spc1_regex "spc_1 8.312298e-06 5.528339e-05")
  set(spc2_regex "spc_2 5.345264e-13 2.816438e-12")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
else()
  message(FATAL_ERROR "Octo-Tiger Tests will not work with this OCTOTIGER_WTIH_GRIDDIM."
  " Compile Octotiger without tests (OCTOTIGER_WITH_TESTS=OFF) to proceed."
  " Your configuration of Octo-Tiger is not verified to work by any tests in this case, so take care!")
endif()

# Init inputfile (independent of kernel variants, it's sufficient to do this once)
add_test(NAME test_problems.rotating_star.init COMMAND gen_rotating_star_init)
set_tests_properties(test_problems.rotating_star.init PROPERTIES
  FIXTURES_SETUP test_problems.rotating_star.init)

function(test_rotating_star_scenario test_name test_log_file kernel_parameters)
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
    
  if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
    # TODO Diff currently comparing the wrong files - collect new ones
    add_test(NAME ${test_name}.diff
      COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
        ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo)
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
test_rotating_star_scenario(test_problems.cpu.rotating_star_vc rotating_star_log.txt " --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
# Rotating Star - OLD CPU
test_rotating_star_scenario(test_problems.cpu.rotating_star_legacy rotating_star_old_log.txt " --monopole_host_kernel_type=LEGACY --multipole_host_kernel_type=LEGACY --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY")
if(OCTOTIGER_WITH_CUDA)
  test_rotating_star_scenario(test_problems.gpu.rotating_star_cuda rotating_star_cuda_log.txt " --cuda_number_gpus=1 --cuda_streams_per_gpu=64 --cuda_buffer_capacity=1 --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=LEGACY")
endif()

if(OCTOTIGER_WITH_KOKKOS)
  test_rotating_star_scenario(test_problems.cpu.rotating_star_kokkos rotating_star_kokkos_log.txt " --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=KOKKOS")
  if(OCTOTIGER_WITH_CUDA)
    test_rotating_star_scenario(test_problems.gpu.rotating_star_kokkos rotating_star_kokkos_cuda_log.txt " --cuda_number_gpus=1 --cuda_streams_per_gpu=64 --cuda_buffer_capacity=1024 --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=KOKKOS")
  endif()
endif()

add_test(test_problems.rotating_star.init.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/rotating_star/rotating_star.bin)
set_tests_properties(test_problems.rotating_star.init.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.rotating_star.init
)
