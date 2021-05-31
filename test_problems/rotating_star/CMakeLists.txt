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
# Adapt in case of physic changes

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

# Rotating Star - CPU
add_test(NAME test_problems.cpu.rotating_star.init COMMAND gen_rotating_star_init)
add_test(NAME test_problems.cpu.rotating_star
  COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY > rotating_star_log.txt")
set_tests_properties(test_problems.cpu.rotating_star PROPERTIES
  FIXTURES_SETUP test_problems.cpu.rotating_star
  FIXTURES_REQUIRED test_problems.cpu.rotating_star.init)
set_tests_properties(test_problems.cpu.rotating_star.init PROPERTIES
  FIXTURES_SETUP test_problems.cpu.rotating_star.init)
set_tests_properties(test_problems.cpu.rotating_star PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star.init)

add_test(NAME test_problems.cpu.rotating_star.rho_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.egas_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.tau_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.pot_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.sx_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.sy_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.sz_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.zx_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.zy_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.zz_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc1_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc2_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc3_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc4_regex COMMAND cat rotating_star_log.txt)
add_test(NAME test_problems.cpu.rotating_star.spc5_regex COMMAND cat rotating_star_log.txt)
set_tests_properties(test_problems.cpu.rotating_star.rho_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${rho_regex})
set_tests_properties(test_problems.cpu.rotating_star.egas_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${egas_regex})
set_tests_properties(test_problems.cpu.rotating_star.tau_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${tau_regex})
set_tests_properties(test_problems.cpu.rotating_star.pot_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${pot_regex})
set_tests_properties(test_problems.cpu.rotating_star.sx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${sx_regex})
set_tests_properties(test_problems.cpu.rotating_star.sy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${sy_regex})
set_tests_properties(test_problems.cpu.rotating_star.sz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${sz_regex})
set_tests_properties(test_problems.cpu.rotating_star.zx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${zx_regex})
set_tests_properties(test_problems.cpu.rotating_star.zy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${zy_regex})
set_tests_properties(test_problems.cpu.rotating_star.zz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${zz_regex})
set_tests_properties(test_problems.cpu.rotating_star.spc1_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${spc1_regex})
set_tests_properties(test_problems.cpu.rotating_star.spc2_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${spc2_regex})
set_tests_properties(test_problems.cpu.rotating_star.spc3_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${spc3_regex})
set_tests_properties(test_problems.cpu.rotating_star.spc4_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${spc4_regex})
set_tests_properties(test_problems.cpu.rotating_star.spc5_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star
  PASS_REGULAR_EXPRESSION ${spc5_regex})
   
add_test(NAME test_problems.cpu.rotating_star.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo)
set_tests_properties(test_problems.cpu.rotating_star.diff PROPERTIES
  FIXTURES_REQUIRED "test_problems.cpu.rotating_star.init;test_problems.cpu.rotating_star"
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

add_test(test_problems.cpu.rotating_star.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo rotating_star_log.txt)
set_tests_properties(test_problems.cpu.rotating_star.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.cpu.rotating_star
)

# Rotating Star - OLD CPU
add_test(NAME test_problems.cpu.rotating_star_old
  COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini --monopole_host_kernel_type=LEGACY --multipole_host_kernel_type=LEGACY --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF --hydro_device_kernel_type=OFF --hydro_host_kernel_type=LEGACY > rotating_star_old_log.txt")
set_tests_properties(test_problems.cpu.rotating_star_old PROPERTIES
  FIXTURES_SETUP test_problems.cpu.rotating_star_old
  FIXTURES_REQUIRED test_problems.cpu.rotating_star.init)
add_test(NAME test_problems.cpu.rotating_star_old.rho_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.egas_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.tau_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.pot_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.sx_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.sy_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.sz_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.zx_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.zy_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.zz_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc1_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc2_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc3_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc4_regex COMMAND cat rotating_star_old_log.txt)
add_test(NAME test_problems.cpu.rotating_star_old.spc5_regex COMMAND cat rotating_star_old_log.txt)
set_tests_properties(test_problems.cpu.rotating_star_old.rho_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${rho_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.egas_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${egas_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.tau_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${tau_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.pot_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${pot_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.sx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${sx_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.sy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${sy_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.sz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${sz_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.zx_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${zx_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.zy_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${zy_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.zz_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${zz_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.spc1_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${spc1_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.spc2_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${spc2_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.spc3_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${spc3_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.spc4_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${spc4_regex})
set_tests_properties(test_problems.cpu.rotating_star_old.spc5_regex PROPERTIES
  FIXTURES_REQUIRED test_problems.cpu.rotating_star_old
  PASS_REGULAR_EXPRESSION ${spc5_regex})

add_test(NAME test_problems.cpu.rotating_star_old.diff
  COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
    ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star/final.silo.data/0.silo)
set_tests_properties(test_problems.cpu.rotating_star_old.diff PROPERTIES
  FIXTURES_REQUIRED "test_problems.cpu.rotating_star.init;test_problems.cpu.rotating_star_old"
  FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

add_test(test_problems.cpu.rotating_star_old.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/rotating_star_old/final.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star_old/final.silo.data/0.silo rotating_star_old_log.txt)
set_tests_properties(test_problems.cpu.rotating_star_old.fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP test_problems.cpu.rotating_star_old
)

# Rotating Star - GPU
if(OCTOTIGER_WITH_CUDA)
  add_test(NAME test_problems.gpu.rotating_star.init COMMAND gen_rotating_star_init)
  add_test(NAME test_problems.gpu.rotating_star
  COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --config_file=${PROJECT_SOURCE_DIR}/test_problems/rotating_star/rotating_star.ini --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=LEGACY > rotating_star_gpu_log.txt")
  add_test(NAME test_problems.gpu.rotating_star.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
      ${PROJECT_BINARY_DIR}/rotating_star.silo ${PROJECT_BINARY_DIR}/final.silo)

  set_tests_properties(test_problems.gpu.rotating_star.init PROPERTIES
    FIXTURES_SETUP test_problems.gpu.rotating_star.init)
  set_tests_properties(test_problems.gpu.rotating_star PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star.init)

  set_tests_properties(test_problems.gpu.rotating_star PROPERTIES
    FIXTURES_SETUP test_problems.gpu.rotating_star
    FIXTURES_REQUIRED test_problems.gpu.rotating_star.init)
  set_tests_properties(test_problems.gpu.rotating_star.diff PROPERTIES
    FIXTURES_REQUIRED "test_problems.gpu.rotating_star.init;test_problems.gpu.rotating_star"
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})


  add_test(NAME test_problems.gpu.rotating_star.rho_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.egas_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.tau_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.pot_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.sx_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.sy_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.sz_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.zx_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.zy_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.zz_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.spc1_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.spc2_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.spc3_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.spc4_regex COMMAND cat rotating_star_gpu_log.txt)
  add_test(NAME test_problems.gpu.rotating_star.spc5_regex COMMAND cat rotating_star_gpu_log.txt)
  set_tests_properties(test_problems.gpu.rotating_star.rho_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${rho_regex})
  set_tests_properties(test_problems.gpu.rotating_star.egas_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${egas_regex})
  set_tests_properties(test_problems.gpu.rotating_star.tau_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${tau_regex})
  set_tests_properties(test_problems.gpu.rotating_star.pot_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${pot_regex})
  set_tests_properties(test_problems.gpu.rotating_star.sx_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${sx_regex})
  set_tests_properties(test_problems.gpu.rotating_star.sy_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${sy_regex})
  set_tests_properties(test_problems.gpu.rotating_star.sz_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${sz_regex})
  set_tests_properties(test_problems.gpu.rotating_star.zx_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${zx_regex})
  set_tests_properties(test_problems.gpu.rotating_star.zy_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${zy_regex})
  set_tests_properties(test_problems.gpu.rotating_star.zz_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${zz_regex})
  set_tests_properties(test_problems.gpu.rotating_star.spc1_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${spc1_regex})
  set_tests_properties(test_problems.gpu.rotating_star.spc2_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${spc2_regex})
  set_tests_properties(test_problems.gpu.rotating_star.spc3_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${spc3_regex})
  set_tests_properties(test_problems.gpu.rotating_star.spc4_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${spc4_regex})
  set_tests_properties(test_problems.gpu.rotating_star.spc5_regex PROPERTIES
    FIXTURES_REQUIRED test_problems.gpu.rotating_star
    PASS_REGULAR_EXPRESSION ${spc5_regex})

  add_test(test_problems.gpu.rotating_star.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/rotating_star_old/final.silo ${PROJECT_BINARY_DIR}/test_problems/rotating_star_old/final.silo.data/0.silo rotating_star_gpu_log.txt)
  set_tests_properties(test_problems.gpu.rotating_star.fixture_cleanup PROPERTIES
      FIXTURES_CLEANUP test_problems.gpu.rotating_star
  )

endif()
