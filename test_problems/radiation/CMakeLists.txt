# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Radiation Diffusion
##############################################################################

# Reference output to compare againsts:
# Collected with scenario:
# octotiger --problem=radiation_diffusion --radiation=on   --gravity=off --max_level=2 --stop_time=10 --hydro=off --odt=1e-5 --hard_dt=25e-1
# Adapt in case of physic changes (note, replace special symbols like + with .)

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  set(rho_regex "rho 0.000000e.00 0.000000e.00 0.000000e.00")
  set(egas_regex "egas 2.454599e-06 2.454599e-06 2.454599e-06")
  set(tau_regex "tau 2.455943e-04 2.455943e-04 2.455943e-04")
  set(pot_regex "pot 0.000000e.00 0.000000e.00 0.000000e.00")
  set(sx_regex "sx 0.000000e.00 0.000000e.00 0.000000e.00")
  set(sy_regex "sy 0.000000e.00 0.000000e.00 0.000000e.00")
  set(sz_regex "sz 0.000000e.00 0.000000e.00 0.000000e.00")
  set(zx_regex "zx 0.000000e.00 0.000000e.00 0.000000e.00")
  set(zy_regex "zy 0.000000e.00 0.000000e.00 0.000000e.00")
  set(zz_regex "zz 0.000000e.00 0.000000e.00 0.000000e.00")
  set(spc1_regex "spc_1 0.000000e.00 0.000000e.00 0.000000e.00")
  set(spc2_regex "spc_2 0.000000e.00 0.000000e.00 0.000000e.00")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00 0.000000e.00")
  set(er_regex "er 9.725649e-07 9.725649e-07 9.725650e-07")
  set(fx_regex "fx 2.329772e-14 2.895043e-14 7.027676e-14")
  set(fy_regex "fy 2.329772e-14 2.895043e-14 7.027676e-14")
  set(fz_regex "fz 2.329772e-14 2.895043e-14 7.027676e-14")
  set(silo_scenario_filename "radiation_diffusion_test.silo")
endif()

function(test_radiation_diffusion_scenario test_name test_log_file reference_filename kernel_parameters)
  add_test(NAME ${test_name}
    COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --problem=radiation_diffusion --radiation=on   --gravity=off --max_level=2 --stop_time=10 --hydro=off --odt=1e-5 --hard_dt=25e-1 ${kernel_parameters} > ${test_log_file}")
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
  add_test(NAME ${test_name}.er_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.fx_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.fy_regex COMMAND cat ${test_log_file})
  add_test(NAME ${test_name}.fz_regex COMMAND cat ${test_log_file})
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
  set_tests_properties(${test_name}.er_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${er_regex})
  set_tests_properties(${test_name}.fx_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${fx_regex})
  set_tests_properties(${test_name}.fy_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${fy_regex})
  set_tests_properties(${test_name}.fz_regex PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    PASS_REGULAR_EXPRESSION ${fz_regex})
    

  if (NOT ${reference_filename} STREQUAL "none" )
    add_test(NAME ${test_name}.diff
      COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-4
       ${PROJECT_SOURCE_DIR}/octotiger-testdata/${reference_filename} ${PROJECT_BINARY_DIR}/test_problems/radiation/final.silo.data/0.silo)
    set_tests_properties(${test_name}.diff PROPERTIES
      FIXTURES_REQUIRED ${test_name}
      FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
  endif()

  add_test(${test_name}.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/radiation_diffusion/final.silo ${PROJECT_BINARY_DIR}/test_problems/radiation_diffusion/final.silo.data/0.silo ${test_log_file})
  set_tests_properties(${test_name}.fixture_cleanup PROPERTIES
      FIXTURES_CLEANUP ${test_name}
  )
endfunction()

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  # Rotating Star - OLD CPU
  # TODO Add radiation kernel variants once there is more than one available...
  test_radiation_diffusion_scenario(test_problems.cpu.am_hydro_on.radiation_diffusion_legacy radiation_diffusion_am_hydro_on_old_log.txt ${silo_scenario_filename} " --hydro_device_kernel_type=OFF ")
endif()

function(test_radiation_coupling_scenario test_name test_log_file kernel_parameters)
  add_test(NAME ${test_name}
    COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --problem=radiation_coupling --radiation=on  --gravity=off --max_level=2 --stop_time=1 --hydro=off --odt=1e-5 --hard_dt=1e-2 ${kernel_parameters} > ${test_log_file}")
  set_tests_properties(${test_name} PROPERTIES
    FIXTURES_SETUP ${test_name})

  add_test(NAME ${test_name}.diff
    COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-4
     ${PROJECT_SOURCE_DIR}/octotiger-testdata/radiation_coupling_test.silo ${PROJECT_BINARY_DIR}/test_problems/radiation/final.silo.data/0.silo)
  set_tests_properties(${test_name}.diff PROPERTIES
    FIXTURES_REQUIRED ${test_name}
    FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})

  add_test(${test_name}.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/radiation_coupling/final.silo ${PROJECT_BINARY_DIR}/test_problems/radiation_coupling/final.silo.data/0.silo ${test_log_file})
  set_tests_properties(${test_name}.fixture_cleanup PROPERTIES
      FIXTURES_CLEANUP ${test_name}
  )
endfunction()

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  # Rotating Star - OLD CPU
  # TODO Add radiation kernel variants once there is more than one available...
  test_radiation_coupling_scenario(test_problems.cpu.am_hydro_on.radiation_coupling_legacy radiation_coupling_am_hydro_on_old_log.txt " --hydro_device_kernel_type=OFF ")
endif()
