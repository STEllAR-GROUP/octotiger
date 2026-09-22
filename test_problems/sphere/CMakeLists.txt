# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################
# Sphere
##############################################################################

# Reference output to compare againsts:
# Collected with the solid sphere scenario!
# Adapt in case of physic changes (note, replace special symbols like + with .)

if (OCTOTIGER_WITH_GRIDDIM EQUAL 8)
  set(rho_regex "rho 1.048130e-03 1.989298e-02")
  set(egas_regex "egas 0.000000e.00 0.000000e.00")
  set(tau_regex "tau 0.000000e.00 0.000000e.00")
  set(pot_regex "pot 4.470774e-01 3.176879e.00")
  set(sx_regex "sx 0.000000e.00 0.000000e.00")
  set(sy_regex "sy 0.000000e.00 0.000000e.00")
  set(sz_regex "sz 0.000000e.00 0.000000e.00")
  set(zx_regex "zx 0.000000e.00 0.000000e.00")
  set(zy_regex "zy 0.000000e.00 0.000000e.00")
  set(zz_regex "zz 0.000000e.00 0.000000e.00")
  set(spc1_regex "spc_1 1.048130e-03 1.989298e-02")
  set(spc2_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(silo_scenario_filename "sphere.silo")
elseif (OCTOTIGER_WITH_GRIDDIM EQUAL 16)
  set(rho_regex "rho 4.665483e-04 1.508687e-02")
  set(egas_regex "egas 0.000000e.00 0.000000e.00")
  set(tau_regex "tau 0.000000e.00 0.000000e.00")
  set(pot_regex "pot 4.493009e-01 3.219940e.00")
  set(sx_regex "sx 0.000000e.00 0.000000e.00")
  set(sy_regex "sy 0.000000e.00 0.000000e.00")
  set(sz_regex "sz 0.000000e.00 0.000000e.00")
  set(zx_regex "zx 0.000000e.00 0.000000e.00")
  set(zy_regex "zy 0.000000e.00 0.000000e.00")
  set(zz_regex "zz 0.000000e.00 0.000000e.00")
  set(spc1_regex "spc_1 4.665483e-04 1.508687e-02")
  set(spc2_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc3_regex "spc_3 0.000000e.00 0.000000e.00")
  set(spc4_regex "spc_4 0.000000e.00 0.000000e.00")
  set(spc5_regex "spc_5 0.000000e.00 0.000000e.00")
  set(silo_scenario_filename "none")
else()
  message(FATAL_ERROR "Octo-Tiger Tests will not work with this OCTOTIGER_WTIH_GRIDDIM."
  " Compile Octotiger without tests (OCTOTIGER_WITH_TESTS=OFF) to proceed."
  " Your configuration of Octo-Tiger is not verified to work by any tests in this case, so take care!")
endif()

function(test_sphere_scenario test_name test_log_file reference_filename kernel_parameters)
  add_test(NAME ${test_name}
    COMMAND sh -c "${PROJECT_BINARY_DIR}/octotiger --runtime.config_file=${PROJECT_SOURCE_DIR}/test_problems/sphere/sphere.ini --output.results_file=${test_log_file}.json ${kernel_parameters} > ${test_log_file}")
  set_tests_properties(${test_name} PROPERTIES
    FIXTURES_SETUP ${test_name})

  addOctotigerAnalyticCheck(${test_name}.numerical ${test_name} ${test_log_file}.json
    "${rho_regex}" "${egas_regex}" "${tau_regex}" "${pot_regex}"
    "${sx_regex}" "${sy_regex}" "${sz_regex}" "${zx_regex}"
    "${zy_regex}" "${zz_regex}" "${spc1_regex}" "${spc2_regex}"
    "${spc3_regex}" "${spc4_regex}" "${spc5_regex}")


  if (NOT ${reference_filename} STREQUAL "none" )
    add_test(NAME ${test_name}.diff
      COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12 -E "lx, ly, lz, idle_rate, locality, _hdf5libinfo, _silolibinfo"
#      COMMAND ${Silo_BROWSER} -e diff -q -x 1.0 -R 1.0e-12
       ${PROJECT_SOURCE_DIR}/octotiger-testdata/${reference_filename} ${PROJECT_BINARY_DIR}/test_problems/sphere/final.silo.data/0.silo)
    set_tests_properties(${test_name}.diff PROPERTIES
      FIXTURES_REQUIRED ${test_name}
      FAIL_REGULAR_EXPRESSION ${OCTOTIGER_SILODIFF_FAIL_PATTERN})
  endif()

  add_test(${test_name}.fixture_cleanup ${CMAKE_COMMAND} -E remove ${PROJECT_BINARY_DIR}/test_problems/sphere/final.silo ${PROJECT_BINARY_DIR}/test_problems/sphere/final.silo.data/0.silo ${test_log_file} ${test_log_file}.json)
  set_tests_properties(${test_name}.fixture_cleanup PROPERTIES
      FIXTURES_CLEANUP ${test_name}
  )
endfunction()

# Sphere - CPU
test_sphere_scenario(test_problems.cpu.sphere_vc  sphere_log.txt ${silo_scenario_filename}
" --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF ")
# Sphere - LEGACY
test_sphere_scenario(test_problems.cpu.sphere_legacy  sphere_old_log.txt ${silo_scenario_filename}
" --monopole_host_kernel_type=LEGACY --multipole_host_kernel_type=LEGACY --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF ")
# Sphere - GPU plain CUDA
if(OCTOTIGER_WITH_CUDA)
  test_sphere_scenario(test_problems.gpu.sphere_cuda  sphere_cuda_log.txt ${silo_scenario_filename}
    " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA ")
endif()
# Sphere - GPU plain HIP
if(OCTOTIGER_WITH_HIP)
  test_sphere_scenario(test_problems.gpu.sphere_hip  sphere_hip_log.txt ${silo_scenario_filename}
    " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=HIP --multipole_device_kernel_type=HIP ")
endif()

# Sphere - Kokkos Host / Device tests
if(OCTOTIGER_WITH_KOKKOS)
  test_sphere_scenario(test_problems.cpu.sphere_kokkos  sphere_kokkos_log.txt ${silo_scenario_filename}
  " --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=OFF --multipole_device_kernel_type=OFF ")
  if(OCTOTIGER_WITH_CUDA)
    test_sphere_scenario(test_problems.gpu.sphere_kokkos_cuda  sphere_kokkos_cuda_log.txt ${silo_scenario_filename}
      " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA ")
  endif()
  if(OCTOTIGER_WITH_HIP)
    test_sphere_scenario(test_problems.gpu.sphere_kokkos_hip  sphere_kokkos_hip_log.txt ${silo_scenario_filename}
      " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_HIP --multipole_device_kernel_type=KOKKOS_HIP ")
  endif()
  if(Kokkos_ENABLE_SYCL)
    test_sphere_scenario(test_problems.gpu.sphere_kokkos_sycl  sphere_kokkos_sycl_log.txt ${silo_scenario_filename}
      " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_SYCL --multipole_device_kernel_type=KOKKOS_SYCL ")
  endif()
endif()
# TODO CPU+GPU tests?
