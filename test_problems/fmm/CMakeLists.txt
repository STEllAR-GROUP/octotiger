# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.

add_hpx_executable(fmm_device_only_fallback_test
  DEPENDENCIES octolib
  SOURCES device_only_fallback.cpp)
separate_arguments(fmm_arch_options NATIVE_COMMAND "${OCTOTIGER_ARCH_FLAG}")
target_compile_options(fmm_device_only_fallback_test PRIVATE ${fmm_arch_options})
add_test(NAME fmm.device_only_fallback COMMAND fmm_device_only_fallback_test)
set_tests_properties(fmm.device_only_fallback PROPERTIES
  PROCESSORS 1 TIMEOUT 120 LABELS fmm)

function(add_fmm_scalar_test name source theta)
  add_executable(fmm_${name}_test ${source})
  target_include_directories(fmm_${name}_test PRIVATE "${PROJECT_SOURCE_DIR}")
  target_compile_definitions(fmm_${name}_test PRIVATE
    OCTOTIGER_GRIDDIM=8 OCTOTIGER_THETA_MINIMUM=${theta}
    OCTOTIGER_KOKKOS_MULTIPOLE_TASKS=1 OCTOTIGER_KOKKOS_MONOPOLE_TASKS=1)
  add_test(NAME fmm.${name} COMMAND fmm_${name}_test)
  set_tests_properties(fmm.${name} PROPERTIES PROCESSORS 1 TIMEOUT 30 LABELS fmm)
endfunction()

add_fmm_scalar_test(stencil_default stencil_bounds.cpp 0.34)
add_fmm_scalar_test(stencil_even_width stencil_bounds.cpp 0.5)
add_fmm_scalar_test(stencil_large_even_width stencil_bounds.cpp 0.3)

add_hpx_executable(fmm_root_leaf_test DEPENDENCIES octolib SOURCES root_leaf.cpp)
target_compile_options(fmm_root_leaf_test PRIVATE ${fmm_arch_options})
add_test(NAME fmm.root_leaf COMMAND fmm_root_leaf_test)
add_test(NAME fmm.root_leaf_device_only COMMAND fmm_root_leaf_test --device-only)
set_tests_properties(fmm.root_leaf fmm.root_leaf_device_only PROPERTIES
  PROCESSORS 1 TIMEOUT 120 LABELS fmm)

add_fmm_scalar_test(dipole dipole.cpp 0.34)
add_hpx_executable(fmm_legacy_dipole_test DEPENDENCIES octolib SOURCES legacy_dipole.cpp)
target_compile_options(fmm_legacy_dipole_test PRIVATE ${fmm_arch_options})
add_test(NAME fmm.legacy_dipole COMMAND fmm_legacy_dipole_test)
set_tests_properties(fmm.legacy_dipole PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS fmm)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  foreach(target fmm_device_only_fallback_test fmm_root_leaf_test fmm_legacy_dipole_test)
    # Keep hydrolib's Kokkos globals owned by the shared library during teardown.
    target_link_libraries(${target} PRIVATE
      "-Wl,--push-state,--no-as-needed" hydrolib "-Wl,--pop-state")
  endforeach()
endif()
