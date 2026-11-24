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
