# Copyright (c) 2026 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_executable(wdeos_sound_speed_test
  DEPENDENCIES octolib
  SOURCES wdeos_sound_speed.cpp)

# Match the instruction set selected for the production SIMD kernels.
separate_arguments(wdeos_arch_options NATIVE_COMMAND "${OCTOTIGER_ARCH_FLAG}")
target_compile_options(wdeos_sound_speed_test PRIVATE ${wdeos_arch_options})
add_test(NAME wdeos_sound_speed COMMAND wdeos_sound_speed_test)
set_tests_properties(wdeos_sound_speed PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS eos)
