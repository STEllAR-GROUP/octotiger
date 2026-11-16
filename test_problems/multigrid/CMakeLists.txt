# This numerical fixture also builds without HPX or the application dependencies.
cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(octotiger_multigrid_numerics LANGUAGES CXX)
  enable_testing()
  set(OCTOTIGER_WITH_GRIDDIM 8 CACHE STRING "Interior cells per block axis")
endif()

add_compile_definitions(OCTOTIGER_GRIDDIM=${OCTOTIGER_WITH_GRIDDIM})

if(TARGET octotiger)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  foreach(case uniform amr multipole nonconvergence)
    set(extra_args --level 1)
    if(case STREQUAL amr)
      set(extra_args --level 3 --amr)
    elseif(case STREQUAL multipole)
      set(extra_args --level 1 --boundary multipole)
    elseif(case STREQUAL nonconvergence)
      set(extra_args --level 1 --max-cycles 1 --expect-failure)
    endif()
    add_test(NAME multigrid_gravity_${case}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_gravity.py
        --exe $<TARGET_FILE:octotiger>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/gravity_${case} ${extra_args})
    set_tests_properties(multigrid_gravity_${case} PROPERTIES
      PROCESSORS 1 TIMEOUT 660 LABELS multigrid RUN_SERIAL TRUE)
  endforeach()
endif()
get_filename_component(multigrid_source_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(multigrid_numerics numerical.cpp
  "${multigrid_source_root}/src/multigrid/block.cpp")
target_include_directories(multigrid_numerics PRIVATE "${multigrid_source_root}")
target_compile_features(multigrid_numerics PRIVATE cxx_std_17)
add_test(NAME multigrid_numerics COMMAND multigrid_numerics)
set_tests_properties(multigrid_numerics PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS multigrid)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_isolated_boundary.cpp")
  add_executable(multigrid_isolated_boundary test_isolated_boundary.cpp)
  target_include_directories(multigrid_isolated_boundary PRIVATE "${multigrid_source_root}")
  target_compile_features(multigrid_isolated_boundary PRIVATE cxx_std_17)
  add_test(NAME multigrid_isolated_boundary COMMAND multigrid_isolated_boundary)
  set_tests_properties(multigrid_isolated_boundary PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS multigrid)
endif()
