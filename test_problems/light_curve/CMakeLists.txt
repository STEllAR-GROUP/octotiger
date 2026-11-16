# This numerical fixture also builds without HPX or application dependencies.
cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(octotiger_light_curve_numerics LANGUAGES CXX)
  enable_testing()
endif()

get_filename_component(light_curve_source_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(light_curve_numerics test_numerics.cpp)
target_include_directories(light_curve_numerics PRIVATE "${light_curve_source_root}")
target_compile_features(light_curve_numerics PRIVATE cxx_std_17)
add_test(NAME light_curve_numerics COMMAND light_curve_numerics)
set_tests_properties(light_curve_numerics PROPERTIES PROCESSORS 1 TIMEOUT 30 LABELS light_curve)

if(TARGET octotiger)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  foreach(case uniform opacity amr fld ipr)
    add_test(NAME light_curve_application_${case}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_light_curve.py
        --exe $<TARGET_FILE:octotiger>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${case}
        --case ${case} --threads 1)
    set_tests_properties(light_curve_application_${case} PROPERTIES
      PROCESSORS 1 TIMEOUT 660 LABELS light_curve RUN_SERIAL TRUE)
  endforeach()
endif()
