# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.

# These tests need no external Silo reference data.
add_executable(ssp_multistep_coefficients coefficients.cpp)
target_include_directories(ssp_multistep_coefficients PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_features(ssp_multistep_coefficients PRIVATE cxx_std_17)
add_test(NAME test_problems.ssp_multistep.coefficients
    COMMAND ssp_multistep_coefficients)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    set(ssp_runtime_scenarios hydro)
    if(NOT OCTOTIGER_DISABLE_ILIST)
        list(APPEND ssp_runtime_scenarios gravity restart)
    endif()
    foreach(scenario IN LISTS ssp_runtime_scenarios)
        add_test(NAME test_problems.ssp_multistep.${scenario}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/runtime.py"
                "$<TARGET_FILE:octotiger>" "${scenario}"
                --work-directory "${CMAKE_CURRENT_BINARY_DIR}")
        set_tests_properties(test_problems.ssp_multistep.${scenario} PROPERTIES
            RUN_SERIAL TRUE
            PROCESSORS 2
            TIMEOUT 600
            LABELS "ssp_multistep;cpu")
    endforeach()
else()
    message(WARNING "Python 3 was not found; SSP multistep runtime tests are unavailable")
endif()
