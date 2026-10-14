# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.

# These tests need no external Silo reference data.
add_executable(ssp_multistep_coefficients coefficients.cpp)
target_include_directories(ssp_multistep_coefficients PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_features(ssp_multistep_coefficients PRIVATE cxx_std_17)
add_test(NAME test_problems.ssp_multistep.coefficients
    COMMAND ssp_multistep_coefficients)

add_executable(ssp_history_amr history_amr.cpp)
target_include_directories(ssp_history_amr PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_features(ssp_history_amr PRIVATE cxx_std_17)
add_test(NAME test_problems.ssp_multistep.history_amr COMMAND ssp_history_amr)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    set(ssp_runtime_scenarios hydro restart_hydro problem_id)
    if(NOT OCTOTIGER_DISABLE_ILIST)
        list(APPEND ssp_runtime_scenarios gravity restart_gravity)
    endif()
    foreach(scenario IN LISTS ssp_runtime_scenarios)
        add_test(NAME test_problems.ssp_multistep.${scenario}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/runtime.py"
                "$<TARGET_FILE:octotiger>" "${scenario}"
                --work-directory "${CMAKE_CURRENT_BINARY_DIR}"
                --silo-library "${Silo_LIBRARY}")
        set_tests_properties(test_problems.ssp_multistep.${scenario} PROPERTIES
            RUN_SERIAL TRUE
            PROCESSORS 2
            TIMEOUT 1800
            LABELS "ssp_multistep;cpu")
    endforeach()

    # These coarse moving-mesh tests cover topology changes, conservation,
    # stage selection and exact restart. They are not resolution studies.
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        "${Python3_EXECUTABLE}" -c "import numpy"
        RESULT_VARIABLE ssp_numpy_result OUTPUT_QUIET ERROR_QUIET)
    if(ssp_numpy_result EQUAL 0)
        # Explicit benchmark target: keep the density-based AMR paired run out of
        # the ordinary correctness suite. The runner checks accuracy as well
        # as the actual RHS work per simulated interval.
        add_custom_target(ssp_rhs_efficiency
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/rhs_efficiency.py"
                --executable "$<TARGET_FILE:octotiger>"
                --work-directory "${CMAKE_CURRENT_BINARY_DIR}"
                --silo-library "${Silo_LIBRARY}"
            DEPENDS octotiger
            USES_TERMINAL
            COMMENT "Comparing SSP and RK3 RHS efficiency on the density-refined advected pulse")
        if(NOT OCTOTIGER_DISABLE_ILIST)
            add_custom_target(ssp_polytrope_efficiency
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/polytrope_efficiency.py"
                    --executable "$<TARGET_FILE:octotiger>"
                    --work-directory "${CMAKE_CURRENT_BINARY_DIR}"
                    --silo-library "${Silo_LIBRARY}"
                DEPENDS octotiger
                USES_TERMINAL
                COMMENT "Comparing SSP and RK3 RHS efficiency on a moving self-gravitating polytrope")
        endif()
        add_test(NAME test_problems.ssp_multistep.rhs_accounting
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/rhs_efficiency_accounting.py")
        set_tests_properties(test_problems.ssp_multistep.rhs_accounting PROPERTIES
            PROCESSORS 1 TIMEOUT 60 LABELS "ssp_multistep;cpu")
        add_test(NAME test_problems.ssp_multistep.amr_hydro
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/amr_accuracy.py"
                --executable "$<TARGET_FILE:octotiger>" --modes transfer --levels 1:3
                --stop-time 0.35 --contact-steepening on --require-post-change-steps
                --work-directory "${CMAKE_CURRENT_BINARY_DIR}"
                --silo-library "${Silo_LIBRARY}")
        set(ssp_amr_tests test_problems.ssp_multistep.amr_hydro)
        if(NOT OCTOTIGER_DISABLE_ILIST)
            add_test(NAME test_problems.ssp_multistep.amr_gravity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/amr_gravity.py"
                    --executable "$<TARGET_FILE:octotiger>" --modes transfer
                    --work-directory "${CMAKE_CURRENT_BINARY_DIR}"
                    --silo-library "${Silo_LIBRARY}")
            list(APPEND ssp_amr_tests test_problems.ssp_multistep.amr_gravity)
        endif()
        set_tests_properties(${ssp_amr_tests} PROPERTIES
            RUN_SERIAL TRUE
            PROCESSORS 6
            TIMEOUT 1800
            LABELS "ssp_multistep;cpu;amr")
    else()
        message(STATUS "NumPy was not found; moving AMR SSP regression tests are unavailable")
    endif()
else()
    message(WARNING "Python 3 was not found; SSP multistep runtime tests are unavailable")
endif()
