# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# *.diff tests fail when the output contains any .vals
set(OCTOTIGER_SILODIFF_FAIL_PATTERN ".vals")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(addOctotigerResultCheck testName fixtureName summaryPath)
  add_test(NAME ${testName}
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/checkRunSummary.py"
      "${summaryPath}" ${ARGN})
  set_tests_properties(${testName} PROPERTIES
    FIXTURES_REQUIRED ${fixtureName}
    LABELS numerical)
endfunction()

function(addOctotigerAnalyticCheck testName fixtureName summaryPath)
  set(checkArguments)
  foreach(expectation IN LISTS ARGN)
	string(REPLACE "e.00" "e+00" expectation "${expectation}")
    string(REPLACE " " ";" expectationParts "${expectation}")
    list(GET expectationParts 0 fieldName)
    list(GET expectationParts 1 expectedL1)
    list(GET expectationParts 2 expectedL2)
    list(APPEND checkArguments
      --expect analytic.${fieldName}.l1 ${expectedL1}
      --expect analytic.${fieldName}.l2 ${expectedL2})
  endforeach()
  addOctotigerResultCheck(${testName} ${fixtureName} ${summaryPath} ${checkArguments})
endfunction()

# Check whether testdata submodule is initialized by checking if the sphere file is in there
if (EXISTS ${PROJECT_SOURCE_DIR}/octotiger-testdata/sphere.silo AND IS_DIRECTORY ${PROJECT_SOURCE_DIR}/octotiger-testdata/)
    if (OCTOTIGER_WITH_BLAST_TEST)
        add_subdirectory(blast)
    endif()
    add_subdirectory(marshak)
    add_subdirectory(rotating_star)
    add_subdirectory(sod)
    add_subdirectory(sphere)
    add_subdirectory(star)
else()
  message(FATAL_ERROR "Could not find directory containing the test reference files (tested with sphere.silo)! "
   "Is the submodule initialized? To fix check out all submodules (run <git submodule update --init --recursive> within the octotiger directory) "
   "or compile with OCTOTIGER_WITH_TESTS=OFF")
endif()
