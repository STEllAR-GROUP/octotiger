# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Reject field differences and mismatched meshes/object inventories.
set(OCTOTIGER_SILODIFF_FAIL_PATTERN "[.]vals|appears only in file")

# Allow reference data to be maintained in an independent checkout.
set(OCTOTIGER_TESTDATA_DIR "${PROJECT_SOURCE_DIR}/octotiger-testdata" CACHE PATH
    "Directory containing Octo-Tiger reference test data")

if (EXISTS "${OCTOTIGER_TESTDATA_DIR}/sphere.silo" AND IS_DIRECTORY "${OCTOTIGER_TESTDATA_DIR}")
    if (OCTOTIGER_WITH_BLAST_TEST)
        add_subdirectory(blast)
    endif()
    add_subdirectory(rotating_star)
    add_subdirectory(sod)
    add_subdirectory(sphere)
    add_subdirectory(star)
else()
  message(FATAL_ERROR "Could not find directory containing the test reference files (tested with sphere.silo)! "
   "Is the submodule initialized? To fix check out all submodules (run <git submodule update --init --recursive> within the octotiger directory) "
   "or set OCTOTIGER_TESTDATA_DIR to an independent checkout, or compile with OCTOTIGER_WITH_TESTS=OFF")
endif()
