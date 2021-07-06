# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# *.diff tests fail when the output contains any .vals
set(OCTOTIGER_SILODIFF_FAIL_PATTERN ".vals")

# Check whether testdata submodule is initialized by checking if the sphere file is in there
if (EXISTS ${PROJECT_SOURCE_DIR}/octotiger-testdata/sphere.silo AND IS_DIRECTORY ${PROJECT_SOURCE_DIR}/octotiger-testdata/)
    if (OCTOTIGER_WITH_BLAST_TEST)
        add_subdirectory(blast)
    endif()
    add_subdirectory(marshak)
    add_subdirectory(rotating_star)
    add_subdirectory(sod)
    add_subdirectory(sphere)
else()
  message(FATAL_ERROR "Could not find directory containing the test reference files (tested with sphere.silo)!"
   "Is the submodule checked out? To fix check out all submodules or compile with OCTOTIGER_WITH_TESTS=OFF")
endif()
