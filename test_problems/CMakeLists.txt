# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(DownloadTestReference)

# *.diff tests fail when the output contains any .vals
set(OCTOTIGER_SILODIFF_FAIL_PATTERN ".vals")

if (OCTOTIGER_WITH_BLAST_TEST)
    add_subdirectory(blast)
endif()
add_subdirectory(marshak)
add_subdirectory(rotating_star)
add_subdirectory(sod)
add_subdirectory(sphere)
