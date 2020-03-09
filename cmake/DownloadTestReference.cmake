# Copyright (c) 2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Used for downloading reference silo files for Octo-Tiger tests

function(download_test_reference test_name url dest)
  message(STATUS "Downloading reference SILO file for ${test_name}")
  if(NOT EXISTS ${dest})
    file(DOWNLOAD ${url} ${dest})
  endif()
  message(STATUS "Downloading reference SILO file for ${test_name} -- Success")
endfunction()