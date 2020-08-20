# Copyright (c) 2020 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Used for checking reference Silo file sizes against produced ones

function(add_file_size_test name file1 file2)
  add_test(
      NAME ${name}
      COMMAND sh -c "test $(wc -c <${file1}) -eq $(wc -c <${file2})"
  )
endfunction()
