//  Copyright (c) 2020 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once

#include <cstdlib>

// Get called once

void start_octotiger(int argc, char* argv[]);
void register_hpx_functions();

// Get called once per locality
void init_executors();
void init_problem();

// Get called once per worker
void cleanup_puddle_on_this_locality(void);
