#pragma once

#include <cstdlib>

// Get called once

void start_octotiger(int argc, char* argv[]);
void register_hpx_functions();

// Get called once per locality
void init_executors();
void init_problem();

// Get called once per worker
void init_stencil(std::size_t worker_id);
void cleanup_puddle_on_this_locality(void);
