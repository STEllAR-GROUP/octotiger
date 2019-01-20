/*
 * real.cpp
 *
 *  Created on: Jan 13, 2016
 *      Author: dmarce1
 */

#include "octotiger/real.hpp"

#include <atomic>
#include <cstddef>

#ifdef DIAGNOSTIC_MODE

std::atomic<std::size_t> real::counter(0);

#endif
