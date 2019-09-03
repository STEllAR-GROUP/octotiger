//  Copyright (c) 2019 Dominic C Marcello
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/real.hpp"

#include <atomic>
#include <cstddef>

#ifdef DIAGNOSTIC_MODE

std::atomic<std::size_t> real::counter(0);

#endif
