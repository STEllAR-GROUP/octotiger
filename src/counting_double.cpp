//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(OCTOTIGER_HAVE_OPERATIONS_COUNT)

#include "defs.hpp"
#include "counting_double.hpp"

boost::atomic<std::size_t> counting_double::additions(0);
boost::atomic<std::size_t> counting_double::subtractions(0);
boost::atomic<std::size_t> counting_double::multiplications(0);
boost::atomic<std::size_t> counting_double::divisions(0);

#endif
