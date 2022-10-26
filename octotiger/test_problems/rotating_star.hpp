//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef ROTATING_STAR_ROTATING_STAR_HPP_
#define ROTATING_STAR_ROTATING_STAR_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/real.hpp"

#include <vector>

#include "octotiger/containers.hpp"
OCTOTIGER_EXPORT oct::vector<real> rotating_star(real x, real y, real z, real);

OCTOTIGER_EXPORT oct::vector<real> rotating_star_a(real x, real y, real z, real);

#endif /* ROTATING_STAR_ROTATING_STAR_HPP_ */
