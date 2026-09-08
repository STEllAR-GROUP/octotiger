//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef ROTATING_STAR_ROTATING_STAR_HPP_
#define ROTATING_STAR_ROTATING_STAR_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/math/Real.hpp"

#include <vector>

OCTOTIGER_EXPORT std::vector<Real> rotating_star(Real x, Real y, Real z, Real);

OCTOTIGER_EXPORT std::vector<Real> rotating_star_a(Real x, Real y, Real z, Real);

#endif /* ROTATING_STAR_ROTATING_STAR_HPP_ */
