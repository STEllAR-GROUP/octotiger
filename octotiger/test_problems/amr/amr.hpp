//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef amr_test_amr_test_HPP_
#define amr_test_amr_test_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/real.hpp"

#include <vector>
#include <array>
#include "octotiger/defs.hpp"


OCTOTIGER_EXPORT oct::vector<real> amr_test(real x, real y, real z, real);

OCTOTIGER_EXPORT oct::vector<real> amr_test_a(real x, real y, real z, real);

OCTOTIGER_EXPORT real amr_test_analytic(real x, real y, real z);

OCTOTIGER_EXPORT bool refine_test_amr(integer level, integer max_level, real x, real y, real z, oct::vector<real> const& U,
		oct::array<oct::vector<real>, NDIM> const& dudx);

#endif /* amr_test_ROTATING_STAR_HPP_ */
