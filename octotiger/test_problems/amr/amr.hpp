//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef amr_test_amr_test_HPP_
#define amr_test_amr_test_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/math/Real.hpp"

#include <vector>
#include <array>
#include "octotiger/defs.hpp"


OCTOTIGER_EXPORT std::vector<Real> amr_test(Real x, Real y, Real z, Real);

OCTOTIGER_EXPORT std::vector<Real> amr_test_a(Real x, Real y, Real z, Real);

OCTOTIGER_EXPORT Real amr_test_analytic(Real x, Real y, Real z);

OCTOTIGER_EXPORT bool refine_test_amr(integer level, integer max_level, Real x, Real y, Real z, std::vector<Real> const& U,
		std::array<std::vector<Real>, NDIM> const& dudx);

#endif /* amr_test_ROTATING_STAR_HPP_ */
