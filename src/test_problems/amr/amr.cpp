//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/test_problems/amr/amr.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>


bool refine_test_amr(integer level, integer max_level, real x, real y, real z, oct::vector<real> const& U,
		oct::array<oct::vector<real>, NDIM> const& dudx) {
	if (level >= max_level) {
		return false;
	} else {
		if (x > 1.0e-20) {
			return true;
		} else {
			return false;
		}
	}

}


real amr_test_analytic(real x, real y, real z) {
	return y;
}

oct::vector<real> amr_test(real x, real y, real z, real) {
	oct::vector<real> u(opts().n_fields, real(0));
	u[rho_i] = u[spc_i] = amr_test_analytic(x,y,z);
	return u;
}
