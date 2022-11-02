//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/roe.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/simd.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

const integer con_i = rho_i;
const integer acl_i = sx_i;
const integer acr_i = sy_i;
const integer sh1_i = sz_i;
const integer sh2_i = egas_i;

real ztwd_sound_speed(real d, real ei) {
	const real A = physcon().A;
	const real B = physcon().B;
	real x, dp_depsilon, dp_drho, cs2;
	const real fgamma = grid::get_fgamma();
	x = pow(d / B, 1.0 / 3.0);
	dp_drho = ((8.0 * A) / (3.0 * B)) * sqr(x) / sqrt(sqr(x) + 1.0) + (fgamma - 1.0) * ei / d;
	dp_depsilon = (fgamma - 1.0) * d;
	const real p = ztwd_pressure(d) + (fgamma - 1.0) * ei;
	cs2 = std::max((p / sqr(d)) * dp_depsilon + dp_drho, real(0));
	return sqrt(cs2);
}

