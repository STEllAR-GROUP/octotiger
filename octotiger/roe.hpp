//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef ROE_HPP_
#define ROE_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/math/Debug.hpp"
#include "octotiger/space_vector.hpp"

#include <algorithm>
#include <vector>

#define de_switch1 (opts().dual_energy_sw1)
#define de_switch2 (opts().dual_energy_sw2)

Real roe_fluxes(hydro_state_t<std::vector<Real>> &F, hydro_state_t<std::vector<Real>> &UL, hydro_state_t<std::vector<Real>> &UR,
		const std::vector<space_vector> &X, Real omega, integer dimension, Real dx);

inline Real ztwd_pressure(Real d, Real A = physcon().A, Real B = physcon().B) {
	const Real x = POWER(d / B, 1.0 / 3.0);
	Real p;
	if (x < 0.01) {
		p = 1.6 * A * POWER(x, 5);
	} else {
		p = A * (x * (2.0 * x * x - 3.0) * SQRT(x * x + 1.0) + 3.0 * asinh(x));
	}
	return p;
}

inline Real ipr_pressure(Real t, Real rho, Real mu) {
        const Real cg = physcon().kb / (mu * physcon().mh);
        const Real cr = (4.0 * physcon().sigma) / (3.0 * physcon().c);
        return cg * rho * t + cr * POWER(t, 4);
}

inline Real ztwd_enthalpy(Real d, Real A = physcon().A, Real B = physcon().B) {
#ifndef NDEBUG
	if (d < 0.0) {
		printf("d = %e in ztwd_enthalpy\n", d);
		abort();
	}
#endif
	const Real x = pow(d / B, 1.0 / 3.0);
	Real h;
	if (x < 0.01) {
		h = 4.0 * A / B * x*x;
	} else {
		h = 8.0 * A / B * (sqrt(x*x + 1.0) - 1.0);
	}
	return h;
}

OCTOTIGER_FORCEINLINE Real ztwd_energy(Real d, Real A = physcon().A, Real B = physcon().B) {
	const Real x = pow(d / B, 1.0 / 3.0);
	if (x < 0.01) {
		return 2.4 * A * POWER(x, 5);
	} else {
		return std::max(ztwd_enthalpy(d) * d - ztwd_pressure(d), Real(0));

	}
}

Real ztwd_sound_speed(Real d, Real ei);

#endif /* ROE_HPP_ */
