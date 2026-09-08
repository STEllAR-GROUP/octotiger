//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include "octotiger/grid.hpp"

#include <hpx/runtime/threads/run_as_os_thread.hpp>

#include "octotiger/test_problems/blast.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>


#if defined(OCTOTIGER_HAVE_BLAST_TEST)

#if !defined(OCTOTIGER_HAVE_BOOST_MULTIPRECISION)
#include <quadmath.h>
using sed_Real = __float128;
#else
#include <boost/multiprecision/cpp_bin_float.hpp>
using sed_Real = boost::multiprecision::cpp_bin_float_quad;
#endif

/*extern "C" {*/
/* Subroutine */int sed_1d__(sed_Real *time, int *nstep, sed_Real *xpos, sed_Real *eblast, sed_Real *omega_in__, sed_Real *xgeom_in__, sed_Real *rho0,
		sed_Real *vel0, sed_Real *ener0, sed_Real *pres0, sed_Real *cs0, sed_Real *gam0, sed_Real *den, sed_Real *ener, sed_Real *pres, sed_Real *vel,
		sed_Real *cs);
//}

#endif

constexpr Real blast_wave_t0 = 7e-4;

#if defined(OCTOTIGER_HAVE_BLAST_TEST)
std::vector<Real> blast_wave_analytic(Real x, Real y, Real z, Real t) {
	static const auto dxmin = 2.0 * opts().xscale / INX / double(1 << opts().max_level);
	Real r = std::sqrt(x * x + y * y + z * z);
	r = std::max(r, dxmin * 1.0e-3);
	t += blast_wave_t0;
	Real rmax = 3.0 * opts().xscale;
	Real d, v, p;
	sedov::solution(t, r, rmax, d, v, p);
	std::vector<Real> u(opts().n_fields, 0.0);
	u[rho_i] = u[spc_i] = std::max(d, 1.0e-20);
	Real s = d * v;
	u[sx_i] = s * x / r;
	u[sy_i] = s * y / r;
	u[sz_i] = s * z / r;
	Real e = std::max(p / (grid::get_fgamma() - 1), 1.0e-20);
	u[egas_i] = e + s * v * 0.5;
	u[tau_i] = std::pow(e, 1 / grid::get_fgamma());
	return u;
}
#endif

std::vector<Real> blast_wave(Real x, Real y, Real z, Real dx) {
	std::vector<Real> u(opts().n_fields, ZERO);
	u[rho_i] = u[spc_i] = 1.0;
	const auto r2 = x * x + y * y + z * z;
	const auto rmax = dx * 3.5;
	if (r2 < rmax * rmax) {
		u[egas_i] = opts().eblast0 / (dx * dx * dx)  / 160.0;
	} else {
		u[egas_i] = 1.0e-20;
	}
	u[tau_i] = std::pow(u[egas_i], 1.0 / grid::get_fgamma());
	return u;

}
