/*
 * roe.cpp
 *
 *  Created on: May 28, 2015
 *      Author: dmarce1
 */

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
    cs2 = std::max(((fgamma - 1.0) * ei / sqr(d)) * dp_depsilon + dp_drho, real(0));
    return sqrt(cs2);
}

real roe_fluxes(hydro_state_t<std::vector<real>>& F, hydro_state_t<std::vector<real>>& UL, hydro_state_t<std::vector<real>>& UR,
	const std::vector<space_vector>& X, real omega, integer dimension, real dx) {

	const real fgamma = grid::get_fgamma();
	const std::size_t sz = UL[0].size();
	const integer u_i = vx_i + dimension;
	const integer v_i = vx_i + (dimension == XDIM ? YDIM : XDIM);
	const integer w_i = vx_i + (dimension == ZDIM ? YDIM : ZDIM);
	real max_lambda = real(0);
	integer this_simd_len;

	for (std::size_t iii = 0; iii < sz; iii += simd_len) {
		hydro_state_t<simd_vector> ur;
		hydro_state_t<simd_vector> ul;
		std::array<simd_vector, NDIM> vf;
		for (integer jjj = 0; jjj != simd_len; ++jjj) {
			const integer index = std::min(integer(iii + jjj), integer(sz - 1));
			for (integer field = 0; field != opts().n_fields; ++field) {
				ur[field][jjj] = UR[field][index];
				ul[field][jjj] = UL[field][index];
			}
			vf[XDIM][jjj] = -omega * X[index][YDIM];
			vf[YDIM][jjj] = +omega * X[index][XDIM];
			vf[ZDIM][jjj] = ZERO;
		}
		this_simd_len = simd_len;
		const simd_vector v_r0 = ur[u_i] / ur[rho_i];
		const simd_vector v_r = v_r0 - vf[u_i - vx_i];
		simd_vector ei_r = ur[egas_i] - HALF * (ur[u_i] * ur[u_i] + ur[v_i] * ur[v_i] + ur[w_i] * ur[w_i]) / ur[rho_i];

		for (integer j = 0; j != this_simd_len; ++j) {
			if( opts().eos == WD) {
				ei_r[j] -= ztwd_energy(ur[rho_i][j]);
			}
			if (ei_r[j] < de_switch2 * ur[egas_i][j]) {
				ei_r[j] = std::pow(ur[tau_i][j], fgamma);
			}
		}

		simd_vector p_r = (fgamma - ONE) * ei_r;
		simd_vector c_r;
		if (opts().eos == WD) {
			for (integer j = 0; j != this_simd_len; ++j) {
				p_r[j] += ztwd_pressure(ur[rho_i][j]);
			}
			for (integer j = 0; j != this_simd_len; ++j) {
				c_r[j] = ztwd_sound_speed(ur[rho_i][j], ei_r[j]);
			}
		} else {
			c_r = sqrt(fgamma * p_r / ur[rho_i]);
		}

		const simd_vector v_l0 = ul[u_i] / ul[rho_i];
		const simd_vector v_l = v_l0 - vf[u_i - vx_i];
		simd_vector ei_l = ul[egas_i] - HALF * (ul[u_i] * ul[u_i] + ul[v_i] * ul[v_i] + ul[w_i] * ul[w_i]) / ul[rho_i];

		for (integer j = 0; j != this_simd_len; ++j) {
			if (opts().eos == WD) {
				ei_l[j] -= ztwd_energy(ul[rho_i][j]);
			}
			if (ei_l[j] < de_switch2 * ul[egas_i][j]) {
				ei_l[j] = std::pow(ul[tau_i][j], fgamma);
			}
		}

		simd_vector p_l = (fgamma - ONE) * ei_l;
		simd_vector c_l;
		if (opts().eos == WD) {
			for (integer j = 0; j != this_simd_len; ++j) {
				p_l[j] += ztwd_pressure(ul[rho_i][j]);
			}
			for (integer j = 0; j != this_simd_len; ++j) {
				c_l[j] = ztwd_sound_speed(ul[rho_i][j], ei_l[j]);
			}
		} else {
			c_l = sqrt(fgamma * p_l / ul[rho_i]);
		}

		const simd_vector a = max(abs(v_r) + c_r, abs(v_l) + c_l);

		hydro_state_t<simd_vector> f;
		for (integer field = 0; field != opts().n_fields; ++field) {
			f[field] = HALF * (ur[field] * v_r + ul[field] * v_l - a * (ur[field] - ul[field]));
		}
		f[u_i] += HALF * (p_r + p_l);
		f[egas_i] += HALF * (p_r * v_r0 + p_l * v_l0);

		for (integer field = 0; field != opts().n_fields; ++field) {
			for (integer j = 0; j != simd_len && iii + j < sz; ++j) {
				F[field][iii + j] = f[field][j];
			}
		}
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
        max_lambda = std::max(max_lambda, a.max());
#else
        using Vc::max;
        using std::max;
		max_lambda = std::max(max_lambda, Vc::reduce(a, [](auto x, auto y) { return (max)(x, y); }));
#endif

	}

	return max_lambda;
}
