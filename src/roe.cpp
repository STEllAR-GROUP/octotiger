/*
 * roe.cpp
 *
 *  Created on: May 28, 2015
 *      Author: dmarce1
 */

#include "roe.hpp"
#include "simd.hpp"
#include <cmath>
#include <cassert>

const integer con_i = rho_i;
const integer acl_i = sx_i;
const integer acr_i = sy_i;
const integer sh1_i = sz_i;
const integer sh2_i = egas_i;

real roe_fluxes(std::array<std::vector<real>, NF>& F, std::array<std::vector<real>, NF>& UL,
		std::array<std::vector<real>, NF>& UR,  const std::vector<space_vector>& X, real omega, integer dimension) {
	const std::size_t sz = UL[0].size();
	const integer u_i = vx_i + dimension;
	const integer v_i = vx_i + (dimension == XDIM ? YDIM : XDIM);
	const integer w_i = vx_i + (dimension == ZDIM ? YDIM : ZDIM);
	real max_lambda = real(0);
	integer this_simd_len;

	for (std::size_t iii = 0; iii < sz; iii += simd_len) {
		std::array<simd_vector, NF> ur;
		std::array<simd_vector, NF> ul;
		std::array<simd_vector, NDIM> vf;
		for (integer jjj = 0; jjj != simd_len; ++jjj) {
			const integer index = std::min(integer(iii + jjj), integer(sz - 1));
			for (integer field = 0; field != NF; ++field) {
				ur[field][jjj] = UR[field][index];
				ul[field][jjj] = UL[field][index];
			}
			vf[XDIM][jjj] = -omega * X[index][YDIM];
			vf[YDIM][jjj] = +omega * X[index][XDIM];
			vf[ZDIM][jjj] = ZERO;
		}
//		this_simd_len = std::min(integer(simd_len), integer(sz - iii));
		this_simd_len = simd_len;
		const simd_vector v_r0 = ur[u_i] / ur[rho_i];
		const simd_vector v_r = v_r0 - vf[u_i - vx_i];
		simd_vector ei_r = ur[egas_i] - HALF * (ur[u_i] * ur[u_i] + ur[v_i] * ur[v_i] + ur[w_i] * ur[w_i]) / ur[rho_i];

		for (integer j = 0; j != this_simd_len; ++j) {
			if (ei_r[j] < de_switch2 * ur[egas_i][j]) {
				ei_r[j] = std::pow(ur[tau_i][j], fgamma);
			}
		}

		const simd_vector p_r = (fgamma - ONE) * ei_r;
		const simd_vector c_r = sqrt(fgamma * p_r / ur[rho_i]);

		const simd_vector v_l0 = ul[u_i] / ul[rho_i];
		const simd_vector v_l = v_l0 - vf[u_i - vx_i];
		simd_vector ei_l = ul[egas_i] - HALF * (ul[u_i] * ul[u_i] + ul[v_i] * ul[v_i] + ul[w_i] * ul[w_i]) / ul[rho_i];

		for (integer j = 0; j != this_simd_len; ++j) {
			if (ei_l[j] < de_switch2 * ul[egas_i][j]) {
				ei_l[j] = std::pow(ul[tau_i][j], fgamma);
			}
		}

		const simd_vector p_l = (fgamma - ONE) * ei_l;
		const simd_vector c_l = sqrt(fgamma * p_l / ul[rho_i]);

		simd_vector a;
		a = max(abs(v_r) + c_r, abs(v_l) + c_l);

		std::array<simd_vector, NF> f;
		for (integer field = 0; field != NF; ++field) {
		//	if( field != rho0_i ) {
				f[field] = HALF * (ur[field] * v_r + ul[field] * v_l - a * (ur[field] - ul[field]));
	//		} else {
//				f[field] = HALF * (ur[field] * v_r0 + ul[field] * v_l0 - a * (ur[field] - ul[field]));
//			}
		}
		for( integer d = 0; d != NDIM; ++d ) {
			const integer field = zx_i + d;
			//		f[field] = -HALF * (a * (ur[field] - ul[field]));
				f[field] = -HALF * (a * (ur[field] - ul[field]));
		}
		f[u_i] += HALF * (p_r + p_l);
		f[egas_i] += HALF * (p_r * v_r0 + p_l * v_l0);

		for (integer field = 0; field != NF; ++field) {
			for (integer j = 0; j != simd_len && iii + j < sz; ++j) {
				F[field][iii + j] = f[field][j];
			}
		}
		max_lambda = std::max(max_lambda, a.max());
	}

	return max_lambda;
}
