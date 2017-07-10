/*
 * diagnostics.hpp
 *
 *  Created on: Apr 20, 2016
 *      Author: dmarce1
 */

#ifndef DIAGNOSTICS_HPP_
#define DIAGNOSTICS_HPP_

#include "defs.hpp"
#include "space_vector.hpp"
#include <vector>
#include <limits>
#include "real.hpp"
#include <taylor.hpp>

struct new_diagnostics_t {
	static constexpr integer nspec = 2;
	real omega;
	real m[nspec];
	real gt[nspec];
	real phi_eff_min[nspec];
	space_vector grid_com;
	space_vector com[nspec];
	space_vector com_dot[nspec];
	real js[nspec];
	real jorb;
	real rL[nspec];
	taylor<3> mom[nspec];
	integer stage;
	real tidal[nspec];
	real a;
	real virial;
	real virial_norm;
	real z_moment[nspec];
	real z_mom_orb;
	real rho_max[nspec];
	std::array<real,NF> grid_sum;
	new_diagnostics_t() {
		stage = 1;
		omega = -1.0;
		grid_com = 0.0;
		for( integer f = 0; f != NF; ++f) {
			grid_sum[f] = 0.0;
		}
		for( integer s = 0; s != nspec; ++s) {
			phi_eff_min[s] = std::numeric_limits<real>::max();
			m[s] = 0.0;
			com[s] = 0.0;
			com_dot[s] = 0.0;
			js[s] = 0.0;
			gt[s] = 0.0;
			mom[s] = 0.0;
			rL[s] = 0.0;
			tidal[s] = 0.0;
			z_moment[s] = 0.0;
			rho_max[s] = 0.0;
			}
		virial_norm = 0.0;
		z_mom_orb = 0.0;
		virial = 0.0;
		a = 0.0;
	}
	static inline real RL_radius(real q ) {
		const real q13 = std::pow(q,1.0/3.0);
		const real q23 = q13 * q13;
		const real n = 0.49 * q23;
		const real d = 0.6 * q23 + std::log(1.0 + q13);
		return n / d;
	}
	const new_diagnostics_t& compute();
	new_diagnostics_t& operator+=(const new_diagnostics_t& other) {
		for( integer f = 0; f != NF; ++f) {
			grid_sum[f] += other.grid_sum[f];
		}
		for( integer s = 0; s != nspec; ++s) {
			for( integer d = 0; d != nspec; ++d) {
				com[s][d] *= m[s];
				com[s][d] += other.com[s][d] * other.m[s];
				com_dot[s][d] *= m[s];
				com_dot[s][d] += other.com_dot[s][d] * other.m[s];
			}
			virial += other.virial;
			virial_norm += other.virial_norm;
			m[s] += other.m[s];
			gt[s] += other.gt[s];
			js[s] += other.js[s];
			rho_max[s] = std::max(rho_max[s], other.rho_max[s]);
			mom[s] += other.mom[s];
			for( integer d = 0; d != NDIM; ++d) {
				if( m[s] > 0.0 ) {
					com[s][d] /= m[s];
					com_dot[s][d] /= m[s];
				}
			}
		}
		return *this;
	}
	friend new_diagnostics_t operator+(const new_diagnostics_t& lhs, const new_diagnostics_t& rhs)
    {
        new_diagnostics_t res(lhs);
        return res += rhs;
    }
	new_diagnostics_t& operator=(const new_diagnostics_t& other) = default;

	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & grid_sum;
		arc & z_mom_orb;
		arc & grid_com;
		arc & virial_norm;
		arc & z_moment;
		arc & virial;
		arc & tidal;
		arc & a;
		arc & rL;
		arc & mom;
		arc & jorb;
		arc & js;
		arc & stage;
		arc & omega;
		arc & phi_eff_min;
		arc & m;
		arc & com;
		arc & com_dot;
		arc & gt;
	}

};

#endif /* DIAGNOSTICS_HPP_ */
