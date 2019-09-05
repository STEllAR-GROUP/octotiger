//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef DIAGNOSTICS_HPP_
#define DIAGNOSTICS_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"
#include "octotiger/real.hpp"
#include "octotiger/space_vector.hpp"
#include "octotiger/taylor.hpp"

#include <array>
#include <limits>
#include <vector>

struct diagnostics_t {
	static constexpr integer nspec = 2;
	bool failed;
	real l1_phi;
	real l2_phi;
	real l3_phi;
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
	real roche_vol[nspec];
	real stellar_vol[nspec];
	real virial;
	real virial_norm;
	real z_moment[nspec];
	real z_mom_orb;
	real rho_max[nspec];
	hydro_state_t<> grid_sum;
	hydro_state_t<> grid_out;
	std::array<real,NDIM> lsum;
	diagnostics_t() {
		failed = false;
		stage = 1;
		omega = -1.0;
		grid_com = 0.0;
		for( integer f = 0; f != opts().n_fields; ++f) {
			grid_sum[f] = 0.0;
			grid_out[f] = 0.0;
		}
		for( integer s = 0; s != nspec; ++s) {
			phi_eff_min[s] = std::numeric_limits<real>::max();
			m[s] = 0.0;
			roche_vol[s] = 0.0;
			stellar_vol[s] = 0.0;
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
		lsum[0] = lsum[1] = lsum[2] = 0.0;
		virial_norm = 0.0;
		z_mom_orb = 0.0;
		virial = 0.0;
		a = 0.0;
		l1_phi = -std::numeric_limits<real>::max();
		l2_phi = -std::numeric_limits<real>::max();
		l3_phi = -std::numeric_limits<real>::max();
	}
	static inline real RL_radius(real q ) {
		const real q13 = std::pow(q,1.0/3.0);
		const real q23 = q13 * q13;
		const real n = 0.49 * q23;
		const real d = 0.6 * q23 + std::log(1.0 + q13);
		return n / d;
	}
	const diagnostics_t& compute();
	diagnostics_t& operator+=(const diagnostics_t& other) {
		failed = failed || other.failed;
		l1_phi = std::max(l1_phi, other.l1_phi);
		l2_phi = std::max(l2_phi, other.l2_phi);
		l3_phi = std::max(l3_phi, other.l3_phi);
		for( integer f = 0; f != opts().n_fields; ++f) {
			grid_sum[f] += other.grid_sum[f];
			grid_out[f] += other.grid_out[f];
		}
		for( integer s = 0; s != nspec; ++s) {
			z_moment[s] += other.z_moment[s];
			for( integer d = 0; d != nspec; ++d) {
				com[s][d] *= m[s];
				com[s][d] += other.com[s][d] * other.m[s];
				com_dot[s][d] *= m[s];
				com_dot[s][d] += other.com_dot[s][d] * other.m[s];
			}
			roche_vol[s] += other.roche_vol[s];
			stellar_vol[s] += other.stellar_vol[s];
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
		lsum[0] += other.lsum[0];
		lsum[1] += other.lsum[1];
		lsum[2] += other.lsum[2];
		return *this;
	}
	friend diagnostics_t operator+(const diagnostics_t& lhs, const diagnostics_t& rhs)
    {
        diagnostics_t res(lhs);
        return res += rhs;
    }
	diagnostics_t& operator=(const diagnostics_t& other) = default;

	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & failed;
		arc & lsum;
		arc & l1_phi;
		arc & l2_phi;
		arc & l3_phi;
		arc & omega;
		arc & m;
		arc & gt;
		arc & phi_eff_min;
		arc & grid_com;
		arc & com;
		arc & com_dot;
		arc & js;
		arc & jorb;
		arc & rL;
		arc & mom;
		arc & stage;
		arc & tidal;
		arc & a;
		arc & roche_vol;
		arc & stellar_vol;
		arc & virial;
		arc & virial_norm;
		arc & z_moment;
		arc & z_mom_orb;
		arc & rho_max;
		arc & grid_sum;
		arc & grid_out;


	}

};

#endif /* DIAGNOSTICS_HPP_ */
