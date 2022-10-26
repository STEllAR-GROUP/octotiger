//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef DIAGNOSTICS_HPP_
#define DIAGNOSTICS_HPP_

#include "octotiger/unitiger/safe_real.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"
#include "octotiger/space_vector.hpp"
#include "octotiger/taylor.hpp"

#include "safe_math.hpp"

#include <array>
#include <limits>
#include <vector>

struct diagnostics_t {
	static constexpr integer nspec = 2;
	bool failed;
	integer stage;
	safe_real a;
	safe_real l1_phi;
	safe_real l2_phi;
	safe_real l3_phi;
	safe_real omega;
	safe_real Torb;
	safe_real jorb;
	safe_real virial;
	safe_real virial_norm;
	safe_real z_mom_orb;
	safe_real munbound1;
	safe_real munbound2;
	space_vector grid_com;
	oct::array<safe_real, nspec> m;
	oct::array<safe_real, nspec> Ts;
	oct::array<safe_real, nspec> phi_eff_min;
	oct::array<safe_real, nspec> js;
	oct::array<safe_real, nspec> ekin;
	oct::array<safe_real, nspec> epot;
	oct::array<safe_real, nspec> eint;
	oct::array<safe_real, nspec> lz1;
	oct::array<safe_real, nspec> lz2;
	oct::array<safe_real, nspec> rL;
	oct::array<safe_real, nspec> tidal;
	oct::array<safe_real, nspec> roche_vol;
	oct::array<safe_real, nspec> stellar_vol;
	oct::array<safe_real, nspec> z_moment;
	oct::array<safe_real, nspec> rho_max;
	oct::array<space_vector, nspec> g;
	oct::array<space_vector, nspec> com;
	oct::array<space_vector, nspec> com_dot;
	oct::array<taylor<3>, nspec> mom;
	hydro_state_t<> grid_sum;
	hydro_state_t<> grid_out;
	oct::array<safe_real, NDIM> lsum;
	safe_real nonvacj;
	safe_real nonvacjlz;
	diagnostics_t() {
		failed = false;
		stage = 1;
		omega = -1.0;
		grid_com = 0.0;
		munbound1 = 0.0;
		munbound2 = 0.0;
		for (integer f = 0; f != opts().n_fields; ++f) {
			grid_sum[f] = 0.0;
			grid_out[f] = 0.0;
		}
		for (integer s = 0; s != nspec; ++s) {
			phi_eff_min[s] = std::numeric_limits<safe_real>::max();
			m[s] = 0.0;
			roche_vol[s] = 0.0;
			stellar_vol[s] = 0.0;
			com[s] = 0.0;
			com_dot[s] = 0.0;
			lz1[s] = 0.0;
			lz2[s] = 0.0;
			ekin[s] = 0.0;
			epot[s] = 0.0;
			eint[s] = 0.0;
			js[s] = 0.0;
			Ts[s] = 0.0;
			g[s] = 0.0;
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
		nonvacj = 0.0;
		nonvacjlz = 0.0;
		l1_phi = -std::numeric_limits<safe_real>::max();
		l2_phi = -std::numeric_limits<safe_real>::max();
		l3_phi = -std::numeric_limits<safe_real>::max();
	}
	static inline safe_real RL_radius(safe_real q) {
		const safe_real q13 = std::pow(q, 1.0 / 3.0);
		const safe_real q23 = q13 * q13;
		const safe_real n = 0.49 * q23;
		const safe_real d = 0.6 * q23 + std::log(1.0 + q13);
		return n / d;
	}
	const diagnostics_t& compute();
	diagnostics_t& operator+=(const diagnostics_t &other) {
		failed = failed || other.failed;
		if (opts().problem == DWD) {
			l1_phi = std::max(l1_phi, other.l1_phi);
			l2_phi = std::max(l2_phi, other.l2_phi);
			l3_phi = std::max(l3_phi, other.l3_phi);
		}
		for (integer f = 0; f != opts().n_fields; ++f) {
			grid_sum[f] += other.grid_sum[f];
			grid_out[f] += other.grid_out[f];
		}
		for (integer s = 0; s != nspec; ++s) {
			z_moment[s] += other.z_moment[s];
			for (integer d = 0; d < NDIM; ++d) {
				com[s][d] *= m[s];
				com[s][d] += other.com[s][d] * other.m[s];
				com_dot[s][d] *= m[s];
				com_dot[s][d] += other.com_dot[s][d] * other.m[s];
			}
			if (opts().problem == DWD) {
				roche_vol[s] += other.roche_vol[s];
				stellar_vol[s] += other.stellar_vol[s];
				virial += other.virial;
				virial_norm += other.virial_norm;
				m[s] += other.m[s];
				Ts[s] += other.Ts[s];
				g[s] += other.g[s];
				ekin[s] += other.ekin[s];
				epot[s] += other.epot[s];
				eint[s] += other.eint[s];
				lz1[s] += other.lz1[s];
				lz2[s] += other.lz2[s];
				js[s] += other.js[s];
				rho_max[s] = std::max(rho_max[s], other.rho_max[s]);
				mom[s] += other.mom[s];
				for (integer d = 0; d < NDIM; ++d) {
					if (m[s] > std::numeric_limits<double>::min()) {
						com[s][d] = com[s][d] * INVERSE(m[s]);
						com_dot[s][d] = com_dot[s][d] * INVERSE(m[s]);
					}
				}
			}
		}
		munbound1 += other.munbound1;
		munbound2 += other.munbound2;
		lsum[0] += other.lsum[0];
		lsum[1] += other.lsum[1];
		lsum[2] += other.lsum[2];
		nonvacj += other.nonvacj;
		nonvacjlz += other.nonvacjlz;
		return *this;
	}
	friend diagnostics_t operator+(const diagnostics_t &lhs, const diagnostics_t &rhs) {
		diagnostics_t res(lhs);
		return res += rhs;
	}
	diagnostics_t& operator=(const diagnostics_t &other) = default;

	template<class Arc>
	void serialize(Arc &arc, const unsigned) {
		arc & munbound1;
		arc & munbound2;
		arc & ekin;
		arc & epot;
		arc & eint;
		arc & failed;
		arc & lsum;
		arc & l1_phi;
		arc & l2_phi;
		arc & l3_phi;
		arc & Torb;
		arc & omega;
		arc & m;
		arc & g;
		arc & Ts;
		arc & phi_eff_min;
		arc & grid_com;
		arc & com;
		arc & com_dot;
		arc & lz1;
		arc & lz2;
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
