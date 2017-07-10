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
	space_vector com[nspec];
	space_vector com_dot[nspec];
	space_vector cop[nspec];
	real js[nspec];
	real jorb;
	real rL[nspec];
	taylor<3> mom[nspec];
	integer stage;
	real tidal[nspec];
	real a;
	new_diagnostics_t() {
		stage = 1;
		omega = -1.0;
		for( integer s = 0; s != nspec; ++s) {
			phi_eff_min[s] = std::numeric_limits<real>::max();
			m[s] = 0.0;
			com[s] = 0.0;
			cop[s] = 0.0;
			com_dot[s] = 0.0;
			js[s] = 0.0;
			gt[s] = 0.0;
			mom[s] = 0.0;
			rL[s] = 0.0;
			tidal[s] = 0.0;
		}
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
		for( integer s = 0; s != nspec; ++s) {
			if( phi_eff_min[s] > other.phi_eff_min[s]) {
				cop[s] = other.cop[s];
				phi_eff_min[s] = other.phi_eff_min[s];
			}
			for( integer d = 0; d != nspec; ++d) {
				com[s][d] *= m[s];
				com[s][d] += other.com[s][d] * other.m[s];
				com_dot[s][d] *= m[s];
				com_dot[s][d] += other.com_dot[s][d] * other.m[s];
			}
			m[s] += other.m[s];
			gt[s] += other.gt[s];
			js[s] += other.js[s];
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
		arc & cop;
		arc & gt;
	}

};

struct diagnostics_t {
	std::vector<real> primary_sum;
	std::vector<real> secondary_sum;
	real z_moment;
	real primary_z_moment;
	real secondary_z_moment;
	std::vector<real> grid_sum;
	real primary_volume, secondary_volume;
	real roche_vol1, roche_vol2;
	space_vector primary_com;
	space_vector secondary_com;
	space_vector grid_com;
	space_vector primary_com_dot;
	space_vector secondary_com_dot;
	space_vector grid_com_dot;
	std::vector<real> outflow_sum;
	std::vector<real> l_sum;
	std::vector<real> field_max;
	std::vector<real> field_min;
	std::vector<real> l1_error;
	std::vector<real> l2_error;
	std::vector<real> gforce_sum;
	std::vector<real> gtorque_sum;
	std::pair<real,real> virial;
	diagnostics_t();
	diagnostics_t& operator+=(const diagnostics_t& other);
	friend diagnostics_t operator+(const diagnostics_t& lhs, const diagnostics_t& rhs)
    {
        diagnostics_t res(lhs);
        return res += rhs;
    }
	diagnostics_t& operator=(const diagnostics_t& other) = default;

	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & z_moment;
		arc & primary_z_moment;
		arc & secondary_z_moment;
		arc & primary_volume;
		arc & roche_vol1;
		arc & roche_vol2;
		arc & secondary_volume;
		arc & primary_com_dot;
		arc & secondary_com_dot;
		arc & grid_com_dot;
		arc & primary_com;
		arc & secondary_com;
		arc & grid_com;
		arc & gforce_sum;
		arc & gtorque_sum;
		arc & primary_sum;
		arc & secondary_sum;
		arc & grid_sum;
		arc & virial;
		arc & outflow_sum;
		arc & l_sum;
		arc & field_max;
		arc & field_min;
		arc & l1_error;
		arc & l2_error;
	}
};



#endif /* DIAGNOSTICS_HPP_ */
