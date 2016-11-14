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
		arc & outflow_sum;
		arc & l_sum;
		arc & field_max;
		arc & field_min;
		arc & l1_error;
		arc & l2_error;
	}
};



#endif /* DIAGNOSTICS_HPP_ */
