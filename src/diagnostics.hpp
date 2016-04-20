/*
 * diagnostics.hpp
 *
 *  Created on: Apr 20, 2016
 *      Author: dmarce1
 */

#ifndef DIAGNOSTICS_HPP_
#define DIAGNOSTICS_HPP_

#include "defs.hpp"

struct diagnostics_t {
	std::vector<real> grid_sum;
	std::vector<real> outflow_sum;
	std::vector<real> l_sum;
	std::vector<real> field_max;
	std::vector<real> field_min;
	std::vector<real> l1_error;
	std::vector<real> l2_error;
	real donor_mass;
	std::vector<real> gforce_sum;
	std::vector<real> gtorque_sum;
	diagnostics_t();
	diagnostics_t& operator+=(const diagnostics_t& other);
	diagnostics_t& operator=(const diagnostics_t& other) = default;

	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & gforce_sum;
		arc & gtorque_sum;
		arc & grid_sum;
		arc & outflow_sum;
		arc & l_sum;
		arc & field_max;
		arc & field_min;
		arc & donor_mass;
		arc & l1_error;
		arc & l2_error;
	}
};



#endif /* DIAGNOSTICS_HPP_ */
