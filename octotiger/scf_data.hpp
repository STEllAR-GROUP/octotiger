//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SCF_DATA_HPP_
#define SCF_DATA_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/real.hpp"

#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

struct scf_data_t {
	real m_x;
	real m;
	real virial_sum;
	real virial_norm;
	real phiA;
	real phiB;
	real phiC;
	real entC;
	real donor_phi_min;
	real accretor_phi_min;
	real donor_phi_max;
	real accretor_phi_max;
	real donor_x;
	real accretor_x;
	real l1_phi;
	real l1_x;
	real accretor_mass;
	real donor_mass;
	real donor_central_enthalpy;
	real accretor_central_enthalpy;
	real donor_central_density;
	real accretor_central_density;
	real xA, xB, xC;
	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & xA;
		arc & xB;
		arc & xC;
		arc & m_x;
		arc & m;
		arc & virial_sum;
		arc & virial_norm;
		arc & phiA;
		arc & phiB;
		arc & phiC;
		arc & entC;
		arc & donor_phi_min;
		arc & accretor_phi_min;
		arc & donor_phi_max;
		arc & accretor_phi_max;
		arc & donor_x;
		arc & accretor_x;
		arc & l1_phi;
		arc & l1_x;
		arc & accretor_mass;
		arc & donor_mass;
		arc & donor_central_enthalpy;
		arc & accretor_central_enthalpy;
		arc & donor_central_density;
		arc & accretor_central_density;
	}
	scf_data_t();
	void accumulate(const scf_data_t& other);
};

HPX_IS_BITWISE_SERIALIZABLE(scf_data_t);


#endif /* SCF_DATA_HPP_ */
