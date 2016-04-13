/*
 * polytrope.hpp
 *
 *  Created on: Feb 22, 2016
 *      Author: dmarce1
 */

#ifndef POLYTROPE_HPP_
#define POLYTROPE_HPP_

#include "defs.hpp"

class polytropic_eos {
private:
	real H0, D0;
	real n;
public:
	polytropic_eos() :
			H0(1.0), D0(1.0), n(1.5) {
	}
	void set_normal_enthalpy(real h) {
		H0 = h;
	}
	void set_normal_density(real d) {
		D0 = d;
	}
	real enthalpy_to_density(real h) {
		return D0 * std::pow(h / H0, n);
	}
	real density_to_enthalpy(real d) {
		return U0 * std::pow(d / D0, 1.0 / n);
	}
};

#endif /* POLYTROPE_HPP_ */
