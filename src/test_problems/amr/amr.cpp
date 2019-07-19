/*
 * amr_test.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: dmarce1
 */

#include "octotiger/test_problems/amr/amr.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>


std::vector<real> amr_test(real x, real y, real z, real) {
	std::vector<real> u(opts().n_fields, real(0));
	u[rho_i] = u[spc_i] = 1 + x;
	return u;
}

std::vector<real> amr_test_a(real x, real y, real z, real) {
	return amr_test(x, y, z, 0);
}
