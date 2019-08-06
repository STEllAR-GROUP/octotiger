/*
 * util.hpp
 *
 *  Created on: Aug 5, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_UTI1L_HPP_
#define OCTOTIGER_UNITIGER_UTI1L_HPP_


template<int a, int b>
constexpr int int_pow() {
	if constexpr (b == 0) {
		return 1;
	} else {
		return a * int_pow<a, b - 1>();
	}
}

template<int NDIM, int INX>
static std::vector<int> find_indices(int lb, int ub) {
	static const cell_geometry<NDIM,INX> geo;
	std::vector<int> I;
	for (int i = 0; i < geo::H_N3; i++) {
		int k = i;
		bool interior = true;
		for (int dim = 0; dim < NDIM; dim++) {
			int this_i = k % geo::H_NX;
			if (this_i < lb || this_i >= ub) {
				interior = false;
				break;
			} else {
				k /= geo::H_NX;
			}
		}
		if (interior) {
			I.push_back(i);
		}
	}
	return I;
}

#endif /* OCTOTIGER_UNITIGER_UTIL_HPP_ */
