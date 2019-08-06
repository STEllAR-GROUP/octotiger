/*
 * util.hpp
 *
 *  Created on: Aug 5, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_UTIL_HPP_
#define OCTOTIGER_UNITIGER_UTIL_HPP_

template<int NDIM, int INX>
std::vector<int> find_indices(int lb, int ub) {
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
