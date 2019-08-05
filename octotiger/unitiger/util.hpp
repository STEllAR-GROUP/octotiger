/*
 * util.hpp
 *
 *  Created on: Aug 5, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_UTIL_HPP_
#define OCTOTIGER_UNITIGER_UTIL_HPP_

template<int NDIM, int H_NX>
std::vector<int> find_indices(int lb, int ub) {
	std::vector<int> I;
	constexpr int H_N3 = std::pow(H_NX, NDIM);
	for (int i = 0; i < H_N3; i++) {
		int k = i;
		bool interior = true;
		for (int dim = 0; dim < NDIM; dim++) {
			int this_i = k % H_NX;
			if (this_i < lb || this_i >= ub) {
				interior = false;
				break;
			} else {
				k /= H_NX;
			}
		}
		if (interior) {
			I.push_back(i);
		}
	}
	return I;
}
#endif /* OCTOTIGER_UNITIGER_UTIL_HPP_ */
