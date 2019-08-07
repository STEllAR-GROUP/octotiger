/*
 * util.hpp
 *
 *  Created on: Aug 5, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_UTI1L_HPP_
#define OCTOTIGER_UNITIGER_UTI1L_HPP_

template<int NDIM, int NX>
std::array<int, NDIM> index_to_dims(int i) {
	std::array<int, NDIM> dims;
	for (int j = 0; j < NDIM; j++) {
		dims[j] = i % NX;
		i /= NX;
	}
	return dims;
}

template<int a, int b>
static constexpr int int_pow() {
	if constexpr (b == 0) {
		return 1;
	} else {
		return a * int_pow<a, b - 1>();
	}
}

static inline void limit_slope(safe_real &ql, safe_real q0, safe_real &qr) {
	const safe_real tmp1 = qr - ql;
	const safe_real tmp2 = qr + ql;

	if (bool(qr < q0) != bool(q0 < ql)) {
		qr = ql = q0;
		return;
	}
	const safe_real tmp3 = tmp1 * tmp1 / 6.0;
	const safe_real tmp4 = tmp1 * (q0 - 0.5 * tmp2);
	if (tmp4 > tmp3) {
		ql = 3.0 * q0 - 2.0 * qr;
	} else if (-tmp3 > tmp4) {
		qr = 3.0 * q0 - 2.0 * ql;
	}
}

static inline safe_real minmod(safe_real a, safe_real b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

static inline safe_real minmod_theta(safe_real a, safe_real b, safe_real c) {
	return minmod(c * minmod(a, b), 0.5 * (a + b));
}

#endif /* OCTOTIGER_UNITIGER_UTIL_HPP_ */
