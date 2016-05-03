/*
 * eos.cpp
 *
 *  Created on: Feb 22, 2016
 *      Author: dmarce1
 */

#include "eos.hpp"

real eos::dhdot_dr(real h, real hdot, real r) const {
	real a, b, d;
	d = this->enthalpy_to_density(h);
	if (r != 0.0) {
		a = 2.0 * hdot / r;
	} else {
		a = 0.0;
	}
	b = 4.0 * M_PI * G * d;
	return -(a + b);
}

real eos::dh_dr(real h, real hdot, real r) const {
	return hdot;
}

eos::eos() :
		M0(2.0), R0(1.0) {
}

void eos::initialize(real& mass, real& radius) {

	const real dr0 = R0 / 1000.0;

	real h, hdot, r, m;
	h = h0();
	hdot = 0.0;
	r = 0.0;
	m = 0.0;
	real dr = dr0;
	real d;
	do {
		if (hdot != 0.0) {
			dr = std::max(std::min(dr0, std::abs(h / hdot) / 2.0),
					dr0 * 1.0e-6);
		}
		d = this->enthalpy_to_density(h);
		//	printf("%e %e %e %e %e\n", r, m, h, d, dr);
		const real dh1 = dh_dr(h, hdot, r) * dr;
		const real dhdot1 = dhdot_dr(h, hdot, r) * dr;
		const real dm1 = 4.0 * M_PI * d * std::pow(r, 2) * dr;
		if (h + dh1 <= ZERO) {
			break;
		}
		d = this->enthalpy_to_density(h + dh1);
		const real dh2 = dh_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		const real dhdot2 = dhdot_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		const real dm2 = 4.0 * M_PI * d * std::pow(r + dr, 2) * dr;
		h += (dh1 + dh2) / 2.0;
		hdot += (dhdot1 + dhdot2) / 2.0;
		r += dr;
		m += (dm1 + dm2) / 2.0;
	} while (h > 0.0);
	mass = m;
	radius = r;
}

real polytropic_eos::enthalpy_to_density(real h) const {
	return d0() * std::pow(h / h0(), n);
}

real polytropic_eos::density_to_enthalpy(real d) const {
	return h0() * std::pow(d / d0(), 1.0 / n);
}

real eos::density_at(real R) const {
	const integer N = 32;
//	const real dx = std::sqrt(h0()) / (G * d0()) / 100.0;
	const real dr = R / N;

	real h, hdot, r;
	h = h0();
	hdot = 0.0;
	for (integer i = 0; i < N; ++i) {
		//	printf("%e %e %e\n", r, h, dr);
		r = i * dr;
		const real dh1 = dh_dr(h, hdot, r) * dr;
		const real dhdot1 = dhdot_dr(h, hdot, r) * dr;
		if (h + dh1 <= ZERO) {
			//	printf( "\n");
			return 0.0;
		}
		const real dh2 = dh_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		const real dhdot2 = dhdot_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		h += (dh1 + dh2) / 2.0;
		hdot += (dhdot1 + dhdot2) / 2.0;
		if (h <= ZERO) {
			//		printf( "\n");
			return 0.0;
		}
	}
	real d = this->enthalpy_to_density(h);
//	printf( "%d\n", d);
	return d;
}
