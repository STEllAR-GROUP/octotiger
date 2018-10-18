/*
 * opacities.hpp
 *
 *  Created on: Sep 25, 2018
 *      Author: dmarce1
 */

#ifndef SRC_RADIATION_OPACITIES_HPP_
#define SRC_RADIATION_OPACITIES_HPP_

#include "../safe_math.hpp"
#include "../physcon.hpp"

template<class U>
U temperature(U rho, U e, U mmw) {
	constexpr U gm1 = U(2.0) / U(3.0);
	return (U(gm1) * U(mmw) * U(physcon.mh) / U(physcon.kb)) * (e * INVERSE(rho));
}

template<class U>
U kappa_R(U rho, U e, U mmw) {

	mmw = U(1.25);
	const U X = U(0.734);
	const U Y = U(0.250);
	const U Z = U(1) - X - Y;

	const U k_m = U(0.1) * Z;

	if (e <= 0.0) {
		return rho * k_m;
	}

	// TODO calculate realistic Z

	const U T = temperature(rho, e, mmw);

	const U f1 = (T * T + U(2.7e+11) * rho);
	const U f2 = (U(1.0) + std::pow(T / U(4.5e+8), U(0.86)));

	const U k_ff_bf = U(4.0e+25) * (U(1) + X) * (Z + U(0.0001)) * rho * POWER(SQRT(INVERSE(T)), U(7));

	const U k_T = (U(1.0) + X) * U(0.2) * T * T / (f1 * f2);

	const U k_hneg = U(1.1e-25) * std::sqrt(rho * Z) * POWER(T, U(7.7));

	const U k_tot = k_m + INVERSE((INVERSE(k_hneg) + INVERSE(k_ff_bf + k_T)));

	return rho * k_tot;

}

template<class U>
U kappa_p(U rho, U e, U mmw) {

	mmw = U(1.25);
	const U X = U(0.734);
	const U Y = U(0.250);
	const U Z = U(1) - X - Y;

	const U k_m = U(0.1) * Z;

	if (e <= 0.0) {
		return U(30.262) * rho * k_m;
	}

	// TODO calculate realistic Z

	const U T = temperature(rho, e, mmw);

	const U k_ff_bf = U(4.0e+25) * (U(1) + X) * (Z + U(0.0001)) * rho * POWER(SQRT(INVERSE(T)), U(7));

	const U k_hneg = U(1.1e-25) * std::sqrt(rho * Z) * POWER(T, U(7.7));

	const U k_tot = k_m + INVERSE((INVERSE(k_hneg) + INVERSE(k_ff_bf)));

	return U(30.262) * rho * k_tot;

}

template<class U>
U B_p(U rho, U e, U mmw) {
	const U T = temperature(rho, e, mmw);
	return (U(physcon.sigma) / U(M_PI)) * T * T * T * T;
}

template<class U>
U dB_p_de(U rho, U e, U mmw) {
	if (e == U(0)) {
		return U(0);
	} else {
		return 4.0 * B_p(rho, e, mmw) / e;
	}
}

#endif /* SRC_RADIATION_OPACITIES_HPP_ */
