/*
 * opacities.hpp
 *
 *  Created on: Sep 25, 2018
 *      Author: dmarce1
 */

#ifndef SRC_RADIATION_OPACITIES_HPP_
#define SRC_RADIATION_OPACITIES_HPP_

#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/safe_math.hpp"

template<class U>
U temperature(U rho, U e, U mmw) {
	constexpr U gm1 = U(2.0) / U(3.0);
	return std::pow((e * INVERSE(rho)), 1.0/4.0);
}

template<class U>
U kappa_R(U rho, U e, U mmw, real X, real Z) {
	if (opts().problem == MARSHAK) {
		return rho * 1.0e+1;
	} else {
		const U T = temperature(rho, e, mmw);
		const U f1 = (T * T + U(2.7e+11) * rho);
		const U f2 = (U(1.0) + std::pow(T / U(4.5e+8), U(0.86)));
		const U k_ff_bf = U(4.0e+25) * (U(1) + X) * (Z + U(0.001)) * rho * POWER(SQRT(INVERSE(T)), U(7));
		const U k_T = (U(1.0) + X) * U(0.2) * T * T / (f1 * f2);
		const U k_tot = k_ff_bf + k_T;
		return rho * k_tot;
	}
}

template<class U>
U kappa_p(U rho, U e, U mmw, real X, real Z) {
	if (opts().problem == MARSHAK) {
		return rho * 1.0e+1;
	} else {
		const U T = temperature(rho, e, mmw);
		const U k_ff_bf = U(30.262) * U(4.0e+25) * (U(1) + X) * (Z + U(0.0001)) * rho * POWER(SQRT(INVERSE(T)), U(7));
		const U k_tot = k_ff_bf;
		return rho * k_tot;
	}
}

template<class U>
U B_p(U rho, U e, U mmw) {
	if( opts().problem == MARSHAK ) {
		return  U((physcon().c/ 4.0 / M_PI )) * e;
	} else {
		const U T = temperature(rho, e, mmw);
		return (U(physcon().sigma) / U(M_PI)) * T * T * T * T;
	}
}

template<class U>
U dB_p_de(U rho, U e, U mmw) {
	if (opts().problem == MARSHAK) {
		return  U((physcon().c/4.0 * M_PI ));
	} else {
		if (e == U(0)) {
			return U(0);
		} else {
			return 4.0 * B_p(rho, e, mmw) / e;
		}
	}
}

#endif /* SRC_RADIATION_OPACITIES_HPP_ */
