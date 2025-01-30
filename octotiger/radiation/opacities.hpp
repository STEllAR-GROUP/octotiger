//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SRC_RADIATION_OPACITIES_HPP_
#define SRC_RADIATION_OPACITIES_HPP_

#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/safe_math.hpp"

template<class U>
U temperature(U rho, U e, U mmw) {
	const U gm1 = U(2.0) / U(3.0);
	return U(physcon().mh * mmw * e * INVERSE(rho * physcon().kb)) * gm1;
}

template<class U>
U kappa_R(U rho, U e, U mmw, U X, U Z) {
	if (opts().problem == MARSHAK) {
		return U(MARSHAK_OPAC);
	} else if (opts().problem == RADIATION_TEST) {
		return U(1e-20);
	} else if (opts().problem == RADIATION_DIFFUSION) {
		return rho;
	} else if (opts().problem == RADIATION_COUPLING) {
		return U(1e-2);
	} else if( opts().problem == STAR) {
		return rho * U(1e-12);
	} else {
		const U T = temperature(rho, e, mmw);
		const U f1 = (T * T + U(2.7e+11) * rho);
		const U f2 = U((U(1.0) + std::pow(T / U(4.5e+8), U(0.86))));
		const U k_ff_bf = U(4.0e+25) * (U(1) + X) * (Z + U(0.001)) * rho * POWER(SQRT(INVERSE(T)), U(7));
		const U k_T = (U(1.0) + X) * U(0.2) * T * T / (f1 * f2);
		const U k_tot = k_ff_bf + k_T;
		return rho * k_tot;
	}
}

template<class U>
U kappa_p(U rho, U e, U mmw, U X, U Z) {
	if (opts().problem == MARSHAK) {
		return U(MARSHAK_OPAC);
	} else if (opts().problem == RADIATION_TEST) {
		return U(1e-20);
	} else if (opts().problem == RADIATION_DIFFUSION) {
		return U(1e2);
	} else if (opts().problem == RADIATION_COUPLING) {
		return U(1e-2);
	} else if( opts().problem == STAR) {
		return rho * U(1e-12);
	} else {
		const U T = temperature(rho, e, mmw);
		const U k_ff_bf = U(U(30.262) * U(4.0e+25) * (U(1) + X) * (Z + U(0.0001)) * rho * U(pow(U(SQRT(INVERSE(T))), U(7))));
		const U k_tot = k_ff_bf;
		return rho * k_tot;
	}
}

template<class U>
U B_p(U rho, U e, U mmw) {
	if (opts().problem == MARSHAK) {
		return U((physcon().c / 4.0 / M_PI)) * e;
	} else {
		const U T = temperature(rho, e, mmw);
	//	printf( "- %e\n", T);
		return (U(physcon().sigma) / U(M_PI)) * T * T * T * T;
	}
}

template<class U>
U dB_p_de(U rho, U e, U mmw) {
	if (opts().problem == MARSHAK) {
		return U((physcon().c / 4.0 * M_PI));
	} else {
		if (e == U(0)) {
			return U(0);
		} else {
			return 4.0 * B_p(rho, e, mmw) / e;
		}
	}
}

#endif /* SRC_RADIATION_OPACITIES_HPP_ */
