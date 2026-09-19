//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SRC_RADIATION_OPACITIES_HPP_
#define SRC_RADIATION_OPACITIES_HPP_

#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/math/Debug.hpp"

template<class U>
U temperature(U rho, U e, U mmw, Real gamma = 5.0 / 3.0) {
    FpeGuard fpeGuard{};
    // Skinner & Ostriker (2013), (2), (44), https://arxiv.org/abs/1306.0010:
    // T = (γ - 1)e μ/(ρ k_B), with their mean particle mass μ = mmw*m_h.
    // This is the EOS used in their ℰ_eq = αe⁴ thermal exchange relation;
    // e here excludes kinetic and cold-degenerate energy, and T is in kelvin.
    const U gm1 = U(gamma - 1);
    return gm1 * mmw * U(physcon().mh / physcon().kb) * e / expectPositive(rho);
}

template<class U>
U kappa_R(U rho, U e, U mmw, Real X, Real Z, Real gamma = 5.0 / 3.0) {
	FpeGuard fpeGuard{};
	if (opts().problem == MARSHAK) {
		return MARSHAK_OPAC;
	} else if (opts().problem == RADIATION_TEST) {
		return 1e-20;
	} else if (opts().problem == RADIATION_DIFFUSION) {
		return 1e2;
	} else if (opts().problem == RADIATION_COUPLING) {
		return 1;
	} else {
		const U T = temperature(rho, e, mmw, gamma);
		const U f1 = (T * T + U(2.7e+11) * rho);
		const U f2 = (U(1.0) + std::pow(T / U(4.5e+8), U(0.86)));
		const U k_ff_bf = U(4.0e+25) * (U(1) + X) * (Z + U(0.001)) * rho * POWER(SQRT(INVERSE(T)), U(7));
		const U k_T = (U(1.0) + X) * U(0.2) * T * T / (f1 * f2);
		const U k_tot = k_ff_bf + k_T;
		return rho * k_tot;
	}
}

template<class U>
U kappa_p(U rho, U e, U mmw, Real X, Real Z, Real gamma = 5.0 / 3.0) {
	FpeGuard fpeGuard{};
	if (opts().problem == MARSHAK) {
		return MARSHAK_OPAC;
	} else if (opts().problem == RADIATION_TEST) {
		return 1e-20;
	} else if (opts().problem == RADIATION_DIFFUSION) {
		return 1e2;
	} else if (opts().problem == RADIATION_COUPLING) {
		return 1e0;
	} else {
		const U T = temperature(rho, e, mmw, gamma);
		const U k_ff_bf = U(30.262) * U(4.0e+25) * (U(1) + X) * (Z + U(0.0001)) * rho * POWER(SQRT(INVERSE(T)), U(7));
		const U k_tot = k_ff_bf;
		return rho * k_tot;
	}
}

// Active M1 selectors. Keep legacy evaluation lazy: a thermal solve must not
// evaluate a Rosseland expression it previously never needed (and vice versa).
inline Real radiationAbsorption(Real rho, Real e, Real mmw, Real X, Real Z, Real gamma) {
    auto const& o = opts();
    if (o.radiationOpacity.model == "grey")
        return radiation::greyCoefficients(o.radiationOpacity, rho, o.code_to_g, o.code_to_cm).absorption;
    return o.rad_opacity >= 0 ? rho * o.rad_opacity : kappa_p(rho,e,mmw,X,Z,gamma);
}
inline Real radiationTransport(Real rho, Real e, Real mmw, Real X, Real Z, Real gamma) {
    auto const& o = opts();
    if (o.radiationOpacity.model == "grey")
        return radiation::greyCoefficients(o.radiationOpacity, rho, o.code_to_g, o.code_to_cm).transport;
    return o.rad_opacity >= 0 ? rho * o.rad_opacity : kappa_R(rho,e,mmw,X,Z,gamma);
}

template<class U>
U B_p(U rho, U e, U mmw, Real gamma = 5.0 / 3.0) {
	FpeGuard fpeGuard{};
	if (opts().problem == MARSHAK) {
		return U((physcon().c / 4.0 / M_PI)) * e;
	} else {
		const U T = temperature(rho, e, mmw, gamma);
		return (U(physcon().sigma) / U(M_PI)) * T * T * T * T;
	}
}

template<class U>
U dB_p_de(U rho, U e, U mmw, Real gamma = 5.0 / 3.0) {
	FpeGuard fpeGuard{};
	if (opts().problem == MARSHAK) {
		return U(physcon().c / (4.0 * M_PI));
	} else {
		if (e == U(0)) {
			return U(0);
		} else {
			return 4.0 * B_p(rho, e, mmw, gamma) / e;
		}
	}
}

#endif /* SRC_RADIATION_OPACITIES_HPP_ */
