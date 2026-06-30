////  Copyright (c) 2019 AUTHORS
////
////  Distributed under the Boost Software License, Version 1.0. (See accompanying
////  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//#ifndef SRC_RADIATION_OPACITIES_HPP_
//#define SRC_RADIATION_OPACITIES_HPP_
//
//#include "octotiger/grid.hpp"
//#include "octotiger/options.hpp"
//#include "octotiger/physcon.hpp"
//#include "octotiger/safe_math.hpp"
//
//template <class U>
//U temperature(U rho, U e, U mmw) {
//	constexpr U gm1 = U(2.0) / U(3.0);
//	return std::pow((e * INVERSE(rho)), 1.0 / 4.0);
//}
//
//struct Opacities {
//	Opacities() {
//		using std::pow;
//		CgsToCode const convert;
//		auto const kB = physcon().kb;
//		auto const mH = physcon().mh;
//		auto const gamma = opts().gas_gamma;
//		kappa_rho_exp = opts().kappa_rho_exp;
//		kappa_T_exp = opts().kappa_T_exp;
//		kappa0 = convert.inverseLength(opts().kappa0);
//		kappa0 *= pow(convert.massDensity(), -kappa_rho_exp);
//		kappa0 *= pow(convert.temperature(), -kappa_T_exp);
//		sigma_rho_exp = opts().sigma_rho_exp;
//		sigma_T_exp = opts().sigma_T_exp;
//		sigma0 = convert.inverseLength(opts().sigma0);
//		sigma0 *= pow(convert.massDensity(), -sigma_rho_exp);
//		sigma0 *= pow(convert.temperature(), -sigma_T_exp);
//		ei2Tcon = mH * inv((gamma - 1_R) * kB);
//	}
//	auto operator()(real rho, real T) const {
//		using std::pow;
//		auto const absorption = kappa0 * pow(rho, kappa_rho_exp) * pow(T, kappa_T_exp);
//		auto const scattering = sigma0 * pow(rho, sigma_rho_exp) * pow(T, sigma_T_exp);
//		return std::pair(absorption, scattering);
//	}
//	auto operator()(real rho, real ei, real mmw) const {
//		return operator()(rho, ei2Tcon * ei * mmw / rho);
//	}
//	std::string absorptionExpression(std::string const &rho = "rho", std::string const &T = "T") const {
//		return hpx::util::format("({:e}) * ({})^{:e} * ({})^{:e}", opts().kappa0, rho, opts().kappa_rho_exp, T, opts().kappa_T_exp);
//	}
//
//	std::string scatteringExpression(std::string const &rho = "rho", std::string const &T = "T") const {
//		return hpx::util::format("({:e}) * ({})^{:e} * ({})^{:e}", opts().sigma0, rho, opts().sigma_rho_exp, T, opts().sigma_T_exp);
//	}
//	std::string extinctionExpression(std::string const &rho = "rho", std::string const &T = "T") const {
//		return "(" + absorptionExpression(rho, T) + ") + (" + scatteringExpression(rho, T) + ")";
//	}
//
//private:
//	real kappa0;
//	real kappa_T_exp;
//	real kappa_rho_exp;
//	real sigma0;
//	real sigma_T_exp;
//	real sigma_rho_exp;
//	real ei2Tcon;
//};
//
////	if (opts().problem == MARSHAK) {
////		return MARSHAK_OPAC;
////	} else if (opts().problem == RADIATION_TEST) {
////		return 1e-2;
////	} else if (opts().problem == RADIATION_DIFFUSION) {
////		return 1e2;
////	} else if (opts().problem == RADIATION_COUPLING) {
////		return 1;
////	} else {
////		const U T = temperature(rho, e, mmw);
////		const U f1 = (T * T + U(2.7e+11) * rho);
////		const U f2 = (U(1.0) + std::pow(T / U(4.5e+8), U(0.86)));
////		const U k_ff_bf = U(4.0e+25) * (U(1) + X) * (Z + U(0.001)) * rho * POWER(SQRT(INVERSE(T)), U(7));
////		const U k_T = (U(1.0) + X) * U(0.2) * T * T / (f1 * f2);
////		const U k_tot = k_ff_bf + k_T;
////		return rho * k_tot;
////	}
//
////	if (opts().problem == MARSHAK) {
////		return MARSHAK_OPAC;
////	} else if (opts().problem == RADIATION_TEST) {
////		return 1e-20;
////	} else if (opts().problem == RADIATION_DIFFUSION) {
////		return 1e2;
////	} else if (opts().problem == RADIATION_COUPLING) {
////		return 1e0;
////	} else {
////		const U T = temperature(rho, e, mmw);
////		const U k_ff_bf = U(30.262) * U(4.0e+25) * (U(1) + X) * (Z + U(0.0001)) * rho * POWER(SQRT(INVERSE(T)), U(7));
////		const U k_tot = k_ff_bf;
////		return rho * k_tot;
////	}
//
//template <class U>
//U B_p(U rho, U e, U mmw) {
//	if (opts().problem == MARSHAK) {
//		return U((physcon().c / 4.0 / M_PI)) * e;
//	} else {
//		const U T = temperature(rho, e, mmw);
//		return (U(physcon().sigma) / U(M_PI)) * T * T * T * T;
//	}
//}
//
//template <class U>
//U dB_p_de(U rho, U e, U mmw) {
//	if (opts().problem == MARSHAK) {
//		return U((physcon().c / 4.0 * M_PI));
//	} else {
//		if (e == U(0)) {
//			return U(0);
//		} else {
//			return 4.0 * B_p(rho, e, mmw) / e;
//		}
//	}
//}
//
//#endif /* SRC_RADIATION_OPACITIES_HPP_ */
