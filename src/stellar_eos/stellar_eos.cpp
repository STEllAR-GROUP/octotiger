/*
 * stellar_eos.cpp
 *
 *  Created on: May 3, 2020
 *      Author: dmarce1
 */

#include "octotiger/stellar_eos/stellar_eos.hpp"
#include "octotiger/options.hpp"

stellar_eos::stellar_eos() {
	fgamma = 5.0 / 3.0;
	g_to_code = s_to_code = cm_to_code = 1.0;
	amu = 1.66053878283e-24;
	kb = 1.380650424e-16;
}

void stellar_eos::set_units(double g, double cm, double s) {
	g_to_code = g;
	cm_to_code = cm;
	s_to_code = s;
	amu = 1.66053878283e-24 / g_to_code;
	kb = 1.380650424e-16 / g_to_code / cm_to_code / cm_to_code * s_to_code * s_to_code;
}

void stellar_eos::set_fgamma(double fg) {
	fgamma = fg;
}

double stellar_eos::kappa_R(double rho, double T, double X, double Z) {
	if (opts().problem == MARSHAK) {
		return MARSHAK_OPAC;
	} else {
		const double f1 = (T * T + double(2.7e+11) * rho);
		const double f2 = (double(1.0) + std::pow(T / double(4.5e+8), double(0.86)));
		const double k_ff_bf = double(4.0e+25) * (double(1) + X) * (Z + double(0.001)) * rho * std::pow(std::sqrt(1.0 / T), double(7));
		const double k_T = (double(1.0) + X) * double(0.2) * T * T / (f1 * f2);
		const double k_tot = k_ff_bf + k_T;
		const double kr = rho * k_tot;
		return kr * cm_to_code;
	}
}

double stellar_eos::kappa_p(double rho, double T, double X, double Z) {
	if (opts().problem == MARSHAK) {
		return MARSHAK_OPAC;
	} else {
		const double k_ff_bf = double(30.262) * double(4.0e+25) * (double(1) + X) * (Z + double(0.0001)) * rho * std::pow(std::sqrt(1.0 / T), double(7));
		const double k_tot = k_ff_bf;
		const double kp = rho * k_tot;
		return kp * cm_to_code;
	}
}

double stellar_eos::B_p(double rho, double T) {
	if (opts().problem == MARSHAK) {
		constexpr auto c = 2.99792458e10;
		return double(c / 4.0 / M_PI) * 1.5 * T * rho * kb * 2.0 / amu;
	} else {
		constexpr auto sigma = 5.67051e-5;
		const double bp = (sigma / double(M_PI)) * T * T * T * T;
		return bp * s_to_code * s_to_code * s_to_code / g_to_code;
	}
}
