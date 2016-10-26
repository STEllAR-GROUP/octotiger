/*
 * units.cpp
 *
 *  Created on: Oct 25, 2016
 *      Author: dmarce1
 */

#include "units.hpp"
#include "helmholtz.hpp"

static real L = 1.0;
static real T = 1.0;
static real M = 1.0;

real get_pressure_factor() {
	return M / (T * T * L);
}

real get_force_factor() {
	return M * L / (T * T);
}

real get_energy_factor() {
	return M * (L * L) / (T * T);
}

real get_length_factor() {
	return L;
}

void set_length_factor(real a) {
	L = a;
	helmholtz_set_cgs_units( L, M, T, 1.0 );
}

real get_mass_factor() {
	return M;
}

void set_mass_factor(real a) {
	M = a;
	helmholtz_set_cgs_units( L, M, T, 1.0 );
}

real get_time_factor() {
	return T;
}

void set_time_factor(real a) {
	T = a;
	helmholtz_set_cgs_units( L, M, T, 1.0 );
}
