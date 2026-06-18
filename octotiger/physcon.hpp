//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SRC_PHYSCON444_HPP_
#define SRC_PHYSCON444_HPP_

#include "octotiger/math/Real.hpp"
#include "octotiger/options.hpp"

#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <initializer_list>
#include <vector>

template <class T = real>
struct specie_state_t : public std::vector<T> {
	specie_state_t() :
		std::vector<T>(opts().n_species, 0_R) {
	}
	specie_state_t(std::initializer_list<T> list) :
		std::vector<T>(list) {
	}
};

struct physcon_t {
	real A;
	real G;
	real B;
	real kb;
	real sigma;
	real c;
	real mh;
	real h;
	template <class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & A;
		arc & G;
		arc & B;
		arc & c;
		arc & sigma;
		arc & kb;
		arc & mh;
		arc & h;
	}
};

HPX_IS_BITWISE_SERIALIZABLE(physcon_t);

physcon_t &physcon();

void mean_ion_weight(const specie_state_t<> species, real &mmw, real &X, real &Z);
void set_AB(real, real);

void set_units(real m, real l, real t, real k);
real stellar_temp_from_rho_mu_s(real rho, real mu, real s);
real stellar_enthalpy_from_rho_mu_s(real rho, real mu, real s);
real stellar_rho_from_enthalpy_mu_s(real h, real mu, real s);
real find_T_rad_gas(real p, real rho, real mu);
real find_ei_rad_gas(real p, real rho, real mu, real gamma, real &T);

OCTOTIGER_EXPORT void normalize_constants();

void these_units(real &m, real &l, real &t, real &k);

void rad_coupling_vars(real rho, real e, real mmw, real &bp, real &kp, real &dkpde, real &dbde);

struct CodeToCgs {
	CodeToCgs() :
		K(1_R), cm(opts().code_to_cm), s(opts().code_to_s), g(opts().code_to_g) {
	}
	real time(real v = 1_R) const {
		return v * s;
	}
	real length(real v = 1_R) const {
		return v * cm;
	}
	real area(real v = 1_R) const {
		return v * sqr(length());
	}
	real volume(real v = 1_R) const {
		return v * area() * length();
	}
	real mass(real v = 1_R) const {
		return v * g;
	}
	real temperature(real v = 1_R) const {
		return v * K;
	}
	real velocity(real v = 1_R) const {
		return v * length() / time();
	}
	real acceleration(real v = 1_R) const {
		return v * length() / sqr(time());
	}
	real force(real v = 1_R) const {
		return v * mass() * acceleration();
	}
	real energy(real v = 1_R) const {
		return v * force() * length();
	}
	real power(real v = 1_R) const {
		return v * energy() / time();
	}
	real massDensity(real v = 1_R) const {
		return v * mass() / volume();
	}
	real specificEnergy(real v = 1_R) const {
		return v * energy() / mass();
	}
	real energyDensity(real v = 1_R) const {
		return v * energy() / volume();
	}
	real pressure(real v = 1_R) const {
		return v * energyDensity();
	}
	real flux(real v = 1_R) const {
		return v * velocity() * energyDensity();
	}
	real inverseLength(real v = 1_R) const {
		return v / length();
	}
	real opacity(real v = 1_R) const {
		return v * area() / mass();
	}

private:
	real K;
	real cm;
	real s;
	real g;
};

struct CgsToCode {
	CgsToCode() :
		K(1_R / 1_R), cm(1_R / opts().code_to_cm), s(1_R / opts().code_to_s), g(1_R / opts().code_to_g) {
	}
	real time(real v = 1_R) const {
		return v * s;
	}
	real length(real v = 1_R) const {
		return v * cm;
	}
	real area(real v = 1_R) const {
		return v * sqr(length());
	}
	real volume(real v = 1_R) const {
		return v * area() * length();
	}
	real mass(real v = 1_R) const {
		return v * g;
	}
	real temperature(real v = 1_R) const {
		return v * K;
	}
	real velocity(real v = 1_R) const {
		return v * length() / time();
	}
	real acceleration(real v = 1_R) const {
		return v * length() / sqr(time());
	}
	real force(real v = 1_R) const {
		return v * mass() * acceleration();
	}
	real energy(real v = 1_R) const {
		return v * force() * length();
	}
	real power(real v = 1_R) const {
		return v * energy() / time();
	}
	real massDensity(real v = 1_R) const {
		return v * mass() / volume();
	}
	real specificEnergy(real v = 1_R) const {
		return v * energy() / mass();
	}
	real energyDensity(real v = 1_R) const {
		return v * energy() / volume();
	}
	real pressure(real v = 1_R) const {
		return v * energyDensity();
	}
	real flux(real v = 1_R) const {
		return v * velocity() * energyDensity();
	}
	real inverseLength(real v = 1_R) const {
		return v / length();
	}
	real opacity(real v = 1_R) const {
		return v * area() / mass();
	}

private:
	real K;
	real cm;
	real s;
	real g;
};

#endif /* SRC_PHYSCON_HPP_ */
