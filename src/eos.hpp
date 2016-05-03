/*
 * polytrope.hpp
 *
 *  Created on: Feb 22, 2016
 *      Author: dmarce1
 */

#ifndef POLYTROPE_HPP_
#define POLYTROPE_HPP_

#include "defs.hpp"

class eos {
public:
	real M0, R0;
private:
	static constexpr real G = 1.0;
protected:
	real dhdot_dr(real h, real hdot, real r) const;
	real dh_dr(real h, real hdot, real r) const;
public:
	real m0() const {
		return M0;
	}
	real r0() const {
		return R0;
	}
	real& m0() {
		return M0;
	}
	real& r0() {
		return R0;
	}
	real d0() const {
		return M0 / (R0 * R0 * R0);
	}

	real h0() const {
		return G * M0 / R0;
	}
	void set_h0(real h) {
		const real d = d0();
		R0 = std::sqrt(h / (G * d));
		M0 = h * R0 / G;
	}
	void set_d0(real d) {
		R0 = std::sqrt(h0() / d / G);
		M0 = R0 * R0 * R0 * d;
	}
	void initialize(real&, real&);
	eos();
	virtual real enthalpy_to_density(real h) const = 0;
	virtual real density_to_enthalpy(real d) const = 0;
	virtual real pressure(real d) const = 0;
	virtual ~eos() = default;
	real density_at(real) const;
};

class cwd_eos: public eos {

private:
	real _x0;
	real A() const {
		const real a = d0() * h0() / (8.0 * std::pow(x0(), 3));
		real b;
		if (x0() < 1.0e-4) {
			b = 0.5 * x0() * x0();
		} else {
			b = std::sqrt(x0() * x0() + 1.0) - 1.0;
		}
		return a / b;
	}
	real B() const {
		return d0() * std::pow(x0(), -3);
	}
	real rho2x(real d) const {
		return std::pow(d / B(), 1.0 / 3.0);
	}
	real x2rho(real x) const {
		return B() * x * x * x;
	}
	static real asing(real z) {
		return std::log(z + std::sqrt(z * z + 1.0));
	}
public:
	real& x0() {
		return _x0;
	}
	real x0() const {
		return _x0;
	}
	cwd_eos() :
			eos(), _x0(0.01) {
	}
	real enthalpy_to_density(real h) const {
		const real x = std::sqrt((std::pow(h * (B() / (8.0 * A())) + 1.0, 2) - 1.0));
		return x2rho(x);
	}
	real density_to_enthalpy(real d) const {
		const real x = rho2x(d);
		return (8.0 * A() / B()) * (std::sqrt(x * x + 1.0) - 1.0);
	}
	real pressure(real d) const {
		const real x = rho2x(d);
		return A() * (x * (2.0 * x * x - 3.0) * std::sqrt(x * x + 1.0) + 3.0 * asinh(x));
	}

};

class polytropic_eos: public eos {
private:
	real n;
public:
	polytropic_eos(real M, real R) :
			n(1.5) {
		real m, r;
		initialize(m, r);
		m0() *= M / m;
		r0() *= R / r;
	}
	virtual ~polytropic_eos() = default;
	virtual real enthalpy_to_density(real h) const;
	virtual real density_to_enthalpy(real d) const;
	virtual real pressure(real d) const {
		const real h = density_to_enthalpy(d);
		return h * d / (n + 1.0);
	}
	;
};

class bipolytropic_eos: public eos {
private:
	real n_C, n_E;
	real f_C, f_E;

	real dC() const {
		return f_C * d0();
	}

public:
	real dE() const {
		return f_E * d0();
	}
	real s0() const {
		return std::pow(P0()/(fgamma-1.0),1.0/fgamma) / dE();
	}
	real P0() const {
		real den = ((1.0 + n_C) / dC()) * std::pow(d0() / dC(), 1.0 / n_C) - (1.0 + n_C) / dC() + (1.0 + n_E) / dE();
		return h0() / den;
	}
	bipolytropic_eos(real M, real R, real _n_C, real _n_E, real _f_C, real _f_E) :
			n_C(_n_C), n_E(_n_E), f_C(_f_C), f_E(_f_E) {
		real m, r;
		initialize(m, r);
		m0() *= M / m;
		r0() *= R / r;
	}
/*	real get_bden() const {
		return f_C * d0();
	}
	void set_bden( real bden ) {
		const real mu = f_E / f_C;
		f_C = bden / d0();
		f_E = f_C * mu;
	}*/
	void set_frac( real f ) {
		real mu = f_E / f_C;
		f_C = f;
		f_E = mu * f_C;
	}
	real get_frac() const {
		return f_C;
	}
	real HC() const {
		return P0() / dC() * (1.0 + n_C);
	}
	real HE() const {
		return P0() / dE() * (1.0 + n_E);
	}
	virtual ~bipolytropic_eos() = default;
	virtual real enthalpy_to_density(real h) const {
		if (h < HE()) {
			return dE() * std::pow(h / HE(), n_E);
		} else {
			return dC() * std::pow((h - HE() + HC()) / HC(), n_C);
		}
	}
	virtual real density_to_enthalpy(real d) const {
		if (d >= dC()) {
			return P0() * (1.0 / dC() * (1.0 + n_C) * (std::pow(d / dC(), 1.0 / n_C) - 1.0) + 1.0 / dE() * (1.0 + n_E));
		} else if (d <= dE()) {
			return P0() / dE() * (1.0 + n_E) * std::pow(d / dE(), 1.0 / n_E);
		} else {
			return P0() / dE() * (1.0 + n_E);
		}
	}
	virtual real pressure(real d) const {
		if (d >= dC()) {
			return P0() * std::pow(d / dC(), 1.0 + 1.0 / n_C);
		} else if (d <= dE()) {
			return P0() * std::pow(d / dE(), 1.0 + 1.0 / n_E);
		} else {
			return P0();
		}
	}
};

#endif /* POLYTROPE_HPP_ */
