/*
 * polytrope.hpp
 *
 *  Created on: Feb 22, 2016
 *      Author: dmarce1
 */

#ifndef POLYTROPE_HPP_
#define POLYTROPE_HPP_

#include "defs.hpp"

class bipolytropic_eos {
public:
	real M0, R0;
private:
	static constexpr real G = 1.0;
	real n_C, n_E;
	real f_C, f_E;

	real dhdot_dr(real h, real hdot, real r) const;
	real dh_dr(real h, real hdot, real r) const;
public:
	real d0() const;
	real h0() const;
	void set_h0(real h);
	void set_d0(real d);
	void initialize(real&, real&);
	real density_at(real,real) const;

public:
	void initialize(real&, real&, real&);
	real dC() const;
	bipolytropic_eos() {
	}
	template<typename Archive>
	void serialize(Archive &ar, const unsigned int version) {
		ar & M0;
		ar & R0;
		ar & n_C;
		ar & n_E;
		ar & f_C;
		ar & f_E;
	}

	void set_d0_using_eos(real newd, bipolytropic_eos& other);
	bipolytropic_eos(real M, real R, real _n_C, real _n_E, real core_frac,
			real mu);
	bipolytropic_eos(real M, real R, real _n_C, real _n_E, real mu,
			const bipolytropic_eos& other);
	bipolytropic_eos(real M, real _n_C, const bipolytropic_eos& other);
	void set_entropy(real other_s0);
	virtual ~bipolytropic_eos() = default;
	virtual real enthalpy_to_density(real h) const;
	real dE() const;
	real s0() const;
	real P0() const;
	void set_frac( real f );
	real get_frac() const;
	real HC() const;
	real HE() const;
	virtual real density_to_enthalpy(real d) const;
	virtual real pressure(real d) const;
};

using accretor_eos = bipolytropic_eos;
using donor_eos = bipolytropic_eos;

#endif /* POLYTROPE_HPP_ */
