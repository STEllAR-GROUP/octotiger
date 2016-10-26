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
protected:
	static constexpr real G = 1.0;
	real dhdot_dr(real h, real hdot, real r) const;
	real dh_dr(real h, real hdot, real r) const;

public:
	real density_at(real, real) const;
	eos() {
	}
	virtual ~eos() = default;
	virtual real enthalpy_to_density(real h) const=0;
	virtual real density_to_enthalpy(real d) const=0;
	virtual real pressure(real d) const=0;
	virtual real h0() const = 0;
};

class wd_eos: public eos {
public:
	real B() const;
	real A, d0;
	void conversion_factors(real& m, real& l, real& t) const;
	void initialize(real&, real&);
	wd_eos();
	wd_eos(real M, real R);
	wd_eos(real M, const wd_eos& other);
	virtual real enthalpy_to_density(real h) const;
	virtual real density_to_enthalpy(real d) const;
	virtual real pressure(real d) const;
	real energy(real d) const;
	void set_d0_using_eos(real newd, const wd_eos& other);
	template<typename Archive>
	void serialize(Archive &ar, const unsigned int version) {
	}
	virtual real h0() const;
	void set_d0(real d);
	void set_h0(real h);
	real get_R0() const;
};

class bipolytropic_eos: public eos {
public:
	real M0, R0;
private:
	real n_C, n_E;
	real f_C, f_E;

public:
	void initialize(real&, real&);
	void initialize(real&, real&, real&);

public:
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

	void set_d0_using_eos(real newd, const bipolytropic_eos& other);
	bipolytropic_eos(real M, real R, real _n_C, real _n_E, real core_frac, real mu);
	bipolytropic_eos(real M, real R, real _n_C, real _n_E, real mu, const bipolytropic_eos& other);
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
	real d0() const;
	real h0() const;
	void set_h0(real h);
	void set_d0(real d);
	virtual real density_to_enthalpy(real d) const;
	virtual real pressure(real d) const;
};

#ifdef WD_EOS
using accretor_eos = wd_eos;
using donor_eos = wd_eos;
#else
using accretor_eos = bipolytropic_eos;
using donor_eos = bipolytropic_eos;
#endif

#endif /* POLYTROPE_HPP_ */
