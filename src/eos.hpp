/*
 * polytrope.hpp
 *
 *  Created on: Feb 22, 2016
 *      Author: dmarce1
 */

#ifndef POLYTROPE_HPP_
#define POLYTROPE_HPP_

#include "defs.hpp"
#include "grid.hpp"


class struct_eos {
protected:
	static constexpr real G = 1.0;
	real dhdot_dr(real h, real hdot, real r) const;
	real dh_dr(real h, real hdot, real r) const;

public:
	real density_at(real, real) const;
	struct_eos() {
	}

//	class wd_struct_eos: public struct_eos {
public:
	real B() const;
	real A, d0_;
	void conversion_factors(real& m, real& l, real& t) const;
	struct_eos(real M, real R);
	struct_eos(real M, const struct_eos& other);
	real energy(real d) const;
	real d0() const;
	template<typename Archive>
	void serialize(Archive &arc, const unsigned int version) {
		arc & A;
		arc & d0_;
		arc & M0;
		arc & R0;
		arc & n_C;
		arc & n_E;
		arc & f_C;
		arc & f_E;
	}

//		class bipolytropic_struct_eos: public struct_eos {
public:
	real M0, R0;
private:
	real n_C, n_E;
	real f_C, f_E;

public:
	void initialize(real&, real&);
	void initialize(real&, real&, real&);

public:
	real get_R0() const;
	real dC() const;


	void set_d0_using_struct_eos(real newd, const struct_eos& other);
	struct_eos(real M, real R, real _n_C, real _n_E, real core_frac, real mu);
	struct_eos(real M, real R, real _n_C, real _n_E, real mu, const struct_eos& other);
	struct_eos(real M, real _n_C, const struct_eos& other);
	void set_entropy(real other_s0);
	~struct_eos() = default;
	real enthalpy_to_density(real h) const;
	real dE() const;
	real s0() const;
	real P0() const;
	void set_frac(real f);
	real get_frac() const;
	real HC() const;
	real HE() const;
	real h0() const;
	void set_h0(real h);
	void set_d0(real d);
	real density_to_enthalpy(real d) const;
	real pressure(real d) const;

};

HPX_IS_BITWISE_SERIALIZABLE(struct_eos);

#endif /* POLYTROPE_HPP_ */
