//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef POLYTROPE_HPP_
#define POLYTROPE_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/math/Real.hpp"

#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#define V1309

class struct_eos {
protected:
	static constexpr Real G = 1.0;
	Real dhdot_dr(Real h, Real hdot, Real r) const;
	Real dh_dr(Real h, Real hdot, Real r) const;

public:
	Real density_at(Real, Real);
	struct_eos() {
	}

//	class wd_struct_eos: public struct_eos {
public:
	Real hfloor() const {
		if (rho_cut > 0.0) {
			const Real h0_ = density_to_enthalpy(rho_cut);
			return h0_ * (1.0 - 2.5 / (1.0 + n_E));
		} else {
			return 0.0;
		}
	}
	Real B() const;
	Real A, d0_, my_radius;
	void conversion_factors(Real &m, Real &l, Real &t) const;
	struct_eos(Real M, Real R);
	struct_eos(Real M, const struct_eos &other);
	Real energy(Real d) const;
	Real d0() const;
	template<typename Archive>
	void serialize(Archive &arc, const unsigned int version) {
		arc & rho_cut;
		arc & A;
		arc & d0_;
		arc & M0;
		arc & R0;
		arc & n_C;
		arc & n_E;
		arc & f_C;
		arc & f_E;
		arc & wd_eps;
		arc & wd_T0;
		arc & wd_core_cut;
	}

//		class bipolytropic_struct_eos: public struct_eos {
public:
	Real M0, R0;
	Real wd_eps, wd_T0;
	Real wd_core_cut;
private:
	Real n_C, n_E;
	Real f_C, f_E;
	Real rho_cut;
public:
	void set_wd_T0(double t, double abar, double zbar);
	void set_cutoff_density(Real d) {
		rho_cut = d;
	}
	Real get_cutoff_density() const {
		return rho_cut;
	}
	OCTOTIGER_EXPORT void initialize(Real&, Real&);
	OCTOTIGER_EXPORT void initialize(Real&, Real&, Real&);

public:
	Real get_R0() const;
	Real dC() const;

	void set_d0_using_struct_eos(Real newd, const struct_eos &other);
	struct_eos(Real M, Real R, Real _n_C, Real _n_E, Real core_frac, Real mu);
	struct_eos(Real M, Real R, Real _n_C, Real _n_E, Real mu, const struct_eos &other);
	struct_eos(Real M, Real _n_C, const struct_eos &other);
	void set_entropy(Real other_s0);
	~struct_eos() = default;
	Real enthalpy_to_density(Real h) const;
	Real dE() const;
	Real s0() const;
	Real P0() const;
	void set_frac(Real f);
	Real get_frac() const;
	Real HC() const;
	Real HE() const;
	Real h0() const;
	void set_h0(Real h);
	void set_d0(Real d);
	Real density_to_enthalpy(Real d) const;
	Real pressure(Real d) const;

};

#endif /* POLYTROPE_HPP_ */
