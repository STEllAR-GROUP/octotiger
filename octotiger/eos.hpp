//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef POLYTROPE_HPP_
#define POLYTROPE_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/real.hpp"
#include <string>
#include <vector>

#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <functional>

#define V1309

class struct_eos {
protected:
	static constexpr real G = 1.0;
	real dhdot_dr(real h, real hdot, real r);
	real dh_dr(real h, real hdot, real r) const;

public:
	real density_at(real, real);
	struct_eos() {
	}

//	class wd_struct_eos: public struct_eos {
public:
	real hfloor() const {
		if (rho_cut > 0.0) {
			const real h0_ = density_to_enthalpy(rho_cut);
			return h0_ * (1.0 - 2.5 / (1.0 + n_E));
		} else {
			return 0.0;
		}
	}
	real B() const;
	real A, d0_;
	void conversion_factors(real &m, real &l, real &t) const;
	struct_eos(real M, real R);
	struct_eos(real M, const struct_eos &other);
	real energy(real d) const;
	real d0() const;
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
		arc & p_mass;
		arc & p_smooth_l;
		arc & my_radius;
		arc & filename;
		arc & HE_prof;
		arc & PE_prof;
		arc & P_vec;
		arc & rho_vec;
		arc & h_vec;
	}

//		class bipolytropic_struct_eos: public struct_eos {
public:
	real M0, R0;
	real wd_eps, wd_T0;
	real wd_core_cut;
private:
	real n_C, n_E;
	real f_C, f_E;
	real rho_cut;
	real p_mass, p_smooth_l, my_radius;
	std::string filename = "";
	std::vector<double> P_vec, rho_vec, h_vec; // the profiles that was reasd from the 1d profile (from, e.g., MESA)
	std::function<double(double)> enthalpy_to_density_prof = nullptr;
	std::function<double(double)> density_to_pressure_prof = nullptr;
	real HE_prof, PE_prof;
public:
	void set_wd_T0(double t, double abar, double zbar);
	void set_cutoff_density(real d) {
		rho_cut = d;
	}
	real get_cutoff_density() const {
		return rho_cut;
	}
	OCTOTIGER_EXPORT void initialize(real&, real&);
	OCTOTIGER_EXPORT void initialize(real&, real&, real&);
	OCTOTIGER_EXPORT void initialize(real&, real&, real&, real&, real&, real&, real);

public:
	real get_R0() const;
	void update_R0();
	real get_p_mass() const;
	real get_RC() const;
	real dC() const;

	void set_d0_using_struct_eos(real newd, const struct_eos &other);
	struct_eos(real M, real R, real _n_C, real _n_E, real core_frac, real mu);
	struct_eos(real M, real R, real _n_C, real _n_E, real core_frac, real mu, real _p_mass);
	struct_eos(real M, real core_frac, const std::string& _filename);
	struct_eos(real M, real R, real _n_C, real _n_E, real mu, const struct_eos &other);
	struct_eos(real M, real _n_C, const struct_eos &other);
	void set_entropy(real other_s0);
	~struct_eos() = default;
	real enthalpy_to_density(real h);
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
	void set_p_mass(real pmass);
	real density_to_enthalpy(real d) const;
	real pressure(real d);

};

#endif /* POLYTROPE_HPP_ */
