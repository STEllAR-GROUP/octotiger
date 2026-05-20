//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef RAD_GRID_HPP_
#define RAD_GRID_HPP_

#include "octotiger/math/Real.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/io/silo.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct.hpp"
#include "octotiger/unitiger/hydro_impl/flux.hpp"
#include "octotiger/unitiger/radiation/radiation_physics.hpp"
//#include "octotiger/sphere_points.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

class rad_grid {
public:
	static constexpr integer er_i = 0;
	static constexpr integer fx_i = 1;
	static constexpr integer fy_i = 2;
	static constexpr integer fz_i = 3;
	static constexpr integer wx_i = 4;
	static constexpr integer wy_i = 5;
	static constexpr integer wz_i = 6;
private:
	static constexpr integer DX = RAD_NX * RAD_NX;
	static constexpr integer DY = RAD_NX;
	static constexpr integer DZ = 1;
	static constexpr int R_DN[NDIM] = { RAD_NX * RAD_NX, RAD_NX, 1 };
	static std::unordered_map<std::string, int> str_to_index;
	static std::unordered_map<int, std::string> index_to_str;
	Real dx;
	std::vector<std::atomic<int>> is_coarse;
	std::vector<std::atomic<int>> has_coarse;
	std::vector<std::vector<Real>> Ushad;
	std::vector<std::vector<Real>> U;
	std::array<std::vector<Real>, NRF> U0;
	std::vector<std::vector<std::vector<Real>>> flux;
	std::array<std::array<std::vector<Real>*, NDIM>, NDIM> P;
	std::vector<std::vector<Real>> X;
	std::vector<Real> mmw, X_spc, Z_spc;
	hydro_computer<NDIM,INX,radiation_physics<NDIM>> hydro;
public:
	static void static_init();
	static std::vector<std::string> get_field_names();
	void set(const std::string name, Real* data);
	std::vector<silo_var_t> var_data() const;
	void set_X( const std::vector<std::vector<Real>>& x );
	void restore();
	void store();

	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & dx;
		arc & U;
	}
	void compute_mmw(const std::vector<std::vector<Real>>& U);
	void change_units(Real m, Real l, Real t, Real k);
	Real rad_imp_comoving(Real& E, Real& e, Real rho, Real mmw, Real X, Real Z, Real dt);
	void sanity_check();
	void compute_flux(Real);
	void initialize_erad(const std::vector<Real> rho, const std::vector<Real> tau);
	void set_dx(Real dx);
	//void compute_fEdd();
	void compute_fluxes();
	void advance(Real dt, Real beta);
	void rad_imp(std::vector<Real>& egas, std::vector<Real>& tau, std::vector<Real>& sx, std::vector<Real>& sy, std::vector<Real>& sz,
			const std::vector<Real>& rho, Real dt);
	std::vector<Real> get_restrict() const;
	std::vector<Real> get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub);
	void set_prolong(const std::vector<Real>&);
	void set_restrict(const std::vector<Real>&, const geo::octant&);
	void set_flux_restrict(const std::vector<Real>& data, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
			const geo::dimension& dim);
	std::vector<Real> get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::dimension& dim) const;
	std::vector<Real> get_intensity(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant&);
	void allocate();
	rad_grid(Real dx);
	rad_grid();
	void set_boundary(const std::vector<Real>& data, const geo::direction& dir);
	Real get_field(integer f, integer i, integer j, integer k) const;
	void set_field(Real v, integer f, integer i, integer j, integer k);
	void set_physical_boundaries(geo::face f, Real t);
	std::vector<Real> get_boundary(const geo::direction& dir);
	using kappa_type = std::function<Real(Real)>;

	Real hydro_signal_speed(const std::vector<Real>& egas, const std::vector<Real>& tau, const std::vector<Real>& sx, const std::vector<Real>& sy, const std::vector<Real>& sz,
			const std::vector<Real>& rho);

	void clear_amr();
	void set_rad_amr_boundary(const std::vector<Real>&, const geo::direction&);
	void complete_rad_amr_boundary();
	std::vector<Real> get_subset(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub);

	friend class node_server;
};




#endif /* RAD_GRID_HPP_ */

