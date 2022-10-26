//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef RAD_GRID_HPP_
#define RAD_GRID_HPP_

#include "octotiger/unitiger/safe_real.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/real.hpp"
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
	static std::unordered_map<std::string, int> str_to_index;
	static std::unordered_map<int, std::string> index_to_str;
	real dx;
	oct::vector<std::atomic<int>> is_coarse;
	oct::vector<std::atomic<int>> has_coarse;
	oct::vector<oct::vector<real>> Ushad;
	oct::vector<oct::vector<real>> U;
	oct::array<oct::vector<real>, NRF> U0;
	oct::vector<oct::vector<oct::vector<real>>> flux;
	oct::array<oct::array<oct::vector<real>*, NDIM>, NDIM> P;
	oct::vector<oct::vector<real>> X;
	oct::vector<real> mmw, X_spc, Z_spc;
	hydro_computer<NDIM,INX,radiation_physics<NDIM>> hydro;
public:
	static void static_init();
	static oct::vector<std::string> get_field_names();
	void set(const std::string name, real* data);
	oct::vector<silo_var_t> var_data() const;
	void set_X( const oct::vector<oct::vector<real>>& x );
	void restore();
	void store();

	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & dx;
		arc & U;
	}
	void compute_mmw(const oct::vector<oct::vector<safe_real>>& U);
	void change_units(real m, real l, real t, real k);
	real rad_imp_comoving(real& E, real& e, real rho, real mmw, real X, real Z, real dt);
	void sanity_check();
	void compute_flux(real);
	void initialize_erad(const oct::vector<safe_real> rho, const oct::vector<safe_real> tau);
	void set_dx(real dx);
	//void compute_fEdd();
	void compute_fluxes();
	void advance(real dt, real beta);
	void rad_imp(oct::vector<real>& egas, oct::vector<real>& tau, oct::vector<real>& sx, oct::vector<real>& sy, oct::vector<real>& sz,
			const oct::vector<real>& rho, real dt);
	oct::vector<real> get_restrict() const;
	oct::vector<real> get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub);
	void set_prolong(const oct::vector<real>&);
	void set_restrict(const oct::vector<real>&, const geo::octant&);
	void set_flux_restrict(const oct::vector<real>& data, const oct::array<integer, NDIM>& lb, const oct::array<integer, NDIM>& ub,
			const geo::dimension& dim);
	oct::vector<real> get_flux_restrict(const oct::array<integer, NDIM>& lb, const oct::array<integer, NDIM>& ub, const geo::dimension& dim) const;
	oct::vector<real> get_intensity(const oct::array<integer, NDIM>& lb, const oct::array<integer, NDIM>& ub, const geo::octant&);
	void allocate();
	rad_grid(real dx);
	rad_grid();
	void set_boundary(const oct::vector<real>& data, const geo::direction& dir);
	real get_field(integer f, integer i, integer j, integer k) const;
	void set_field(real v, integer f, integer i, integer j, integer k);
	void set_physical_boundaries(geo::face f, real t);
	oct::vector<real> get_boundary(const geo::direction& dir);
	using kappa_type = std::function<real(real)>;

	real hydro_signal_speed(const oct::vector<real>& egas, const oct::vector<real>& tau, const oct::vector<real>& sx, const oct::vector<real>& sy, const oct::vector<real>& sz,
			const oct::vector<real>& rho);

	void clear_amr();
	void set_rad_amr_boundary(const oct::vector<real>&, const geo::direction&);
	void complete_rad_amr_boundary();
	oct::vector<real> get_subset(const oct::array<integer, NDIM>& lb, const oct::array<integer, NDIM>& ub);

	friend class node_server;
};




#endif /* RAD_GRID_HPP_ */

