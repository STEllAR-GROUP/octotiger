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
private:
	static constexpr integer DX = RAD_NX * RAD_NX;
	static constexpr integer DY = RAD_NX;
	static constexpr integer DZ = 1;
	static std::unordered_map<std::string, int> str_to_index;
	static std::unordered_map<int, std::string> index_to_str;
	real dx;
	std::vector<std::atomic<int>> is_coarse;
	std::vector<std::atomic<int>> has_coarse;
	std::vector<real> Ushad;
	std::vector<real> U;
	std::vector<real> U0;
	std::vector<std::vector<real>> flux;
	std::array<std::array<std::vector<real>,NDIM>,NDIM> P;
	std::vector<std::vector<real>> X;
	std::vector<real> mmw, X_spc, Z_spc;
	hydro_computer<NDIM,INX,radiation_physics<NDIM>> hydro;
public:
	static void static_init();
	static std::vector<std::string> get_field_names();
	void set(const std::string name, real* data);
	std::vector<silo_var_t> var_data() const;
	void set_X( const std::vector<std::vector<real>>& x );
	void restore();
	void store();

	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & dx;
		arc & U;
	}
	void compute_mmw(const std::vector<std::vector<safe_real>>& U);
	void change_units(real m, real l, real t, real k);
	real rad_imp_comoving(real& E, real& e, real rho, real mmw, real X, real Z, real dt);
	void sanity_check();
	void compute_flux(real,const std::vector<real> &egas, const std::vector<real> &tau,
			const std::vector<real> &sx, const std::vector<real> &sy, const std::vector<real> &sz,
			const std::vector<real> &rho);
	void initialize_erad(const std::vector<safe_real> rho, const std::vector<safe_real> tau, const std::vector<std::vector<safe_real>>& X);
	void set_dx(real dx);
	//void compute_fEdd();
	void set_fluxes(std::vector<real>& egas, std::vector<real>& tau, std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
			const std::vector<real>& rho);
	void compute_fluxes();
	void advance(real dt, real beta, std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
			const std::vector<real>& rho);
	void rad_imp(std::vector<real>& egas, std::vector<real>& tau, std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
			const std::vector<real>& rho, real dt);
	std::vector<real> get_restrict() const;
	std::vector<real> get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub);
	void set_prolong(const std::vector<real>&);
	void set_restrict(const std::vector<real>&, const geo::octant&);
	void set_flux_restrict(const std::vector<real>& data, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
			const geo::dimension& dim);
	std::vector<real> get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::dimension& dim) const;
	std::vector<real> get_intensity(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant&);
	void allocate();
	rad_grid(real dx);
	rad_grid();
	static std::vector<real> radiation_diffusion_analytic(real x, real y, real z, real t);
	void set_boundary(const std::vector<real>& data, const geo::direction& dir);
	real get_er(integer i, integer j, integer k) const;
	void set_er(real v, integer i, integer j, integer k);
	void set_physical_boundaries(geo::face f, real t);
	std::vector<real> get_boundary(const geo::direction& dir);
	using kappa_type = std::function<real(real)>;

	real hydro_signal_speed(const std::vector<real>& egas, const std::vector<real>& tau, const std::vector<real>& sx, const std::vector<real>& sy, const std::vector<real>& sz,
			const std::vector<real>& rho);

	void clear_amr();
	void set_rad_amr_boundary(const std::vector<real>&, const geo::direction&);
	void complete_rad_amr_boundary();
	std::vector<real> get_subset(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub);

	friend class node_server;
};




#endif /* RAD_GRID_HPP_ */

