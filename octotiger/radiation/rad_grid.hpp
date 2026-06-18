//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef RAD_GRID_HPP_
#define RAD_GRID_HPP_


#include "octotiger/defs.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/io/silo.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/hydro_impl/flux.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct.hpp"
#include "octotiger/unitiger/radiation/radiation_physics.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/math/Matrix.hpp"
#include "octotiger/math/Vector.hpp"

#include <string>
#include <unordered_map>


struct RadiationSource {
	Real dEr_dt = 0_R;
	Vector<Real, NDIM> dFr_dt = 0_R;
	Vector<Real, NDIM> dS_dt = 0_R;
	Real dEg_dt = 0_R;
	Real dtau_dt = 0_R;
	int iters = 0;
	bool converged = false;
};


class rad_grid {
public:
	static constexpr int er_i = 0;
	static constexpr int fx_i = 1;
	static constexpr int fy_i = 2;
	static constexpr int fz_i = 3;
	static constexpr int wx_i = 4;
	static constexpr int wy_i = 5;
	static constexpr int wz_i = 6;

private:
	static constexpr int DX = RAD_NX * RAD_NX;
	static constexpr int DY = RAD_NX;
	static constexpr int DZ = 1;
	static constexpr int R_DN[NDIM] = {RAD_NX * RAD_NX, RAD_NX, 1};
	static std::unordered_map<std::string, int> str_to_index;
	static std::unordered_map<int, std::string> index_to_str;
	Real dx;
	std::vector<std::atomic<int>> is_coarse;
	std::vector<std::atomic<int>> has_coarse;
	std::vector<std::vector<Real>> Ushad;
	std::vector<std::vector<Real>> U;
	std::array<std::vector<Real>, NRF> U0;
	std::vector<std::vector<std::vector<Real>>> flux;
	std::array<std::array<std::vector<Real> *, NDIM>, NDIM> P;
	std::vector<std::vector<Real>> X;
	std::vector<Real> mmw, kappa, chi;
	hydro_computer<NDIM, INX, radiation_physics<NDIM>> hydro;

public:
	static void static_init();
	static std::vector<std::string> get_field_names();
	void set(const std::string name, Real *data);
	std::vector<silo_var_t> var_data() const;
	void set_X(const std::vector<std::vector<Real>> &x);
	void restore();
	void store();

	template <class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & dx;
		arc & U;
	}
	using HydroState = std::vector<std::vector<Real>>;
	void computeFlux();
	std::vector<RadiationSource> computeSource(HydroState &, Real dt);
	void applySource(HydroState&, std::vector<RadiationSource> const &, Real);
	void applyFlux(Real, Real);
	void computeMaterialProperties(const std::vector<std::vector<Real>> &);
	void change_units(Real m, Real l, Real t, Real k);
	void sanity_check();
	void initialize_erad(const std::vector<Real> rho, const std::vector<Real> tau);
	void set_dx(Real dx);
	// void compute_fEdd();
	std::vector<Real> get_restrict() const;
	std::vector<Real> get_prolong(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub);
	void set_prolong(const std::vector<Real> &);
	void set_restrict(const std::vector<Real> &, const geo::octant &);
	void set_flux_restrict(const std::vector<Real> &data, const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub,
						   const geo::dimension &dim);
	std::vector<Real> get_flux_restrict(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub,
										const geo::dimension &dim) const;
	std::vector<Real> get_intensity(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub, const geo::octant &);
	void allocate();
	rad_grid(Real dx);
	rad_grid();
	void set_boundary(const std::vector<Real> &data, const geo::direction &dir);
	Real get_field(int f, int i, int j, int k) const;
	void set_field(Real v, int f, int i, int j, int k);
	void set_physical_boundaries(geo::face f, Real t);
	std::vector<Real> get_boundary(const geo::direction &dir);
	using kappa_type = std::function<Real(Real)>;

	Real hydro_signal_speed(const std::vector<Real> &egas, const std::vector<Real> &tau, const std::vector<Real> &sx,
							const std::vector<Real> &sy, const std::vector<Real> &sz, const std::vector<Real> &rho);

	void clear_amr();
	void set_rad_amr_boundary(const std::vector<Real> &, const geo::direction &);
	void complete_rad_amr_boundary();
	std::vector<Real> get_subset(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub);

	friend class node_server;
};

#endif /* RAD_GRID_HPP_ */
