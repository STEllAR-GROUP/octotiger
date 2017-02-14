/*
 * rad_grid.hpp
 *
 *  Created on: May 20, 2016
 *      Author: dmarce1
 */

#ifndef RAD_GRID_HPP_
#define RAD_GRID_HPP_

#include "defs.hpp"

#ifdef RADIATION

#define R_BW 2
#define R_NX (INX+2*R_BW)
#define R_N3 (R_NX*R_NX*R_NX)

#include "geometry.hpp"
#include "sphere_points.hpp"
#include <cmath>
#include <vector>
#include <hpx/lcos/local/mutex.hpp>

typedef real rad_type;

class rad_grid_init {
public:
	rad_grid_init();
};

static constexpr auto kappa_p = [](real rho, real e) {
	return rho;
};

static constexpr auto dkappa_p_de = [](real rho, real e) {
	return 0.0;
};

static constexpr auto kappa_R = [](real rho, real e) {
	return rho;
};

static constexpr auto kappa_s = [](real rho, real e) {
	return rho;
};

static constexpr auto B_p = [](real rho, real e) {
	return std::pow( e / rho, 4.0);
};
static constexpr auto dB_p_de = [](real rho, real e) {
	return 4.0 * std::pow( e / rho, 4.0) / e;
};

class rad_grid: public rad_grid_init {
private:
	static constexpr integer er_i = 0;
	static constexpr integer fx_i = 1;
	static constexpr integer fy_i = 2;
	static constexpr integer fz_i = 3;
	static constexpr integer DX = R_NX * R_NX;
	static constexpr integer DY = R_NX;
	static constexpr integer DZ = 1;

	friend class rad_grid_init;
	static std::vector<sphere_point> sphere_points;
	static void initialize();
	static integer rindex(integer, integer, integer);
	real dx;
	std::vector<rad_type> J;
	std::vector<rad_type> sigma_a;
	std::vector<rad_type> sigma_s;
	std::vector<std::vector<rad_type>> I;
	std::vector<rad_type> E;
	std::vector<rad_type> Fx;
	std::vector<rad_type> Fy;
	std::vector<rad_type> Fz;
	std::vector<rad_type> Beta_x;
	std::vector<rad_type> Beta_y;
	std::vector<rad_type> Beta_z;
	std::vector<rad_type> fEdd_xx;
	std::vector<rad_type> fEdd_xy;
	std::vector<rad_type> fEdd_xz;
	std::vector<rad_type> fEdd_yy;
	std::vector<rad_type> fEdd_yz;
	std::vector<rad_type> fEdd_zz;
	std::array<std::vector<rad_type>, NRF> U;
	std::array<std::vector<rad_type>, NRF> U0;
	std::array<std::array<std::vector<rad_type>, NRF>,NDIM> flux;
	std::array<std::array<std::vector<rad_type>*, NDIM>, NDIM> fEdd;
	hpx::lcos::local::mutex Pmtx;

public:
	void restore();
	void store();
	std::size_t load(FILE* fp);
	std::size_t save(FILE* fp) const;

	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & dx;
		arc & U;
	}

	void rad_imp_comoving(real& E, real& e, real rho, real dt);
	void sanity_check();
	void initialize_erad(const std::vector<real> rho, const std::vector<real> tau);
	void set_dx(real dx);
	void compute_fEdd();
	void compute_flux();
	void advance(real dt, real beta);
	void rad_imp(std::vector<real>& egas, std::vector<real>& tau, std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
			const std::vector<real>& rho, real dt);
	std::vector<real> get_restrict(const geo::octant&) const;
	std::vector<real> get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant&);
	void set_prolong(const std::vector<real>&, const geo::octant&);
	void set_restrict(const std::vector<real>&, const geo::octant&, const geo::octant&);
	void compute_intensity(const geo::octant& oct);
	void accumulate_intensity(const geo::octant& oct);
	void free_octant(const geo::octant& oct);
	void set_intensity(const std::vector<rad_type>& data, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant&);
	void set_emissivity(const std::vector<real>& rho, const std::vector<real>&, const std::vector<real>&, const std::vector<real>&, const std::vector<real>& e);
	std::vector<rad_type> get_intensity(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant&);
	void allocate();
	void get_output(std::array<std::vector<real>, NF + NGF + NRADF + NPF>& v, integer, integer, integer) const;
	rad_grid(real dx);
	rad_grid();
	void set_boundary(const std::vector<real>& data, const geo::direction& dir);
	void set_field(rad_type v, integer f, integer i, integer j, integer k);
	void set_physical_boundaries(geo::face f);
	std::vector<real> get_boundary(const geo::direction& dir);
	using kappa_type = std::function<real(real)>;
};

#endif /* RAD_GRID_HPP_ */

#endif
