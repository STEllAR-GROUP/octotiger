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
//#include "sphere_points.hpp"
#include <cmath>
#include <vector>
#include "physcon.hpp"
#include <hpx/lcos/local/mutex.hpp>

typedef real rad_type;

class rad_grid_init {
public:
	rad_grid_init();
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
	static void initialize();
	real dx;
	std::array<std::vector<rad_type>, NRF> U;
	std::array<std::vector<rad_type>, NRF> U0;
	std::array<std::array<std::vector<rad_type>, NRF>,NDIM> flux;
	std::array<std::array<std::vector<rad_type>*, NDIM>, NDIM> P;
	std::vector<std::vector<real>> X;
	std::vector<real> mmw;
	static std::array<std::array<real,NDIM>,NDIM> compute_p( real E, real Fx, real Fy, real Fz);
public:
	void set_X( const std::vector<std::vector<real>>& x );
	void restore();
	void store();
	std::size_t load(std::istream& strm);
	std::size_t save(std::ostream& strm) const;

	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & dx;
		arc & dx;
		arc & U;
	}
	void compute_mmw(const std::vector<std::vector<real>>& U);
	void change_units(real m, real l, real t, real k);
	real rad_imp_comoving(real& E, real& e, real rho, real mmw, real dt);
	void sanity_check();
	void initialize_erad(const std::vector<real> rho, const std::vector<real> tau);
	void set_dx(real dx);
	//void compute_fEdd();
	void compute_flux();
	void advance(real dt, real beta);
	void rad_imp(std::vector<real>& egas, std::vector<real>& tau, std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
			const std::vector<real>& rho, real dt);
	std::vector<real> get_restrict() const;
	std::vector<real> get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub);
	void set_prolong(const std::vector<real>&);
	void set_restrict(const std::vector<real>&, const geo::octant&);
	void set_flux_restrict(const std::vector<rad_type>& data, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
			const geo::dimension& dim);
	std::vector<rad_type> get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::dimension& dim) const;
	std::vector<rad_type> get_intensity(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant&);
	void allocate();
	void get_output(std::array<std::vector<real>, NF + NGF + NRF + NPF>& v, integer, integer, integer) const;
	rad_grid(real dx);
	rad_grid();
	void set_boundary(const std::vector<real>& data, const geo::direction& dir);
	void set_field(rad_type v, integer f, integer i, integer j, integer k);
	void set_physical_boundaries(geo::face f);
	std::vector<real> get_boundary(const geo::direction& dir);
	using kappa_type = std::function<real(real)>;

	real hydro_signal_speed(const std::vector<real>& egas, const std::vector<real>& tau, const std::vector<real>& sx, const std::vector<real>& sy, const std::vector<real>& sz,
			const std::vector<real>& rho);

	friend class node_server;
};

#endif /* RAD_GRID_HPP_ */

#endif
