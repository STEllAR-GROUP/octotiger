/*
 * rad_grid.hpp
 *
 *  Created on: May 20, 2016
 *      Author: dmarce1
 */

#ifndef RAD_GRID_HPP_
#define RAD_GRID_HPP_

#include "defs.hpp"

#define R_BW 1
#define R_NX (INX+2*R_BW)
#define R_N3 (R_NX*R_NX*R_NX)

#include "geometry.hpp"
#include "sphere_points.hpp"
#include <cmath>
#include <vector>

typedef real rad_type;

class rad_grid_init {
public:
	rad_grid_init();
};

class rad_grid: public rad_grid_init {
private:

	friend class rad_grid_init;
	static std::vector<sphere_point> sphere_points;
	static void initialize();
	static integer rindex(integer, integer, integer);
	real dx;
	std::vector<rad_type> J;
	std::vector<rad_type> sigma_a;
	std::vector<std::vector<rad_type>> I;
	std::vector<rad_type> E;
	std::vector<rad_type> Pxx;
	std::vector<rad_type> Pxy;
	std::vector<rad_type> Pxz;
	std::vector<rad_type> Pyy;
	std::vector<rad_type> Pyz;
	std::vector<rad_type> Pzz;
	std::array<std::array<const std::vector<rad_type>*, NDIM >, NDIM> P;
	hpx::mutex Pmtx;

public:
	std::vector<real> get_P(const geo::dimension& dim1,const geo::dimension& dim2 ) const;
	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & dx;
	}

	std::vector<real> get_restrict(const geo::octant&) const;
	std::vector<real> get_prolong(const std::array<integer, NDIM>& lb,
			const std::array<integer, NDIM>& ub, const geo::octant&);
	void set_prolong(const std::vector<real>&, const geo::octant&);
	void set_restrict(const std::vector<real>&, const geo::octant&, const geo::octant&);
	void compute_intensity(const geo::octant& oct);
	void free_octant(const geo::octant& oct);
	void set_intensity(const std::vector<rad_type>& data, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant&);
	void set_emissivity(const std::vector<real>& rho, const std::vector<real>& e);
	std::vector<rad_type> get_intensity(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
		const geo::octant&);

	rad_grid(real dx);
};

#endif /* RAD_GRID_HPP_ */
