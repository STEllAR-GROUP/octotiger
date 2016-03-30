/*
 * fmm.hpp
 *
 *  Created on: Feb 3, 2016
 *      Author: dmarce1
 */

#ifndef FMM_HPP_
#define FMM_HPP_

#include "defs.hpp"
#include "geometry.hpp"
#include "harmonics.hpp"

#define USE_MOM_CORRECTION

using multipole_type = multipole_expansion<M_POLES>;
using expansion_type = local_expansion<L_POLES>;

class fmm {
private:
	static constexpr integer IN3 = INX * INX * INX;
	std::vector<multipole_type> M;
	std::vector<expansion_type> L;
	real dx;
	bool dx_set;
	static integer index(integer i, integer j, integer k) {
		return (i * INX + j) * INX + k;
	}
	static integer coarse_index(integer i, integer j, integer k) {
		return (i * (INX / 2) + j) * (INX / 2) + k;
	}
public:
	void set_dx(real d);
	fmm();
	void set_source(real this_m, integer i, integer j, integer k);
	std::vector<multipole_type> M2M();
	std::vector<multipole_type> get_multipoles(const std::array<integer, NDIM>& lb,
			const std::array<integer, NDIM>& ub) const;
	std::array<real, NDIM + 1> four_force(integer i, integer j, integer k) const;
	real get_phi(integer i, integer j, integer k) const;
	real get_gx(integer i, integer j, integer k) const;
	real get_gy(integer i, integer j, integer k) const;
	real get_gz(integer i, integer j, integer k) const;
	std::vector<expansion_type> get_expansions(const geo::octant octant) const;
	void set_multipoles(const std::vector<multipole_type>& Mfine, const geo::octant octant);
	void L2L(const std::vector<expansion_type>& Lcoarse);
	void self_M2L(bool is_root, bool is_leaf);
	void other_M2L(const std::vector<multipole_type>& Q, const std::array<integer, NDIM>& lb,
			const std::array<integer, NDIM>& ub, bool is_leaf, integer width = 1);
};

#endif /* FMM_HPP_ */
