/*
 * hydro.hpp
 *
 *  Created on: Jul 31, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_HYDRO_HPP_
#define OCTOTIGER_UNITIGER_HYDRO_HPP_
#include <vector>

#include "safe_real.hpp"

#define SAFE_MATH_ON
#ifdef NOHPX
#include "/home/dmarce1/workspace/octotiger/octotiger/safe_math.hpp"
#else
#include "../../octotiger/safe_math.hpp"
#endif

#ifdef NOHPX
#include <future>
using std::future;
using std::async;
using std::launch;
#endif

#include "./cell_geometry.hpp"
#include "./util.hpp"

namespace hydro {

template<int NDIM>
using x_type = std::array<std::vector<safe_real>, NDIM>;

using flux_type = std::vector<std::vector<std::vector<safe_real>>>;

template<int NDIM>
using recon_type =std::vector<std::vector<std::array<safe_real, int_pow<3,NDIM>()>>>;

using state_type = std::vector<std::vector<safe_real>>;
}

template<int NDIM, int INX>
struct hydro_computer: public cell_geometry<NDIM, INX> {
	using geo = cell_geometry<NDIM,INX>;

	enum bc_type {OUTFLOW, PERIODIC};

	const hydro::recon_type<NDIM> reconstruct(hydro::state_type &U, const hydro::x_type<NDIM>&, safe_real );

	safe_real flux(const hydro::state_type& U, const hydro::recon_type<NDIM> &Q, hydro::flux_type &F, hydro::x_type<NDIM> &X, safe_real omega);

	void post_process(hydro::state_type &U, safe_real dx);

	void boundaries(hydro::state_type &U);

	void advance(const hydro::state_type &U0, hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type<NDIM> &X, safe_real dx, safe_real dt,
			safe_real beta, safe_real omega);

	void output(const hydro::state_type &U, const hydro::x_type<NDIM> &X, int num, safe_real);

	void use_angmom_correction(int index, int count);

	void use_smooth_recon(int field);

	std::vector<safe_real> get_field_sums(const hydro::state_type &U, safe_real dx);

	std::vector<safe_real> get_field_mags(const hydro::state_type &U, safe_real dx);

	hydro_computer();

	void set_bc( int face, bc_type bc) {
		bc_types[face] = bc;
	}

private:

	int nf_;
	int angmom_index_;
	int angmom_count_;
	std::vector<std::array<safe_real, geo::NDIR / 2>> D1;
	std::vector<std::vector<std::array<safe_real, geo::NDIR>>> Q;
	std::vector<std::vector<std::vector<std::array<safe_real, geo::NFACEDIR>>>> fluxes;
	std::vector<bool> smooth_field_;
	std::vector<bc_type> bc_types;
}
;

#include "impl/hydro.hpp"

#endif /* OCTOTIGER_UNITIGER_HYDRO_HPP_ */
