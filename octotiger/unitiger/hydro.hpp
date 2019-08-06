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
#include "basis.hpp"

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
namespace hydro {

void filter_cell1d(std::array<safe_real, 3> &C, safe_real C0);
void filter_cell2d(std::array<safe_real, 9> &C, safe_real C0);
void filter_cell3d(std::array<safe_real, 27> &C, safe_real C0);
void output_cell2d(FILE *fp, const std::array<safe_real, 9> &C, int joff, int ioff);
}

#include "./cell_geometry.hpp"
#include "./util.hpp"

namespace hydro {

template<int NDIM>
using x_type = std::vector<std::array<safe_real, NDIM>>;

using flux_type = std::vector<std::vector<std::vector<safe_real>>>;

template<int NDIM>
using recon_type =std::vector<std::vector<std::array<safe_real, int_pow<3,NDIM>()>>>;

using state_type = std::vector<std::vector<safe_real>>;
}

template<int NDIM, int INX>
struct hydro_computer: public cell_geometry<NDIM, INX> {
	using geo = cell_geometry<NDIM,INX>;

	const hydro::recon_type<NDIM> reconstruct(hydro::state_type &U, safe_real dx);

	safe_real flux(const hydro::recon_type<NDIM> &Q, hydro::flux_type &F, hydro::x_type<NDIM> &X, safe_real omega);

	void post_process(hydro::state_type &U, safe_real dx);

	void boundaries(hydro::state_type &U);

	void advance(const hydro::state_type &U0, hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type<NDIM> &X, safe_real dx, safe_real dt,
			safe_real beta, safe_real omega);

	void output(const hydro::state_type &U, const hydro::x_type<NDIM> &X, int num);

	void use_angmom_correction(int index, int count) {
		angmom_index_ = index;
		angmom_count_ = count;
	}



	hydro_computer();

private:

	int nf;
	int angmom_index_, angmom_count_;
	std::vector<std::array<safe_real, geo::NDIR / 2>> D1;
	std::vector<std::vector<std::array<safe_real, geo::NDIR>>> Q;
	std::vector<std::vector<std::array<safe_real, geo::NDIR>>> L;
	std::vector<std::vector<std::vector<std::array<safe_real, geo::NFACEDIR>>>> fluxes;

	void filter_cell(std::array<safe_real, geo::NDIR> &C, safe_real c0) {
		if constexpr (NDIM == 1) {
			hydro::filter_cell1d(C, c0);
		} else if constexpr (NDIM == 2) {
			hydro::filter_cell2d(C, c0);
		} else {
			hydro::filter_cell3d(C, c0);
		}
	}
//
//	safe_real z_error(const std::vector<std::vector<safe_real>> &U) {
//		safe_real err = 0.0;
//		for (auto &i : find_indices<NDIM, INX>(geo::H_BW, geo::H_NX - geo::H_BW)) {
//			err += std::abs(U[zx_i][i]);
//		}
//		return err;
//	}

}
;

#include "impl/impl.hpp"

#endif /* OCTOTIGER_UNITIGER_HYDRO_HPP_ */
