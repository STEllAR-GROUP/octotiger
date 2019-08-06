/*
 * impl.hpp
 *
 *  Created on: Aug 3, 2019
 *      Author: dmarce1
 */

#ifndef HYDRO_IMPL_HPP_
#define HYDRO_IMPL_HPP_

#ifndef NOHPX
//#include <hpx/include/async.hpp>
//#include <hpx/include/future.hpp>
//#include <hpx/lcos/when_all.hpp>

//using namespace hpx;
#endif

#include "./flux.hpp"
#include "./reconstruct.hpp"
#include "./boundaries.hpp"
#include "./advance.hpp"
#include "./output.hpp"

template<int NDIM, int INX>
hydro_computer<NDIM, INX>::hydro_computer() {
	nf_ = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));
	angmom_count_ = 0;
	D1 = decltype(D1)(geo::H_N3);
	Q = decltype(Q)(nf_, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));
	fluxes = decltype(fluxes)(NDIM,
			std::vector < std::vector<std::array<safe_real, geo::NFACEDIR>> > (nf_, std::vector<std::array<safe_real, geo::NFACEDIR>>(geo::H_N3)));

	for (const auto &i : geo::find_indices(0, geo::H_NX)) {
		for (int d = 0; d < geo::NDIR / 2; d++) {
			D1[i][d] = NAN;
		}
	}
	for( int f = 0; f < nf_; f++) {
		smooth_field_.push_back(false);
	}
}

template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::filter_cell(std::array<safe_real, geo::NDIR> &C, safe_real c0) {
	if constexpr (NDIM == 1) {
		hydro::filter_cell1d(C, c0);
	} else if constexpr (NDIM == 2) {
		hydro::filter_cell2d(C, c0);
	} else {
		hydro::filter_cell3d(C, c0);
	}
}


template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::use_smooth_recon(int field) {
	smooth_field_[field] = true;
}

template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::use_angmom_correction(int index, int count) {
	angmom_index_ = index;
	angmom_count_ = count;
}

template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::post_process(hydro::state_type &U, safe_real dx) {
	physics < NDIM > p;
	p.template post_process < NDIM > (U, dx);
}

#endif /* IMPL_HPP_ */
