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
	nf_ = physics<NDIM>::nf;
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
	bc_.resize(2*NDIM,OUTFLOW);
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

template<int NDIM, int INX>
std::vector<safe_real>  hydro_computer<NDIM, INX>::get_field_sums(const hydro::state_type &U, safe_real dx) {
	std::vector<safe_real> sums(nf_,0.0);
	static const auto indices = geo::find_indices(geo::H_BW,geo::H_NX-geo::H_BW);
	for( int f = 0; f < nf_; f++) {
		for( auto i : indices) {
			sums[f] += U[f][i] * (dx * dx * dx);
		}
	}
	return sums;
}

template<int NDIM, int INX>
std::vector<safe_real>  hydro_computer<NDIM, INX>::get_field_mags(const hydro::state_type &U, safe_real dx) {
	std::vector<safe_real> sums(nf_,0.0);
	static const auto indices = geo::find_indices(geo::H_BW,geo::H_NX-geo::H_BW);
	for( int f = 0; f < nf_; f++) {
		for( auto i : indices) {
			sums[f] += std::abs(U[f][i] * (dx * dx * dx));
		}
	}
	return sums;
}

#endif /* IMPL_HPP_ */
