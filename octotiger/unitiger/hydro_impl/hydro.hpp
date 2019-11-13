//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRO_IMPL_HPP_
#define HYDRO_IMPL_HPP_



#include "octotiger/unitiger/unitiger.hpp"
#include "octotiger/unitiger/physics.hpp"

template<int NDIM, int INX, class PHYS>
hydro_computer<NDIM, INX, PHYS>::hydro_computer() {
	nf_ = PHYS::field_count();

	angmom_index_ = -1;

	for( int f = 0; f < nf_; f++) {
		smooth_field_.push_back(false);
		disc_detect_.push_back(false);
	}
	bc_.resize(2*NDIM,OUTFLOW);
}

template<int NDIM, int INX, class PHYS>
void hydro_computer<NDIM, INX, PHYS>::use_smooth_recon(int field) {
	smooth_field_[field] = true;
}

template<int NDIM, int INX, class PHYS>
void hydro_computer<NDIM, INX, PHYS>::use_disc_detect(int field) {
	disc_detect_[field] = true;
}



template<int NDIM, int INX, class PHYS>
void hydro_computer<NDIM, INX, PHYS>::use_angmom_correction(int index) {
	angmom_index_ = index;
}

template<int NDIM, int INX, class PHYS>
void hydro_computer<NDIM, INX, PHYS>::post_process(hydro::state_type &U, safe_real dx) {
	PHYS p;
	p.template post_process < NDIM > (U, dx);
}

template<int NDIM, int INX, class PHYS>
std::vector<safe_real>  hydro_computer<NDIM, INX, PHYS>::get_field_sums(const hydro::state_type &U, safe_real dx) {
	std::vector<safe_real> sums(nf_,0.0);
	static const auto indices = geo::find_indices(geo::H_BW,geo::H_NX-geo::H_BW);
	for( int f = 0; f < nf_; f++) {
		for( auto i : indices) {
			sums[f] += U[f][i] * (dx * dx * dx);
		}
	}
	return sums;
}

template<int NDIM, int INX, class PHYS>
std::vector<safe_real>  hydro_computer<NDIM, INX, PHYS>::get_field_mags(const hydro::state_type &U, safe_real dx) {
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
