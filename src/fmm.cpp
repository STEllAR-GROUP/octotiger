/*
 * fmm.cpp
 *
 *  Created on: Feb 22, 2016
 *      Author: dmarce1
 */

#include "fmm.hpp"

void fmm::set_dx(real d) {
	dx = d;
	dx_set = true;
}

fmm::fmm() :
		M(IN3), L(IN3), dx(1.0), dx_set(false) {
}

void fmm::set_source(real this_m, integer i, integer j, integer k) {
	assert(dx_set);
	const integer l = index(i, j, k);
	M[l].at(0, 0) = this_m;
	for (int l = 1; l < M_POLES; ++l) {
		for (int m = 0; m <= l; ++m) {
			M[l](l, m) = std::complex < real > (0.0, 0.0);
		}
	}
	//	printf( "%e %e\n", this_m, M[index(i, j, k)](0,0).real());
}

std::vector<multipole_type> fmm::M2M() {
	assert(dx_set);
	std::vector<multipole_type> Mcoarse(INX * INX * INX / 8);
	for (integer i = 0; i != INX; ++i) {
		const real x = (real(i % 2) - 0.5) * dx;
		for (integer j = 0; j != INX; ++j) {
			const real y = (real(j % 2) - 0.5) * dx;
			for (integer k = 0; k != INX; ++k) {
				const real z = (real(k % 2) - 0.5) * dx;
				const auto ic = coarse_index(i / 2, j / 2, k / 2);
				const auto ii = index(i, j, k);
				Mcoarse[ic].M2M(M[ii], x, y, z);
				L[ic] = expansion_type();
			}
		}
	}
	return Mcoarse;
}

std::vector<multipole_type> fmm::get_multipoles(const std::array<integer, NDIM>& lb,
		const std::array<integer, NDIM>& ub) const {
	assert(dx_set);
	integer sz = 1;
	std::vector<multipole_type> Q;
	for (integer d = 0; d != NDIM; ++d) {
		sz *= (ub[d] - lb[d] + 1);
	}
	Q.reserve(sz);
	for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
			for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
				Q.push_back(M[index(i, j, k)]);
			}
		}
	}
	return Q;
}

std::array<real, NDIM + 1> fmm::four_force(integer i, integer j, integer k) const {
	return {get_phi(i,j,k),get_gx(i,j,k),get_gy(i,j,k),get_gz(i,j,k)};
}

real fmm::get_phi(integer i, integer j, integer k) const {
	return -L[index(i, j, k)](0, 0).real();
}

real fmm::get_gx(integer i, integer j, integer k) const {
	const auto& ii = index(i, j, k);
	const auto& gp = -L[ii](1, +1);
	const auto& gm = -L[ii](1, -1);
	return 0.5 * (gp - gm).real();
}

real fmm::get_gy(integer i, integer j, integer k) const {
	const auto& ii = index(i, j, k);
	const auto& gp = L[ii](1, +1);
	const auto& gm = L[ii](1, -1);
	return 0.5 * (gp + gm).imag();
}

real fmm::get_gz(integer i, integer j, integer k) const {
	return L[index(i, j, k)](1, 0).real();
}

std::vector<expansion_type> fmm::get_expansions(const geo::octant octant) const {
	assert(dx_set);
	std::vector<expansion_type> l;
	const auto w = INX / 2;
	l.reserve(INX * INX * INX / 8);
	for (integer i = w * octant[XDIM]; i < w * (1 + octant[XDIM]); ++i) {
		for (integer j = w * octant[YDIM]; j < w * (1 + octant[YDIM]); ++j) {
			for (integer k = w * octant[ZDIM]; k < w * (1 + octant[ZDIM]); ++k) {
				l.push_back(L[index(i, j, k)]);
			}
		}
	}
	return l;
}

void fmm::set_multipoles(const std::vector<multipole_type>& Mfine, const geo::octant octant) {
	assert(dx_set);
	const auto w = INX / 2;
	integer l = 0;
	for (integer i = w * octant[XDIM]; i < w * (1 + octant[XDIM]); ++i) {
		for (integer j = w * octant[YDIM]; j < w * (1 + octant[YDIM]); ++j) {
			for (integer k = w * octant[ZDIM]; k < w * (1 + octant[ZDIM]); ++k) {
				M[index(i, j, k)] = Mfine[l];
				++l;
			}
		}
	}
}

void fmm::L2L(const std::vector<expansion_type>& Lcoarse) {
	assert(dx_set);
	integer l = 0;
	const real d = 0.5 * dx;
	for (integer i = 0; i < INX / 2; ++i) {
		for (integer j = 0; j < INX / 2; ++j) {
			for (integer k = 0; k < INX / 2; ++k) {
				L[index(2 * i + 0, 2 * j + 0, 2 * k + 0)].L2L(Lcoarse[l], -d, -d, -d);
				L[index(2 * i + 0, 2 * j + 0, 2 * k + 1)].L2L(Lcoarse[l], -d, -d, +d);
				L[index(2 * i + 0, 2 * j + 1, 2 * k + 0)].L2L(Lcoarse[l], -d, +d, -d);
				L[index(2 * i + 0, 2 * j + 1, 2 * k + 1)].L2L(Lcoarse[l], -d, +d, +d);
				L[index(2 * i + 1, 2 * j + 0, 2 * k + 0)].L2L(Lcoarse[l], +d, -d, -d);
				L[index(2 * i + 1, 2 * j + 0, 2 * k + 1)].L2L(Lcoarse[l], +d, -d, +d);
				L[index(2 * i + 1, 2 * j + 1, 2 * k + 0)].L2L(Lcoarse[l], +d, +d, -d);
				L[index(2 * i + 1, 2 * j + 1, 2 * k + 1)].L2L(Lcoarse[l], +d, +d, +d);
				++l;
			}
		}
	}
}

void fmm::self_M2L(bool is_root, bool is_leaf) {
	other_M2L(M, { 0, 0, 0 }, { INX, INX, INX }, is_leaf, is_root ? INX : 1);
}

void fmm::other_M2L(const std::vector<multipole_type>& Q, const std::array<integer, NDIM>& lb,
		const std::array<integer, NDIM>& ub, bool is_leaf, integer width) {
	assert(dx_set);
	const auto this_index = [&](integer i, integer j, integer k) {
		const auto ny = ub[YDIM] - lb[YDIM];
		const auto nz = ub[ZDIM] - lb[ZDIM];
		return ((i - lb[XDIM]) * ny + (j - lb[YDIM])) * nz + (k - lb[ZDIM]);
	};
	for (integer i0 = lb[XDIM]; i0 < ub[XDIM]; ++i0) {
		const integer i1b = std::max(2 * (((i0 + 2 * width) / 2) - width) - 2 * width, integer(0));
		const integer i1e = std::min(2 * (((i0 + 2 * width) / 2) + width) + 2 - 2 * width, INX);
		for (integer j0 = lb[YDIM]; j0 < ub[YDIM]; ++j0) {
			const integer j1b = std::max(2 * (((j0 + 2 * width) / 2) - width) - 2 * width, integer(0));
			const integer j1e = std::min(2 * (((j0 + 2 * width) / 2) + width) + 2 - 2 * width, INX);
			for (integer k0 = lb[ZDIM]; k0 < ub[ZDIM]; ++k0) {
				const integer k1b = std::max(2 * (((k0 + 2 * width) / 2) - width) - 2 * width, integer(0));
				const integer k1e = std::min(2 * (((k0 + 2 * width) / 2) + width) + 2 - 2 * width, INX);
				for (integer i1 = i1b; i1 < i1e; ++i1) {
					for (integer j1 = j1b; j1 < j1e; ++j1) {
						for (integer k1 = k1b; k1 < k1e; ++k1) {
							const integer ii0 = this_index(i0, j0, k0);
							const integer ii1 = index(i1, j1, k1);
							const auto max_dist = std::max(std::max(std::abs(i0 - i1), std::abs(j0 - j1)),
									std::abs(k0 - k1));
							if (max_dist > 1 || (is_leaf && max_dist > 0)) {
								const real x = (i0 - i1) * dx;
								const real y = (j0 - j1) * dx;
								const real z = (k0 - k1) * dx;
								/*			multipole_expansion<M_POLES> MM;
								 multipole_expansion<M_POLES> MP(Q[ii0]);
								 for (integer l = 0; l < M_POLES - 1; ++l) {
								 for (integer m = 0; m <= l; ++m) {
								 MM.at(l, m) = std::complex < real > (0, 0);
								 }
								 }
								 #ifdef USE_MOM_CORRECTION
								 auto l = M_POLES - 1;
								 const auto m1 = Q[ii0](0, 0).real();
								 const auto m0 = M[ii1](0, 0).real();
								 const auto mu = m1 / m0;
								 for (integer m = 0; m <= l; ++m) {
								 MM.at(l, m) = Q[ii0](l, m) - mu * M[ii1](l, m);
								 //MP.at(l, m) = std::complex < real > (0, 0);
								 }
								 #endif
								 auto Lc = MM.M2L(x, y, z);*/
								L[ii1] += Q[ii0].M2L(x, y, z, M[ii1]);
								//		L[ii1].at(1, 0) += Lc(1, 0);
								//		L[ii1].at(1, 1) += Lc(1, 1);
							}
						}
					}
				}
			}
		}
	}
}
