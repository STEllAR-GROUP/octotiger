/*
 * grid_fmm.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: dmarce1
 */
#include "grid.hpp"
#include "simd.hpp"
#include "profiler.hpp"
#include "taylor.hpp"

void find_eigenvectors(real q[3][3], real e[3][3], real lambda[3]) {
	PROF_BEGIN;
	real b0[3], b1[3], A, bdif;
	int iter = 0;
	for (int l = 0; l < 3; l++) {
		b0[0] = b0[1] = b0[2] = 0.0;
		b0[l] = 1.0;
		do {
			iter++;
			for (int i = 0; i < 3; i++) {
				b1[i] = 0.0;
			}
			for (int i = 0; i < 3; i++) {
				for (int m = 0; m < 3; m++) {
					b1[i] += q[i][m] * b0[m];
				}
			}
			A = 0.0;
			for (int i = 0; i < 3; i++) {
				A += b1[i] * b1[i];
			}
			A = sqrt(A);
			bdif = 0.0;
			for (int i = 0; i < 3; i++) {
				b1[i] = b1[i] / A;
				bdif += pow(b0[i] - b1[i], 2);
			}
			for (int i = 0; i < 3; i++) {
				b0[i] = b1[i];
			}

		} while (fabs(bdif) > 1.0e-14);
		for (int m = 0; m < 3; m++) {
			e[l][m] = b0[m];
		}
		for (int i = 0; i < 3; i++) {
			A += b0[i] * q[l][i];
		}
		lambda[l] = sqrt(A) / sqrt(e[l][0] * e[l][0] + e[l][1] * e[l][1] + e[l][2] * e[l][2]);
	}
	PROF_END;
}

std::pair<space_vector, space_vector> grid::find_axis() const {
	PROF_BEGIN;
	real quad_moment[NDIM][NDIM];
	real eigen[NDIM][NDIM];
	real lambda[NDIM];
	space_vector this_com;
	real mtot = 0.0;
	for (integer i = 0; i != NDIM; ++i) {
		this_com[i] = 0.0;
		for (integer j = 0; j != NDIM; ++j) {
			quad_moment[i][j] = 0.0;
		}
	}

	for (integer i = G_BW; i != G_NX - G_BW; ++i) {
		for (integer j = G_BW; j != G_NX - G_BW; ++j) {
			for (integer k = G_BW; k != G_NX - G_BW; ++k) {
				const integer iii1 = gindex(i, j, k);
				const integer iii0 = gindex(i + H_BW - G_BW, j + H_BW - G_BW, k + H_BW - G_BW);
				for (integer n = 0; n != NDIM; ++n) {
					this_com[n] += M[0][iii1]() * com[0][iii1][n];
					mtot += M[0][iii1]();
					for (integer m = 0; m != NDIM; ++m) {
						quad_moment[n][m] += M[0][iii1](n, m);
						quad_moment[n][m] += M[0][iii1]() * com[0][iii1][n] * com[0][iii1][m];
					}
				}
			}
		}
	}
	for (integer j = 0; j != NDIM; ++j) {
		this_com[j] /= mtot;
	}

	find_eigenvectors(quad_moment, eigen, lambda);
	integer index;
	if (lambda[0] > lambda[1] && lambda[0] > lambda[2]) {
		index = 0;
	} else if (lambda[1] > lambda[2]) {
		index = 1;
	} else {
		index = 2;
	}
	space_vector rc;
	for (integer j = 0; j != NDIM; ++j) {
		rc[j] = eigen[index][j];
	}
	std::pair<space_vector, space_vector> pair;
	pair.first = rc;
	pair.second = this_com;
	PROF_END;
	return pair;
}

void grid::solve_gravity(gsolve_type type) {

	compute_multipoles(type);
	compute_interactions(type);
	compute_expansions(type);
}

void grid::compute_interactions(gsolve_type type) {
	PROF_BEGIN;
	npair np;
	dpair dp;
	for (integer lev = 0; lev != nlevel; ++lev) {
		std::fill(std::begin(L[lev]), std::end(L[lev]), ZERO);
		std::fill(std::begin(L_c[lev]), std::end(L_c[lev]), ZERO);
	}
	const integer list_size = ilist_n.size();
	taylor<4, simd_vector> m0;
	taylor<4, simd_vector> m1;
	taylor<4, simd_vector> n0;
	taylor<4, simd_vector> n1;
	std::array<simd_vector, NDIM> dX;
	std::array < simd_vector, NDIM > X;
	std::array<simd_vector, NDIM> Y;
	for (integer li = 0; li < list_size; li += simd_len) {
		for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
			const integer iii0 = ilist_n[li + i].loc.first;
			const integer iii1 = ilist_n[li + i].loc.second;
			const integer lev = ilist_n[li + i].lev;
			for (auto& d : geo::dimension::full_set()) {
				X[d][i] = com[lev][iii0][d];
				Y[d][i] = com[lev][iii1][d];
			}
			for (integer j = 0; j != 20; ++j) {
				m0.ptr()[j][i] = M[lev][iii1].ptr()[j];
				m1.ptr()[j][i] = M[lev][iii0].ptr()[j];
			}
			for (integer j = 10; j != 20; ++j) {
				if (type == RHO) {
					n0.ptr()[j][i] = M[lev][iii1].ptr()[j] - M[lev][iii0].ptr()[j] * (M[lev][iii1]() / M[lev][iii0]());
					n1.ptr()[j][i] = M[lev][iii0].ptr()[j] - M[lev][iii1].ptr()[j] * (M[lev][iii0]() / M[lev][iii1]());
				} else {
					n0.ptr()[j][i] = ZERO;
					n1.ptr()[j][i] = ZERO;
				}
			}
		}
		for (auto& d : geo::dimension::full_set()) {
			dX[d] = X[d] - Y[d];
		}
		taylor<5, simd_vector> D;
		taylor<4, simd_vector> A0, A1;
		std::array<simd_vector, NDIM> B0 = { ZERO, ZERO, ZERO };
		std::array<simd_vector, NDIM> B1 = { ZERO, ZERO, ZERO };
		D.set_basis(dX);
		A0() = m0() * D();
		A1() = m1() * D();
		for (auto& a : geo::dimension::full_set()) {
			if (type != RHO) {
				A0() -= m0(a) * D(a);
				A1() += m1(a) * D(a);
			}
			A0(a) = +m0() * D(a);
			A1(a) = -m1() * D(a);
			for (auto& b : geo::dimension::full_set()) {
				const auto tmp = D(a, b) * (real(1) / real(2));
				A0() += m0(a, b) * tmp;
				A1() += m1(a, b) * tmp;
				if (type != RHO) {
					A0(a) -= m0(a) * D(a, b);
					A1(a) -= m1(a) * D(a, b);
				}
				A0(a, b) = m0() * D(a, b);
				A1(a, b) = m1() * D(a, b);

				for (auto& c : geo::dimension::full_set()) {
					const auto tmp0 = D(a, b, c) * (real(1) / real(6));
					A0() -= m0(a, b, c) * tmp0;
					A1() += m1(a, b, c) * tmp0;

					const auto tmp1 = D(a, b, c) * (real(1) / real(2));
					A0(a) += m0(c, b) * tmp1;
					A1(a) -= m1(c, b) * tmp1;

					const auto tmp2 = D(a, b, c);
					A0(a, b) -= m0(c) * tmp2;
					A1(a, b) += m1(c) * tmp2;

					A1(a, b, c) = -m1() * tmp2;
					A0(a, b, c) = +m0() * tmp2;

					if (type == RHO) {
						for (integer d = 0; d != NDIM; ++d) {
							const auto tmp = D(a, b, c, d) * (real(1) / real(6));
							B0[a] -= n0(b, c, d) * tmp;
							B1[a] -= n1(b, c, d) * tmp;
						}
					}
				}

			}
		}

		for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
			const integer iii0 = ilist_n[li + i].loc.first;
			const integer iii1 = ilist_n[li + i].loc.second;
			const integer lev = ilist_n[li + i].lev;
			for (integer j = 0; j != 20; ++j) {
				L[lev][iii0].ptr()[j] += A0.ptr()[j][i];
				L[lev][iii1].ptr()[j] += A1.ptr()[j][i];
			}
			if (type == RHO) {
				for (integer j = 0; j != NDIM; ++j) {
					L_c[lev][iii0][j] += B0[j][i];
					L_c[lev][iii1][j] += B1[j][i];
				}
			}
		}
	}

	const integer dsize = ilist_d.size();
	const integer lev = 0;
	for (integer li = 0; li < dsize; li += simd_len) {
		simd_vector m0, m1;
		for (integer i = 0; i != simd_len && li + i < dsize; ++i) {
			const integer iii0 = ilist_d[li + i].loc.first;
			const integer iii1 = ilist_d[li + i].loc.second;
			for (integer d = 0; d != NDIM; ++d) {
				X[d][i] = com[lev][iii0][d];
				Y[d][i] = com[lev][iii1][d];
			}
			m0[i] = M[lev][iii1]();
			m1[i] = M[lev][iii0]();
		}
		simd_vector phi0, phi1, gx0, gx1, gy0, gy1, gz0, gz1;
		std::array < simd_vector, NDIM > dX;
		simd_vector r = ZERO;
		for (auto& d : geo::dimension::full_set()) {
			dX[d] = X[d] - Y[d];
			r += dX[d] * dX[d];
		}
		r = sqrt(r);
		const simd_vector rinv = ONE / r;
		const simd_vector r3inv = ONE / (r * r * r);
		phi0 = -m0 * rinv;
		phi1 = -m1 * rinv;
		for (auto& d : geo::dimension::full_set()) {
			dX[d] *= r3inv;
		}
		if (type == RHO) {
			gx0 = +m0 * dX[XDIM];
			gy0 = +m0 * dX[YDIM];
			gz0 = +m0 * dX[ZDIM];
			gx1 = -m1 * dX[XDIM];
			gy1 = -m1 * dX[YDIM];
			gz1 = -m1 * dX[ZDIM];
		}
		for (integer i = 0; i != simd_len && i + li < dsize; ++i) {
			const integer iii0 = ilist_d[li + i].loc.first;
			const integer iii1 = ilist_d[li + i].loc.second;
			L[lev][iii0]() += phi0[i];
			L[lev][iii1]() += phi1[i];
			if (type == RHO) {
				L[lev][iii1](XDIM) += gx1[i];
				L[lev][iii1](YDIM) += gy1[i];
				L[lev][iii1](ZDIM) += gz1[i];
				L[lev][iii0](XDIM) += gx0[i];
				L[lev][iii0](YDIM) += gy0[i];
				L[lev][iii0](ZDIM) += gz0[i];
			}
		}
	}
	PROF_END;
}

void grid::compute_boundary_interactions(gsolve_type type, const geo::direction& dir, bool is_monopole) {
	if (!is_leaf) {
		if (!is_monopole) {
			compute_boundary_interactions_multipole_multipole(type, ilist_n_bnd[dir]);
		} else {
			compute_boundary_interactions_monopole_multipole(type, ilist_d_bnd[dir]);
		}
	} else {
		if (!is_monopole) {
			compute_boundary_interactions_multipole_monopole(type, ilist_d_bnd[dir]);
		} else {
			compute_boundary_interactions_monopole_monopole(type, ilist_d_bnd[dir]);
		}
	}

}

void grid::compute_boundary_interactions_multipole_multipole(gsolve_type type, const std::vector<npair>& ilist_n_bnd) {
	PROF_BEGIN;
	npair np;
	const integer list_size = ilist_n_bnd.size();
	taylor<4, simd_vector> m0;
	taylor<4, simd_vector> n0;
	std::array<simd_vector, NDIM> dX;
	std::array < simd_vector, NDIM > X;
	std::array<simd_vector, NDIM> Y;
	for (integer li = 0; li < list_size; li += simd_len) {
		for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
			const integer iii0 = ilist_n_bnd[li + i].loc.first;
			const integer iii1 = ilist_n_bnd[li + i].loc.second;
			const integer lev = ilist_n_bnd[li + i].lev;
			for (auto& d : geo::dimension::full_set()) {
				X[d][i] = com[lev][iii0][d];
				Y[d][i] = com[lev][iii1][d];
			}
			for (integer j = 0; j != 20; ++j) {
				m0.ptr()[j][i] = M[lev][iii1].ptr()[j];
			}
			for (integer j = 10; j != 20; ++j) {
				if (type == RHO) {
					n0.ptr()[j][i] = M[lev][iii1].ptr()[j] - M[lev][iii0].ptr()[j] * (M[lev][iii1]() / M[lev][iii0]());
				} else {
					n0.ptr()[j][i] = ZERO;
				}
			}
		}
		for (auto& d : geo::dimension::full_set()) {
			dX[d] = X[d] - Y[d];
		}

		taylor<5, simd_vector> D;
		taylor<4, simd_vector> A0;
		std::array<simd_vector, NDIM> B0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

		D.set_basis(dX);

		A0() = m0() * D();
		for (auto& a : geo::dimension::full_set()) {
			if (type != RHO) {
				A0() -= m0(a) * D(a);
			}
			A0(a) = +m0() * D(a);
			for (auto& b : geo::dimension::full_set()) {
				A0() += m0(a, b) * D(a, b) * (real(1) / real(2));
				if (type != RHO) {
					A0(a) -= m0(a) * D(a, b);
				}
				A0(a, b) = m0() * D(a, b);
				for (auto& c : geo::dimension::full_set()) {
					A0() -= m0(a, b, c) * D(a, b, c) * (real(1) / real(6));

					A0(a) += m0(c, b) * D(a, b, c) * (real(1) / real(2));

					if (type == RHO) {
						for (auto& d : geo::dimension::full_set()) {
							const auto tmp = D(a, b, c, d) * (real(1) / real(6));
							B0[a] -= n0(b, c, d) * tmp;
						}
					}
					A0(a, b) -= m0(c) * D(a, b, c);
					A0(a, b, c) = +m0() * D(a, b, c);
				}
			}
		}

		for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
			const integer iii0 = ilist_n_bnd[li + i].loc.first;
			const integer lev = ilist_n_bnd[li + i].lev;
			for (integer j = 0; j != 20; ++j) {
				L[lev][iii0].ptr()[j] += A0.ptr()[j][i];
			}
			if (type == RHO) {
				for (integer j = 0; j != NDIM; ++j) {
					L_c[lev][iii0][j] += B0[j][i];
				}
			}
		}
	}
	PROF_END;
}

void grid::compute_boundary_interactions_multipole_monopole(gsolve_type type, const std::vector<npair>& ilist_n_bnd) {
	PROF_BEGIN;
	npair np;
	const integer list_size = ilist_n_bnd.size();
	taylor<4, simd_vector> m0;
	taylor<4, simd_vector> n0;
	std::array<simd_vector, NDIM> dX;
	std::array < simd_vector, NDIM > X;
	std::array<simd_vector, NDIM> Y;
	for (integer li = 0; li < list_size; li += simd_len) {
		for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
			const integer iii0 = ilist_n_bnd[li + i].loc.first;
			const integer iii1 = ilist_n_bnd[li + i].loc.second;
			const integer lev = ilist_n_bnd[li + i].lev;
			for (auto& d : geo::dimension::full_set()) {
				X[d][i] = com[lev][iii0][d];
				Y[d][i] = com[lev][iii1][d];
			}
			for (integer j = 0; j != 20; ++j) {
				m0.ptr()[j][i] = M[lev][iii1].ptr()[j];
			}
			for (integer j = 10; j != 20; ++j) {
				if (type == RHO) {
					n0.ptr()[j][i] = M[lev][iii1].ptr()[j] - M[lev][iii0].ptr()[j] * (M[lev][iii1]() / M[lev][iii0]());
				} else {
					n0.ptr()[j][i] = ZERO;
				}
			}
		}
		for (auto& d : geo::dimension::full_set()) {
			dX[d] = X[d] - Y[d];
		}

		taylor<5, simd_vector> D;
		taylor<2, simd_vector> A0;
		std::array<simd_vector, NDIM> B0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

		D.set_basis(dX);

		A0() = m0() * D();
		for (auto& a : geo::dimension::full_set()) {
			if (type != RHO) {
				A0() -= m0(a) * D(a);
			}
			A0(a) = +m0() * D(a);
			for (auto& b : geo::dimension::full_set()) {
				A0() += m0(a, b) * D(a, b) * (real(1) / real(2));
				if (type != RHO) {
					A0(a) -= m0(a) * D(a, b);
				}
				for (auto& c : geo::dimension::full_set()) {
					A0() -= m0(a, b, c) * D(a, b, c) * (real(1) / real(6));

					A0(a) += m0(c, b) * D(a, b, c) * (real(1) / real(2));

					if (type == RHO) {
						for (auto& d : geo::dimension::full_set()) {
							const auto tmp = D(a, b, c, d) * (real(1) / real(6));
							B0[a] -= n0(b, c, d) * tmp;
						}
					}
				}
			}
		}

		for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
			const integer iii0 = ilist_n_bnd[li + i].loc.first;
			const integer lev = ilist_n_bnd[li + i].lev;
			for (integer j = 0; j != 4; ++j) {
				L[lev][iii0].ptr()[j] += A0.ptr()[j][i];
			}
			if (type == RHO) {
				for (integer j = 0; j != NDIM; ++j) {
					L_c[lev][iii0][j] += B0[j][i];
				}
			}
		}
	}
	PROF_END;
}

void grid::compute_boundary_interactions_monopole_multipole(gsolve_type type, const std::vector<npair>& ilist_n_bnd) {
	PROF_BEGIN;
	npair np;
	const integer list_size = ilist_n_bnd.size();
	simd_vector m0;
	taylor<4, simd_vector> n0;
	std::array<simd_vector, NDIM> dX;
	std::array < simd_vector, NDIM > X;
	std::array<simd_vector, NDIM> Y;
	for (integer li = 0; li < list_size; li += simd_len) {
		for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
			const integer iii0 = ilist_n_bnd[li + i].loc.first;
			const integer iii1 = ilist_n_bnd[li + i].loc.second;
			const integer lev = ilist_n_bnd[li + i].lev;
			for (auto& d : geo::dimension::full_set()) {
				X[d][i] = com[lev][iii0][d];
				Y[d][i] = com[lev][iii1][d];
			}
			m0[i] = M[lev][iii1]();
			for (integer j = 10; j != 20; ++j) {
				if (type == RHO) {
					n0.ptr()[j][i] = -M[lev][iii0].ptr()[j] * (M[lev][iii1]() / M[lev][iii0]());
				} else {
					n0.ptr()[j][i] = ZERO;
				}
			}
		}
		for (auto& d : geo::dimension::full_set()) {
			dX[d] = X[d] - Y[d];
		}

		taylor<5, simd_vector> D;
		taylor<4, simd_vector> A0;
		std::array<simd_vector, NDIM> B0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

		D.set_basis(dX);

		A0() = m0 * D();
		for (auto& a : geo::dimension::full_set()) {
			A0(a) = +m0 * D(a);
			for (auto& b : geo::dimension::full_set()) {
				A0(a, b) = m0 * D(a, b);
				for (auto& c : geo::dimension::full_set()) {
					if (type == RHO) {
						for (auto& d : geo::dimension::full_set()) {
							const auto tmp = D(a, b, c, d) * (real(1) / real(6));
							B0[a] -= n0(b, c, d) * tmp;
						}
					}
					A0(a, b, c) = +m0 * D(a, b, c);
				}
			}
		}

		for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
			const integer iii0 = ilist_n_bnd[li + i].loc.first;
			const integer lev = ilist_n_bnd[li + i].lev;
			for (integer j = 0; j != 20; ++j) {
				L[lev][iii0].ptr()[j] += A0.ptr()[j][i];
			}
			if (type == RHO) {
				for (integer j = 0; j != NDIM; ++j) {
					L_c[lev][iii0][j] += B0[j][i];
				}
			}
		}
	}
	PROF_END;
}

void grid::compute_boundary_interactions_monopole_monopole(gsolve_type type, const std::vector<npair>& ilist_d_bnd) {
	PROF_BEGIN;
	const integer dsize = ilist_d_bnd.size();
	dpair dp;
	std::array < simd_vector, NDIM > X;
	std::array<simd_vector, NDIM> Y;
	const integer lev = 0;
	for (integer li = 0; li < dsize; li += simd_len) {
		simd_vector m0;
		for (integer i = 0; i != simd_len && li + i < dsize; ++i) {
			const integer iii0 = ilist_d_bnd[li + i].loc.first;
			const integer iii1 = ilist_d_bnd[li + i].loc.second;
			for (auto& d : geo::dimension::full_set()) {
				X[d][i] = com[lev][iii0][d];
				Y[d][i] = com[lev][iii1][d];
			}
			m0[i] = M[lev][iii1]();
		}
		simd_vector phi0, gx0, gy0, gz0;
		std::array<simd_vector, NDIM> dX;
		simd_vector r = ZERO;
		for (auto& d : geo::dimension::full_set()) {
			dX[d] = X[d] - Y[d];
			r += dX[d] * dX[d];
		}
		r = sqrt(r);
		const simd_vector rinv = ONE / r;
		const simd_vector r3inv = ONE / (r * r * r);
		phi0 = -m0 * rinv;
		for (auto& d : geo::dimension::full_set()) {
			dX[d] *= r3inv;
		}
		if (type == RHO) {
			gx0 = +m0 * dX[XDIM];
			gy0 = +m0 * dX[YDIM];
			gz0 = +m0 * dX[ZDIM];
		}
		for (integer i = 0; i != simd_len && i + li < dsize; ++i) {
			const integer iii0 = ilist_d_bnd[li + i].loc.first;
			L[lev][iii0]() += phi0[i];
			if (type == RHO) {
				L[lev][iii0](XDIM) += gx0[i];
				L[lev][iii0](YDIM) += gy0[i];
				L[lev][iii0](ZDIM) += gz0[i];
			}
		}
	}
	PROF_END;
}

void grid::compute_ilist() {
	PROF_BEGIN;
	std::vector<std::vector<geo::direction> > neighbor_num(nlevel, std::vector<geo::direction>(G_N3, geo::direction(-1)));
	integer lev = nlevel - 2;
	for (integer inx = 4; inx <= INX; inx <<= 1) {
		const integer nx = inx + 2 * G_BW;
		for (integer i0 = 0; i0 != nx; ++i0) {
			for (integer j0 = 0; j0 != nx; ++j0) {
				for (integer k0 = 0; k0 != nx; ++k0) {
					bool on_bnd = false;
					const integer iii0 = i0 * nx * nx + j0 * nx + k0;
					integer x, y, z;
					if (k0 < G_BW) {
						z = -1;
						on_bnd = true;
					} else if (k0 >= nx - G_BW) {
						z = +1;
						on_bnd = true;
					} else {
						z = 0;
					}
					if (j0 < G_BW) {
						y = -1;
						on_bnd = true;
					} else if (j0 >= nx - G_BW) {
						y = +1;
						on_bnd = true;
					} else {
						y = 0;
					}
					if (i0 < G_BW) {
						x = -1;
						on_bnd = true;
					} else if (i0 >= nx - G_BW) {
						x = +1;
						on_bnd = true;
					} else {
						x = 0;
					}
					if (on_bnd) {
						neighbor_num[lev][iii0].set(x, y, z);
					}
				}
			}
		}
		--lev;
	}
	lev = nlevel - 2;
	npair np;
	dpair dp;
	std::vector<npair> ilist_n0;
	std::vector<dpair> ilist_d0;
	std::array<std::vector<npair>, geo::direction::count()> ilist_n0_bnd;
	std::array<std::vector<dpair>, geo::direction::count()> ilist_d0_bnd;
	const auto W = G_BW / 2;
	for (integer inx = 4; inx <= INX; inx <<= 1) {
		if (is_root || lev == 0) {
			const integer nx = inx + 2 * G_BW;
			for (integer i0 = 0; i0 != nx; ++i0) {
				for (integer j0 = 0; j0 != nx; ++j0) {
					for (integer k0 = 0; k0 != nx; ++k0) {
						const integer iii0 = i0 * nx * nx + j0 * nx + k0;
						const integer imin = std::max(integer(0), 2 * ((i0 / 2) - W));
						const integer jmin = std::max(integer(0), 2 * ((j0 / 2) - W));
						const integer kmin = std::max(integer(0), 2 * ((k0 / 2) - W));
						const integer imax = std::min(integer(nx - 1), 2 * ((i0 / 2) + W) + 1);
						const integer jmax = std::min(integer(nx - 1), 2 * ((j0 / 2) + W) + 1);
						const integer kmax = std::min(integer(nx - 1), 2 * ((k0 / 2) + W) + 1);
						for (integer i1 = imin; i1 <= imax; ++i1) {
							for (integer j1 = jmin; j1 <= jmax; ++j1) {
								for (integer k1 = kmin; k1 <= kmax; ++k1) {
									const integer iii1 = i1 * nx * nx + j1 * nx + k1;
									integer max_dist = std::max(std::abs(k0 - k1), std::max(std::abs(i0 - i1), std::abs(j0 - j1)));
									dp.lev = lev;
									dp.loc.first = iii0;
									dp.loc.second = iii1;
									if (neighbor_num[lev][iii1] != -1 && neighbor_num[lev][iii0] == -1) {
										ilist_d0_bnd[neighbor_num[lev][iii1]].push_back(dp);
									}
									if ((max_dist > W) && ((lev != 0) || !is_leaf)) {
										np.lev = lev;
										np.loc.first = iii0;
										np.loc.second = iii1;
										if (neighbor_num[lev][iii1] == -1 && neighbor_num[lev][iii0] == -1) {
											if (iii1 > iii0) {
												ilist_n0.push_back(np);
											}
										} else if (neighbor_num[lev][iii0] == -1) {
											ilist_n0_bnd[neighbor_num[lev][iii1]].push_back(np);
										}
									} else {
										if ((lev == 0) && (max_dist > 0) && is_leaf) {
											if (neighbor_num[lev][iii1] == -1 && neighbor_num[lev][iii0] == -1) {
												if (iii1 > iii0) {
													ilist_d0.push_back(dp);
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		--lev;
	}
	ilist_n = std::vector<npair>(ilist_n0.begin(), ilist_n0.end());
	ilist_d = std::vector<dpair>(ilist_d0.begin(), ilist_d0.end());
	for (auto& dir : geo::direction::full_set()) {
		ilist_n_bnd[dir] = std::vector<npair>(ilist_n0_bnd[dir].begin(), ilist_n0_bnd[dir].end());
		ilist_d_bnd[dir] = std::vector<dpair>(ilist_d0_bnd[dir].begin(), ilist_d0_bnd[dir].end());
	}
	PROF_END;
}

expansion_pass_type grid::compute_expansions(gsolve_type type, const expansion_pass_type* parent_expansions) {
	PROF_BEGIN;
	integer lev = is_root ? nlevel - 1 : 1;
	expansion_pass_type exp_ret;
	if (!is_leaf) {
		exp_ret.first.resize(INX * INX * INX);
		if (type == RHO) {
			exp_ret.second.resize(INX * INX * INX);
		}
	}
	for (integer inx = is_root ? 4 : INX; inx <= INX; inx <<= 1) {
		--lev;
		const integer nxp = (inx / 2) + 2 * G_BW;
		auto child_index = [=](integer ip, integer jp, integer kp, integer ci, integer bw=G_BW) -> integer {
			const integer ic = (2 * (ip - G_BW)+bw) + ((ci >> 0) & 1);
			const integer jc = (2 * (jp - G_BW)+bw) + ((ci >> 1) & 1);
			const integer kc = (2 * (kp - G_BW)+bw) + ((ci >> 2) & 1);
			return (inx+2*bw) * (inx+2*bw) * ic + (inx+2*bw) * jc + kc;
		};

		for (integer ip = G_BW; ip != nxp - G_BW; ++ip) {
			for (integer jp = G_BW; jp != nxp - G_BW; ++jp) {
				for (integer kp = G_BW; kp != nxp - G_BW; ++kp) {
					const integer iiip = nxp * nxp * ip + nxp * jp + kp;
					if (!((lev == 1) && !is_root)) {
						std::array < simd_vector, NDIM > X;
						std::array<simd_vector, NDIM> dX;
						taylor<4, simd_vector> l;
						std::array<simd_vector, NDIM> lc;
						if (!is_root && lev == 0) {
							const integer index = (INX * INX / 4) * (ip - G_BW) + (INX / 2) * (jp - G_BW) + (kp - G_BW);
							for (integer j = 0; j != 20; ++j) {
								l.ptr()[j] = parent_expansions->first[index].ptr()[j];
							}
							if (type == RHO) {
								for (integer j = 0; j != NDIM; ++j) {
									lc[j] = parent_expansions->second[index][j];
								}
							}
						} else {
							for (integer j = 0; j != 20; ++j) {
								l.ptr()[j] = L[lev + 1][iiip].ptr()[j];
							}
							for (integer j = 0; j != NDIM; ++j) {
								lc[j] = L_c[lev + 1][iiip][j];
							}
						}
						for (integer ci = 0; ci != NCHILD; ++ci) {
							const integer iiic = child_index(ip, jp, kp, ci);
							for (auto& d : geo::dimension::full_set()) {
								X[d][ci] = com[lev][iiic][d];
							}
						}
						const auto& Y = com[lev + 1][iiip];
						for (auto& d : geo::dimension::full_set()) {
							dX[d] = X[d] - Y[d];
						}
						l <<= dX;
						//		if (type == RHO) {
						//			lc <<= dX;
						//		}
						for (integer ci = 0; ci != NCHILD; ++ci) {
							const integer iiic = child_index(ip, jp, kp, ci);
							for (integer j = 0; j != 20; ++j) {
								L[lev][iiic].ptr()[j] += l.ptr()[j][ci];
							}
							if (type == RHO) {
								for (integer j = 0; j != NDIM; ++j) {
									L_c[lev][iiic][j] += lc[j][ci];
								}
							}

							if (!is_leaf && lev == 0) {
								integer index = child_index(ip, jp, kp, ci, 0);
								exp_ret.first[index] = L[lev][iiic];
								if (type == RHO) {
									exp_ret.second[index] = L_c[lev][iiic];
								}
							}
						}
					}
				}
			}
		}
	}

	if (is_leaf) {
		for (integer i = G_BW; i != G_NX - G_BW; ++i) {
			for (integer j = G_BW; j != G_NX - G_BW; ++j) {
				for (integer k = G_BW; k != G_NX - G_BW; ++k) {
					const integer iii = gindex(i, j, k);
					const integer iiih = hindex(i + H_BW - G_BW, j + H_BW - G_BW, k + H_BW - G_BW);
					if (type == RHO) {
						G[phi_i][iii] = L[0][iii]();
						for (auto& d : geo::dimension::full_set()) {
							G[gx_i + d][iii] = -L[0][iii](d) - L_c[0][iii][d];
						}
						U[pot_i][iiih] = G[phi_i][iii] * U[rho_i][iiih];
					} else {
						dphi_dt[iiih] = L[0][iii]();
					}
				}
			}
		}
	}
	PROF_END;
	return exp_ret;
}

multipole_pass_type grid::compute_multipoles(gsolve_type type, const multipole_pass_type* child_poles) {
	PROF_BEGIN;
	integer lev = 0;
	const real dx3 = dx * dx * dx;

	if (/*is_leaf && */type == RHO) {
		const integer iii0 = hindex(H_BW, H_BW, H_BW);
		const std::array<real, NDIM> x0 = { X[XDIM][iii0], X[YDIM][iii0], X[ZDIM][iii0] };
		std::array<integer, 3> i;
		for (i[0] = 0; i[0] != G_NX; ++i[0]) {
			for (i[1] = 0; i[1] != G_NX; ++i[1]) {
				for (i[2] = 0; i[2] != G_NX; ++i[2]) {
					const integer iii = gindex(i[0], i[1], i[2]);
					for (integer dim = 0; dim != NDIM; ++dim) {
						com[0][iii][dim] = x0[dim] + (i[dim] - G_BW) * dx;
					}
				}
			}
		}
	}

	multipole_pass_type mret;
	if (!is_root) {
		mret.first.resize(INX * INX * INX / NCHILD);
		mret.second.resize(INX * INX * INX / NCHILD);
	}
	integer index = 0;
	for (integer inx = INX; (inx >= std::max(INX / 2, integer(1))) || (inx > 1 && is_root); inx >>= 1) {

		const integer nxp = inx + 2 * G_BW;
		const integer nxc = (2 * inx) + 2 * G_BW;

		auto child_index = [=](integer ip, integer jp, integer kp, integer ci) -> integer {
			const integer ic = (2 * ip - G_BW) + ((ci >> 0) & 1);
			const integer jc = (2 * jp - G_BW) + ((ci >> 1) & 1);
			const integer kc = (2 * kp - G_BW) + ((ci >> 2) & 1);
			return nxc * nxc * ic + nxc * jc + kc;
		};

		for (integer ip = G_BW; ip != nxp - G_BW; ++ip) {
			for (integer jp = G_BW; jp != nxp - G_BW; ++jp) {
				for (integer kp = G_BW; kp != nxp - G_BW; ++kp) {
					const integer iiip = nxp * nxp * ip + nxp * jp + kp;
					M[lev][iiip] = ZERO;
					if (lev != 0) {
						if (type == RHO) {
							simd_vector mc;
							std::array < simd_vector, NDIM > X;
							for (integer ci = 0; ci != NCHILD; ++ci) {
								const integer iiic = child_index(ip, jp, kp, ci);
								mc[ci] = M[lev - 1][iiic]();
								for (auto& d : geo::dimension::full_set()) {
									X[d][ci] = com[lev - 1][iiic][d];
								}
							}
							real mtot = mc.sum();
							assert(mtot != ZERO);
							for (auto& d : geo::dimension::full_set()) {
								com[lev][iiip][d] = (X[d] * mc).sum() / mtot;
								//						com[lev][iiip][d] = X[d].sum() / real(8);
							}
						}
						taylor<4, simd_vector> mc, mp;
						std::array<simd_vector, NDIM> x, y, dx;
						for (integer ci = 0; ci != NCHILD; ++ci) {
							const integer iiic = child_index(ip, jp, kp, ci);
							const space_vector& X = com[lev - 1][iiic];
							for (integer j = 0; j != 20; ++j) {
								mc.ptr()[j][ci] = M[lev - 1][iiic].ptr()[j];
								for (auto& d : geo::dimension::full_set()) {
									x[d][ci] = X[d];
								}
							}
						}
						const space_vector& Y = com[lev][iiip];
						for (auto& d : geo::dimension::full_set()) {
							simd_vector y = Y[d];
							dx[d] = x[d] - y;
						}
						mp = mc >> dx;
						for (integer j = 0; j != 20; ++j) {
							M[lev][iiip].ptr()[j] = mp.ptr()[j].sum();
						}
					} else {
						if (child_poles == nullptr) {
							const integer iiih = hindex(ip + H_BW - G_BW, jp + H_BW - G_BW, kp + H_BW - G_BW);
							if (type == RHO) {
								M[lev][iiip]() = U[rho_i][iiih] * dx3;
							} else {
								M[lev][iiip]() = dUdt[rho_i][iiih] * dx3;
							}
						} else {
							M[lev][iiip] = child_poles->first[index];
							if (type == RHO) {
								com[lev][iiip] = child_poles->second[index];
							}
							++index;
						}
					}
					if (!is_root && (lev == 1)) {
						mret.first[index] = M[lev][iiip];
						mret.second[index] = com[lev][iiip];
						++index;
					}
				}
			}
		}
		++lev;
		index = 0;
	}
	PROF_END;
	return mret;
}

std::vector<real> grid::get_gravity_boundary(const geo::direction& dir) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size = get_boundary_size(lb, ub, dir, INNER, INX, G_BW);
	const bool is_refined = !is_leaf;
	if (is_refined) {
		size *= 20 + 3;
	}
	data.resize(size);
	integer iter = 0;

	for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
			for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
				const auto& m = multipole_value(0, i, j, k);
				const auto& com = center_of_mass_value(i, j, k);
				const integer top = is_refined ? m.size() : 1;
				for (integer l = 0; l < top; ++l) {
					data[iter] = m.ptr()[l];
					++iter;
				}
				if (is_refined) {
					for (integer d = 0; d != NDIM; ++d) {
						data[iter] = com[d];
						++iter;
					}
				}
			}
		}
	}
	PROF_END;
	return data;
}

void grid::set_gravity_boundary(const std::vector<real>& data, const geo::direction& dir, bool monopole) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, dir, OUTER, INX, G_BW);
	integer iter = 0;
	for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
			for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
				auto& m = multipole_value(0, i, j, k);
				auto& com = center_of_mass_value(i, j, k);
				const integer top = monopole ? 1 : m.size();
				for (integer l = 0; l < top; ++l) {
					m.ptr()[l] = data[iter];
					++iter;
				}
				if (!monopole) {
					for (integer d = 0; d != NDIM; ++d) {
						com[d] = data[iter];
						++iter;
					}
				}
			}
		}
	}
	PROF_END;
}
