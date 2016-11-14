/*
 * grid_fmm.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: dmarce1
 */
#include "grid.hpp"
#include "simd.hpp"
#include "profiler.hpp"
#include "options.hpp"
#include "taylor.hpp"

static std::vector<interaction_type> ilist_n;
static std::vector<interaction_type> ilist_d;
static std::vector<interaction_type> ilist_r;
static std::vector<std::vector<boundary_interaction_type>> ilist_d_bnd(geo::direction::count());
static std::vector<std::vector<boundary_interaction_type>> ilist_n_bnd(geo::direction::count());
static taylor<4, real> factor;
extern options opts;

template<class T>
void load_multipole(taylor<4, T>& m, space_vector& c, const gravity_boundary_type& data, integer iter, bool monopole) {
	if (monopole) {
		m = T(0.0);
		m = T((*(data.m))[iter]);
	} else {
		for (int i = 0; i != 20; ++i) {
			m.ptr()[i] = (*(data.M))[iter].ptr()[i];
		}
		for (integer d = 0; d != NDIM; ++d) {
			c[d] = (*(data.x))[iter][d];
		}
	}
}

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

	for (integer i = 0; i != G_NX; ++i) {
		for (integer j = 0; j != G_NX; ++j) {
			for (integer k = 0; k != G_NX; ++k) {
				const integer iii1 = gindex(i, j, k);
				const integer iii0 = gindex(i + H_BW , j + H_BW , k + H_BW );
				for (integer n = 0; n != NDIM; ++n) {
					real mass;
					if (is_leaf) {
						mass = mon[iii1];
					} else {
						mass = M[iii1]();
					}
					this_com[n] += mass * com[0][iii1][n];
					mtot += mass;
					for (integer m = 0; m != NDIM; ++m) {
						if (!is_leaf) {
							quad_moment[n][m] += M[iii1](n, m);
						}
						quad_moment[n][m] += mass * com[0][iii1][n] * com[0][iii1][m];
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
	std::array<simd_vector, NDIM> dX;
	std::array < simd_vector, NDIM > X;
	std::array<simd_vector, NDIM> Y;
	std::fill(std::begin(L), std::end(L), ZERO);
	std::fill(std::begin(L_c), std::end(L_c), ZERO);
	if (!is_leaf) {
		const auto& this_ilist = is_root ? ilist_r : ilist_n;
		interaction_type np;
		interaction_type dp;
		const integer list_size = this_ilist.size();
		taylor<4, simd_vector> m0;
		taylor<4, simd_vector> m1;
		taylor<4, simd_vector> n0;
		taylor<4, simd_vector> n1;
		for (integer li = 0; li < list_size; li += simd_len) {
			for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
				const integer iii0 = this_ilist[li + i].first;
				const integer iii1 = this_ilist[li + i].second;
				for (integer d = 0; d < NDIM; ++d) {
					X[d][i] = com[0][iii0][d];
					Y[d][i] = com[0][iii1][d];
				}
				for (integer j = 0; j != 20; ++j) {
					m0.ptr()[j][i] = M[iii1].ptr()[j];
					m1.ptr()[j][i] = M[iii0].ptr()[j];
				}
				for (integer j = 10; j != 20; ++j) {
					if (type == RHO) {
						n0.ptr()[j][i] = M[iii1].ptr()[j] - M[iii0].ptr()[j] * (M[iii1]() / M[iii0]());
						n1.ptr()[j][i] = M[iii0].ptr()[j] - M[iii1].ptr()[j] * (M[iii0]() / M[iii1]());
					} else {
						n0.ptr()[j][i] = ZERO;
						n1.ptr()[j][i] = ZERO;
					}
				}
			}
			for (integer d = 0; d < NDIM; ++d) {
				dX[d] = X[d] - Y[d];
			}
			taylor<5, simd_vector> D;
			taylor<4, simd_vector> A0, A1;
			std::array<simd_vector, NDIM> B0 = { simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO) };
			std::array<simd_vector, NDIM> B1 = { simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO) };
			D.set_basis(dX);

			A0() = m0() * D();
			A1() = m1() * D();
			for (integer a = 0; a < NDIM; ++a) {
				if (type != RHO) {
					A0() -= m0(a) * D(a);
					A1() += m1(a) * D(a);
				}
				for (integer b = a; b < NDIM; ++b) {
					const auto tmp = D(a, b) * (real(1) / real(2)) * factor(a, b);
					A0() += m0(a, b) * tmp;
					A1() += m1(a, b) * tmp;
					for (integer c = b; c < NDIM; ++c) {
						const auto tmp0 = D(a, b, c) * (real(1) / real(6)) * factor(a, b, c);
						A0() -= m0(a, b, c) * tmp0;
						A1() += m1(a, b, c) * tmp0;
					}

				}
			}

			for (integer a = 0; a < NDIM; ++a) {
				A0(a) = +m0() * D(a);
				A1(a) = -m1() * D(a);
				for (integer b = 0; b < NDIM; ++b) {
					if (type != RHO) {
						A0(a) -= m0(a) * D(a, b);
						A1(a) -= m1(a) * D(a, b);
					}
					for (integer c = b; c < NDIM; ++c) {
						const auto tmp1 = D(a, b, c) * (real(1) / real(2)) * factor(c, b);
						A0(a) += m0(c, b) * tmp1;
						A1(a) -= m1(c, b) * tmp1;
					}

				}
			}

			if (type == RHO) {
				for (integer a = 0; a < NDIM; ++a) {
					for (integer b = 0; b < NDIM; ++b) {
						for (integer c = b; c < NDIM; ++c) {
							for (integer d = c; d != NDIM; ++d) {
								const auto tmp = D(a, b, c, d) * (real(1) / real(6)) * factor(b, c, d);
								B0[a] -= n0(b, c, d) * tmp;
								B1[a] -= n1(b, c, d) * tmp;
							}
						}
					}

				}
			}

			for (integer a = 0; a < NDIM; ++a) {
				for (integer b = a; b < NDIM; ++b) {
					A0(a, b) = m0() * D(a, b);
					A1(a, b) = m1() * D(a, b);
					for (integer c = 0; c < NDIM; ++c) {
						const auto tmp2 = D(a, b, c);
						A0(a, b) -= m0(c) * tmp2;
						A1(a, b) += m1(c) * tmp2;
					}

				}
			}

			for (integer a = 0; a < NDIM; ++a) {
				for (integer b = a; b < NDIM; ++b) {
					for (integer c = b; c < NDIM; ++c) {
						const auto tmp2 = D(a, b, c);
						A1(a, b, c) = -m1() * tmp2;
						A0(a, b, c) = +m0() * tmp2;
					}

				}
			}

			for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
				const integer iii0 = this_ilist[li + i].first;
				const integer iii1 = this_ilist[li + i].second;
				for (integer j = 0; j != 20; ++j) {
					L[iii0].ptr()[j] += A0.ptr()[j][i];
					L[iii1].ptr()[j] += A1.ptr()[j][i];
				}
				if (type == RHO) {
					for (integer j = 0; j != NDIM; ++j) {
						L_c[iii0][j] += B0[j][i];
						L_c[iii1][j] += B1[j][i];
					}
				}
			}
		}
	} else {
#if !defined(HPX_HAVE_DATAPAR)
		const v4sd d0 = { 1.0 / dx, +1.0 / (dx * dx), +1.0 / (dx * dx), +1.0 / (dx * dx) };
		const v4sd d1 = { 1.0 / dx, -1.0 / (dx * dx), -1.0 / (dx * dx), -1.0 / (dx * dx) };
#else
        const std::array<double, 4> di0 = { 1.0 / dx, +1.0 / (dx * dx), +1.0 / (dx * dx), +1.0 / (dx * dx) };
        const v4sd d0(di0.data());

        const std::array<double, 4> di1 = { 1.0 / dx, -1.0 / (dx * dx), -1.0 / (dx * dx), -1.0 / (dx * dx) };
        const v4sd d1(di1.data());
#endif

		const integer dsize = ilist_d.size();
		const integer lev = 0;
		for (integer li = 0; li < dsize; ++li) {
			const integer iii0 = ilist_d[li].first;
			const integer iii1 = ilist_d[li].second;
			const auto& ele = ilist_d[li];
			v4sd m0, m1;
			for (integer i = 0; i != 4; ++i) {
				m0[i] = mon[iii1];
			}
			for (integer i = 0; i != 4; ++i) {
				m1[i] = mon[iii0];
			}
			v4sd* l0ptr = (v4sd*) L[iii0].ptr();
			v4sd* l1ptr = (v4sd*) L[iii1].ptr();
			*l0ptr += m0 * ele.four * d0;
			*l1ptr += m1 * ele.four * d1;
		}
	}
	PROF_END;
}

void grid::compute_boundary_interactions(gsolve_type type, const geo::direction& dir, bool is_monopole, const gravity_boundary_type& mpoles) {
	if (!is_leaf) {
		if (!is_monopole) {
			compute_boundary_interactions_multipole_multipole(type, ilist_n_bnd[dir], mpoles);
		} else {
			compute_boundary_interactions_monopole_multipole(type, ilist_d_bnd[dir], mpoles);
		}
	} else {
		if (!is_monopole) {
			compute_boundary_interactions_multipole_monopole(type, ilist_d_bnd[dir], mpoles);
		} else {
			compute_boundary_interactions_monopole_monopole(type, ilist_d_bnd[dir], mpoles);
		}
	}

}

void grid::compute_boundary_interactions_multipole_multipole(gsolve_type type, const std::vector<boundary_interaction_type>& ilist_n_bnd,
	const gravity_boundary_type& mpoles) {
	PROF_BEGIN;
	taylor<4, simd_vector> m0;
	taylor<4, simd_vector> n0;
	std::array<simd_vector, NDIM> dX;
	std::array < simd_vector, NDIM > X;
	space_vector Y;
//	space_vector c;
	for (integer si = 0; si != ilist_n_bnd.size(); ++si) {
		integer index = mpoles.is_local ? ilist_n_bnd[si].second : si;
		load_multipole(m0, Y, mpoles, index, false);
		const integer list_size = ilist_n_bnd[si].first.size();
		const integer iii1 = ilist_n_bnd[si].second;
		for (integer li = 0; li < list_size; li += simd_len) {
			for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
				const integer iii0 = ilist_n_bnd[si].first[li + i];
				for (integer d = 0; d < NDIM; ++d) {
					X[d][i] = com[0][iii0][d];
				}
				for (integer j = 10; j != 20; ++j) {
					if (type == RHO) {
						n0.ptr()[j][i] = m0.ptr()[j][i] - M[iii0].ptr()[j] * (m0()[i] / M[iii0]());
					} else {
						n0.ptr()[j][i] = ZERO;
					}
				}
			}
			for (integer d = 0; d < NDIM; ++d) {
				dX[d] = X[d] - simd_vector(Y[d]);
			}

			taylor<5, simd_vector> D;
			taylor<4, simd_vector> A0;
			std::array<simd_vector, NDIM> B0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

			D.set_basis(dX);

			A0() = m0() * D();
			for (integer a = 0; a < NDIM; ++a) {
				if (type != RHO) {
					A0() -= m0(a) * D(a);
				}
				for (integer b = a; b < NDIM; ++b) {
					A0() += m0(a, b) * D(a, b) * (real(1) / real(2)) * factor(a, b);
					for (integer c = b; c < NDIM; ++c) {
						A0() -= m0(a, b, c) * D(a, b, c) * (real(1) / real(6)) * factor(a, b, c);
					}
				}
			}

			for (integer a = 0; a < NDIM; ++a) {
				A0(a) = +m0() * D(a);
				for (integer b = 0; b < NDIM; ++b) {
					if (type != RHO) {
						A0(a) -= m0(a) * D(a, b);
					}
					for (integer c = b; c < NDIM; ++c) {
						A0(a) += m0(c, b) * D(a, b, c) * (real(1) / real(2)) * factor(c, b);
					}
				}
			}

			if (type == RHO) {
				for (integer a = 0; a < NDIM; ++a) {
					for (integer b = 0; b < NDIM; ++b) {
						for (integer c = b; c < NDIM; ++c) {
							for (integer d = c; d < NDIM; ++d) {
								const auto tmp = D(a, b, c, d) * (real(1) / real(6));
								B0[a] -= n0(b, c, d) * tmp * factor(b, c, d);
							}
						}
					}
				}
			}

			for (integer a = 0; a < NDIM; ++a) {
				for (integer b = a; b < NDIM; ++b) {
					A0(a, b) = m0() * D(a, b);
					for (integer c = 0; c < NDIM; ++c) {
						A0(a, b) -= m0(c) * D(a, b, c);
					}
				}
			}

			for (integer a = 0; a < NDIM; ++a) {
				for (integer b = a; b < NDIM; ++b) {
					for (integer c = b; c < NDIM; ++c) {
						A0(a, b, c) = +m0() * D(a, b, c);
					}
				}
			}

			for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
				const integer iii0 = ilist_n_bnd[si].first[li + i];
				for (integer j = 0; j != 20; ++j) {
					L[iii0].ptr()[j] += A0.ptr()[j][i];
				}
				if (type == RHO) {
					for (integer j = 0; j != NDIM; ++j) {
						L_c[iii0][j] += B0[j][i];
					}
				}
			}
		}
	}
	PROF_END;
}

void grid::compute_boundary_interactions_multipole_monopole(gsolve_type type, const std::vector<boundary_interaction_type>& ilist_n_bnd,
	const gravity_boundary_type& mpoles) {
	PROF_BEGIN;
	taylor<4, simd_vector> m0;
	taylor<4, simd_vector> n0;
	std::array<simd_vector, NDIM> dX;
	std::array < simd_vector, NDIM > X;
	space_vector Y;
//	space_vector c;
	for (integer si = 0; si != ilist_n_bnd.size(); ++si) {
		const integer list_size = ilist_n_bnd[si].first.size();
		const integer iii1 = ilist_n_bnd[si].second;
		integer index = mpoles.is_local ? ilist_n_bnd[si].second : si;
		load_multipole(m0, Y, mpoles, index, false);
		for (integer j = 10; j != 20; ++j) {
			if (type == RHO) {
				n0.ptr()[j] = m0.ptr()[j];
			} else {
				n0.ptr()[j] = ZERO;
			}
		}
		for (integer li = 0; li < list_size; li += simd_len) {
			for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
				const integer iii0 = ilist_n_bnd[si].first[li + i];
				for (integer d = 0; d < NDIM; ++d) {
					X[d][i] = com[0][iii0][d];
				}
			}
			for (integer d = 0; d < NDIM; ++d) {
				dX[d] = X[d] - Y[d];
			}

			taylor<5, simd_vector> D;
			taylor<2, simd_vector> A0;
			std::array<simd_vector, NDIM> B0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

			D.set_basis(dX);

			A0() = m0() * D();
			for (integer a = 0; a < NDIM; ++a) {
				if (type != RHO) {
					A0() -= m0(a) * D(a);
				}
				for (integer b = a; b < NDIM; ++b) {
					A0() += m0(a, b) * D(a, b) * (real(1) / real(2)) * factor(a, b);
					for (integer c = b; c < NDIM; ++c) {
						A0() -= m0(a, b, c) * D(a, b, c) * (real(1) / real(6)) * factor(a, b, c);
					}
				}
			}

			for (integer a = 0; a < NDIM; ++a) {
				A0(a) = +m0() * D(a);
				for (integer b = 0; b < NDIM; ++b) {
					if (type != RHO) {
						A0(a) -= m0(a) * D(a, b);
					}
					for (integer c = b; c < NDIM; ++c) {
						A0(a) += m0(c, b) * D(a, b, c) * (real(1) / real(2)) * factor(b, c);
					}
				}
			}

			if (type == RHO) {
				for (integer a = 0; a < NDIM; ++a) {
					for (integer b = 0; b < NDIM; ++b) {
						for (integer c = b; c < NDIM; ++c) {
							for (integer d = c; d < NDIM; ++d) {
								const auto tmp = D(a, b, c, d) * (real(1) / real(6)) * factor(b, c, d);
								B0[a] -= n0(b, c, d) * tmp;
							}
						}
					}
				}
			}

			for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
				const integer iii0 = ilist_n_bnd[si].first[li + i];
				for (integer j = 0; j != 4; ++j) {
					L[iii0].ptr()[j] += A0.ptr()[j][i];
				}
				if (type == RHO) {
					for (integer j = 0; j != NDIM; ++j) {
						L_c[iii0][j] += B0[j][i];
					}
				}
			}
		}
	}
	PROF_END;
}

void grid::compute_boundary_interactions_monopole_multipole(gsolve_type type, const std::vector<boundary_interaction_type>& ilist_n_bnd,
	const gravity_boundary_type& mpoles) {
	PROF_BEGIN;
	interaction_type np;
	simd_vector m0;
	taylor<4, simd_vector> n0;
	std::array<simd_vector, NDIM> dX;
	std::array < simd_vector, NDIM > X;
	std::array<simd_vector, NDIM> Y;
	for (integer si = 0; si != ilist_n_bnd.size(); ++si) {
		integer index = mpoles.is_local ? ilist_n_bnd[si].second : si;
		const integer list_size = ilist_n_bnd[si].first.size();
		const integer iii1 = ilist_n_bnd[si].second;
		for (integer d = 0; d != NDIM; ++d) {
			Y[d] = ilist_n_bnd[si].x[d] * dx + this->X[d][hindex(H_BW,H_BW,H_BW)];
		}
		m0 = (*(mpoles.m))[index];
		for (integer li = 0; li < list_size; li += simd_len) {
			for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
				const integer iii0 = ilist_n_bnd[si].first[li + i];
				for (integer d = 0; d < NDIM; ++d) {
					X[d][i] = com[0][iii0][d];
				}
				for (integer j = 10; j != 20; ++j) {
					if (type == RHO) {
						n0.ptr()[j][i] = -M[iii0].ptr()[j] * (m0[i] / M[iii0]());
					} else {
						n0.ptr()[j][i] = ZERO;
					}
				}
			}
			for (integer d = 0; d < NDIM; ++d) {
				dX[d] = X[d] - Y[d];
			}

			taylor<5, simd_vector> D;
			taylor<4, simd_vector> A0;
			std::array<simd_vector, NDIM> B0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

			D.set_basis(dX);

			A0() = m0 * D();
			for (integer a = 0; a < NDIM; ++a) {
				A0(a) = +m0 * D(a);
				for (integer b = 0; b < NDIM; ++b) {
					A0(a, b) = m0 * D(a, b);
					for (integer c = 0; c < NDIM; ++c) {
						if (type == RHO) {
							for (integer d = 0; d < NDIM; ++d) {
								const auto tmp = D(a, b, c, d) * (real(1) / real(6));
								B0[a] -= n0(b, c, d) * tmp;
							}
						}
						A0(a, b, c) = +m0 * D(a, b, c);
					}
				}
			}

			A0() = m0 * D();

			for (integer a = 0; a < NDIM; ++a) {
				A0(a) = +m0 * D(a);
			}

			for (integer a = 0; a < NDIM; ++a) {
				for (integer b = a; b < NDIM; ++b) {
					A0(a, b) = m0 * D(a, b);
				}
			}

			for (integer a = 0; a < NDIM; ++a) {
				for (integer b = a; b < NDIM; ++b) {
					for (integer c = b; c < NDIM; ++c) {
						if (type == RHO) {
							A0(a, b, c) = +m0 * D(a, b, c);
						}
					}
				}
			}

			if (type == RHO) {
				for (integer a = 0; a < NDIM; ++a) {
					for (integer b = 0; b < NDIM; ++b) {
						for (integer c = b; c < NDIM; ++c) {
							for (integer d = c; d < NDIM; ++d) {
								const auto tmp = D(a, b, c, d) * (real(1) / real(6)) * factor(b, c, d);
								B0[a] -= n0(b, c, d) * tmp;
							}
						}
					}
				}
			}

			for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
				const integer iii0 = ilist_n_bnd[si].first[li + i];
				for (integer j = 0; j != 20; ++j) {
					L[iii0].ptr()[j] += A0.ptr()[j][i];
				}
				if (type == RHO) {
					for (integer j = 0; j != NDIM; ++j) {
						L_c[iii0][j] += B0[j][i];
					}
				}
			}
		}
	}
	PROF_END;
}

void grid::compute_boundary_interactions_monopole_monopole(gsolve_type type, const std::vector<boundary_interaction_type>& ilist_n_bnd,
	const gravity_boundary_type& mpoles) {
	PROF_BEGIN;
	simd_vector m0;
	std::array<simd_vector, NDIM> dX;
	std::array < simd_vector, NDIM > X;
	std::array<simd_vector, NDIM> Y;
	integer index = 0;

#if !defined(HPX_HAVE_DATAPAR)
	const v4sd d0 = { 1.0 / dx, +1.0 / (dx * dx), +1.0 / (dx * dx), +1.0 / (dx * dx) };
#else
    const std::array<double, 4> di0 = { 1.0 / dx, +1.0 / (dx * dx), +1.0 / (dx * dx), +1.0 / (dx * dx) };
    const v4sd d0(di0.data());
#endif
	for (integer si = 0; si != ilist_n_bnd.size(); ++si) {
		const integer dsize = ilist_n_bnd[si].first.size();
		const integer lev = 0;
		integer index = mpoles.is_local ? ilist_n_bnd[si].second : si;
		const integer iii1 = ilist_n_bnd[si].second;
		for (integer li = 0; li < dsize; ++li) {
			const integer iii0 = ilist_n_bnd[si].first[li];
			const auto& four = ilist_n_bnd[si].four[li];
			v4sd m0;
			const auto tmp = (*(mpoles).m)[index];
			for (integer i = 0; i != 4; ++i) {
				m0[i] = tmp;
			}
			v4sd* l0ptr = (v4sd*) L[iii0].ptr();
			*l0ptr += m0 * four * d0;
		}
	}
	PROF_END;
}

void compute_ilist() {

	factor = 0.0;
	factor() += 1.0;
	for (integer a = 0; a < NDIM; ++a) {
		factor(a) += 1.0;
		for (integer b = 0; b < NDIM; ++b) {
			factor(a, b) += 1.0;
			for (integer c = 0; c < NDIM; ++c) {
				factor(a, b, c) += 1.0;
			}
		}
	}

	const integer inx = INX;
	const integer nx = inx;
	interaction_type np;
	interaction_type dp;
	std::vector<interaction_type> ilist_r0;
	std::vector<interaction_type> ilist_n0;
	std::vector<interaction_type> ilist_d0;
	std::array < std::vector<interaction_type>, geo::direction::count() > ilist_n0_bnd;
	std::array < std::vector<interaction_type>, geo::direction::count() > ilist_d0_bnd;
	const real theta0 = opts.theta;
	const auto theta = [](integer i0, integer j0, integer k0, integer i1, integer j1, integer k1) {
		real tmp = std::sqrt((i0-i1)*(i0-i1)+(j0-j1)*(j0-j1)+(k0-k1)*(k0-k1));
		if( tmp > 0.0 ) {
			return 1.0 / tmp;
		} else {
			return 1.0e+10;
		}
	};
	const auto interior = [](integer i, integer j, integer k) {
		bool rc = true;
		if( i < 0 || i >= G_NX ) {
			rc = false;
		}
		else if( j < 0 || j >= G_NX ) {
			rc = false;
		}
		else if( k < 0 || k >= G_NX ) {
			rc = false;
		}
		return rc;
	};
	const auto neighbor_dir = [](integer i, integer j, integer k) {
		integer i0 = 0, j0 = 0, k0 = 0;
		if( i < 0) {
			i0 = -1;
		} else if( i >= G_NX ) {
			i0 = +1;
		}
		if( j < 0) {
			j0 = -1;
		} else if( j >= G_NX ) {
			j0 = +1;
		}
		if( k < 0) {
			k0 = -1;
		} else if( k >= G_NX ) {
			k0 = +1;
		}
		geo::direction d;
		d.set(i0,j0,k0);
		return d;
	};
	integer max_d = 0;
	integer width = INX;
	for (integer i0 = 0; i0 != nx; ++i0) {
		for (integer j0 = 0; j0 != nx; ++j0) {
			for (integer k0 = 0; k0 != nx; ++k0) {
				integer this_d = 0;
				integer ilb = std::min(i0 - width, integer(0));
				integer jlb = std::min(j0 - width, integer(0));
				integer klb = std::min(k0 - width, integer(0));
				integer iub = std::max(i0 + width + 1, G_NX);
				integer jub = std::max(j0 + width + 1, G_NX);
				integer kub = std::max(k0 + width + 1, G_NX);
				for (integer i1 = ilb; i1 < iub; ++i1) {
					for (integer j1 = jlb; j1 < jub; ++j1) {
						for (integer k1 = klb; k1 < kub; ++k1) {
							const real x = i0 - i1;
							const real y = j0 - j1;
							const real z = k0 - k1;
							const real r = std::sqrt(x * x + y * y + z * z);
							const real r3 = r * r * r;
							v4sd four;
							if (r > 0.0) {
								four[0] = -1.0 / r;
								four[1] = x / r3;
								four[2] = y / r3;
								four[3] = z / r3;
							} else {
								for (integer i = 0; i != 4; ++i) {
									four[i] = 0.0;
								}
							}
							if (i0 == i1 && j0 == j1 && k0 == k1) {
								continue;
							}
							const integer i0_c = (i0 + INX) / 2 - INX / 2;
							const integer j0_c = (j0 + INX) / 2 - INX / 2;
							const integer k0_c = (k0 + INX) / 2 - INX / 2;
							const integer i1_c = (i1 + INX) / 2 - INX / 2;
							const integer j1_c = (j1 + INX) / 2 - INX / 2;
							const integer k1_c = (k1 + INX) / 2 - INX / 2;
							const real theta_f = theta(i0, j0, k0, i1, j1, k1);
							const real theta_c = theta(i0_c, j0_c, k0_c, i1_c, j1_c, k1_c);
							const integer iii0 = gindex(i0, j0, k0);
							const integer iii1n = gindex((i1+INX)%INX, (j1+INX)%INX, (k1+INX)%INX);
							const integer iii1 = gindex(i1, j1, k1);
							if (theta_c > theta0 && theta_f <= theta0) {
								np.first = iii0;
								np.second = iii1n;
								np.four = four;
								np.x[XDIM] = i1;
								np.x[YDIM] = j1;
								np.x[ZDIM] = k1;
								if (interior(i1, j1, k1) && interior(i0, j0, k0)) {
									if (iii1 > iii0) {
										ilist_n0.push_back(np);
									}
								} else if (interior(i0, j0, k0)) {
									ilist_n0_bnd[neighbor_dir(i1, j1, k1)].push_back(np);
								}
							}
							if (theta_c > theta0) {
								++this_d;
								dp.first = iii0;
								dp.second = iii1n;
								dp.x[XDIM] = i1;
								dp.x[YDIM] = j1;
								dp.x[ZDIM] = k1;
								dp.four = four;
								if (interior(i1, j1, k1) && interior(i0, j0, k0)) {
									if (iii1 > iii0) {
										ilist_d0.push_back(dp);
									}
								} else if (interior(i0, j0, k0)) {
									ilist_d0_bnd[neighbor_dir(i1, j1, k1)].push_back(dp);
								}
							}
							if (theta_f <= theta0) {
								np.first = iii0;
								np.second = iii1n;
								np.x[XDIM] = i1;
								np.x[YDIM] = j1;
								np.x[ZDIM] = k1;
								np.four = four;
								if (interior(i1, j1, k1) && interior(i0, j0, k0)) {
									if (iii1 > iii0) {
										ilist_r0.push_back(np);
									}
								}
							}
						}
					}
				}
				max_d = std::max(max_d, this_d);
			}
		}
	}
	printf("# direct = %i\n", int(max_d));
	ilist_n = std::vector < interaction_type > (ilist_n0.begin(), ilist_n0.end());
	ilist_d = std::vector < interaction_type > (ilist_d0.begin(), ilist_d0.end());
	ilist_r = std::vector < interaction_type > (ilist_r0.begin(), ilist_r0.end());
	for (auto& dir : geo::direction::full_set()) {
		auto& d = ilist_d_bnd[dir];
		auto& d0 = ilist_d0_bnd[dir];
		auto& n = ilist_n_bnd[dir];
		auto& n0 = ilist_n0_bnd[dir];
		for (auto i0 : d0) {
			bool found = false;
			for (auto& i : d) {
				if (i.second == i0.second) {
					i.first.push_back(i0.first);
					i.four.push_back(i0.four);
					found = true;
					break;
				}
			}
			if (!found) {
				boundary_interaction_type i;
				i.second = i0.second;
				i.x = i0.x;
				n.push_back(i);
				i.first.push_back(i0.first);
				i.four.push_back(i0.four);
				d.push_back(i);
			}
		}
		for (auto i0 : n0) {
			bool found = false;
			for (auto& i : n) {
				if (i.second == i0.second) {
					i.first.push_back(i0.first);
					i.four.push_back(i0.four);
					found = true;
					break;
				}
			}
			assert(found);
		}
	}
}

expansion_pass_type grid::compute_expansions(gsolve_type type, const expansion_pass_type* parent_expansions) {
	PROF_BEGIN;
	expansion_pass_type exp_ret;
	if (!is_leaf) {
		exp_ret.first.resize(INX * INX * INX);
		if (type == RHO) {
			exp_ret.second.resize(INX * INX * INX);
		}
	}
	const integer inx = INX;
	const integer nxp = (inx / 2);
	auto child_index = [=](integer ip, integer jp, integer kp, integer ci, integer bw=0) -> integer {
		const integer ic = (2 * (ip )+bw) + ((ci >> 0) & 1);
		const integer jc = (2 * (jp )+bw) + ((ci >> 1) & 1);
		const integer kc = (2 * (kp )+bw) + ((ci >> 2) & 1);
		return (inx+2*bw) * (inx+2*bw) * ic + (inx+2*bw) * jc + kc;
	};

	for (integer ip = 0; ip != nxp; ++ip) {
		for (integer jp = 0; jp != nxp; ++jp) {
			for (integer kp = 0; kp != nxp; ++kp) {
				const integer iiip = nxp * nxp * ip + nxp * jp + kp;
				std::array < simd_vector, NDIM > X;
				std::array<simd_vector, NDIM> dX;
				taylor<4, simd_vector> l;
				std::array<simd_vector, NDIM> lc;
				if (!is_root) {
					const integer index = (INX * INX / 4) * (ip) + (INX / 2) * (jp) + (kp);
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
						l.ptr()[j] = 0.0;
					}
					for (integer j = 0; j != NDIM; ++j) {
						lc[j] = 0.0;
					}
				}
				for (integer ci = 0; ci != NCHILD; ++ci) {
					const integer iiic = child_index(ip, jp, kp, ci);
					for (integer d = 0; d < NDIM; ++d) {
						X[d][ci] = com[0][iiic][d];
					}
				}
				const auto& Y = com[1][iiip];
				for (integer d = 0; d < NDIM; ++d) {
					dX[d] = X[d] - Y[d];
				}
				l <<= dX;
				for (integer ci = 0; ci != NCHILD; ++ci) {
					const integer iiic = child_index(ip, jp, kp, ci);
					for (integer j = 0; j != 20; ++j) {
						L[iiic].ptr()[j] += l.ptr()[j][ci];
					}
					if (type == RHO) {
						for (integer j = 0; j != NDIM; ++j) {
							L_c[iiic][j] += lc[j][ci];
						}
					}

					if (!is_leaf) {
						integer index = child_index(ip, jp, kp, ci, 0);
						exp_ret.first[index] = L[iiic];
						if (type == RHO) {
							exp_ret.second[index] = L_c[iiic];
						}
					}
				}
			}
		}
	}

	if (is_leaf) {
		for (integer i = 0; i != G_NX; ++i) {
			for (integer j = 0; j != G_NX; ++j) {
				for (integer k = 0; k != G_NX; ++k) {
					const integer iii = gindex(i, j, k);
					const integer iii0 = h0index(i, j, k);
					const integer iiih = hindex(i + H_BW , j + H_BW , k + H_BW );
					if (type == RHO) {
						G[iii][phi_i] = L[iii]();
						for (integer d = 0; d < NDIM; ++d) {
							G[iii][gx_i + d] = -L[iii](d);
							if (opts.ang_con == true) {
								G[iii][gx_i + d] -= L_c[iii][d];
							}
						}
						U[pot_i][iiih] = G[iii][phi_i] * U[rho_i][iiih];
					} else {
						dphi_dt[iii0] = L[iii]();
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
	if (is_leaf) {
		M.resize(0);
		mon.resize(G_N3);
	} else {
		M.resize(G_N3);
		mon.resize(0);
	}
	if (type == RHO) {
		const integer iii0 = hindex(H_BW, H_BW, H_BW);
		const std::array<real, NDIM> x0 = { X[XDIM][iii0], X[YDIM][iii0], X[ZDIM][iii0] };
		std::array<integer, 3> i;
		for (i[0] = 0; i[0] != G_NX; ++i[0]) {
			for (i[1] = 0; i[1] != G_NX; ++i[1]) {
				for (i[2] = 0; i[2] != G_NX; ++i[2]) {
					const integer iii = gindex(i[0], i[1], i[2]);
					for (integer dim = 0; dim != NDIM; ++dim) {
						com[0][iii][dim] = x0[dim] + (i[dim]) * dx;
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
	taylor<4, real> MM;
	integer index = 0;
	for (integer inx = INX; (inx >= INX / 2); inx >>= 1) {

		const integer nxp = inx;
		const integer nxc = (2 * inx);

		auto child_index = [=](integer ip, integer jp, integer kp, integer ci) -> integer {
			const integer ic = (2 * ip ) + ((ci >> 0) & 1);
			const integer jc = (2 * jp ) + ((ci >> 1) & 1);
			const integer kc = (2 * kp ) + ((ci >> 2) & 1);
			return nxc * nxc * ic + nxc * jc + kc;
		};

		for (integer ip = 0; ip != nxp; ++ip) {
			for (integer jp = 0; jp != nxp; ++jp) {
				for (integer kp = 0; kp != nxp; ++kp) {
					const integer iiip = nxp * nxp * ip + nxp * jp + kp;
					if (lev != 0) {
						if (type == RHO) {
							simd_vector mc;
							std::array < simd_vector, NDIM > X;
							for (integer ci = 0; ci != NCHILD; ++ci) {
								const integer iiic = child_index(ip, jp, kp, ci);
								if (is_leaf) {
									mc[ci] = mon[iiic];
								} else {
									mc[ci] = M[iiic]();
								}
								for (integer d = 0; d < NDIM; ++d) {
									X[d][ci] = com[0][iiic][d];
								}
							}
							real mtot = mc.sum();
							for (integer d = 0; d < NDIM; ++d) {
								com[1][iiip][d] = (X[d] * mc).sum() / mtot;
							}
						}
						taylor<4, simd_vector> mc, mp;
						std::array<simd_vector, NDIM> x, y, dx;
						for (integer ci = 0; ci != NCHILD; ++ci) {
							const integer iiic = child_index(ip, jp, kp, ci);
							const space_vector& X = com[lev - 1][iiic];
							if (is_leaf) {
								mc()[ci] = mon[iiic];
								for (integer j = 1; j != 20; ++j) {
									mc.ptr()[j][ci] = 0.0;
								}
							} else {
								for (integer j = 0; j != 20; ++j) {
									mc.ptr()[j][ci] = M[iiic].ptr()[j];
								}
							}
							for (integer d = 0; d < NDIM; ++d) {
								x[d][ci] = X[d];
							}
						}
						const space_vector& Y = com[lev][iiip];
						for (integer d = 0; d < NDIM; ++d) {
							simd_vector y(Y[d]);
							dx[d] = x[d] - y;
						}
						mp = mc >> dx;
						for (integer j = 0; j != 20; ++j) {
							MM.ptr()[j] = mp.ptr()[j].sum();
						}
					} else {
						if (child_poles == nullptr) {
							const integer iiih = hindex(ip + H_BW , jp + H_BW , kp + H_BW );
							const integer iii0 = h0index(ip, jp, kp);
							if (type == RHO) {
								mon[iiip] = U[rho_i][iiih] * dx3;
							} else {
								mon[iiip] = dUdt[rho_i][iii0] * dx3;
							}
						} else {
							M[iiip] = child_poles->first[index];
							if (type == RHO) {
								com[lev][iiip] = child_poles->second[index];
							}
							++index;
						}
					}
					if (!is_root && (lev == 1)) {
						mret.first[index] = MM;
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

gravity_boundary_type grid::get_gravity_boundary(const geo::direction& dir, bool is_local) {
	PROF_BEGIN;
//	std::array<integer, NDIM> lb, ub;
	gravity_boundary_type data;
	data.is_local = is_local;
	if (!is_local) {
		data.allocate();
		integer iter = 0;
		const std::vector<boundary_interaction_type>& list = ilist_n_bnd[dir.flip()];
		if (is_leaf) {
			data.m->reserve(list.size());
			for (auto i : list) {
				const integer iii = i.second;
				data.m->push_back(mon[iii]);
			}
		} else {
			data.M->reserve(list.size());
			data.x->reserve(list.size());
			for (auto i : list) {
				const integer iii = i.second;
				const integer top = M[iii].size();
				data.M->push_back(M[iii]);
//				space_vector tmp;
				data.x->push_back(com[0][iii]);
			}
		}
	} else {
		static const auto nuldel = [](void*) {};
		if (is_leaf) {
			data.m = std::shared_ptr < std::vector < real >> (&mon, nuldel);
		} else {
			data.M = std::shared_ptr < std::vector < multipole >> (&M, nuldel);
			data.x = std::shared_ptr < std::vector < space_vector >> (&com[0], nuldel);
		}
	}
	PROF_END;
	return data;
}
