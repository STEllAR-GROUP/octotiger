//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/grid_flattened_indices.hpp"
#include "octotiger/grid_fmm.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/profiler.hpp"
#include "octotiger/real.hpp"
#include "octotiger/simd.hpp"
#include "octotiger/space_vector.hpp"
#include "octotiger/taylor.hpp"

#include <hpx/include/parallel_for_loop.hpp>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

extern taylor<4, real> factor;
extern taylor<4, m2m_vector> factor_half_v;
extern taylor<4, m2m_vector> factor_sixth_v;

#ifdef OCTOTIGER_HAVE_GRAV_PAR
const auto for_loop_policy = hpx::parallel::execution::par;
#else
const auto for_loop_policy = hpx::parallel::execution::seq;
#endif

// why not lists per cell? (David)
// is a grid actually a cell? (David)
// what is the semantics of a space_vector? (David)

// non-root and non-leaf node (David)
static std::vector<interaction_type> ilist_n;
// monopole-monopole interactions? (David)
static std::vector<interaction_type> ilist_d;
// interactions of root node (empty list? boundary stuff?) (David)
static std::vector<interaction_type> ilist_r;
static std::vector<std::vector<boundary_interaction_type>> ilist_d_bnd(geo::direction::count());
static std::vector<std::vector<boundary_interaction_type>> ilist_n_bnd(geo::direction::count());
extern taylor<4, real> factor;

template<class T>
void load_multipole(taylor<4, T> &m, space_vector &c, const gravity_boundary_type &data, integer iter, bool monopole) {
	if (monopole) {
		m = T(0.0);
		m = T((*(data.m))[iter]);
	} else {
		auto const &tmp1 = (*(data.M))[iter];
#pragma GCC ivdep
		for (int i = 0; i != 20; ++i) {
			m[i] = tmp1[i];
		}
		auto const &tmp2 = (*(data.x))[iter];
#pragma GCC ivdep
		for (integer d = 0; d != NDIM; ++d) {
			c[d] = tmp2[d];
		}
	}
}

real compute_part_pot(real r) {
        if (opts().p_smooth == EXACT) {
                return -1.0 / r;
        } else if (opts().p_smooth == RUFFERT) { // As in Ruffert 1993 (as well as in Sandquist et al. 1998)
                const real r2 = sqr(r);
                const real l2 = sqr(opts().p_smooth_l);
                const real denom = sqrt(r2 + l2 * exp(-r2/l2));
                return -1.0 / denom;
        }
        else if ((opts().p_smooth == MONAGHAN) || (opts().p_smooth == MONAGHAN_MULTI)) { // As in Monaghan 1992
                const real h = opts().p_smooth_l;
                if ( h <= 0.0 ) {
                        std::cerr << "Could not procceed with zero smoothing length!";
                        abort();
                } else {
                        const real x = r / h;
                        if ((0.0 <= x) && (x <= 1.0)) {
                        	const real pol = (2.0 / 3.0) * pow(x, 2) - 0.3 * pow(x, 4) + 0.1 * pow(x, 5) - 1.4;
                        	return pol / h;
                        } else if ((1.0 <= x) && (x <= 2.0)) {
                                const real pol = (4.0 / 3.0) * pow(x, 2) - 1.0 * pow(x, 3) + 0.3 * pow(x, 4) - (1.0 / 30.0) * pow(x, 5) - 1.6 + 1.0 / (15.0 * x);
                                return pol / h;
                        } else {
                                return -1.0 / r;
                        }
                }
        } else {
                std::cerr << "Could not find particles smoothing type!";
                abort();
        }
        return 0;
}

real compute_part_force(real r, real y) {
        if (opts().p_smooth == EXACT) {
                return y / (r * r * r);
        } else if (opts().p_smooth == RUFFERT) { // As in Ruffert 1993 (as well as in Sandquist et al. 1998)
                const real r2 = sqr(r);
                const real l2 = sqr(opts().p_smooth_l);
                const real denom = sqrt(r2 + l2 * exp(-r2/l2));
                return y * (1.0 - exp(-r2/l2)) / (denom * denom * denom);
        }
        else if ((opts().p_smooth == MONAGHAN) || (opts().p_smooth == MONAGHAN_MULTI)) { // As in Monaghan 1992
                const real h = opts().p_smooth_l;
                const real x = r / h;
                if ( h <= 0.0 ) {
                        std::cerr << "Could not procceed with zero smoothing length!";
                        abort();
                }
                else {
                        if ((0.0 <= x) && (x <= 1.0)) {
                                const real pol = (4.0 / 3.0) - 1.2 * pow(x, 2) + 0.5 * pow(x, 3);
                                return y * pol / (h * h * h);
                        } else if ((1.0 <= x) && (x <= 2.0)) {
                                const real pol = (8.0 / 3.0) - 3 * x + 1.2 * pow(x, 2) - (1.0 / 6.0) * pow(x, 3) - 1.0 / (15.0 * x * x * x);
                                return y * pol / (h * h * h);
                        } else {
                                return y / (r * r * r);
                        }
                }
        } else {
                std::cerr << "Could not find particles smoothing type!";
                abort();
        }
        return 0;
}

v4sd make_four(space_vector orig, space_vector dest) {
	auto x = dest[XDIM] - orig[XDIM];
	auto y = dest[YDIM] - orig[YDIM];
	auto z = dest[ZDIM] - orig[ZDIM];
        const real tmp = sqr(x) + sqr(y) + sqr(z);
        const real r = (tmp == 0) ? 0 : std::sqrt(tmp);
        const real r3 = r * r * r;
        v4sd four;
        if (r > 0.0) {
                four[0] = compute_part_pot(r);
                four[1] = compute_part_force(r, x);
                four[2] = compute_part_force(r, y);
                four[3] = compute_part_force(r, z);
        } else {
        	for (integer i = 0; i != 4; ++i) {
		        four[i] = 0.0;
        	}
        }
	return four;
}

v4sd make_four_partner(v4sd four) {
	v4sd partner;
        partner[0] = four[0];
        partner[1] = -four[1];
        partner[2] = -four[2];
        partner[3] = -four[3];
	return partner;
}


void find_eigenvectors(real q[3][3], real e[3][3], real lambda[3]) {

	real b0[3], b1[3], A, bdif;
	int iter = 0;
	for (int l = 0; l < 3; l++) {
		b0[0] = b0[1] = b0[2] = 0.0;
		b0[l] = 1.0;
		do {
			iter++;
			b1[0] = b1[1] = b1[2] = 0.0;
			for (int i = 0; i < 3; i++) {
				for (int m = 0; m < 3; m++) {
					b1[i] += q[i][m] * b0[m];
				}
			}
			A = sqrt(sqr(b1[0]) + sqr(b1[1]) + sqr(b1[2]));
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
		lambda[l] = sqrt(A) / sqrt(sqr(e[l][0]) + sqr(e[l][1]) + sqr(e[l][2]));
	}

}

std::pair<space_vector, space_vector> grid::find_axis() const {

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

	auto &M = *M_ptr;
	auto &mon = *mon_ptr;
	std::vector<space_vector> const &com0 = *(com_ptr[0]);
	for (integer i = 0; i != G_NX; ++i) {
		for (integer j = 0; j != G_NX; ++j) {
			for (integer k = 0; k != G_NX; ++k) {
				const integer iii1 = gindex(i, j, k);
				const integer iii0 = gindex(i + H_BW, j + H_BW, k + H_BW);
				space_vector const &com0iii1 = com0[iii1];
				multipole const &Miii1 = M[iii1];
				for (integer n = 0; n != NDIM; ++n) {
					real mass;
					if (is_leaf) {
						mass = mon[iii1];
					} else {
						mass = Miii1();
					}
					this_com[n] += mass * com0iii1[n];
					mtot += mass;
					for (integer m = 0; m != NDIM; ++m) {
						if (!is_leaf) {
							quad_moment[n][m] += Miii1(n, m);
						}
						quad_moment[n][m] += mass * com0iii1[n] * com0iii1[m];
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
	std::pair<space_vector, space_vector> pair { rc, this_com };

	return pair;
}

void grid::solve_gravity(gsolve_type type) {
	compute_multipoles(type);
	compute_interactions(type);
	compute_expansions(type);
}

// constexpr int to_ab_idx_map3[3][3] = {
//     {  4,  5,  6  },
//     {  5,  7,  8  },
//     {  6,  8,  9  }
// };

// constexpr int cb_idx_map[6] = {
//      4,  5,  6,  7,  8,  9,
// };

// constexpr int to_abc_idx_map3[3][6] = {
//     {  10, 11, 12, 13, 14, 15, },
//     {  11, 13, 14, 16, 17, 18, },
//     {  12, 14, 15, 17, 18, 19, }
// };

// constexpr int to_abcd_idx_map3[3][10] = {
//     { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 },
//     { 21, 23, 24, 26, 27, 28, 30, 31, 32, 33 },
//     { 22, 24, 25, 27, 28, 29, 31, 32, 33, 34 }
// };

// constexpr int bcd_idx_map[10] = {
//     10, 11, 12, 13, 14, 15, 16, 17, 18, 19
// };

// constexpr int to_abc_idx_map6[6][3] = {
//     { 10, 11, 12 }, { 11, 13, 14 },
//     { 12, 14, 15 }, { 13, 16, 17 },
//     { 14, 17, 18 }, { 15, 18, 19 }
// };

// constexpr int ab_idx_map6[6] = {
//     4, 5, 6, 7, 8, 9
// };

void grid::compute_interactions(gsolve_type type) {
	PROFILE();

	// calculating the contribution of all the inner cells
	// calculating the interaction

	auto &mon = *mon_ptr;
	// L stores the gravitational potential
	// L should be L, (10) in the paper
	// L_c stores the correction for angular momentum
	// L_c, (20) in the paper (Dominic)
	std::fill(std::begin(L), std::end(L), ZERO);
	std::fill(std::begin(L_c), std::end(L_c), ZERO);

	//printf("in interaction\n");
        for (auto &p : particles) {
		p.L = 0.0;
		for (integer i = 0; i < NDIM; i++) {
			p.L_c[i] = 0.0;
		}
	}

	// Non-leaf nodes use taylor expansion
	// For leaf node, calculates the gravitational potential between two particles
	if (!is_leaf) {
		// Non-leaf means that multipole-multipole interaction (David)
		// Fetch the interaction list for the current cell (David)
		const auto &this_ilist = is_root ? ilist_r : ilist_n;
		const integer list_size = this_ilist.size();

		// Space vector is a vector pack (David)
		// Center of masses of each cell (com_ptr1 is the center of mass of the parent cell)
		std::vector<space_vector> const &com0 = (*(com_ptr[0]));
		// Do 8 cell-cell interactions at the same time (simd.hpp) (Dominic)
		// This could lead to problems either on Haswell or on KNL as the number is hardcoded
		// (David)
		// It is unclear what the underlying vectorization library does, if simd_len > architectural
		// length (David)
		hpx::parallel::for_loop_strided(for_loop_policy, 0, list_size, simd_len, [&com0, &this_ilist, list_size, type, this](std::size_t li) {

			// dX is distance between X and Y
			// X and Y are the two cells interacting
			// X and Y store the 3D center of masses (per simd element, SoA style)
			std::array<simd_vector, NDIM> dX;
			std::array<simd_vector, NDIM> X;
			std::array<simd_vector, NDIM> Y;

			// m multipole moments of the cells
			taylor<4, simd_vector> m0;
			taylor<4, simd_vector> m1;
			// n angular momentum of the cells
			taylor<4, simd_vector> n0;
			taylor<4, simd_vector> n1;

			// vector of multipoles (== taylor<4>s)
			// here the multipoles are used as a storage of expansion coefficients
			auto &M = *M_ptr;

			// FIXME: replace with vector-pack gather-loads
			// TODO: only uses first 10 coefficients? ask Dominic
#ifndef __GNUG__
#pragma GCC ivdep
#endif
			for (integer i = 0; i != simd_len && integer(li + i) < list_size; ++i) {
				const integer iii0 = this_ilist[li + i].first;
				const integer iii1 = this_ilist[li + i].second;
				space_vector const &com0iii0 = com0[iii0];
				space_vector const &com0iii1 = com0[iii1];
				for (integer d = 0; d < NDIM; ++d) {
					// load the 3D center of masses
					X[d][i] = com0iii0[d];
					Y[d][i] = com0iii1[d];
				}

				// cell specific taylor series coefficients
				multipole const &Miii0 = M[iii0];
				multipole const &Miii1 = M[iii1];
				for (integer j = 0; j != taylor_sizes[3]; ++j) {
					m0[j][i] = Miii1[j];
					m1[j][i] = Miii0[j];
				}

				// RHO computes the gravitational potential based on the mass density
				// non-RHO computes the time derivative (hydrodynamics code V_t)
				if (type == RHO || type == RHOM) {
					// this branch computes the angular momentum correction, (20) in the paper
					// divide by mass of other cell
					real const tmp1 = Miii1() / Miii0();
					real const tmp2 = Miii0() / Miii1();
					// calculating the coefficients for formula (M are the octopole moments)
					// the coefficients are calculated in (17) and (18)
					// TODO: Dominic
					for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
						n0[j][i] = Miii1[j] - Miii0[j] * tmp1;
						n1[j][i] = Miii0[j] - Miii1[j] * tmp2;
					}
				}
			}

			if (type == DRHODT) {
#pragma GCC ivdep
			for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
				n0[j] = ZERO;
				n1[j] = ZERO;
			}
		}

// distance between cells in all dimensions
#pragma GCC ivdep
			for (integer d = 0; d < NDIM; ++d) {
				dX[d] = X[d] - Y[d];
			}

			// R_i in paper is the dX in the code
			// D is taylor expansion value for a given X expansion of the gravitational
			// potential (multipole expansion)
			taylor<5, simd_vector> D;
			// A0, A1 are the contributions to L (for each cell)
			taylor<4, simd_vector> A0, A1;
			// B0, B1 are the contributions to L_c (for each cell)
			std::array<simd_vector, NDIM> BB0 = { simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO) };
			std::array<simd_vector, NDIM> B1 = { simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO) };

			// calculates all D-values, calculate all coefficients of 1/r (not the potential),
			// formula (6)-(9) and (19)
                        if ((type != RHOM) || (opts().p_smooth != MONAGHAN_MULTI) || (particles.size() == 0) || (/*(octotiger::fmm::STENCIL_WIDTH - */1.0 * dx > 2.0 * opts().p_smooth_l)) {
                                D.set_basis(dX);
                        } else {
                                const auto is_smooth = contain_particles(particles, X, opts().p_smooth_multi_f * dx) + contain_particles(particles, Y, opts().p_smooth_multi_f * dx);
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
                                const real has_particles = is_smooth.sum();
#else
                                const real has_particles = Vc::reduce(is_smooth);
#endif
                                if (has_particles == 0) {
                                        D.set_basis(dX);
                                } else {
/*					if (has_particles > 1.0) {
						const auto contain1 = contain_particles(particles, X, opts().p_smooth_multi_f * dx);
						const auto contain2 = contain_particles(particles, Y, opts().p_smooth_multi_f * dx);
						const real has_particles1 = contain1.sum();
						const real has_particles2 = contain2.sum();
						printf("contain 1 %e, Xcom (%e, %e, %e), contain 2 %e, Ycom (%e, %e, %e)\n", has_particles1, (contain1 * X[0]).sum(), (contain1 * X[1]).sum(), (contain1 * X[2]).sum(), has_particles2, (contain2 * Y[0]).sum(), (contain2 * Y[1]).sum(), (contain2 * Y[2]).sum());
					}
					auto dx_r = sqrt(sqr(dX[0]) + sqr(dX[1]) + sqr(dX[2]));
					real dx_r_min = dx_r[0];
					for (integer i = 1; i != simd_len; ++i) {
						if (dx_r[i] < dx_r_min) {
							dx_r_min = dx_r[i];
						}
					}
					if (dx_r_min < 2*opts().p_smooth_l) {
						printf("set basis MULTI in multi-multi, dx %e, smooth: %e, 2smooth: %e, min r: %e\n", dx, opts().p_smooth_l, 2*opts().p_smooth_l, dx_r_min);
						printf("particle in (%e, %e, %e), Xcom in (%e, %e, %e) Ycom (%e, %e, %e), has particles %e\n", particles[0].pos[0], particles[0].pos[1], particles[0].pos[2], (is_smooth * X[0]).sum(), (is_smooth * X[1]).sum(), (is_smooth * X[2]).sum(), (is_smooth * Y[0]).sum(), (is_smooth * Y[1]).sum(), (is_smooth * Y[2]).sum(), has_particles);
					}*/
					D.set_basis_monaghan(dX, opts().p_smooth_l, is_smooth);
                                }
                        }
			
			// the following loops calculate formula (10), potential from A->B and B->A
			// (basically alternating)
			A0[0] = m0[0] * D[0];
			A1[0] = m1[0] * D[0];
			if (type == DRHODT) {
				for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
					A0[0] -= m0[i] * D[i];
					A1[0] += m1[i] * D[i];
				}
			}
			for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
				const auto tmp = D[i] * (factor[i] * HALF);
				A0[0] += m0[i] * tmp;
				A1[0] += m1[i] * tmp;
			}
			for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
				const auto tmp = D[i] * (factor[i] * SIXTH);
				A0[0] -= m0[i] * tmp;
				A1[0] += m1[i] * tmp;
			}

//                 for (integer a = 0; a < NDIM; ++a) {
//                     A0(a) = m0() * D(a);
//                     A1(a) = -m1() * D(a);
//                     for (integer b = 0; b < NDIM; ++b) {
//                         if (type != RHO) {
//                             A0(a) -= m0(a) * D(a, b);
//                             A1(a) -= m1(a) * D(a, b);
//                         }
//                         for (integer c = b; c < NDIM; ++c) {
//                             const auto tmp1 = D(a, b, c) * (factor(c, b) * HALF);
//                             A0(a) += m0(c, b) * tmp1;
//                             A1(a) -= m1(c, b) * tmp1;
//                         }
//                     }
//                 }
			for (integer a = 0; a < NDIM; ++a) {
				int const *ab_idx_map = to_ab_idx_map3[a];
				int const *abc_idx_map = to_abc_idx_map3[a];

				A0(a) = m0() * D(a);
				A1(a) = -m1() * D(a);
				for (integer i = 0; i != 6; ++i) {
					if (type == DRHODT && i < 3) {
						A0(a) -= m0(a) * D[ab_idx_map[i]];
						A1(a) -= m1(a) * D[ab_idx_map[i]];
					}
					const integer cb_idx = cb_idx_map[i];
					const auto tmp1 = D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
					A0(a) += m0[cb_idx] * tmp1;
					A1(a) -= m1[cb_idx] * tmp1;
				}
			}

			if (type == RHO || type == RHOM) {
//                     for (integer a = 0; a != NDIM; ++a) {
//                         for (integer b = 0; b != NDIM; ++b) {
//                             for (integer c = b; c != NDIM; ++c) {
//                                 for (integer d = c; d != NDIM; ++d) {
//                                     const auto tmp = D(a, b, c, d) * (factor(b, c, d) * SIXTH);
//                                     B0[a] -= n0(b, c, d) * tmp;
//                                     B1[a] -= n1(b, c, d) * tmp;
//                                 }
//                             }
//                         }
//                     }
				for (integer a = 0; a != NDIM; ++a) {
					int const *abcd_idx_map = to_abcd_idx_map3[a];
					for (integer i = 0; i != 10; ++i) {
						const integer bcd_idx = bcd_idx_map[i];
						const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
						BB0[a] -= n0[bcd_idx] * tmp;
						B1[a] -= n1[bcd_idx] * tmp;
					}
				}
			}

//                 for (integer a = 0; a < NDIM; ++a) {
//                     for (integer b = a; b < NDIM; ++b) {
//                         A0(a, b) = m0() * D(a, b);
//                         A1(a, b) = m1() * D(a, b);
//                         for (integer c = 0; c < NDIM; ++c) {
//                             const auto& tmp = D(a, b, c);
//                             A0(a, b) -= m0(c) * tmp;
//                             A1(a, b) += m1(c) * tmp;
//                         }
//                     }
//                 }
			for (integer i = 0; i != 6; ++i) {
				int const *abc_idx_map6 = to_abc_idx_map6[i];

				integer const ab_idx = ab_idx_map6[i];
				A0[ab_idx] = m0() * D[ab_idx];
				A1[ab_idx] = m1() * D[ab_idx];
				for (integer c = 0; c < NDIM; ++c) {
					const auto &tmp = D[abc_idx_map6[c]];
					A0[ab_idx] -= m0(c) * tmp;
					A1[ab_idx] += m1(c) * tmp;
				}
			}

			for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
				const auto &tmp = D[i];
				A1[i] = -m1[0] * tmp;
				A0[i] = m0[0] * tmp;
			}
#ifdef OCTOTIGER_HAVE_GRAV_PAR
                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif

			// potential and correction have been computed, no scatter the results
			for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
				const integer iii0 = this_ilist[li + i].first;
				const integer iii1 = this_ilist[li + i].second;

				expansion tmp1, tmp2;
#pragma GCC ivdep
			for (integer j = 0; j != taylor_sizes[3]; ++j) {
				tmp1[j] = A0[j][i];
				tmp2[j] = A1[j][i];
			}
			L[iii0] += tmp1;
			L[iii1] += tmp2;

			if (type == RHO || type == RHOM) {
				space_vector &L_ciii0 = L_c[iii0];
				space_vector &L_ciii1 = L_c[iii1];
				for (integer j = 0; j != NDIM; ++j) {
					L_ciii0[j] += BB0[j][i];
					L_ciii1[j] += B1[j][i];
				}
			}
		}
	}	);
	} else {
		// first compute potential and the the three components of the force
		// vectorizing across the dimensions, therefore 1 component for the potential + 3
		// components for the force

		// Coefficients for potential evaluation? (David)
		alignas(64) double di0[8] = { 1. / dx, 1. / sqr(dx), 1. / sqr(dx), 1. / sqr(dx) };
		const v4sd d0(di0, Vc::Aligned);

		// negative of d0 because it's the force in the opposite direction
		alignas(64) double di1[8] = { 1. / dx, -1. / sqr(dx), -1. / sqr(dx), -1. / sqr(dx) };
		const v4sd d1(di1, Vc::Aligned);

		// Number of body-body interactions current leaf cell, probably includes interactions with
		// bodies in neighboring cells  (David)
		const integer dsize = ilist_d.size();

		// Iterate interaction description between bodies in the current cell and closeby bodies
		// (David)
		for (integer li = 0; li < dsize; ++li) {
			// Retrieve the interaction description for the current body (David)
			const auto &ele = ilist_d[li];
			// Extract the indices of the two interacting bodies (David)
			const integer iii0 = ele.first;
			const integer iii1 = ele.second;
			// fetch both interacting bodies (monopoles) (David)
			// broadcasts a single value

			L[iii0] += mon[iii1] * ele.four * d0;
			L[iii1] += mon[iii0] * ele.four * d1;

			if (type == RHOM) {

                        	space_vector const &com_f = (*(com_ptr[0]))[iii0];
                        	space_vector const &com_s = (*(com_ptr[0]))[iii1];

                        	std::vector<integer> particles_f = get_particles_inds(particles, iii0);
                        	std::vector<integer> particles_s = get_particles_inds(particles, iii1);
			
				// particles in cell 1 interact with cell 2 - p1c2
				for (auto f_ind : particles_f) {
					particle& p_f = particles[f_ind]; 
                        		v4sd four = make_four(com_s, p_f.pos);
                                        v4sd four_partner = make_four_partner(four);
                                	//printf("mono-mono interaction of particle %i, (%e, %e, %e) with second cell (%e, %e, %e), rel (%e, %e, %e), m %e phi %e, g (%e, %e, %e)\n", p_f.id, p_f.pos[0], p_f.pos[1], p_f.pos[2], com_s[0], com_s[1], com_s[2], (com_s[0] - com_f[0]) / dx, (com_s[1] - com_f[1]) / dx, (com_s[2] - com_f[2]) / dx, mon[iii0], mon[iii0] * four[0], mon[iii0] * four[1], mon[iii0] * four[2], mon[iii0] * four[3]);
                                        L[iii1] +=  p_f.mass * four_partner;
                                        p_f.L += mon[iii1] * four;

					// p1p2 interactions
		                        for (auto s_ind : particles_s) {
						particle& p_s = particles[s_ind];
						printf("pp interaction\n");
                	                        v4sd four = make_four(p_s.pos, p_f.pos);
                        	                v4sd four_partner = make_four_partner(four);
                                	        p_s.L +=  p_f.mass * four_partner;
                                        	p_f.L +=  p_s.mass * four;
						//printf("pf L %e ps L %e\n", p_f.L[0], p_s.L[0]);
		                        }
				}
				// p2c1 interactions
                        	for (auto s_ind : particles_s) {
					particle& p_s = particles[s_ind];
                                        v4sd four = make_four(com_f, p_s.pos);
                                        v4sd four_partner = make_four_partner(four);
					//printf("mono-mono interaction of particle %i, (%e, %e, %e) with first cell (%e, %e, %e), m %e phi %e, g (%e, %e, %e)\n", p_s.id, p_s.pos[0], p_s.pos[1], p_s.pos[2], com_f[0], com_f[1], com_f[2], mon[iii0], mon[iii0] * four[0], mon[iii0] * four[1], mon[iii0] * four[2], mon[iii0] * four[3]);
                                        L[iii0] +=  p_s.mass * four_partner;
                                        p_s.L += mon[iii0] * four;
                        	}
			}
		}
		if (opts().part_self_interact && type == RHOM) {
			// particles can interact with the particles within the same cell and with the containing cell itself
	                for (auto &p : particles) {

				// particles interact with the containing cell (pc interactions)
	                        const auto p_inds = p.containing_cell_index;
               		        const integer iii0 = gindex(p_inds[XDIM], p_inds[YDIM], p_inds[ZDIM]);
				space_vector const &com_cell = (*(com_ptr[0]))[iii0];

                                v4sd four = make_four(com_cell, p.pos);
                                v4sd four_partner = make_four_partner(four);
                                L[iii0] +=  p.mass * four_partner;
				//printf("mono-mono interaction of particle %i, (%e, %e, %e) with containing cell (%e, %e, %e), m %e phi %e, g (%e, %e, %e)\n", p.id, p.pos[0], p.pos[1], p.pos[2], com_cell[0], com_cell[1], com_cell[2], mon[iii0], mon[iii0] * four[0], mon[iii0] * four[1], mon[iii0] * four[2], mon[iii0] * four[3]);
                                p.L += mon[iii0] * four;

				auto particles_in_cells = get_particles_inds(particles, iii0);
				// pp interactions - needs factor half because of double iterations
                                for (auto partner_ind : particles_in_cells) {
					particle& p_partner = particles[partner_ind];
					if (p_partner.id != p.id) {
						printf("pp self interaction\n");
                               	        	v4sd four = make_four(p_partner.pos, p.pos);
                                       		v4sd four_partner = make_four_partner(four);
                                        	p_partner.L += 0.5 * p.mass * four_partner;
       	                                	p.L += 0.5 * p_partner.mass * four;
					}
				}

			}
		}
	}
}

void grid::compute_boundary_interactions(gsolve_type type, const geo::direction &dir, bool is_monopole, const gravity_boundary_type &mpoles) {
	PROFILE();
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

void grid::compute_boundary_interactions_multipole_multipole(gsolve_type type, const std::vector<boundary_interaction_type> &ilist_n_bnd,
		const gravity_boundary_type &mpoles) {
	PROFILE();

	auto &M = *M_ptr;
	auto &mon = *mon_ptr;
        std::vector<particle> p_bnd(0); // particles in exterior multipole
        if ((*(mpoles).p).size() > 0) {
                p_bnd = (*(mpoles).p)[0];
        }

	std::vector<space_vector> const &com0 = *(com_ptr[0]);
	hpx::parallel::for_loop(for_loop_policy, 0, ilist_n_bnd.size(), [&mpoles, &com0, &ilist_n_bnd, type, this, &M, &p_bnd](std::size_t si) {

		taylor<4, simd_vector> m0;
		taylor<4, simd_vector> n0;
		std::array<simd_vector, NDIM> dX;
		std::array<simd_vector, NDIM> X;
		space_vector Y;

		boundary_interaction_type const &bnd = ilist_n_bnd[si];
		integer index = (mpoles.local_semaphore != nullptr) ? bnd.second : si;

		load_multipole(m0, Y, mpoles, index, false);

		std::array<simd_vector, NDIM> simdY = { simd_vector(real(Y[0])), simd_vector(real(Y[1])), simd_vector(real(Y[2])), };
		const integer list_size = bnd.first.size();
		for (integer li = 0; li < list_size; li += simd_len) {
#ifndef __GNUG__
#pragma GCC ivdep
#endif
		for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
			const integer iii0 = bnd.first[li + i];
			space_vector const &com0iii0 = com0[iii0];
			for (integer d = 0; d < NDIM; ++d) {
				X[d][i] = com0iii0[d];
			}
			if (type == RHO || type == RHOM) {
				multipole const &Miii0 = M[iii0];
				real const tmp = m0()[i] / Miii0();
				for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
					n0[j][i] = m0[j][i] - Miii0[j] * tmp;
				}
			}
		}
		if (type == DRHODT) {
#pragma GCC ivdep
		for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
			n0[j] = ZERO;
		}
	}
#pragma GCC ivdep
		for (integer d = 0; d < NDIM; ++d) {
			dX[d] = X[d] - simdY[d];
		}

		taylor<5, simd_vector> D;
		taylor<4, simd_vector> A0;
		std::array<simd_vector, NDIM> BB0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

                if ((type != RHOM) || (opts().p_smooth != MONAGHAN_MULTI) || ((p_bnd.size() == 0) && (particles.size() == 0)) || (dx > 2 * opts().p_smooth_l)) {
                        D.set_basis(dX);
                } else {
                        const auto is_smooth = contain_particles(particles, X, opts().p_smooth_multi_f * dx) + contain_particles(p_bnd, simdY, opts().p_smooth_multi_f * dx);
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
                        const real has_particles = is_smooth.sum();
#else
                        const real has_particles = Vc::reduce(is_smooth);
#endif
                        if (has_particles == 0) {
                                D.set_basis(dX);
                        } else {
/*	                        auto dx_r = sqrt(sqr(dX[0]) + sqr(dX[1]) + sqr(dX[2]));
                                real dx_r_min = dx_r[0];
                                for (integer i = 1; i != simd_len; ++i) {
                	                if (dx_r[i] < dx_r_min) {
        	        	                dx_r_min = dx_r[i];
                                        }
                                }
				if (dx_r_min < 2 * opts().p_smooth_l) {
					printf("set basis MULTI in boundary multi-multi, dx = %e, p in grid: %i, p in bnd: %i, smooth: %e, 2smooth: %e, min r: %e\n", dx, particles.size(), p_bnd.size(), opts().p_smooth_l, 2*opts().p_smooth_l, dx_r_min);
					if (particles.size() > 0) {
						printf("particle in (%e, %e, %e), X in (%e, %e, %e)\n", particles[0].pos[0], particles[0].pos[1], particles[0].pos[2], (is_smooth * X[0]).sum(), (is_smooth * X[1]).sum(), (is_smooth * X[2]).sum());
					} else {
                                        	printf("particle in bnd (%e, %e, %e), Ybnd in (%e, %e, %e), has particles %e\n", p_bnd[0].pos[0], p_bnd[0].pos[1], p_bnd[0].pos[2], Y[0],  Y[1], Y[2], has_particles);

					}
				}*/
				D.set_basis_monaghan(dX, opts().p_smooth_l, is_smooth);
                        }
                }

		A0[0] = m0[0] * D[0];
		if (type == DRHODT) {
			for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
				A0[0] -= m0[i] * D[i];
			}
		}
		for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
			A0[0] += m0[i] * D[i] * (factor[i] * HALF);
		}
		for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
			A0[0] -= m0[i] * D[i] * (factor[i] * SIXTH);
		}

//                 for (integer a = 0; a < NDIM; ++a) {
//                     A0(a) = m0() * D(a);
//                     for (integer b = 0; b < NDIM; ++b) {
//                         if (type != RHO) {
//                             A0(a) -= m0(a) * D(a, b);
//                         }
//                         for (integer c = b; c < NDIM; ++c) {
//                             const auto tmp = D(a, b, c) * (factor(c, b) * HALF);
//                             A0(a) += m0(c, b) * tmp;
//                         }
//                     }
//                 }
		for (integer a = 0; a < NDIM; ++a) {
			int const *ab_idx_map = to_ab_idx_map3[a];
			int const *abc_idx_map = to_abc_idx_map3[a];

			A0(a) = m0() * D(a);
			for (integer i = 0; i != 6; ++i) {
				if (type == DRHODT && i < 3) {
					A0(a) -= m0(a) * D[ab_idx_map[i]];
				}
				const integer cb_idx = cb_idx_map[i];
				const auto tmp = D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
				A0(a) += m0[cb_idx] * tmp;
			}
		}

		if (type == RHO || type == RHOM) {
//                     for (integer a = 0; a < NDIM; ++a) {
//                         for (integer b = 0; b < NDIM; ++b) {
//                             for (integer c = b; c < NDIM; ++c) {
//                                 for (integer d = c; d < NDIM; ++d) {
//                                     const auto tmp = D(a, b, c, d) * (factor(b, c, d) * SIXTH);
//                                     B0[a] -= n0(b, c, d) * tmp;
//                                 }
//                             }
//                         }
//                     }
			for (integer a = 0; a < NDIM; ++a) {
				int const *abcd_idx_map = to_abcd_idx_map3[a];
				for (integer i = 0; i != 10; ++i) {
					const integer bcd_idx = bcd_idx_map[i];
					const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
					BB0[a] -= n0[bcd_idx] * tmp;
				}
			}
		}

//                 for (integer a = 0; a < NDIM; ++a) {
//                     for (integer b = a; b < NDIM; ++b) {
//                         A0(a, b) = m0() * D(a, b);
//                         for (integer c = 0; c < NDIM; ++c) {
//                             A0(a, b) -= m0(c) * D(a, b, c);
//                         }
//                     }
//                 }
		for (integer i = 0; i != 6; ++i) {
			int const *abc_idx_map6 = to_abc_idx_map6[i];

			integer const ab_idx = ab_idx_map6[i];
			A0[ab_idx] = m0() * D[ab_idx];
			for (integer c = 0; c < NDIM; ++c) {
				A0[ab_idx] -= m0(c) * D[abc_idx_map6[c]];
			}
		}

		for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
			A0[i] = m0[0] * D[i];
		}

#ifdef OCTOTIGER_HAVE_GRAV_PAR
                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
		for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
			const integer iii0 = bnd.first[li + i];
			expansion &Liii0 = L[iii0];

			expansion tmp;
#pragma GCC ivdep
		for (integer j = 0; j != taylor_sizes[3]; ++j) {
			tmp[j] = A0[j][i];
		}
		Liii0 += tmp;

		if (type == RHO || type == RHOM) {
			space_vector &L_ciii0 = L_c[iii0];
#pragma GCC ivdep
		for (integer j = 0; j != NDIM; ++j) {
			L_ciii0[j] += BB0[j][i];
		}
	}
}
}
}	);

}

void grid::compute_boundary_interactions_multipole_monopole(gsolve_type type, const std::vector<boundary_interaction_type> &ilist_n_bnd,
		const gravity_boundary_type &mpoles) {
	PROFILE();

	auto &M = *M_ptr;
	auto &mon = *mon_ptr;
        std::vector<particle> p_bnd(0); // particles in exterior multipole
        if ((*(mpoles).p).size() > 0) {
                p_bnd = (*(mpoles).p)[0];
        }

	std::vector<space_vector> const &com0 = *(com_ptr[0]);
	hpx::parallel::for_loop(for_loop_policy, 0, ilist_n_bnd.size(), [&mpoles, &com0, &ilist_n_bnd, type, this, &p_bnd](std::size_t si) {

		taylor<4, simd_vector> m0;
		taylor<4, simd_vector> n0;
		std::array<simd_vector, NDIM> dX;
		std::array<simd_vector, NDIM> X;
		space_vector Y;

		boundary_interaction_type const &bnd = ilist_n_bnd[si];
		const integer list_size = bnd.first.size();
		integer index = (mpoles.local_semaphore != nullptr) ? bnd.second : si;
		load_multipole(m0, Y, mpoles, index, false);

		std::array<simd_vector, NDIM> simdY = { simd_vector(real(Y[0])), simd_vector(real(Y[1])), simd_vector(real(Y[2])), };

		if (type == RHO || type == RHOM) {
#pragma GCC ivdep
		for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
			n0[j] = m0[j];
		}
	} else {
#pragma GCC ivdep
		for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
			n0[j] = ZERO;
		}
	}
#pragma GCC ivdep
		for (integer li = 0; li < list_size; li += simd_len) {
			for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
				const integer iii0 = bnd.first[li + i];
				space_vector const &com0iii0 = com0[iii0];
				for (integer d = 0; d < NDIM; ++d) {
					X[d][i] = com0iii0[d];
				}
			}
#pragma GCC ivdep
		for (integer d = 0; d < NDIM; ++d) {
			dX[d] = X[d] - simdY[d];
		}

		taylor<5, simd_vector> D;
		taylor<2, simd_vector> A0;
		std::array<simd_vector, NDIM> BB0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

                if ((type != RHOM) || (opts().p_smooth != MONAGHAN_MULTI) || (p_bnd.size() == 0)) {
                        D.set_basis(dX);
                } else {
                        const auto is_smooth = contain_particles(p_bnd, simdY, opts().p_smooth_multi_f * dx);
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
                        const real has_particles = is_smooth.sum();
#else
                        const real has_particles = Vc::reduce(is_smooth);
#endif
                        if (has_particles == 0) {
                                D.set_basis(dX);
                        } else {
				printf("set basis multi in boundary multi-mono 1\n");
				D.set_basis_monaghan(dX, opts().p_smooth_l, is_smooth);
                        }
                }

		A0[0] = m0[0] * D[0];
		if (type == DRHODT) {
			for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
				A0[0] -= m0[i] * D[i];
			}
		}
		for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
			A0[0] += m0[i] * D[i] * (factor[i] * HALF);
		}
		for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
			A0[0] -= m0[i] * D[i] * (factor[i] * SIXTH);
		}

//                 for (integer a = 0; a < NDIM; ++a) {
//                     A0(a) = m0() * D(a);
//                     for (integer b = 0; b < NDIM; ++b) {
//                         if (type != RHO) {
//                             A0(a) -= m0(a) * D(a, b);
//                         }
//                         for (integer c = b; c < NDIM; ++c) {
//                             A0(a) += m0(c, b) * D(a, b, c) * (factor(b, c) * HALF);
//                         }
//                     }
//                 }
		for (integer a = 0; a < NDIM; ++a) {
			int const *ab_idx_map = to_ab_idx_map3[a];
			int const *abc_idx_map = to_abc_idx_map3[a];

			A0(a) = m0() * D(a);
			for (integer i = 0; i != 6; ++i) {
				if (type == DRHODT && i < 3) {
					A0(a) -= m0(a) * D[ab_idx_map[i]];
				}
				const integer cb_idx = cb_idx_map[i];
				A0(a) += m0[cb_idx] * D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
			}
		}

		if (type == RHO || type == RHOM) {
//                     for (integer a = 0; a < NDIM; ++a) {
//                         for (integer b = 0; b < NDIM; ++b) {
//                             for (integer c = b; c < NDIM; ++c) {
//                                 for (integer d = c; d < NDIM; ++d) {
//                                     const auto tmp = D(a, b, c, d) * (factor(b, c, d) * SIXTH);
//                                     B0[a] -= n0(b, c, d) * tmp;
//                                 }
//                             }
//                         }
//                     }
			for (integer a = 0; a < NDIM; ++a) {
				int const *abcd_idx_map = to_abcd_idx_map3[a];
				for (integer i = 0; i != 10; ++i) {
					const integer bcd_idx = bcd_idx_map[i];
					const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
					BB0[a] -= n0[bcd_idx] * tmp;
				}
			}
		}

#ifdef OCTOTIGER_HAVE_GRAV_PAR
                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
		for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
			const integer iii0 = bnd.first[li + i];
			expansion &Liii0 = L[iii0];
#pragma GCC ivdep
		for (integer j = 0; j != 4; ++j) {
			Liii0[j] += A0[j][i];
		}
		if (type == RHO || type == RHOM) {
			space_vector &L_ciii0 = L_c[iii0];
#pragma GCC ivdep
		for (integer j = 0; j != NDIM; ++j) {
			L_ciii0[j] += BB0[j][i];
		}
	}
}
}
if ((type == RHOM) && (particles.size() > 0)) {
		// adding multipoles from boundary to particles within the cells
                for (integer li = 0; li < list_size; li++) {
                	const integer iii0 = bnd.first[li];
                	//space_vector const &com0iii0 = com0[iii0];
			std::vector<integer> part_mono = get_particles_inds(particles, iii0);
			// assume no more than simd_len (8) particles in one cell
			if (part_mono.size() > 0) {
                            for (integer i = 0; i != simd_len && i < part_mono.size(); ++i) {
                                space_vector const part_pos = particles[part_mono[i]].pos;
                                for (integer d = 0; d < NDIM; ++d) {
                                        X[d][i] = part_pos[d];
                                }
                            }
#pragma GCC ivdep
                for (integer d = 0; d < NDIM; ++d) {
                        dX[d] = X[d] - simdY[d];
                }

                taylor<5, simd_vector> D;
                taylor<2, simd_vector> A0;
                std::array<simd_vector, NDIM> BB0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };
                
		if ((opts().p_smooth != MONAGHAN_MULTI)) { // || ((octotiger::fmm::STENCIL_WIDTH - 1) * dx > 2.0 * opts().p_smooth_l)) {
                        D.set_basis(dX);
                } else {
			printf("set basis multi in boundary multi-mono 2\n");
			D.set_basis_monaghan(dX, opts().p_smooth_l, int_simd_vector(1));
		}

                A0[0] = m0[0] * D[0];
                for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
                        A0[0] += m0[i] * D[i] * (factor[i] * HALF);
                }
                for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                        A0[0] -= m0[i] * D[i] * (factor[i] * SIXTH);
                }

                for (integer a = 0; a < NDIM; ++a) {
                        int const *ab_idx_map = to_ab_idx_map3[a];
                        int const *abc_idx_map = to_abc_idx_map3[a];

                        A0(a) = m0() * D(a);
                        for (integer i = 0; i != 6; ++i) {
                                if (type == DRHODT && i < 3) {
                                        A0(a) -= m0(a) * D[ab_idx_map[i]];
                                }
                                const integer cb_idx = cb_idx_map[i];
                                A0(a) += m0[cb_idx] * D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
                        }
                }
                for (integer a = 0; a < NDIM; ++a) {
                	int const *abcd_idx_map = to_abcd_idx_map3[a];
                	for (integer i = 0; i != 10; ++i) {
                		const integer bcd_idx = bcd_idx_map[i];
                		const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                        	BB0[a] -= n0[bcd_idx] * tmp;
                        }
                }
#ifdef OCTOTIGER_HAVE_GRAV_PAR
                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
		space_vector const &com0iii0 = com0[iii0];
		auto const &Liii0 = L[iii0];
                for (integer i = 0; i != simd_len && i < part_mono.size(); ++i) {
                        particle& p = particles[part_mono[i]];
			//printf("interaction of part %i, (%e, %e, %e) in cell (%e, %e, %e) with multipole (%e, %e, %e)\n", p.id, p.pos[0], p.pos[1], p.pos[2], com0iii0[0], com0iii0[1], com0iii0[2], Y[0], Y[1], Y[2]);
			//printf("before %e (%e, %e, %e)\n", p.L(), p.L_c[0], p.L_c[1], p.L_c[2]);
#pragma GCC ivdep
                	for (integer j = 0; j != 4; ++j) {
			//	printf("adding to p.L[%i]\n", j);
                        	p.L[j] += A0[j][i];
                	}
#pragma GCC ivdep
                	for (integer j = 0; j != NDIM; ++j) {
			//	printf("adding to p.L_c[%i]\n", j);
                        	p.L_c[j] += BB0[j][i];
               		}
			//printf("after %e (%e, %e, %e), cell %e\n", p.L(), p.L_c[0], p.L_c[1], p.L_c[2], Liii0());
        	}
            } // close if part_mono > 0
	} //close ilist loop
} // close type==RHO condition
}	); // close hpx loop

} // close function

void grid::compute_boundary_interactions_monopole_multipole(gsolve_type type, const std::vector<boundary_interaction_type> &ilist_n_bnd,
		const gravity_boundary_type &mpoles) {
	PROFILE();

	auto &M = *M_ptr;
	auto &mon = *mon_ptr;

	std::array<real, NDIM> Xbase = { X[0][hindex(H_BW, H_BW, H_BW)], X[1][hindex(H_BW, H_BW, H_BW)], X[2][hindex(H_BW, H_BW, H_BW)] };

	std::vector<space_vector> const &com0 = *(com_ptr[0]);
	hpx::parallel::for_loop(for_loop_policy, 0, ilist_n_bnd.size(), [&mpoles, &Xbase, &com0, &ilist_n_bnd, type, this, &M](std::size_t si) {
		taylor<4, simd_vector> n0;
		std::array<simd_vector, NDIM> dX;
		std::array<simd_vector, NDIM> X;
		std::array<simd_vector, NDIM> Y;

		boundary_interaction_type const &bnd = ilist_n_bnd[si];
		integer index = (mpoles.local_semaphore != nullptr) ? bnd.second : si;
		const integer list_size = bnd.first.size();

#pragma GCC ivdep
		for (integer d = 0; d != NDIM; ++d) {
			Y[d] = bnd.x[d] * dx + Xbase[d];
		}

		simd_vector m0 = (*(mpoles.m))[index];
		for (integer li = 0; li < list_size; li += simd_len) {
			for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
				const integer iii0 = bnd.first[li + i];
				space_vector const &com0iii0 = com0[iii0];
#pragma GCC ivdep
		for (integer d = 0; d < NDIM; ++d) {
			X[d][i] = com0iii0[d];
		}
		if (type == RHO || type == RHOM) {
			multipole const &Miii0 = M[iii0];
			real const tmp = m0[i] / Miii0();
#pragma GCC ivdep
		for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
			n0[j][i] = -Miii0[j] * tmp;
		}
	}
}

if (type == DRHODT) {
#pragma GCC ivdep
		for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
			n0[j] = ZERO;
		}
	}
#pragma GCC ivdep
		for (integer d = 0; d < NDIM; ++d) {
			dX[d] = X[d] - Y[d];
		}

		taylor<5, simd_vector> D;
		taylor<4, simd_vector> A0;
		std::array<simd_vector, NDIM> BB0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

                if ((type != RHOM) || (opts().p_smooth != MONAGHAN_MULTI) || (particles.size() == 0)) {
                        D.set_basis(dX);
                } else {
                        const auto is_smooth = contain_particles(particles, X, opts().p_smooth_multi_f * dx);
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
                        const real has_particles = is_smooth.sum();
#else
                        const real has_particles = Vc::reduce(is_smooth);
#endif
                        if (has_particles == 0) {
                                D.set_basis(dX);
                        } else {
				printf("set basis multi in boundary mono-multi 1\n");
                                D.set_basis_monaghan(dX, opts().p_smooth_l, is_smooth);
                        }
                }

		A0[0] = m0 * D[0];
		for (integer i = taylor_sizes[0]; i != taylor_sizes[2]; ++i) {
			A0[i] = m0 * D[i];
		}
		for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
			A0[i] = m0 * D[i];
		}

		if (type == RHO || type == RHOM) {
//                     for (integer a = 0; a != NDIM; ++a) {
//                         for (integer b = 0; b != NDIM; ++b) {
//                             for (integer c = b; c != NDIM; ++c) {
//                                 for (integer d = c; d != NDIM; ++d) {
//                                     const auto tmp = D(a, b, c, d) * (factor(b, c, d) * SIXTH);
//                                     B0[a] -= n0(b, c, d) * tmp;
//                                 }
//                             }
//                         }
//                     }
			for (integer a = 0; a != NDIM; ++a) {
				int const *abcd_idx_map = to_abcd_idx_map3[a];
				for (integer i = 0; i != 10; ++i) {
					const integer bcd_idx = bcd_idx_map[i];
					const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
					BB0[a] -= n0[bcd_idx] * tmp;
				}
			}
		}

#ifdef OCTOTIGER_HAVE_GRAV_PAR
                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
		for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
			const integer iii0 = bnd.first[li + i];

			expansion tmp;
#pragma GCC ivdep
		for (integer j = 0; j != taylor_sizes[3]; ++j) {
			tmp[j] = A0[j][i];
		}
		L[iii0] += tmp;

		if (type == RHO || type == RHOM) {
			space_vector &L_ciii0 = L_c[iii0];
#pragma GCC ivdep
		for (integer j = 0; j != NDIM; ++j) {
			L_ciii0[j] += BB0[j][i];
		}
	}
}
}
	if (type == RHOM) {
                std::vector<particle> p_bnd(0); // particles in exterior cell
                if ((*(mpoles).p).size() > 0) {
                        p_bnd = (*(mpoles).p)[si];
                }
		for (auto p : p_bnd) {
	                m0 = p.mass;
	                for (integer d = 0; d != NDIM; ++d) {
				Y[d] = (real)p.pos[d];
			}
        	        for (integer li = 0; li < list_size; li += simd_len) {
                	        for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
                        	        const integer iii0 = bnd.first[li + i];
                                	space_vector const &com0iii0 = com0[iii0];
#pragma GCC ivdep
	                		for (integer d = 0; d < NDIM; ++d) {
	        		                X[d][i] = com0iii0[d];
        		        	}
		        	        multipole const &Miii0 = M[iii0];
                			real const tmp = m0[i] / Miii0();
#pragma GCC ivdep
	 		               for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
        	        		        n0[j][i] = -Miii0[j] * tmp;
		                	}	        		
				}
		                for (integer d = 0; d < NDIM; ++d) {
        		                dX[d] = X[d] - Y[d];
		                }

                		taylor<5, simd_vector> D;
		                taylor<4, simd_vector> A0;
                		std::array<simd_vector, NDIM> BB0 = { simd_vector(0.0), simd_vector(0.0), simd_vector(0.0) };

                                if ((opts().p_smooth != MONAGHAN_MULTI)) { // || ((octotiger::fmm::STENCIL_WIDTH - 1) * dx > 2.0 * opts().p_smooth_l)) {
                                        D.set_basis(dX);
                                } else {
					printf("set basis multi in boundary  mono-multi 2\n"); 
					D.set_basis_monaghan(dX, opts().p_smooth_l, int_simd_vector(1));
				}

 		                A0[0] = m0 * D[0];
                		for (integer i = taylor_sizes[0]; i != taylor_sizes[2]; ++i) {
		                        A0[i] = m0 * D[i];
                		}
		                for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                		        A0[i] = m0 * D[i];
		                }

	                        for (integer a = 0; a != NDIM; ++a) {
        	                        int const *abcd_idx_map = to_abcd_idx_map3[a];
                	                for (integer i = 0; i != 10; ++i) {
                        	                const integer bcd_idx = bcd_idx_map[i];
                                	        const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                                        	BB0[a] -= n0[bcd_idx] * tmp;
	                                }
        	                }                
#ifdef OCTOTIGER_HAVE_GRAV_PAR
		                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
                		for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
		                        const integer iii0 = bnd.first[li + i];

                	        	expansion tmp;
#pragma GCC ivdep
			                for (integer j = 0; j != taylor_sizes[3]; ++j) {
        	        		        tmp[j] = A0[j][i];
			                }
                			L[iii0] += tmp;

                        		space_vector &L_ciii0 = L_c[iii0];
#pragma GCC ivdep
			                for (integer j = 0; j != NDIM; ++j) {
        	        	        	L_ciii0[j] += BB0[j][i];
	                		}

				}
			}
		}
	}
}
);
}

void grid::compute_boundary_interactions_monopole_monopole(gsolve_type type, const std::vector<boundary_interaction_type> &ilist_n_bnd,
		const gravity_boundary_type &mpoles) {
	PROFILE();

	auto &M = *M_ptr;
	auto &mon = *mon_ptr;

	// Coefficients for potential evaluation? (David)
	alignas(64) double di0[8] = { 1. / dx, 1. / sqr(dx), 1. / sqr(dx), 1. / sqr(dx) };
	const v4sd d0(di0, Vc::Aligned);

	hpx::parallel::for_loop(for_loop_policy, 0, ilist_n_bnd.size(), [&type, &mpoles, &ilist_n_bnd, &d0, this](std::size_t si) {
	// iterating over the exterior cells
		boundary_interaction_type const &bnd = ilist_n_bnd[si];
		const integer dsize = bnd.first.size();
		integer index = (mpoles.local_semaphore != nullptr) ? bnd.second : si;
		v4sd m0 = (*(mpoles).m)[index];
		v4sd m0_d0 = m0 * d0;

                std::vector<particle> p_bnd(0); // particles in exterior cell
                if ((*(mpoles).p).size() > 0) {
                        p_bnd = (*(mpoles).p)[si];
                }

                const std::array<real, NDIM> Xbase = { X[0][hindex(H_BW, H_BW, H_BW)], X[1][hindex(H_BW, H_BW, H_BW)], X[2][hindex(H_BW, H_BW, H_BW)] };
                space_vector bnd_pos;
                for (integer d = 0; d < NDIM; ++d) {
                        bnd_pos[d] = Xbase[d] + bnd.x[d] * dx;
                }

		for (integer li = 0; li < dsize; ++li) { // iterating over all interior cells that interact with the exterior cell
			// exterior cell interacting with interior cell
			const integer iii0 = bnd.first[li];
			auto tmp1 = m0_d0 * bnd.four[li];
			expansion &Liii0 = L[iii0];
			for (integer i = 0; i != 4; ++i) {
				Liii0[i] += tmp1[i];
			}

			if (type == RHOM) { 
				// particles in interior cell interacting with exterior cell
                        	std::vector<integer> particles_in_cell = get_particles_inds(particles, iii0);
                        	for (auto p_ind : particles_in_cell) {
                                	particle& pcell = particles[p_ind];
                                	v4sd four = make_four(bnd_pos, pcell.pos);
                                	pcell.L += m0 * four;
                        	}
				// interior cell interacting with particles in exterior cell
				if (p_bnd.size() > 0) {
                                	space_vector const &com_cell = (*(com_ptr[0]))[iii0];
                                	std::vector<integer> particles_in_cell = get_particles_inds(particles, iii0);
                                	for (auto p : p_bnd) {
						// interior cell interacting with particles in exterior cell
                                        	v4sd four = make_four(p.pos, com_cell);
                                        	auto tmp1 = p.mass * four;
                                        	expansion &Liii0 = L[iii0];
                                        	for (integer i = 0; i != 4; ++i) {
                                                	Liii0[i] += tmp1[i];
                                        	}
						// particles in interior cell interacting with particles in exterior cell
                                        	for (auto p_ind : particles_in_cell) {
                                                	particle& pcell = particles[p_ind];
                                                	v4sd four = make_four(p.pos, pcell.pos);
                                                	pcell.L += p.mass * four;
                                        	}
                                	}
                        	}
			}	
		}
	});
}

void compute_ilist() {
	// factor = 0.0;
	// factor() += 1.0;
	// for (integer a = 0; a < NDIM; ++a) {
	//     factor(a) += 1.0;
	//     for (integer b = 0; b < NDIM; ++b) {
	//         factor(a, b) += 1.0;
	//         for (integer c = 0; c < NDIM; ++c) {
	//             factor(a, b, c) += 1.0;
	//         }
	//     }
	// }

	const integer inx = INX;
	const integer nx = inx;
	interaction_type np;
	interaction_type dp;
	std::vector<interaction_type> ilist_r0;
	std::vector<interaction_type> ilist_n0;
	std::vector<interaction_type> ilist_d0;
	std::array<std::vector<interaction_type>, geo::direction::count()> ilist_n0_bnd;
	std::array<std::vector<interaction_type>, geo::direction::count()> ilist_d0_bnd;
	const real theta0 = opts().theta;
	const auto theta = [](integer i0, integer j0, integer k0, integer i1, integer j1, integer k1) {
		real tmp = (sqr(i0 - i1) + sqr(j0 - j1) + sqr(k0 - k1));
		// protect against sqrt(0)
		if (tmp > 0.0) {
			return 1.0 / (std::sqrt(tmp));
		} else {
			return 1.0e+10;
		}
	};
	const auto interior = [](integer i, integer j, integer k) {
		bool rc = true;
		if (i < 0 || i >= G_NX) {
			rc = false;
		} else if (j < 0 || j >= G_NX) {
			rc = false;
		} else if (k < 0 || k >= G_NX) {
			rc = false;
		}
		return rc;
	};
	const auto neighbor_dir = [](integer i, integer j, integer k) {
		integer i0 = 0, j0 = 0, k0 = 0;
		if (i < 0) {
			i0 = -1;
		} else if (i >= G_NX) {
			i0 = +1;
		}
		if (j < 0) {
			j0 = -1;
		} else if (j >= G_NX) {
			j0 = +1;
		}
		if (k < 0) {
			k0 = -1;
		} else if (k >= G_NX) {
			k0 = +1;
		}
		geo::direction d;
		d.set(i0, j0, k0);
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
							// protect against sqrt(0)
							const real tmp = sqr(x) + sqr(y) + sqr(z);
							const real r = (tmp == 0) ? 0 : std::sqrt(tmp);
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
							const integer iii1n = gindex((i1 + INX) % INX, (j1 + INX) % INX, (k1 + INX) % INX);
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
//										if ((iii0 == gindex(4, 0, 0)) || (iii1 == gindex(4, 0, 0))) {
//											printf("i0 = (%i, %i, %i), i1 = (%i, %i, %i), n len = %i\n",
//											i0, j0, k0, i1, j1, k1, ilist_n0.size());
//										}
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
//										if ((iii0 == gindex(4, 0, 0)) || (iii1 == gindex(4, 0, 0))) {
//                                                                                	printf("i0 = (%i, %i, %i), i1 = (%i, %i, %i), %e, d len = %i\n",
//                                                                                        i0, j0, k0, i1, j1, k1, r, ilist_d0.size());
//										}
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
//										if ((iii0 == gindex(4, 0, 0)) || (iii1 == gindex(4, 0, 0))) {
//                                                                                	printf("i0 = (%i, %i, %i), i1 = (%i, %i, %i), r len = %i\n",
//                                                                                        i0, j0, k0, i1, j1, k1, ilist_r0.size());
//										}
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

//     printf("# direct = %i\n", int(max_d));
	ilist_n = std::vector<interaction_type>(ilist_n0.begin(), ilist_n0.end());
	ilist_d = std::vector<interaction_type>(ilist_d0.begin(), ilist_d0.end());
	ilist_r = std::vector<interaction_type>(ilist_r0.begin(), ilist_r0.end());
	for (auto &dir : geo::direction::full_set()) {
		auto &d = ilist_d_bnd[dir];
		auto &d0 = ilist_d0_bnd[dir];
//		printf("boundary interaction dir %i d0 length: %i\n", dir, d0.size());
		auto &n = ilist_n_bnd[dir];
		auto &n0 = ilist_n0_bnd[dir];
		for (auto i0 : d0) {
			bool found = false;
			for (auto &i : d) {
				if (i.second == i0.second) {
//					if (i0.first == gindex(4, 0, 0)) {
//						printf("boundary with first 0, (%e, %e, %e)\n", i0.x[0], i0.x[1], i0.x[2]);
//					}
					i.first.push_back(i0.first);
					i.four.push_back(i0.four);
					found = true;
					break;
				}
			}
			if (!found) {
				boundary_interaction_type i;
//				if (i0.first == gindex(4, 0, 0)) {
//					printf("first boundary with first 0, (%e, %e, %e)\n", i0.x[0], i0.x[1], i0.x[2]);
//				}
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
			for (auto &i : n) {
				if (i.second == i0.second) {
					i.first.push_back(i0.first);
					i.four.push_back(i0.four);
					found = true;
					break;
				}
			}
			assert(found);
		}
//		printf("boundary interaction d length: %i\n", d.size());
	}
}

// expansion_pass_type grid::compute_expansions_soa(
//     gsolve_type type, const expansion_pass_type* parent_expansions) {
//     // Create return type expansion_pass_type
//     // which is typedef std::pair<std::vector<expansion>, std::vector<space_vector>>
//     // expansion_pass_type;
//     expansion_pass_type exp_ret;

//     // load data into soa arrays
//     std::vector<expansion>& local_expansions = this->get_L();
//     auto center_of_masses = std::vector<space_vector>(octotiger::fmm::INNER_CELLS);
//     std::vector<space_vector> const& com0 = *(com_ptr[0]);
//     octotiger::fmm::iterate_inner_cells_not_padded([this, &local_expansions, &center_of_masses,
//         com0](const octotiger::fmm::multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
//         local_expansions.at(flat_index_unpadded) = L.at(flat_index_unpadded);
//         center_of_masses.at(flat_index_unpadded) = com0.at(flat_index_unpadded);
//     });

//     octotiger::fmm::struct_of_array_data<expansion, real, 20, octotiger::fmm::ENTRIES,
//         octotiger::fmm::SOA_PADDING>
//         potential_expansions_SoA(local_expansions);
//     octotiger::fmm::struct_of_array_data<space_vector, real, 3, octotiger::fmm::ENTRIES,
//         octotiger::fmm::SOA_PADDING>
//         center_of_masses_SoA(center_of_masses);
//     // Get parents expansions as dummies
//     octotiger::fmm::struct_of_array_data<expansion, real, 20, octotiger::fmm::ENTRIES,
//         octotiger::fmm::SOA_PADDING>
//         parent_expansions_SoA(local_expansions);
//     octotiger::fmm::struct_of_array_data<space_vector, real, 3, octotiger::fmm::ENTRIES,
//         octotiger::fmm::SOA_PADDING>
//         parent_corrections_SoA(center_of_masses);

//     // get child indices
//     const integer inx = INX;
//     const integer nxp = (inx / 2);
//     auto child_index = [=](
//         integer ip, integer jp, integer kp, integer ci, integer bw = 0) -> integer {
//         const integer ic = (2 * (ip) + bw) + ((ci >> 0) & 1);
//         const integer jc = (2 * (jp) + bw) + ((ci >> 1) & 1);
//         const integer kc = (2 * (kp) + bw) + ((ci >> 2) & 1);
//         return (inx + 2 * bw) * (inx + 2 * bw) * ic + (inx + 2 * bw) * jc + kc;
//     };

//     // Iterate through all cells in this block
//     // TODO make version with stride for vectorization
//     octotiger::fmm::iterate_inner_cells_not_padded([this, &local_expansions, &center_of_masses_SoA,
//                                                     com0, &parent_expansions_SoA, &parent_corrections_SoA, &exp_ret, child_index,
//         type, inx](const octotiger::fmm::multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
//         std::array<m2m_vector, NDIM> parent_corrections;
//         const integer index =
//             (INX * INX / 4) * (i_unpadded.x) + (INX / 2) * (i_unpadded.y) + (i_unpadded.z);
//         const integer ic = (2 * (i_unpadded.x));
//         const integer jc = (2 * (i_unpadded.y));
//         const integer kc = (2 * (i_unpadded.z));

//         // Get position of child nodes
//         std::array<std::array<m2m_vector, NDIM>, NCHILD> X;
//         integer childindex = inx * inx * (ic + 0) + inx * (jc + 0) + (kc + 0);
//         X[0][0] = center_of_masses_SoA.value<0>(childindex);
//         X[0][1] = center_of_masses_SoA.value<1>(childindex);
//         X[0][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex = inx * inx * (ic + 1) + inx * (jc + 0) + (kc + 0);
//         X[1][0] = center_of_masses_SoA.value<0>(childindex);
//         X[1][1] = center_of_masses_SoA.value<1>(childindex);
//         X[1][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex = inx * inx * (ic + 0) + inx * (jc + 1) + (kc + 0);
//         X[2][0] = center_of_masses_SoA.value<0>(childindex);
//         X[2][1] = center_of_masses_SoA.value<1>(childindex);
//         X[2][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex = inx * inx * (ic + 1) + inx * (jc + 1) + (kc + 0);
//         X[3][0] = center_of_masses_SoA.value<0>(childindex);
//         X[3][1] = center_of_masses_SoA.value<1>(childindex);
//         X[3][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex = inx * inx * (ic + 0) + inx * (jc + 0) + (kc + 1);
//         X[4][0] = center_of_masses_SoA.value<0>(childindex);
//         X[4][1] = center_of_masses_SoA.value<1>(childindex);
//         X[4][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex = inx * inx * (ic + 1) + inx * (jc + 0) + (kc + 1);
//         X[5][0] = center_of_masses_SoA.value<0>(childindex);
//         X[5][1] = center_of_masses_SoA.value<1>(childindex);
//         X[5][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex = inx * inx * (ic + 0) + inx * (jc + 1) + (kc + 1);
//         X[6][0] = center_of_masses_SoA.value<0>(childindex);
//         X[6][1] = center_of_masses_SoA.value<1>(childindex);
//         X[6][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex = inx * inx * (ic + 1) + inx * (jc + 1) + (kc + 1);
//         X[7][0] = center_of_masses_SoA.value<0>(childindex);
//         X[7][1] = center_of_masses_SoA.value<1>(childindex);
//         X[7][2] = center_of_masses_SoA.value<2>(childindex);

//         // Get expansions and angular corrections of parent node
//         // TODO parent expansions contain zeros if we are at the root node
//         std::array<std::array<m2m_vector, 20>, NCHILD> m_partner;
//         for (auto i = 0; i < NCHILD; ++i) {
//             m_partner[i][0] = parent_expansions_SoA.value<0>(index);
//             m_partner[i][1] = parent_expansions_SoA.value<1>(index);
//             m_partner[i][2] = parent_expansions_SoA.value<2>(index);
//             m_partner[i][3] = parent_expansions_SoA.value<3>(index);
//             m_partner[i][4] = parent_expansions_SoA.value<4>(index);
//             m_partner[i][5] = parent_expansions_SoA.value<5>(index);
//             m_partner[i][6] = parent_expansions_SoA.value<6>(index);
//             m_partner[i][7] = parent_expansions_SoA.value<7>(index);
//             m_partner[i][8] = parent_expansions_SoA.value<8>(index);
//             m_partner[i][9] = parent_expansions_SoA.value<9>(index);
//             m_partner[i][10] = parent_expansions_SoA.value<10>(index);
//             m_partner[i][11] = parent_expansions_SoA.value<11>(index);
//             m_partner[i][12] = parent_expansions_SoA.value<12>(index);
//             m_partner[i][13] = parent_expansions_SoA.value<13>(index);
//             m_partner[i][14] = parent_expansions_SoA.value<14>(index);
//             m_partner[i][15] = parent_expansions_SoA.value<15>(index);
//             m_partner[i][16] = parent_expansions_SoA.value<16>(index);
//             m_partner[i][17] = parent_expansions_SoA.value<17>(index);
//             m_partner[i][18] = parent_expansions_SoA.value<18>(index);
//             m_partner[i][19] = parent_expansions_SoA.value<19>(index);
//         }

//         if (type == RHO) {
//             parent_corrections[0] = parent_corrections_SoA.value<0>(index);
//             parent_corrections[1] = parent_corrections_SoA.value<1>(index);
//             parent_corrections[2] = parent_corrections_SoA.value<2>(index);
//         }

//         // Get position of all interaction partners
//         std::array<std::array<m2m_vector, NDIM>, NCHILD> Y;
//         childindex = index;
//         Y[0][0] = center_of_masses_SoA.value<0>(childindex);
//         Y[0][1] = center_of_masses_SoA.value<1>(childindex);
//         Y[0][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex++;
//         Y[1][0] = center_of_masses_SoA.value<0>(childindex);
//         Y[1][1] = center_of_masses_SoA.value<1>(childindex);
//         Y[1][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex++;
//         Y[2][0] = center_of_masses_SoA.value<0>(childindex);
//         Y[2][1] = center_of_masses_SoA.value<1>(childindex);
//         Y[2][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex++;
//         Y[3][0] = center_of_masses_SoA.value<0>(childindex);
//         Y[3][1] = center_of_masses_SoA.value<1>(childindex);
//         Y[3][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex++;
//         Y[4][0] = center_of_masses_SoA.value<0>(childindex);
//         Y[4][1] = center_of_masses_SoA.value<1>(childindex);
//         Y[4][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex++;
//         Y[5][0] = center_of_masses_SoA.value<0>(childindex);
//         Y[5][1] = center_of_masses_SoA.value<1>(childindex);
//         Y[5][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex++;
//         Y[6][0] = center_of_masses_SoA.value<0>(childindex);
//         Y[6][1] = center_of_masses_SoA.value<1>(childindex);
//         Y[6][2] = center_of_masses_SoA.value<2>(childindex);

//         childindex++;
//         Y[7][0] = center_of_masses_SoA.value<0>(childindex);
//         Y[7][1] = center_of_masses_SoA.value<1>(childindex);
//         Y[7][2] = center_of_masses_SoA.value<2>(childindex);

//         // Calculate distances
//         std::array<std::array<m2m_vector, NDIM>, NCHILD> dX;
//         for (auto i = 0; i < NCHILD; ++i) {
//             dX[i][0] = X[i][0] - Y[i][0];
//             dX[i][1] = X[i][1] - Y[i][1];
//             dX[i][2] = X[i][2] - Y[i][2];
//         }

//         // Calculate taylor expansions for this cell (<<=)
//         taylor_consts tc;
//         std::array<std::array<m2m_vector, NDIM>, NCHILD> cur_expansion;
//         for (auto i = 0; i < NCHILD; ++i) {
//             cur_expansion[i][0] = m_partner[i][0] * dX[i][0];
//             cur_expansion[i][0] = m_partner[i][1] * dX[i][1];
//             cur_expansion[i][0] = m_partner[i][2] * dX[i][2];
//         }
//         for (auto i = 0; i < NCHILD; ++i) {
//             for (auto d1 = 0; d1 < NDIM; ++d1) {
//                 for (auto d2 = 0; d2 < NDIM; ++d2) {
//                     cur_expansion[i][0] += m_partner[i][tc.map2[d1][d2]] * dX[i][d1] * dX[i][d2] * 0.5;
//                     cur_expansion[i][1+d1] += m_partner[i][tc.map2[d1][d2]] * dX[i][d2];
//                 }
//             }
//         }
//         for (auto i = 0; i < NCHILD; ++i) {
//             for (auto d1 = 0; d1 < NDIM; ++d1) {
//                 for (auto d2 = 0; d2 < NDIM; ++d2) {
//                     for (auto d3 = 0; d3 < NDIM; ++d3) {
//                         cur_expansion[i][0] += m_partner[i][tc.map3[d1][d2][d3]] * dX[i][d1] *
//                                                dX[i][d2] * dX[i][d3] * 1.0 / 6.0;
//                         cur_expansion[i][1+d1] += m_partner[i][tc.map3[d1][d2][d3]] * dX[i][d2] * dX[i][d3] * 0.5;
//                         cur_expansion[i][tc.map2[d1][d3]] += m_partner[i][tc.map3[d1][d2][d3]] * dX[i][d3];
//                     }
//                 }
//             }
//             for (auto j = 0; j < NDIM; ++j) {
//                 m_partner[i][j] = cur_expansion[i][j];
//             }

//         }

//         // TODO Hand down expansions to all childs

//         // TODO Create Multipole expansions which is going to handed down to the children

//     });
//     // If we are in a leaf: Calculate the actual expansions
//     // Return multipole expansions which are to be handed down to the children
//     return exp_ret;
// }
expansion_pass_type grid::compute_expansions(gsolve_type type, const expansion_pass_type *parent_expansions) {
	PROFILE();

	expansion_pass_type exp_ret;
	if (!is_leaf) {
		exp_ret.first.resize(INX * INX * INX);
		if (type == RHO || type == RHOM) {
			exp_ret.second.resize(INX * INX * INX);
		}
	}
	const integer inx = INX;
	const integer nxp = (inx / 2);
	auto child_index = [=](integer ip, integer jp, integer kp, integer ci, integer bw = 0) -> integer {
		const integer ic = (2 * (ip) + bw) + ((ci >> 0) & 1);
		const integer jc = (2 * (jp) + bw) + ((ci >> 1) & 1);
		const integer kc = (2 * (kp) + bw) + ((ci >> 2) & 1);
		return (inx + 2 * bw) * (inx + 2 * bw) * ic + (inx + 2 * bw) * jc + kc;
	};
	//printf("in expansion\n");
	for (integer ip = 0; ip != nxp; ++ip) {
		for (integer jp = 0; jp != nxp; ++jp) {
			for (integer kp = 0; kp != nxp; ++kp) {
				const integer iiip = nxp * nxp * ip + nxp * jp + kp;
				std::array<simd_vector, NDIM> X;
				std::array<simd_vector, NDIM> dX;
				taylor<4, simd_vector> l;
				std::array<simd_vector, NDIM> lc;
				std::vector<integer> particles_in_cell;
                                taylor<4, simd_vector> l_part;
                                std::array<simd_vector, NDIM> lc_part;
				if (!is_root) {
					const integer index = (INX * INX / 4) * (ip) + (INX / 2) * (jp) + (kp);
					for (integer j = 0; j != 20; ++j) {
						l[j] = parent_expansions->first[index][j];
						l_part[j] = parent_expansions->first[index][j];
					}
					if (type == RHO || type == RHOM) {
						for (integer j = 0; j != NDIM; ++j) {
							lc[j] = parent_expansions->second[index][j];
							lc_part[j] = parent_expansions->second[index][j];
						}
					}
				} else {
					for (integer j = 0; j != 20; ++j) {
						l[j] = 0.0;
						l_part[j] = 0.0;
					}
					for (integer j = 0; j != NDIM; ++j) {
						lc[j] = 0.0;
						lc_part[j] = 0.0;
					}
				}
				for (integer ci = 0; ci != NCHILD; ++ci) {
					const integer iiic = child_index(ip, jp, kp, ci);
					for (integer d = 0; d < NDIM; ++d) {
						X[d][ci] = (*(com_ptr[0]))[iiic][d];
					}
				}
                                if (is_leaf) { // find particles within the parent cell
                                        space_vector b_min;
                                        space_vector b_max;
                                        for (integer d = 0; d < NDIM; ++d) {
                                                b_min[d] = X[d].min() - 0.5 * dx;
                                                b_max[d] = X[d].max() + 0.5 * dx;
                                        }
                                        for (integer part_i = 0; part_i < particles.size(); ++part_i) {
                                                particle p = particles[part_i];
                                                if (p.is_in_boundary(b_min, b_max)) {
                                                        particles_in_cell.push_back(part_i);
                                                }
                                        }
				}
				const auto &Y = (*(com_ptr[1]))[iiip];
				for (integer d = 0; d < NDIM; ++d) {
					dX[d] = X[d] - Y[d];
				}
				l <<= dX;
				for (integer ci = 0; ci != NCHILD; ++ci) {
					const integer iiic = child_index(ip, jp, kp, ci);
					expansion &Liiic = L[iiic];
					for (integer j = 0; j != 20; ++j) {
						Liiic[j] += l[j][ci];
					}
					space_vector &L_ciiic = L_c[iiic];
					if (type == RHO || type == RHOM) {
						for (integer j = 0; j != NDIM; ++j) {
							L_ciiic[j] += lc[j][ci];
						}
					}

					if (!is_leaf) {
						integer index = child_index(ip, jp, kp, ci, 0);
						for (integer j = 0; j != 20; ++j) {
							exp_ret.first[index][j] = Liiic[j];
						}

						if (type == RHO || type == RHOM) {
							for (integer j = 0; j != 3; ++j) {
								exp_ret.second[index][j] = L_ciiic[j];
							}
						}
					}
				}
				
				// compute expansions to particle locations
				if ((is_leaf) && (type == RHOM)) {
					if (particles_in_cell.size() > 0) {
						for (integer ci = 0; ci != NCHILD; ++ci) { // assume no more than NCHILD (8) particles in parent cell
							if (ci < particles_in_cell.size()) {
                                        			for (integer d = 0; d < NDIM; ++d) {
                                                			X[d][ci] = particles[particles_in_cell[ci]].pos[d];
								}
							}
							else {
                                                                for (integer d = 0; d < NDIM; ++d) {
                                                                        X[d][ci] = 0.0;
                                                                }
							}
						}
                                		for (integer d = 0; d < NDIM; ++d) {
                                        		dX[d] = X[d] - Y[d];
                                		}
                                		l_part <<= dX;
                                
						for (integer ci = 0; ci != NCHILD; ++ci) {
							if (ci < particles_in_cell.size()) {
                                        			particle& p = particles[particles_in_cell[ci]];
                                        			for (integer j = 0; j != 20; ++j) {
                                                			p.L[j] += l_part[j][ci];
                                        			}
                                        			if (type == RHOM) {
                                                			for (integer j = 0; j != NDIM; ++j) {
                                                        			p.L_c[j] += lc_part[j][ci];
                                                			}
								}
                                        		}
                                        	} 
					} // end particle size condition
                                } // end leaf condition
			}
		}
	}

	if (is_leaf) {
		for (integer i = 0; i != G_NX; ++i) {
			for (integer j = 0; j != G_NX; ++j) {
				for (integer k = 0; k != G_NX; ++k) {
					const integer iii = gindex(i, j, k);
					const integer iii0 = h0index(i, j, k);
					const integer iiih = hindex(i + H_BW, j + H_BW, k + H_BW);
					if (type == RHO || type == RHOM) {
						G[iii][phi_i] = physcon().G * L[iii]();
						for (integer d = 0; d < NDIM; ++d) {
							G[iii][gx_i + d] = -physcon().G * L[iii](d);
							if (opts().correct_am_grav) {
								G[iii][gx_i + d] -= physcon().G * L_c[iii][d];
							}
						}
						U[pot_i][iiih] = G[iii][phi_i] * U[rho_i][iiih];
						if (type == RHOM) {
							phi_part[iii0] = physcon().G * L[iii]();
						}
					} else {
						dphi_dt[iii0] = physcon().G * L[iii]();
					}
                                        //if ((xmin[0] == 0.0) && (xmin[1] == 0.0) && (xmin[2] == 0.0) && (i == 0) && (j == 0) && (k == 0)) {
					//	printf("after update expansion gx=%e, gy=%e, gz=%e, phi=%e\n", G[iii][gx_i], G[iii][gy_i], G[iii][gz_i], G[iii][phi_i]);
					//}
				}
			}
		}
		if (type == RHOM) {
			for (auto& p : particles) {
				p.pot = physcon().G * p.L();
				for (integer d = 0; d < NDIM; ++d) {
					p.g[d] = -physcon().G * p.L(d);
                                        if (opts().correct_am_grav) {
                                        	p.g[d] -= physcon().G * p.L_c[d];
                                        }
				}
				//printf("part %i, (%e, %e, %e), phi %e, m %e, g (%e, %e, %e), v (%e, %e, %e)\n", p.id, p.pos[0], p.pos[1], p.pos[2], p.L[0], p.mass, p.g[0], p.g[1], p.g[2], p.vel[0], p.vel[1], p.vel[2]);
			}
		}
	}
	return exp_ret;
}

multipole_pass_type grid::compute_multipoles(gsolve_type type, const multipole_pass_type *child_poles) {
	PROFILE();

	integer lev = 0;
	const real dx3 = dx * dx * dx;
	M_ptr = std::make_shared<std::vector<multipole>>();
	mon_ptr = std::make_shared<std::vector<real>>();
	auto &M = *M_ptr;
	auto &mon = *mon_ptr;
	if (is_leaf) {
		M.resize(0);
		mon.resize(G_N3);
	} else {
		M.resize(G_N3);
		mon.resize(0);
	}
	if (com_ptr[1] == nullptr) {
		com_ptr[1] = std::make_shared<std::vector<space_vector>>(G_N3 / 8);
	}
	if (type == RHO || type == RHOM) {
		com_ptr[0] = std::make_shared<std::vector<space_vector>>(G_N3);
		const integer iii0 = hindex(H_BW, H_BW, H_BW);
		const std::array<real, NDIM> x0 = { X[XDIM][iii0], X[YDIM][iii0], X[ZDIM][iii0] };
		//printf("grid bmin = (%e, %e, %e), index=%i, dx=%e\n", x0[0] - 0.5 * dx, x0[1] - 0.5 * dx, x0[2] - 0.5 * dx, iii0, dx);
		const integer iiin = hindex(H_BW + G_NX - 1, H_BW + G_NX - 1, H_BW + G_NX - 1);
		//printf("grid bmax = (%e, %e, %e), index=%i, dx=%e\n", X[XDIM][iiin] + 0.5 * dx, X[YDIM][iiin] + 0.5 * dx, X[ZDIM][iiin] + 0.5 * dx, iiin, dx);
		for (integer i = 0; i != G_NX; ++i) {
			for (integer j = 0; j != G_NX; ++j) {
				for (integer k = 0; k != G_NX; ++k) {
					auto &com0iii = (*(com_ptr[0]))[gindex(i, j, k)];
					com0iii[0] = x0[0] + i * dx;
					com0iii[1] = x0[1] + j * dx;
					com0iii[2] = x0[2] + k * dx;
				}
			}
		}
	}
	if (com_ptr[0] == nullptr) {
		printf("Failed to call RHO before DRHODT\n");
		abort();
	}
	multipole_pass_type mret;
	if (!is_root) {
		mret.first.resize(INX * INX * INX / NCHILD);
		mret.second.resize(INX * INX * INX / NCHILD);
	} else {
	}
	taylor<4, real> MM;
	integer index = 0;
	for (integer inx = INX; (inx >= INX / 2); inx >>= 1) {
		//printf("inx = %i, lev = %i\n", inx, lev);
		const integer nxp = inx;
		const integer nxc = (2 * inx);

		auto child_index = [=](integer ip, integer jp, integer kp, integer ci) -> integer {
			const integer ic = (2 * ip) + ((ci >> 0) & 1);
			const integer jc = (2 * jp) + ((ci >> 1) & 1);
			const integer kc = (2 * kp) + ((ci >> 2) & 1);
//			printf("cell (%i, %i, %i), child = %i -> (%i, %i, %i)\n", ip, jp, kp, ci, ic, jc, kc);
			return nxc * nxc * ic + nxc * jc + kc;
		};

		for (integer ip = 0; ip != nxp; ++ip) {
			for (integer jp = 0; jp != nxp; ++jp) {
				for (integer kp = 0; kp != nxp; ++kp) {
					const integer iiip = nxp * nxp * ip + nxp * jp + kp;
					if (lev != 0) {
						std::vector<particle> particles_in_cell;
						if (type == RHO || type == RHOM) {
							simd_vector mc;
							std::array<simd_vector, NDIM> Xc;
							for (integer ci = 0; ci != NCHILD; ++ci) {
								const integer iiic = child_index(ip, jp, kp, ci);
						//		printf("in child loop: index = %i, iiip = %i, iiic = %i\n", index, iiip, iiic);
								if (is_leaf) {
						//			printf("leaf\n");
									mc[ci] = mon[iiic];
								} else {
						//			printf("no leaf\n");
									mc[ci] = M[iiic]();
								}
								for (integer d = 0; d < NDIM; ++d) {
									Xc[d][ci] = (*(com_ptr[0]))[iiic][d];
						//			printf("cell center in %i direction of child %i is: %e\n", d, ci, (*(com_ptr[0]))[iiic][d]);
								}
							}
							if (is_leaf && type == RHOM) {
//								integer part_index = NCHILD;
								space_vector b_min;
								space_vector b_max;
								for (integer d = 0; d < NDIM; ++d) {
									b_min[d] = Xc[d].min() - 0.5 * dx;
									b_max[d] = Xc[d].max() + 0.5 * dx;
								}
                                                                space_vector b_min2;
                                                                space_vector b_max2;
                                                                for (integer d = 0; d < NDIM; ++d) {
                                                                        b_min2[d] = Xc[d][0] - 0.5 * dx;
                                                                        b_max2[d] = Xc[d][7] + 0.5 * dx;
                                                                }
								for (integer d = 0; d < NDIM; ++d) {
									if ((b_min[d] != b_min2[d]) || (b_max[d] != b_max2[d])) {
										printf("oh no, %e != %e or %e != %e, %i!!!!!\n\n", b_min[d], b_min2[d], b_max[d], b_max2[d], d);
									}
                                                                }
								//printf("min boundary: (%e, %e, %e), max boundary: (%e, %e, %e), particles: %i\n", b_min[0], b_min[1], b_min[2], b_max[0], b_max[1], b_max[2], particles.size());
								//for (integer part_i = 0; part_i < opts().Part_M.size(); part_i++) {
								for (integer part_i = 0; part_i < particles.size(); part_i++) {
								//	space_vector part_pos;
								//	part_pos[XDIM] = opts().Part_X[part_i];
								//	if (NDIM > 1) {
							//			part_pos[YDIM] = opts().Part_Y[part_i];
							//			if (NDIM > 2) {
							//				part_pos[ZDIM] = opts().Part_Z[part_i];
							//			}
							//		}
//									printf("particle %i, mc index %i, mass %e, pos: (%e, %e, %e)\n", part_i, part_index, opts().Part_M[part_i], opts().Part_X[part_i], opts().Part_Y[part_i], opts().Part_Z[part_i]);
									//particle p = particle(opts().Part_M[part_i], part_pos);
									auto &p = particles[part_i];
									if (p.is_in_boundary(b_min, b_max)) {
//										printf("multipole bmin (%e, %e, %e) bmax (%e, %e, %e)\n", b_min[0], b_min[1], b_min[2], b_max[0], b_max[1], b_max[2]);
										particles_in_cell.push_back(p);
										const integer iii0 = hindex(H_BW, H_BW, H_BW);
										const std::array<real, NDIM> x0 = { X[XDIM][iii0] - 0.5 * dx, X[YDIM][iii0] - 0.5 * dx, X[ZDIM][iii0] - 0.5 * dx };
										const integer i_part = std::min(int((p.pos[XDIM] - x0[XDIM]) / dx), INX - 1);
										const integer j_part = std::min(int((p.pos[YDIM] - x0[YDIM]) / dx), INX - 1);
										const integer k_part = std::min(int((p.pos[ZDIM] - x0[ZDIM]) / dx), INX - 1);
										const std::array<integer, NDIM> p_in_cell = { int(i_part), int(j_part), int(k_part) };
										const std::array<integer, NDIM> p_in_parent_cell = {ip , jp, kp };
										p.update_cell_indexes(p_in_cell, p_in_parent_cell);
//										for (integer d = 0; d < NDIM; ++d) {
//											X[d][part_index] = p.pos[d];
//										}										
  //                                                                              mc[part_index++] = p.mass;
//								printf("part index: %i, mc size: %i\n", part_index, mc.size());
//										printf("inside cell, mass: %e\n", p.mass);
//										printf("inside cell, x: %e\n", p.pos[0]);
//										printf("inside cell, y: %e\n", p.pos[1]);
//										printf("inside cell, z: %e\n", p.pos[2]);
									}
								}
							}
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
							real mtot = mc.sum();
#else
                            				real mtot = Vc::reduce(mc);
#endif
							if (type == RHOM) {
                                                        	for (integer part_i = 0; part_i < particles_in_cell.size(); part_i++) {
                                                                	mtot += particles_in_cell[part_i].mass;
                                                        	}
							} 
							for (integer d = 0; d < NDIM; ++d) {
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
								(*(com_ptr[1]))[iiip][d] = (Xc[d] * mc).sum() / mtot;
#else
                               					(*(com_ptr[1]))[iiip][d] = Vc::reduce(Xc[d] * mc) / mtot;
#endif
								if (type == RHOM) {
        	                                                	for (integer part_i = 0; part_i < particles_in_cell.size(); part_i++) {
	                                                                	(*(com_ptr[1]))[iiip][d] += particles_in_cell[part_i].pos[d] * particles_in_cell[part_i].mass / mtot;
               		                                        	}
								}								
							}							
						}

						taylor<4, simd_vector> mc, mp;
						std::array<simd_vector, NDIM> x, y, dx;
						for (integer ci = 0; ci != NCHILD; ++ci) {

							const integer iiic = child_index(ip, jp, kp, ci);
							const space_vector &Xc = (*(com_ptr[lev - 1]))[iiic];
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

								x[d][ci] = Xc[d];

							}
						}
						//printf("com of %i (index %i): (%e, %e, %e)\n", iiip, index, (*(com_ptr[lev]))[iiip][0], (*(com_ptr[lev]))[iiip][1], (*(com_ptr[lev]))[iiip][2]);
						const space_vector &Y = (*(com_ptr[lev]))[iiip];
						for (integer d = 0; d < NDIM; ++d) {
							dx[d] = x[d] - simd_vector(Y[d]);
						}
						mp = mc >> dx;
						for (integer j = 0; j != 20; ++j) {
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
							MM[j] = mp[j].sum();
#else
                            				MM[j] = Vc::reduce(mp[j]);
#endif
						}
                                                if (type == RHOM && particles_in_cell.size() > 0) { // for now assume no more than 8 particles per cell
	                                                taylor<4, simd_vector> mc_part, mp_part;
							taylor<4, real> MM_part;
        	                                        std::array<simd_vector, NDIM> dx_part;
                	                                for (integer ci = 0; ci != NCHILD; ++ci) {
								if (ci < particles_in_cell.size()) {
	                                                                mc_part()[ci] = particles_in_cell[ci].mass;
                                                			for (integer d = 0; d < NDIM; ++d) {
			                                                        dx_part[d][ci] = particles_in_cell[ci].pos[d] - Y[d];
                        			                        }
								}
								else {
									mc_part()[ci] = 0.0;
                                                                        for (integer d = 0; d < NDIM; ++d) {
                                                                                dx_part[d][ci] = 0.0;
                                                                        }
								}
                                                                for (integer j = 1; j != 20; ++j) {
                                                                        mc_part.ptr()[j][ci] = 0.0;
                                                                }
							}
							mp_part = mc_part >> dx_part;
	                                                for (integer j = 0; j != 20; ++j) {
        	                                                //printf("mp_part[%i] size = %i, part1 = %e, part2 = %e, sum = %e, sum cells = %e\n", j, mp_part[j].size(), (double)(mp_part[j][0]), (double)(mp_part[j][1]), mp_part[j].sum(), MM[j]);
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
	                                                        MM[j] += mp_part[j].sum();
#else
	                                                        MM[j] += Vc::reduce(mp_part[j]);
#endif
        	                                        }
                                                }
					} else {

						if (child_poles == nullptr) {
							//printf("in comp mult: no childs. index = %i, iiip = %i\n", index, iiip);
							const integer iiih = hindex(ip + H_BW, jp + H_BW, kp + H_BW);
							const integer iii0 = h0index(ip, jp, kp);
							if (type == RHO || type == RHOM) {
								mon[iiip] = U[rho_i][iiih] * dx3;
							} else {
								mon[iiip] = dUdt[rho_i][iii0] * dx3;
							}
						} else {
							//printf("in comp mult: has childs. index = %i, iiip = %i\n", index, iiip);
							assert(M.size());
							M[iiip] = child_poles->first[index];
							if (type == RHO || type == RHOM) {
								(*(com_ptr)[lev])[iiip] = child_poles->second[index];
							}
							++index;
						}

					}
					if (!is_root && (lev == 1)) {

						mret.first[index] = MM;
						mret.second[index] = (*(com_ptr[lev]))[iiip];
						++index;

					}
				}
			}
		}
		++lev;
		index = 0;
	}

	return mret;
}

gravity_boundary_type grid::get_gravity_boundary(const geo::direction &dir, bool is_local) {
	PROFILE();

	//	std::array<integer, NDIM> lb, ub;
	gravity_boundary_type data;
	auto &M = *M_ptr;
	auto &mon = *mon_ptr;
	if (!is_local) {
		data.allocate();
		// constexpr size_t blocksize = INX * INX * INX;
		// if (is_leaf) {
		//     data.m->reserve(blocksize);
		//     for (auto i = 0; i < blocksize; ++i) {
		//         data.m->push_back(mon[i]);
		//     }
		// } else {
		//     data.M->reserve(blocksize);
		//     data.x->reserve(blocksize);
		//     for (auto i = 0; i < blocksize; ++i) {
		//         data.M->push_back(M[i]);
		//         data.x->push_back((*(com_ptr[0]))[i]);
		//     }
		// }
		integer iter = 0;
		const std::vector<boundary_interaction_type> &list = ilist_n_bnd[dir.flip()];
		if (is_leaf) {
			data.m->reserve(list.size());
			data.p->resize(list.size());
                        integer p_cnt = 0;
                        for (auto i : list) { 
                                const integer iii = i.second;
                                data.m->push_back(mon[iii]);
                                const auto p_inds = get_particles_inds(particles, iii);
                                auto* cur_p = data.p->data();
                                cur_p[p_cnt].reserve(p_inds.size());
                                for (auto p_i : p_inds) {
                                        cur_p[p_cnt].push_back(particles[p_i]);
                                }
                                p_cnt++;
                        }
		} else {
			data.M->reserve(list.size());
			data.x->reserve(list.size());
			for (auto i : list) {
				const integer iii = i.second;
				const integer top = M[iii].size();
				data.M->push_back(M[iii]);
				data.x->push_back((*(com_ptr[0]))[iii]);
			}
                        if ((opts().p_smooth == MONAGHAN_MULTI)) { //) && ((octotiger::fmm::STENCIL_WIDTH - 1) * dx <= 2.0 * opts().p_smooth_l)) {
                                data.p->resize(1);
                                auto* cur_p = data.p->data();
                                cur_p[0].reserve(particles.size());
                                for (auto p_i : particles) {
                                        cur_p[0].push_back(p_i);
                                }
			}
		}
	} else {
		if (is_leaf) {
			data.m = mon_ptr;
                        if (particles.size() > 0) {
                                const std::vector<boundary_interaction_type> &list = ilist_n_bnd[dir.flip()];
                                data.p->resize(list.size());
                                for (integer i = 0; i < list.size(); i++) {
                                        const integer iii = list[i].second;
                                        const auto p_inds = get_particles_inds(particles, iii);
                                        auto* cur_p = data.p->data();
                                        cur_p[i].reserve(p_inds.size());
                                        for (auto p_i : p_inds) {
                                                cur_p[i].push_back(particles[p_i]);
                                        }
                                }
                        }
		} else {
			data.M = M_ptr;
			data.x = com_ptr[0];
                        if ((opts().p_smooth == MONAGHAN_MULTI)) { // && ((octotiger::fmm::STENCIL_WIDTH - 1) * dx < 4.0 * opts().p_smooth_l)) {
                                data.p->resize(1);
                                auto* cur_p = data.p->data();
                                cur_p[0].reserve(particles.size());
                                for (auto p_i : particles) {
                                        cur_p[0].push_back(p_i);
                                }
			}
		}
	}

	return data;
}

neighbor_gravity_type grid::fill_received_array(neighbor_gravity_type raw_input) {
	PROFILE();
	const std::vector<boundary_interaction_type> &list = ilist_n_bnd[raw_input.direction];
	neighbor_gravity_type filled_array;
	filled_array.data.allocate();
	constexpr size_t blocksize = INX * INX * INX;
	filled_array.direction = raw_input.direction;
	filled_array.is_monopole = raw_input.is_monopole;
	if (raw_input.is_monopole) {
		// fill with unimportant data
		filled_array.data.m->resize(blocksize);
		// Overwrite data that matters
		int counter = 0;
		for (auto i : list) {
			const integer iii = i.second;
			filled_array.data.m->at(iii) = raw_input.data.m->at(counter);
			counter++;
		}
	} else {
		// fill with unimportant data
		filled_array.data.M->resize(blocksize);
		filled_array.data.x->resize(blocksize);
		// Overwrite data that matters
		int counter = 0;
		for (auto i : list) {
			const integer iii = i.second;
			filled_array.data.M->at(iii) = raw_input.data.M->at(counter);
			filled_array.data.x->at(iii) = raw_input.data.x->at(counter);
			counter++;
		}
	}
	return filled_array;
}

const std::vector<boundary_interaction_type>& grid::get_ilist_n_bnd(const geo::direction &dir) {
	return ilist_n_bnd[dir];
}
