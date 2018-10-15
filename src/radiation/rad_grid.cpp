#include "defs.hpp"
#include "rad_grid.hpp"
#include "grid.hpp"
#include "options.hpp"
#include "node_server.hpp"
#include "opacities.hpp"
#include "physcon.hpp"
#include "../roe.hpp"

//#define IMPLICIT_OFF
#include <iostream>
#include "implicit.hpp"

extern options opts;

using hiprec = double;

constexpr auto _0 = hiprec(0);
constexpr auto _1 = hiprec(1);
constexpr auto _2 = hiprec(2);
constexpr auto _3 = hiprec(3);
constexpr auto _4 = hiprec(4);
constexpr auto _5 = hiprec(5);

integer rindex(integer x, integer y, integer z) {
	return z + R_NX * (y + R_NX * x);
}

typedef node_server::set_rad_grid_action set_rad_grid_action_type;
HPX_REGISTER_ACTION (set_rad_grid_action_type);

hpx::future<void> node_client::set_rad_grid(std::vector<real>&& g/*, std::vector<real>&& o*/) const {
	return hpx::async<typename node_server::set_rad_grid_action>(get_unmanaged_gid(), g/*, o*/);
}

void node_server::set_rad_grid(const std::vector<real>& data/*, std::vector<real>&& outflows*/) {
	rad_grid_ptr->set_prolong(data/*, std::move(outflows)*/);
}

typedef node_server::send_rad_boundary_action send_rad_boundary_action_type;
HPX_REGISTER_ACTION (send_rad_boundary_action_type);

typedef node_server::send_rad_flux_correct_action send_rad_flux_correct_action_type;
HPX_REGISTER_ACTION (send_rad_flux_correct_action_type);

void node_client::send_rad_flux_correct(std::vector<real>&& data, const geo::face& face, const geo::octant& ci) const {
	hpx::apply<typename node_server::send_rad_flux_correct_action>(get_unmanaged_gid(), std::move(data), face, ci);
}

void node_server::recv_rad_flux_correct(std::vector<real>&& data, const geo::face& face, const geo::octant& ci) {
	const geo::quadrant index(ci, face.get_dimension());
	niece_rad_channels[face][index].set_value(std::move(data));
}

hpx::future<void> node_client::send_rad_boundary(std::vector<rad_type>&& data, const geo::direction& dir,
		std::size_t cycle) const {
	return hpx::async<typename node_server::send_rad_boundary_action>(get_gid(), std::move(data), dir, cycle);
}

void node_server::recv_rad_boundary(std::vector<rad_type>&& bdata, const geo::direction& dir, std::size_t cycle) {
	sibling_rad_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

typedef node_server::send_rad_children_action send_rad_children_action_type;
HPX_REGISTER_ACTION (send_rad_children_action_type);

void node_server::recv_rad_children(std::vector<real>&& data, const geo::octant& ci, std::size_t cycle) {
	child_rad_channels[ci].set_value(std::move(data), cycle);
}

hpx::future<void> node_client::send_rad_children(std::vector<real>&& data, const geo::octant& ci, std::size_t cycle) const {
	return hpx::async<typename node_server::send_rad_children_action>(get_unmanaged_gid(), std::move(data), ci, cycle);
}

void rad_grid::rad_imp(std::vector<real>& egas, std::vector<real>& tau, std::vector<real>& sx, std::vector<real>& sy,
		std::vector<real>& sz, const std::vector<real>& rho, real dt) {
#ifdef IMPLICIT_OFF
	return;
#endif

	const integer d = H_BW - R_BW;
	const real clight = physcon.c;
	const real clightinv = INVERSE(clight);
	const real fgamma = grid::get_fgamma();
	for (integer i = R_BW; i != R_NX - R_BW; ++i) {
		for (integer j = R_BW; j != R_NX - R_BW; ++j) {
			for (integer k = R_BW; k != R_NX - R_BW; ++k) {
				const integer iiih = hindex(i + d, j + d, k + d);
				const integer iiir = rindex(i, j, k);
				const real den = rho[iiih];
				const real deninv = INVERSE(den);
				real vx = sx[iiih] * deninv;
				real vy = sy[iiih] * deninv;
				real vz = sz[iiih] * deninv;

				/* Compute e0 from dual energy formalism */
				real e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * den;
				e0 -= 0.5 * vy * vy * den;
				e0 -= 0.5 * vz * vz * den;
				if (opts.eos == WD) {
					e0 -= ztwd_energy(den);
				}
				if (e0 < egas[iiih] * de_switch2) {
					e0 = std::pow(tau[iiih], fgamma);
				}
				real E0 = U[er_i][iiir];
				space_vector F0;
				space_vector u0;
				F0[0] = U[fx_i][iiir];
				F0[1] = U[fy_i][iiir];
				F0[2] = U[fz_i][iiir];
				u0[0] = vx;
				u0[1] = vy;
				u0[2] = vz;
				real E1 = E0;
				space_vector F1 = F0;
				space_vector u1 = u0;
				real e1 = e0;

				const auto ddt = implicit_radiation_step_2nd_order(E1, e1, F1, u1, den, mmw[iiir], dt);
				const real dE_dt = ddt.first;
				const real dFx_dt = ddt.second[0];
				const real dFy_dt = ddt.second[1];
				const real dFz_dt = ddt.second[2];

				/* Accumulate derivatives */
				U[er_i][iiir] += dE_dt * dt;
				U[fx_i][iiir] += dFx_dt * dt;
				U[fy_i][iiir] += dFy_dt * dt;
				U[fz_i][iiir] += dFz_dt * dt;

				egas[iiih] -= dE_dt * dt;
				sx[iiih] -= dFx_dt * dt * clightinv * clightinv;
				sy[iiih] -= dFy_dt * dt * clightinv * clightinv;
				sz[iiih] -= dFz_dt * dt * clightinv * clightinv;

				/* Find tau with dual energy formalism*/
				real e = egas[iiih];
				e -= 0.5 * sx[iiih] * sx[iiih] * deninv;
				e -= 0.5 * sy[iiih] * sy[iiih] * deninv;
				e -= 0.5 * sz[iiih] * sz[iiih] * deninv;
				if (opts.eos == WD) {
					e -= ztwd_energy(den);
				}
				if (e < de_switch1 * egas[iiih]) {
					e = e1;
				}
				if (U[er_i][iiir] <= 0.0) {
					printf("Er = %e %e %e %e\n", E0, E1, U[er_i][iiir], dt);
					abort();
				}
				tau[iiih] = std::pow(e, INVERSE(fgamma));
				if (tau[iiih] < 0.0) {
					abort_error()
					;
				}
				if (U[er_i][iiir] <= 0.0) {
					printf("2231242!!! %e %e %e \n", E0, U[er_i][iiir], dE_dt * dt);
					abort();
				}
			}
		}
	}
}
/*void node_server::recv_rad_children(std::vector<rad_type>&& bdata, const geo::octant& oct, const geo::octant& ioct) {
 child_rad_channels[ioct][oct]->set_value(std::move(bdata));
 }*/

void rad_grid::get_output(std::array<std::vector<real>, OUTPUT_COUNT>& v, integer i, integer j, integer k) const {
	const integer iii = rindex(i, j, k);
	v[NF + 0].push_back(real(U[er_i][iii]));
	v[NF + 1].push_back(real(U[fx_i][iii]));
	v[NF + 2].push_back(real(U[fy_i][iii]));
	v[NF + 3].push_back(real(U[fz_i][iii]));

}

void rad_grid::set_dx(real _dx) {
	dx = _dx;
}

void rad_grid::set_X(const std::vector<std::vector<real>>& x) {
	X.resize(NDIM);
	for (integer d = 0; d != NDIM; ++d) {
		X[d].resize(R_N3);
		for (integer xi = 0; xi != R_NX; ++xi) {
			for (integer yi = 0; yi != R_NX; ++yi) {
				for (integer zi = 0; zi != R_NX; ++zi) {
					const auto D = H_BW - R_BW;
					const integer iiir = rindex(xi, yi, zi);
					const integer iiih = hindex(xi + D, yi + D, zi + D);
					//		printf( "%i %i %i %i %i %i \n", d, iiir, xi, yi, zi, iiih);
					X[d][iiir] = x[d][iiih];
				}
			}
		}
	}
}

real rad_grid::hydro_signal_speed(const std::vector<real>& egas, const std::vector<real>& tau, const std::vector<real>& sx,
		const std::vector<real>& sy, const std::vector<real>& sz, const std::vector<real>& rho) {
	real a = 0.0;
	const real fgamma = grid::get_fgamma();
	for (integer xi = R_BW; xi != R_NX - R_BW; ++xi) {
		for (integer yi = R_BW; yi != R_NX - R_BW; ++yi) {
			for (integer zi = R_BW; zi != R_NX - R_BW; ++zi) {
				const integer D = H_BW - R_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				const real rhoinv = INVERSE(rho[iiih]);
				real vx = sx[iiih] * rhoinv;
				real vy = sy[iiih] * rhoinv;
				real vz = sz[iiih] * rhoinv;
				real e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * rho[iiih];
				e0 -= 0.5 * vy * vy * rho[iiih];
				e0 -= 0.5 * vz * vz * rho[iiih];
				if (opts.eos == WD) {
					e0 -= ztwd_energy(rho[iiih]);
				}
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}

				real this_a = (4.0 / 9.0) * U[er_i][iiir] * rhoinv;
				//		printf( "%e %e %e %e\n",rho[iiih], e0, mmw[iiir],dx );
				const real cons = kappa_R(rho[iiih], e0, mmw[iiir]) * dx;
				if (cons < 32.0) {
					this_a *= std::max(1.0 - std::exp(-cons), 0.0);
				}
				a = std::max(this_a, a);
			}
		}
	}
	return SQRT(a);
}

void rad_grid::compute_mmw(const std::vector<std::vector<real>>& U) {
	mmw.resize(R_N3);
	for (integer i = 0; i != R_NX; ++i) {
		for (integer j = 0; j != R_NX; ++j) {
			for (integer k = 0; k != R_NX; ++k) {
				const integer d = H_BW - R_BW;
				const integer iiir = rindex(i, j, k);
				const integer iiih = hindex(i + d, j + d, k + d);
				std::array<real, NSPECIES> spc;
				for (integer si = 0; si != NSPECIES; ++si) {
					spc[si] = U[spc_i + si][iiih];
					mmw[iiir] = mean_ion_weight(spc);
				}
			}
		}
	}

}

void node_server::compute_radiation(real dt) {
//	physcon.c = 1.0;
	if (my_location.level() == 0) {
		printf("c = %e\n", physcon.c);
	}

	rad_grid_ptr->set_dx(grid_ptr->get_dx());
	auto rgrid = rad_grid_ptr;
	rad_grid_ptr->compute_mmw(grid_ptr->U);
	const real min_dx = TWO * grid::get_scaling_factor() / real(INX << opts.max_level);
	const real clight = physcon.c;
	const real max_dt = min_dx / clight * 0.4;
	const real ns = std::ceil(dt * INVERSE(max_dt));
	if (ns > std::numeric_limits<int>::max()) {
		printf("Number of substeps greater than %i. dt = %e max_dt = %e\n", std::numeric_limits<int>::max(), dt, max_dt);
	}
	integer nsteps = std::max(int(ns), 1);

	const real this_dt = dt * INVERSE(real(nsteps));
	auto& egas = grid_ptr->get_field(egas_i);
	const auto& rho = grid_ptr->get_field(rho_i);
	auto& tau = grid_ptr->get_field(tau_i);
	auto& sx = grid_ptr->get_field(sx_i);
	auto& sy = grid_ptr->get_field(sy_i);
	auto& sz = grid_ptr->get_field(sz_i);

	if (my_location.level() == 0) {
		printf("Explicit\n");
	}
	rgrid->store();
	for (integer i = 0; i != nsteps; ++i) {
		rgrid->sanity_check();
		if (my_location.level() == 0) {
			printf("rad sub-step %i of %i\r", int(i + 1), int(nsteps));
			fflush(stdout);
		}
		all_rad_bounds();
		rgrid->compute_flux();
		GET(exchange_rad_flux_corrections());
		rgrid->advance(this_dt, 1.0);
	}
	if (my_location.level() == 0) {
		printf("\nImplicit\n");
	}
	rgrid->sanity_check();
	rgrid->rad_imp(egas, tau, sx, sy, sz, rho, dt);
	all_rad_bounds();
	if (my_location.level() == 0) {
		printf("Rad done\n");
	}
}

std::array<std::array<hiprec, NDIM>, NDIM> compute_p2(real E_, real Fx_, real Fy_, real Fz_) {


	const hiprec E = E_;
	const hiprec Fx = Fx_;
	const hiprec Fy = Fy_;
	const hiprec Fz = Fz_;

	const hiprec clight = physcon.c;
	std::array<std::array<hiprec, NDIM>, NDIM> P;
	hiprec f = SQRT(Fx * Fx + Fy * Fy + Fz * Fz) * INVERSE(clight * E);
	hiprec nx, ny, nz;
	assert(E > _0);
	if (f > _0) {
		const hiprec finv = INVERSE(clight * E * f);
		nx = Fx * finv;
		ny = Fy * finv;
		nz = Fz * finv;
	} else {
		nx = ny = nz = _0;
	}
	const hiprec chi = (_3 + _4 * f * f) * INVERSE((_5 + _2 * SQRT(_4 - _3 * f * f)));
	const hiprec f1 = ((_1 - chi) / _2);
	const hiprec f2 = ((_3 * chi - _1) / _2);
	P[XDIM][YDIM] = P[YDIM][XDIM] = f2 * nx * ny * E;
	P[XDIM][ZDIM] = P[ZDIM][XDIM] = f2 * nx * nz * E;
	P[ZDIM][YDIM] = P[YDIM][ZDIM] = f2 * ny * nz * E;
	P[XDIM][XDIM] = (f1 + f2 * nx * nx) * E;
	P[YDIM][YDIM] = (f1 + f2 * ny * ny) * E;
	P[ZDIM][ZDIM] = (f1 + f2 * nz * nz) * E;
	return P;
}

void rad_grid::allocate() {
	rad_grid::dx = dx;
	for (integer f = 0; f != NRF; ++f) {
		U0[f].resize(R_N3);
		U[f].resize(R_N3);
		for (integer d = 0; d != NDIM; ++d) {
			flux[d][f].resize(R_N3);
		}
	}
}

void rad_grid::store() {
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = 0; i != R_N3; ++i) {
			U0[f][i] = U[f][i];
		}
	}
}

void rad_grid::restore() {
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = 0; i != R_N3; ++i) {
			U[f][i] = U0[f][i];
		}
	}
}

inline real minmod(real a, real b) {
//	return 0;
	bool a_is_neg = a < 0;
	bool b_is_neg = b < 0;
	if (a_is_neg != b_is_neg)
		return ZERO;

	real val = std::min(std::abs(a), std::abs(b));
	return a_is_neg ? -val : val;
}

void rad_grid::sanity_check() {
	for (integer xi = R_BW; xi != R_NX - R_BW; ++xi) {
		for (integer yi = R_BW; yi != R_NX - R_BW; ++yi) {
			for (integer zi = R_BW; zi != R_NX - R_BW; ++zi) {
				const integer iiir = rindex(xi, yi, zi);
				if (U[er_i][iiir] <= 0.0) {
					printf("INSANE\n");
					//		printf("%e %i %i %i\n", U[er_i][iiir], xi, yi, zi);
					abort();
				}
			}
		}
	}
}

std::size_t rad_grid::load(std::istream& strm) {
//	printf( "LOADING\n");
	std::size_t cnt = 0;
	cnt += read(strm, dx);
	for (integer i = R_BW; i < R_NX - R_BW; ++i) {
		for (integer j = R_BW; j < R_NX - R_BW; ++j) {
			const integer iiir = rindex(i, j, R_BW);
			for (integer f = 0; f != NRF; ++f) {
				cnt += read(strm, &U[f][iiir], INX);
			}
		}
	}
	return cnt;
}

std::size_t rad_grid::save(std::ostream& strm) const {
	std::size_t cnt = 0;

	cnt += write(strm, dx);
	for (integer i = R_BW; i < R_NX - R_BW; ++i) {
		for (integer j = R_BW; j < R_NX - R_BW; ++j) {
			const integer iiir = rindex(i, j, R_BW);
			for (integer f = 0; f != NRF; ++f) {
				cnt += write(strm, &U[f][iiir], INX);
			}
		}
	}
	return cnt;
}

static inline real vanleer2(real a, real b) {
	if (a == 0.0 || b == 0.0) {
		return 0.0;
	} else {
		const real ainv = INVERSE(std::abs(a));
		const real binv = INVERSE(std::abs(b));
		return (b * binv + a * ainv) / (ainv + binv);
	}
}

void rad_grid::compute_flux() {
	const hiprec clight = physcon.c;


	const auto lambda_max = []( hiprec mu, hiprec er, hiprec absf) {
		if( er > 0.0 ) {
			const hiprec clight = physcon.c;
			hiprec f = absf * INVERSE (clight*er);
			const hiprec tmp = SQRT(_4-_3*f*f);
			const hiprec tmp2 = SQRT((_2/_3)*(_4-_3*f*f -tmp)+_2*mu*mu*(_2-f*f-tmp));
			return hiprec((tmp2 + std::abs(mu*f)) * INVERSE( tmp ));
		} else {
			return _0;
		}
	};

	for (integer i = 0; i != R_N3; ++i) {
		for (integer f = 0; f != NRF; ++f) {
			for (integer d = 0; d != NDIM; ++d) {
				flux[d][f][i] = 0.0;
			}
		}
	}

	const integer D[3] = { DX, DY, DZ };
	for (int face_dim = 0; face_dim < NDIM; face_dim++) {
		for (integer l = R_BW; l != R_NX - R_BW + (face_dim == XDIM ? 1 : 0); ++l) {
			for (integer j = R_BW; j != R_NX - R_BW + (face_dim == YDIM ? 1 : 0); ++j) {
				for (integer k = R_BW; k != R_NX - R_BW + (face_dim == ZDIM ? 1 : 0); ++k) {
					integer i = rindex(l, j, k);
					hiprec f_p[3], f_m[3];
					hiprec absf_m = _0, absf_p = _0;
					const hiprec er_m = U[er_i][i-D[face_dim]];
					const hiprec er_p = U[er_i][i];
					for (integer d = 0; d != NDIM; ++d) {
						f_m[d] = U[fx_i + d][i - D[face_dim]];
						f_p[d] = U[fx_i + d][i];
					}
					const auto P_p = compute_p2(er_p, f_p[0], f_p[1], f_p[2]);
					const auto P_m = compute_p2(er_m, f_m[0], f_m[1], f_m[2]);

					hiprec mu_m = _0;
					hiprec mu_p = _0;
					for (int d = 0; d < 3; d++) {
						absf_m += f_m[d] * f_m[d];
						absf_p += f_p[d] * f_p[d];
					}
					absf_m = SQRT(absf_m);
					absf_p = SQRT(absf_p);
					if (absf_m > _0) {
						mu_m = f_m[face_dim] * INVERSE(absf_m);
					}
					if (absf_p > _0) {
						mu_p = f_p[face_dim] * INVERSE(absf_p);
					}
					constexpr hiprec half = _1/_2;
					const hiprec a_m = lambda_max(mu_m, er_m, absf_m);
					const hiprec a_p = lambda_max(mu_p, er_p, absf_p);
					const hiprec a = std::max(a_m, a_p) * clight;
					flux[face_dim][er_i][i] = (f_p[face_dim] + f_m[face_dim]) * half - (er_p - er_m) * half * a;
					for (integer flux_dim = 0; flux_dim != NDIM; ++flux_dim) {
						flux[face_dim][fx_i + flux_dim][i] = clight * clight
								* (P_p[flux_dim][face_dim] + P_m[flux_dim][face_dim]) * half
								- (f_p[flux_dim] - f_m[flux_dim]) * half * a;
					}
				}
			}
		}
	}
}

void rad_grid::change_units(real m, real l, real t, real k) {
	const real l2 = l * l;
	const real t2 = t * t;
	const real t2inv = 1.0 * INVERSE(t2);
	const real tinv = 1.0 * INVERSE(t);
	const real l3 = l2 * l;
	const real l3inv = 1.0 * INVERSE(l3);
	for (integer i = 0; i != R_N3; ++i) {
		U[er_i][i] *= (m * l2 * t2inv) * l3inv;
		U[fx_i][i] *= tinv * (m * t2inv);
		U[fy_i][i] *= tinv * (m * t2inv);
		U[fz_i][i] *= tinv * (m * t2inv);
	}
}

void rad_grid::advance(real dt, real beta) {
	const real l = dt * INVERSE(dx);
	const integer D[3] = { DX, DY, DZ };
	for (integer f = 0; f != NRF; ++f) {
		for (integer xi = R_BW; xi != R_NX - R_BW; ++xi) {
			for (integer yi = R_BW; yi != R_NX - R_BW; ++yi) {
				for (integer zi = R_BW; zi != R_NX - R_BW; ++zi) {
					const integer iii = rindex(xi, yi, zi);
					const real& u0 = U0[f][iii];
					real u1 = U[f][iii];
					for (integer d = 0; d != NDIM; ++d) {
						u1 -= l * (flux[d][f][iii + D[d]] - flux[d][f][iii]);
					}
					for (integer d = 0; d != NDIM; ++d) {
						U[f][iii] = u0 * (1.0 - beta) + beta * u1;
					}
				}
			}
		}
	}
}

void rad_grid::set_physical_boundaries(geo::face face) {
	for (integer i = 0; i != R_NX; ++i) {
		for (integer j = 0; j != R_NX; ++j) {
			for (integer k = 0; k != R_BW; ++k) {
				integer iii1, iii0;
				switch (face) {
				case 0:
					iii1 = rindex(k, i, j);
					iii0 = rindex(R_BW, i, j);
					break;
				case 1:
					iii1 = rindex(R_NX - 1 - k, i, j);
					iii0 = rindex(R_NX - 1 - R_BW, i, j);
					break;
				case 2:
					iii1 = rindex(i, k, j);
					iii0 = rindex(i, R_BW, j);
					break;
				case 3:
					iii1 = rindex(i, R_NX - 1 - k, j);
					iii0 = rindex(i, R_NX - 1 - R_BW, j);
					break;
				case 4:
					iii1 = rindex(i, j, k);
					iii0 = rindex(i, j, R_BW);
					break;
				case 5:
				default:
					iii1 = rindex(i, j, R_NX - 1 - k);
					iii0 = rindex(i, j, R_NX - 1 - R_BW);
				}
				for (integer f = 0; f != NRF; ++f) {
					U[f][iii1] = U[f][iii0];
					//		U[f][iii1] = 0.0;
				}
				for (integer d = 0; d != NDIM; ++d) {
				}
				switch (face) {
				case 0:
					U[fx_i][iii1] = std::min(U[fx_i][iii1], 0.0);
					break;
				case 1:
					U[fx_i][iii1] = std::max(U[fx_i][iii1], 0.0);
					break;
				case 2:
					U[fy_i][iii1] = std::min(U[fy_i][iii1], 0.0);
					break;
				case 3:
					U[fy_i][iii1] = std::max(U[fy_i][iii1], 0.0);
					break;
				case 4:
					U[fz_i][iii1] = std::min(U[fz_i][iii1], 0.0);
					break;
				case 5:
					U[fz_i][iii1] = std::max(U[fz_i][iii1], 0.0);
					break;
				}
			}
		}
	}
}

hpx::future<void> node_server::exchange_rad_flux_corrections() {
	const geo::octant ci = my_location.get_child_index();
	constexpr auto full_set = geo::face::full_set();
	for (auto& f : full_set) {
		const auto face_dim = f.get_dimension();
		auto const& this_aunt = aunts[f];
		if (!this_aunt.empty()) {
			std::array<integer, NDIM> lb, ub;
			lb[XDIM] = lb[YDIM] = lb[ZDIM] = R_BW;
			ub[XDIM] = ub[YDIM] = ub[ZDIM] = INX + R_BW;
			if (f.get_side() == geo::MINUS) {
				lb[face_dim] = R_BW;
			} else {
				lb[face_dim] = INX + R_BW;
			}
			ub[face_dim] = lb[face_dim] + 1;
			auto data = rad_grid_ptr->get_flux_restrict(lb, ub, face_dim);
			this_aunt.send_rad_flux_correct(std::move(data), f.flip(), ci);
		}
	}

	return hpx::async(hpx::util::annotated_function([this]() {
		constexpr integer size = geo::face::count() * geo::quadrant::count();
		std::array<hpx::future<void>, size> futs;
		integer index = 0;
		for (auto const& f : geo::face::full_set()) {
			if (this->nieces[f] == +1) {
				for (auto const& quadrant : geo::quadrant::full_set()) {
					futs[index++] =
					niece_rad_channels[f][quadrant].get_future().then(
							hpx::util::annotated_function(
									[this, f, quadrant](hpx::future<std::vector<real> > && fdata) -> void
									{
										const auto face_dim = f.get_dimension();
										std::array<integer, NDIM> lb, ub;
										switch (face_dim) {
											case XDIM:
											lb[XDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + R_BW;
											lb[YDIM] = quadrant.get_side(0) * (INX / 2) + R_BW;
											lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + R_BW;
											ub[XDIM] = lb[XDIM] + 1;
											ub[YDIM] = lb[YDIM] + (INX / 2);
											ub[ZDIM] = lb[ZDIM] + (INX / 2);
											break;
											case YDIM:
											lb[XDIM] = quadrant.get_side(0) * (INX / 2) + R_BW;
											lb[YDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + R_BW;
											lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + R_BW;
											ub[XDIM] = lb[XDIM] + (INX / 2);
											ub[YDIM] = lb[YDIM] + 1;
											ub[ZDIM] = lb[ZDIM] + (INX / 2);
											break;
											case ZDIM:
											default:
											lb[XDIM] = quadrant.get_side(0) * (INX / 2) + R_BW;
											lb[YDIM] = quadrant.get_side(1) * (INX / 2) + R_BW;
											lb[ZDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + R_BW;
											ub[XDIM] = lb[XDIM] + (INX / 2);
											ub[YDIM] = lb[YDIM] + (INX / 2);
											ub[ZDIM] = lb[ZDIM] + 1;
											break;
										}
										rad_grid_ptr->set_flux_restrict(GET(fdata), lb, ub, face_dim);
									}, "node_server::exchange_rad_flux_corrections::set_flux_restrict"
							));
				}
			}
		}
		return hpx::future<void>(hpx::when_all(std::move(futs)));
	}, "node_server::set_rad_flux_restrict"));
}

void rad_grid::set_flux_restrict(const std::vector<rad_type>& data, const std::array<integer, NDIM>& lb,
		const std::array<integer, NDIM>& ub, const geo::dimension& dim) {
	PROF_BEGIN;
	integer index = 0;
	for (integer field = 0; field != NRF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					const integer iii = rindex(i, j, k);
					flux[dim][field][iii] = data[index];
					++index;
				}
			}
		}
	}PROF_END;
}

std::vector<rad_type> rad_grid::get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
		const geo::dimension& dim) const {
	PROF_BEGIN;
	std::vector<rad_type> data;
	integer size = 1;
	for (auto& dim : geo::dimension::full_set()) {
		size *= (ub[dim] - lb[dim]);
	}
	size /= (NCHILD / 2);
	size *= NRF;
	data.reserve(size);
	const integer stride1 = (dim == XDIM) ? (R_NX) : (R_NX) * (R_NX);
	const integer stride2 = (dim == ZDIM) ? (R_NX) : 1;
	for (integer field = 0; field != NRF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; i += 2) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; j += 2) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; k += 2) {
					const integer i00 = rindex(i, j, k);
					const integer i10 = i00 + stride1;
					const integer i01 = i00 + stride2;
					const integer i11 = i00 + stride1 + stride2;
					real value = ZERO;
					value += flux[dim][field][i00];
					value += flux[dim][field][i10];
					value += flux[dim][field][i01];
					value += flux[dim][field][i11];
					//		const real f = dx / TWO;
					/*if (opts.ang_con) {
					 if (field == zx_i) {
					 if (dim == YDIM) {
					 value += F[dim][sy_i][i00] * f;
					 value += F[dim][sy_i][i10] * f;
					 value -= F[dim][sy_i][i01] * f;
					 value -= F[dim][sy_i][i11] * f;
					 } else if (dim == ZDIM) {
					 value -= F[dim][sz_i][i00] * f;
					 value -= F[dim][sz_i][i10] * f;
					 value += F[dim][sz_i][i01] * f;
					 value += F[dim][sz_i][i11] * f;
					 } else if (dim == XDIM) {
					 value += F[dim][sy_i][i00] * f;
					 value += F[dim][sy_i][i10] * f;
					 value -= F[dim][sy_i][i01] * f;
					 value -= F[dim][sy_i][i11] * f;
					 value -= F[dim][sz_i][i00] * f;
					 value += F[dim][sz_i][i10] * f;
					 value -= F[dim][sz_i][i01] * f;
					 value += F[dim][sz_i][i11] * f;
					 }
					 } else if (field == zy_i) {
					 if (dim == XDIM) {
					 value -= F[dim][sx_i][i00] * f;
					 value -= F[dim][sx_i][i10] * f;
					 value += F[dim][sx_i][i01] * f;
					 value += F[dim][sx_i][i11] * f;
					 } else if (dim == ZDIM) {
					 value += F[dim][sz_i][i00] * f;
					 value -= F[dim][sz_i][i10] * f;
					 value += F[dim][sz_i][i01] * f;
					 value -= F[dim][sz_i][i11] * f;
					 } else if (dim == YDIM) {
					 value -= F[dim][sx_i][i00] * f;
					 value -= F[dim][sx_i][i10] * f;
					 value += F[dim][sx_i][i01] * f;
					 value += F[dim][sx_i][i11] * f;
					 value += F[dim][sz_i][i00] * f;
					 value -= F[dim][sz_i][i10] * f;
					 value += F[dim][sz_i][i01] * f;
					 value -= F[dim][sz_i][i11] * f;
					 }
					 } else if (field == zz_i) {
					 if (dim == XDIM) {
					 value += F[dim][sx_i][i00] * f;
					 value -= F[dim][sx_i][i10] * f;
					 value += F[dim][sx_i][i01] * f;
					 value -= F[dim][sx_i][i11] * f;
					 } else if (dim == YDIM) {
					 value -= F[dim][sy_i][i00] * f;
					 value += F[dim][sy_i][i10] * f;
					 value -= F[dim][sy_i][i01] * f;
					 value += F[dim][sy_i][i11] * f;
					 } else if (dim == ZDIM) {
					 value -= F[dim][sy_i][i00] * f;
					 value += F[dim][sy_i][i10] * f;
					 value -= F[dim][sy_i][i01] * f;
					 value += F[dim][sy_i][i11] * f;
					 value += F[dim][sx_i][i00] * f;
					 value += F[dim][sx_i][i10] * f;
					 value -= F[dim][sx_i][i01] * f;
					 value -= F[dim][sx_i][i11] * f;
					 }
					 }
					 }*/
					value /= real(4);
					data.push_back(value);
				}
			}
		}
	}PROF_END;
	return data;
}

void node_server::all_rad_bounds() {
	GET(exchange_interlevel_rad_data());
	collect_radiation_bounds();
	send_rad_amr_bounds();
	rcycle++;
}

hpx::future<void> node_server::exchange_interlevel_rad_data() {

	hpx::future<void> f = hpx::make_ready_future();
	integer ci = my_location.get_child_index();

	if (is_refined) {
		for (auto const& ci : geo::octant::full_set()) {
			auto data = GET(child_rad_channels[ci].get_future(rcycle));
			rad_grid_ptr->set_restrict(data, ci);
		}
	}
	if (my_location.level() > 0) {
		auto data = rad_grid_ptr->get_restrict();
		f = parent.send_rad_children(std::move(data), ci, rcycle);
	}
	return std::move(f);
}

void node_server::collect_radiation_bounds() {
	for (auto const& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			auto bdata = rad_grid_ptr->get_boundary(dir);
			neighbors[dir].send_rad_boundary(std::move(bdata), dir.flip(), rcycle);
		}
	}

	std::array<hpx::future<void>, geo::direction::count()> results;
	for (auto& r : results) {
		r = hpx::make_ready_future();
	}
	integer index = 0;
	for (auto const& dir : geo::direction::full_set()) {
		if (!(neighbors[dir].empty() && my_location.level() == 0)) {
			results[index++] = sibling_rad_channels[dir].get_future(rcycle).then(
					hpx::util::annotated_function([this](hpx::future<sibling_rad_type> && f) -> void
					{
						auto&& tmp = GET(f);
						rad_grid_ptr->set_boundary(tmp.data, tmp.direction );
					}, "node_server::collect_rad_bounds::set_rad_boundary"));
		}
	}
	for (auto& f : results) {
		GET(f);
	}
//	wait_all_and_propagate_exceptions(std::move(results));

	for (auto& face : geo::face::full_set()) {
		if (my_location.is_physical_boundary(face)) {
			rad_grid_ptr->set_physical_boundaries(face);
		}
	}
}

void rad_grid::initialize_erad(const std::vector<real> rho, const std::vector<real> tau) {
	const real fgamma = grid::get_fgamma();
	for (integer xi = 0; xi != R_NX; ++xi) {
		for (integer yi = 0; yi != R_NX; ++yi) {
			for (integer zi = 0; zi != R_NX; ++zi) {
				const auto D = H_BW - R_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				const real ei = POWER(tau[iiih], fgamma);
				U[er_i][iiir] = B_p(rho[iiih], ei, mmw[iiir]) * (4.0 * M_PI / physcon.c);
				U[fx_i][iiir] = U[fy_i][iiir] = U[fz_i][iiir] = 0.0;
			}
		}
	}
}

rad_grid::rad_grid(real _dx) :
		dx(_dx) {
	allocate();
}

rad_grid::rad_grid() {
	allocate();
}

void rad_grid::set_boundary(const std::vector<real>& data, const geo::direction& dir) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, dir, OUTER, INX, R_BW);
	integer iter = 0;

	for (integer field = 0; field != NRF; ++field) {
		auto& Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					Ufield[rindex(i, j, k)] = data[iter];
					++iter;
				}
			}
		}
	}PROF_END;
}

std::vector<real> rad_grid::get_boundary(const geo::direction& dir) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size;
	size = NRF * get_boundary_size(lb, ub, dir, INNER, INX, R_BW);
	data.resize(size);
	integer iter = 0;

	for (integer field = 0; field != NRF; ++field) {
		auto& Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					data[iter] = Ufield[rindex(i, j, k)];
					++iter;
				}
			}
		}
	}

	PROF_END;
	return data;
}

void rad_grid::set_field(rad_type v, integer f, integer i, integer j, integer k) {
	U[f][rindex(i, j, k)] = v;
}

void rad_grid::set_prolong(const std::vector<real>& data) {
	integer index = 0;
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = R_BW; i != R_NX - R_BW; ++i) {
			for (integer j = R_BW; j != R_NX - R_BW; ++j) {
				for (integer k = R_BW; k != R_NX - R_BW; ++k) {
					const integer iii = rindex(i, j, k);
					U[f][iii] = data[index];
					++index;
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub) {
	std::vector<real> data;
	integer size = NRF;
	for (integer dim = 0; dim != NDIM; ++dim) {
		size *= (ub[dim] - lb[dim]);
	}
	auto lb0 = lb;
	auto ub0 = ub;
	for (integer d = 0; d != NDIM; ++d) {
		lb0[d] /= 2;
		ub0[d] /= 2;
	}

	for (integer f = 0; f != NRF; ++f) {
		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
					const integer iii = rindex(i / 2, j / 2, k / 2);
					real value = U[f][iii];
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

std::vector<real> rad_grid::get_restrict() const {
	std::vector<real> data;
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = R_BW; i < R_NX - R_BW; i += 2) {
			for (integer j = R_BW; j < R_NX - R_BW; j += 2) {
				for (integer k = R_BW; k < R_NX - R_BW; k += 2) {
					const integer iii = rindex(i, j, k);
					real v = ZERO;
					for (integer x = 0; x != 2; ++x) {
						for (integer y = 0; y != 2; ++y) {
							for (integer z = 0; z != 2; ++z) {
								const integer jjj = iii + x * R_NX * R_NX + y * R_NX + z;
								v += U[f][jjj];
							}
						}
					}
					v /= real(NCHILD);
					data.push_back(v);
				}
			}
		}
	}
	return data;
}

void rad_grid::set_restrict(const std::vector<real>& data, const geo::octant& octant) {
	integer index = 0;
	const integer i0 = octant.get_side(XDIM) * (INX / 2);
	const integer j0 = octant.get_side(YDIM) * (INX / 2);
	const integer k0 = octant.get_side(ZDIM) * (INX / 2);
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = R_BW; i != R_NX / 2; ++i) {
			for (integer j = R_BW; j != R_NX / 2; ++j) {
				for (integer k = R_BW; k != R_NX / 2; ++k) {
					const integer iii = rindex(i + i0, j + j0, k + k0);
					U[f][iii] = data[index];
					++index;
					if (index > int(data.size())) {
						printf("rad_grid::set_restrict error %i %i\n", int(index), int(data.size()));
					}
				}
			}
		}
	}

}
;

void node_server::send_rad_amr_bounds() {
	if (is_refined) {
		constexpr auto full_set = geo::octant::full_set();
		for (auto& ci : full_set) {
			const auto& flags = amr_flags[ci];
			for (auto& dir : geo::direction::full_set()) {
				if (flags[dir]) {
					std::array<integer, NDIM> lb, ub;
					std::vector<real> data;
					get_boundary_size(lb, ub, dir, OUTER, INX, R_BW);
					for (integer dim = 0; dim != NDIM; ++dim) {
						lb[dim] = ((lb[dim] - R_BW)) + 2 * R_BW + ci.get_side(dim) * (INX);
						ub[dim] = ((ub[dim] - R_BW)) + 2 * R_BW + ci.get_side(dim) * (INX);
					}
					data = rad_grid_ptr->get_prolong(lb, ub);
					children[ci].send_rad_boundary(std::move(data), dir, rcycle);
				}
			}
		}
	}
}

typedef node_server::erad_init_action erad_init_action_type;
HPX_REGISTER_ACTION (erad_init_action_type);

hpx::future<void> node_client::erad_init() const {
	return hpx::async<typename node_server::erad_init_action>(get_unmanaged_gid());
}

void node_server::erad_init() {
	std::array<hpx::future<void>, NCHILD> futs;
	int index = 0;
	if (is_refined) {
		for (auto& child : children) {
			futs[index++] = child.erad_init();
		}
	}
	grid_ptr->rad_init();
	if (is_refined) {
		hpx::wait_all(futs);
	}
}
