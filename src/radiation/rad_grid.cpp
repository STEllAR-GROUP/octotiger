//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/radiation/implicit.hpp"
#include "octotiger/radiation/kernel_interface.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/real.hpp"
#include "octotiger/roe.hpp"
#include "octotiger/space_vector.hpp"

#include <hpx/include/future.hpp>

#include <array>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "octotiger/unitiger/radiation/radiation_physics_impl.hpp"

using real = double;

std::unordered_map<std::string, int> rad_grid::str_to_index;
std::unordered_map<int, std::string> rad_grid::index_to_str;

void rad_grid::static_init() {
	str_to_index["er"] = er_i;
	str_to_index["fx"] = fx_i;
	str_to_index["fy"] = fy_i;
	str_to_index["fz"] = fz_i;
	for (const auto &s : str_to_index) {
		index_to_str[s.second] = s.first;
	}
}

std::vector<std::string> rad_grid::get_field_names() {
	std::vector<std::string> rc;
	for (auto i : str_to_index) {
		rc.push_back(i.first);
	}
	return rc;
}

void rad_grid::set(const std::string name, real *data) {
	auto iter = str_to_index.find(name);
	real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
	if (iter != str_to_index.end()) {
		int f = iter->second;
		int jjj = 0;
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
					data[jjj] /= f == er_i ? eunit : funit;
					U[f][iii] = data[jjj];
					jjj++;
				}
			}
		}
	}

}

std::vector<silo_var_t> rad_grid::var_data() const {
	std::vector<silo_var_t> s;
	real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
	for (auto l : str_to_index) {
		const int f = l.second;
		std::string this_name = l.first;
		int jjj = 0;
		silo_var_t this_s(this_name);
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
					this_s(jjj) = U[f][iii];
					this_s(jjj) *= f == er_i ? eunit : funit;
					this_s.set_range(this_s(jjj));
					jjj++;
				}
			}
		}
		s.push_back(std::move(this_s));
	}
	return std::move(s);
}

constexpr auto _0 = real(0);
constexpr auto _1 = real(1);
constexpr auto _2 = real(2);
constexpr auto _3 = real(3);
constexpr auto _4 = real(4);
constexpr auto _5 = real(5);

using set_rad_grid_action_type = node_server::set_rad_grid_action;
HPX_REGISTER_ACTION(set_rad_grid_action_type);

hpx::future<void> node_client::set_rad_grid(std::vector<real> &&g/*, std::vector<real>&& o*/) const {
	return hpx::async<typename node_server::set_rad_grid_action>(get_unmanaged_gid(), g/*, o*/);
}

void node_server::set_rad_grid(const std::vector<real> &data/*, std::vector<real>&& outflows*/) {
	rad_grid_ptr->set_prolong(data/*, std::move(outflows)*/);
}

using send_rad_boundary_action_type = node_server::send_rad_boundary_action;
HPX_REGISTER_ACTION(send_rad_boundary_action_type);

using send_rad_flux_correct_action_type = node_server::send_rad_flux_correct_action;
HPX_REGISTER_ACTION(send_rad_flux_correct_action_type);

void node_client::send_rad_flux_correct(std::vector<real> &&data, const geo::face &face, const geo::octant &ci) const {
	hpx::apply<typename node_server::send_rad_flux_correct_action>(get_unmanaged_gid(), std::move(data), face, ci);
}

void node_server::recv_rad_flux_correct(std::vector<real> &&data, const geo::face &face, const geo::octant &ci) {
	const geo::quadrant index(ci, face.get_dimension());
	niece_rad_channels[face][index].set_value(std::move(data));
}

hpx::future<void> node_client::send_rad_boundary(std::vector<real> &&data, const geo::direction &dir, std::size_t cycle) const {
	return hpx::async<typename node_server::send_rad_boundary_action>(get_gid(), std::move(data), dir, cycle);
}

void node_server::recv_rad_boundary(std::vector<real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_real tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

using send_rad_children_action_type = node_server::send_rad_children_action;
HPX_REGISTER_ACTION(send_rad_children_action_type);

void node_server::recv_rad_children(std::vector<real> &&data, const geo::octant &ci, std::size_t cycle) {
	child_rad_channels[ci].set_value(std::move(data), cycle);
}

hpx::future<void> node_client::send_rad_children(std::vector<real> &&data, const geo::octant &ci, std::size_t cycle) const {
	return hpx::async<typename node_server::send_rad_children_action>(get_unmanaged_gid(), std::move(data), ci, cycle);
}

void rad_grid::rad_imp(std::vector<real> &egas, std::vector<real> &tau, std::vector<real> &sx, std::vector<real> &sy, std::vector<real> &sz,
		const std::vector<real> &rho, real dt) {
	PROFILE()
	;
	const integer d = H_BW - RAD_BW;
	const real clight = physcon().c;
	const real clightinv = INVERSE(clight);
	const real fgamma = grid::get_fgamma();
	octotiger::radiation::radiation_kernel<er_i, fx_i, fy_i, fz_i>(d, rho, sx, sy, sz, egas, tau, fgamma, U, mmw, X_spc, Z_spc, dt, clightinv);
}

void rad_grid::set_dx(real _dx) {
	dx = _dx;
}

void rad_grid::set_X(const std::vector<std::vector<real>> &x) {
	X.resize(NDIM);
	for (integer d = 0; d != NDIM; ++d) {
		X[d].resize(RAD_N3);
		for (integer xi = 0; xi != RAD_NX; ++xi) {
			for (integer yi = 0; yi != RAD_NX; ++yi) {
				for (integer zi = 0; zi != RAD_NX; ++zi) {
					const auto D = H_BW - RAD_BW;
					const integer iiir = rindex(xi, yi, zi);
					const integer iiih = hindex(xi + D, yi + D, zi + D);
					//		printf( "%i %i %i %i %i %i \n", d, iiir, xi, yi, zi, iiih);
					X[d][iiir] = x[d][iiih];
				}
			}
		}
	}
}

real rad_grid::hydro_signal_speed(const std::vector<real> &egas, const std::vector<real> &tau, const std::vector<real> &sx, const std::vector<real> &sy,
		const std::vector<real> &sz, const std::vector<real> &rho) {
	real a = 0.0;
	const real fgamma = grid::get_fgamma();
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer D = H_BW - RAD_BW;
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
				if (opts().eos == WD) {
					e0 -= ztwd_energy(rho[iiih]);
				}
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}

				real this_a = (4.0 / 9.0) * U[er_i][iiir] * rhoinv;
				//		printf( "%e %e %e %e\n",rho[iiih], e0, mmw[iiir],dx );
				const real cons = kappa_R(rho[iiih], e0, mmw[iiir], X_spc[iiir], Z_spc[iiir]) * dx;
				if (cons < 32.0) {
					this_a *= std::max(1.0 - std::exp(-cons), 0.0);
				}
				a = std::max(this_a, a);
			}
		}
	}
	return SQRT(a);
}

void rad_grid::compute_mmw(const std::vector<std::vector<safe_real>> &U) {
	mmw.resize(RAD_N3);
	X_spc.resize(RAD_N3);
	Z_spc.resize(RAD_N3);
	for (integer i = 0; i != RAD_NX; ++i) {
		for (integer j = 0; j != RAD_NX; ++j) {
			for (integer k = 0; k != RAD_NX; ++k) {
				const integer d = H_BW - RAD_BW;
				const integer iiir = rindex(i, j, k);
				const integer iiih = hindex(i + d, j + d, k + d);
				specie_state_t<real> spc;
				for (integer si = 0; si != opts().n_species; ++si) {
					spc[si] = U[spc_i + si][iiih];
				}
				mean_ion_weight(spc, mmw[iiir], X_spc[iiir], Z_spc[iiir]);
			}
		}
	}

}

void node_server::compute_radiation(real dt, real omega) {
//	physcon().c = 1.0;
	if (my_location.level() == 0) {
//		printf("c = %e\n", physcon().c);
	}

	rad_grid_ptr->set_dx(grid_ptr->get_dx());
	auto rgrid = rad_grid_ptr;
	rad_grid_ptr->compute_mmw(grid_ptr->U);
	const real min_dx = TWO * grid::get_scaling_factor() / real(INX << opts().max_level);
	const real clight = physcon().c / 10.0;
	const real max_dt = min_dx / clight * 0.2;
	const real ns = std::ceil(dt * INVERSE(max_dt));
	if (ns > std::numeric_limits<int>::max()) {
		printf("Number of substeps greater than %i. dt = %e max_dt = %e\n", std::numeric_limits<int>::max(), dt, max_dt);
	}
	integer nsteps = std::max(int(ns), 1);

	const real this_dt = dt * INVERSE(real(nsteps));
	auto &egas = grid_ptr->get_field(egas_i);
	const auto &rho = grid_ptr->get_field(rho_i);
	auto &tau = grid_ptr->get_field(tau_i);
	auto &sx = grid_ptr->get_field(sx_i);
	auto &sy = grid_ptr->get_field(sy_i);
	auto &sz = grid_ptr->get_field(sz_i);
	rad_grid_ptr->set_X(grid_ptr->get_X());

//	if (my_location.level() == 0) {
//		printf("Explicit\n");
//	}
	if (opts().rad_implicit) {
		rgrid->rad_imp(egas, tau, sx, sy, sz, rho, 0.5 * dt);
	}
	for (integer i = 0; i != nsteps; ++i) {
		//	rgrid->sanity_check();
		if (my_location.level() == 0) {
			printf("radiation sub-step %i of %i\r", int(i + 1), int(nsteps));
			fflush(stdout);
		}

		rgrid->store();
		const double beta[3] = { 1.0, 0.25, 2.0 / 3.0 };
		for (int rk = 0; rk < 3; rk++) {
			all_rad_bounds();
			rgrid->compute_flux(omega);
			if( my_location.level() == 0 ) printf( "\nbounds 10\n");
			GET(exchange_rad_flux_corrections());
			if( my_location.level() == 0 ) printf( "\nbounds 11\n");
			rgrid->advance(this_dt, beta[rk]);
		}

	}
	if (opts().rad_implicit) {
		rgrid->rad_imp(egas, tau, sx, sy, sz, rho, 0.5 * dt);
	}
//	rgrid->sanity_check();
	all_rad_bounds();
	if (my_location.level() == 0) {
		printf("\n");
//		printf("Rad done\n");
	}
}

template<class T>
T minmod(T a, T b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

void rad_grid::allocate() {
	rad_grid::dx = dx;
	U.resize(NRF);
	flux.resize(NDIM);
	for (int d = 0; d < NDIM; d++) {
		flux[d].resize(NRF);
	}
	for (integer f = 0; f != NRF; ++f) {
		U0[f].resize(RAD_N3);
		U[f].resize(RAD_N3);
		for (integer d = 0; d != NDIM; ++d) {
			flux[d][f].resize(RAD_N3);
		}
	}
}

void rad_grid::store() {
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = 0; i != RAD_N3; ++i) {
			U0[f][i] = U[f][i];
		}
	}
}

void rad_grid::restore() {
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = 0; i != RAD_N3; ++i) {
			U[f][i] = U0[f][i];
		}
	}
}

void rad_grid::sanity_check() {
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
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

void rad_grid::compute_flux(real omega) {
	PROFILE()
	;
	radiation_physics<NDIM>::set_clight(physcon().c / 10.0);
	if (opts().correct_am_hydro) {
		hydro.use_angmom_correction(fx_i);
	}
	const auto &q = hydro.reconstruct(U, X, omega);
	hydro.flux(U, q, flux, X, omega);
}

void rad_grid::change_units(real m, real l, real t, real k) {
	const real l2 = l * l;
	const real t2 = t * t;
	const real t2inv = 1.0 * INVERSE(t2);
	const real tinv = 1.0 * INVERSE(t);
	const real l3 = l2 * l;
	const real l3inv = 1.0 * INVERSE(l3);
	for (integer i = 0; i != RAD_N3; ++i) {
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
		for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
			for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
				for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
					const integer iii = rindex(xi, yi, zi);
					const real &u0 = U0[f][iii];
					real u1 = U[f][iii];
					for (integer d = 0; d != NDIM; ++d) {
						u1 -= l * (flux[d][f][iii + D[d]] - flux[d][f][iii]);
					}
					U[f][iii] = u0 * (1.0 - beta) + beta * u1;
				}
			}
		}
	}
}

void rad_grid::set_physical_boundaries(geo::face face, real t) {
	for (integer i = 0; i != RAD_NX; ++i) {
		for (integer j = 0; j != RAD_NX; ++j) {
			for (integer k = 0; k != RAD_BW; ++k) {
				integer iii1, iii0;
				switch (face) {
				case 0:
					iii1 = rindex(k, i, j);
					iii0 = rindex(RAD_BW, i, j);
					break;
				case 1:
					iii1 = rindex(RAD_NX - 1 - k, i, j);
					iii0 = rindex(RAD_NX - 1 - RAD_BW, i, j);
					break;
				case 2:
					iii1 = rindex(i, k, j);
					iii0 = rindex(i, RAD_BW, j);
					break;
				case 3:
					iii1 = rindex(i, RAD_NX - 1 - k, j);
					iii0 = rindex(i, RAD_NX - 1 - RAD_BW, j);
					break;
				case 4:
					iii1 = rindex(i, j, k);
					iii0 = rindex(i, j, RAD_BW);
					break;
				case 5:
				default:
					iii1 = rindex(i, j, RAD_NX - 1 - k);
					iii0 = rindex(i, j, RAD_NX - 1 - RAD_BW);
				}
				for (integer f = 0; f != NRF; ++f) {
					U[f][iii1] = U[f][iii0];
				}
				switch (face) {
				case 0:
					if (opts().problem == MARSHAK) {
						if (t > 0) {
							auto u = marshak_wave_analytic(-opts().xscale, 0, 0, t);
							U[fx_i][iii1] = u[opts().n_fields + fx_i];
							U[er_i][iii1] = std::max(u[opts().n_fields + er_i], 1.0e-10);
						} else {
							U[fx_i][iii1] = 0.0;
							U[er_i][iii1] = 1.0e-10;
						}
					} else {
						U[fx_i][iii1] = std::min(U[fx_i][iii1], 0.0);
					}
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
	for (auto &f : full_set) {
		const auto face_dim = f.get_dimension();
		auto const &this_aunt = aunts[f];
		if (!this_aunt.empty()) {
			std::array<integer, NDIM> lb, ub;
			lb[XDIM] = lb[YDIM] = lb[ZDIM] = RAD_BW;
			ub[XDIM] = ub[YDIM] = ub[ZDIM] = INX + RAD_BW;
			if (f.get_side() == geo::MINUS) {
				lb[face_dim] = RAD_BW;
			} else {
				lb[face_dim] = INX + RAD_BW;
			}
			ub[face_dim] = lb[face_dim] + 1;
			auto data = rad_grid_ptr->get_flux_restrict(lb, ub, face_dim);
			this_aunt.send_rad_flux_correct(std::move(data), f.flip(), ci);
		}
	}

	constexpr integer size = geo::face::count() * geo::quadrant::count();
	std::array<future<void>, size> futs;
	for (auto &f : futs) {
		f = hpx::make_ready_future();
	}
	integer index = 0;
	for (auto const &f : geo::face::full_set()) {
		if (this->nieces[f] == +1) {
			for (auto const &quadrant : geo::quadrant::full_set()) {
				futs[index++] = niece_rad_channels[f][quadrant].get_future().then([this, f, quadrant](hpx::future<std::vector<real> > && fdata) -> void
				{
					const auto face_dim = f.get_dimension();
					std::array<integer, NDIM> lb, ub;
					switch (face_dim) {
						case XDIM:
						lb[XDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
						lb[YDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
						lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
						ub[XDIM] = lb[XDIM] + 1;
						ub[YDIM] = lb[YDIM] + (INX / 2);
						ub[ZDIM] = lb[ZDIM] + (INX / 2);
						break;
						case YDIM:
						lb[XDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
						lb[YDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
						lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
						ub[XDIM] = lb[XDIM] + (INX / 2);
						ub[YDIM] = lb[YDIM] + 1;
						ub[ZDIM] = lb[ZDIM] + (INX / 2);
						break;
						case ZDIM:
						default:
						lb[XDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
						lb[YDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
						lb[ZDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
						ub[XDIM] = lb[XDIM] + (INX / 2);
						ub[YDIM] = lb[YDIM] + (INX / 2);
						ub[ZDIM] = lb[ZDIM] + 1;
						break;
					}
					rad_grid_ptr->set_flux_restrict(GET(fdata), lb, ub, face_dim);
				});
			}
		}
	}
	return hpx::when_all(std::move(futs)).then([](future<decltype(futs)> fout) {
		auto fin = GET(fout);
		for (auto &f : fin) {
			GET(f);
		}
	});
}

void rad_grid::set_flux_restrict(const std::vector<real> &data, const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub,
		const geo::dimension &dim) {

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
	}
}

std::vector<real> rad_grid::get_flux_restrict(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub, const geo::dimension &dim) const {

	std::vector<real> data;
	integer size = 1;
	for (auto &dim : geo::dimension::full_set()) {
		size *= (ub[dim] - lb[dim]);
	}
	size /= (NCHILD / 2);
	size *= NRF;
	data.reserve(size);
	const integer stride1 = (dim == XDIM) ? (RAD_NX) : (RAD_NX) * (RAD_NX);
	const integer stride2 = (dim == ZDIM) ? (RAD_NX) : 1;
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
					value /= real(4);
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

void node_server::all_rad_bounds() {
	if( my_location.level() == 0 ) printf( "\nbounds 1\n");
	GET(exchange_interlevel_rad_data());
	if( my_location.level() == 0 ) printf( "\nbounds 2\n");
	collect_radiation_bounds();
	if( my_location.level() == 0 ) printf( "\nbounds 3\n");
	send_rad_amr_bounds();
	if( my_location.level() == 0 ) printf( "\nbounds 4\n");
	rcycle++;
}

hpx::future<void> node_server::exchange_interlevel_rad_data() {

	hpx::future<void> f = hpx::make_ready_future();
	integer ci = my_location.get_child_index();

	if (is_refined) {
		for (auto const &ci : geo::octant::full_set()) {
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
	for (auto const &dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			auto bdata = rad_grid_ptr->get_boundary(dir);
			neighbors[dir].send_rad_boundary(std::move(bdata), dir.flip(), rcycle);
		}
	}

	std::array<hpx::future<void>, geo::direction::count()> results;
	for (auto &r : results) {
		r = hpx::make_ready_future();
	}
	integer index = 0;
	for (auto const &dir : geo::direction::full_set()) {
		if (!(neighbors[dir].empty() && my_location.level() == 0)) {
			results[index++] = sibling_rad_channels[dir].get_future(rcycle).then([this](hpx::future<sibling_real> &&f) -> void {
				auto &&tmp = GET(f);
				rad_grid_ptr->set_boundary(tmp.data, tmp.direction);
			});
		}
	}
	for (auto &f : results) {
		GET(f);
	}
//	wait_all_and_propagate_exceptions(std::move(results));

	for (auto &face : geo::face::full_set()) {
		if (my_location.is_physical_boundary(face)) {
			rad_grid_ptr->set_physical_boundaries(face, current_time);
		}
	}
}

void rad_grid::initialize_erad(const std::vector<safe_real> rho, const std::vector<safe_real> tau) {
	const real fgamma = grid::get_fgamma();
	for (integer xi = 0; xi != RAD_NX; ++xi) {
		for (integer yi = 0; yi != RAD_NX; ++yi) {
			for (integer zi = 0; zi != RAD_NX; ++zi) {
				const auto D = H_BW - RAD_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				const real ei = POWER(tau[iiih], fgamma);
				U[er_i][iiir] = B_p((double) rho[iiih], (double) ei, (double) mmw[iiir]) * (4.0 * M_PI / physcon().c);
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

void rad_grid::set_boundary(const std::vector<real> &data, const geo::direction &dir) {

	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, dir, OUTER, INX, RAD_BW);
	integer iter = 0;

	for (integer field = 0; field != NRF; ++field) {
		auto &Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					Ufield[rindex(i, j, k)] = data[iter];
					++iter;
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_boundary(const geo::direction &dir) {

	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size = NRF * get_boundary_size(lb, ub, dir, INNER, INX, RAD_BW);
	data.resize(size);
	integer iter = 0;

	for (integer field = 0; field != NRF; ++field) {
		auto &Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					data[iter] = Ufield[rindex(i, j, k)];
					++iter;
				}
			}
		}
	}

	return data;
}

void rad_grid::set_field(real v, integer f, integer i, integer j, integer k) {
	U[f][rindex(i, j, k)] = v;
}

real rad_grid::get_field(integer f, integer i, integer j, integer k) const {
	return U[f][rindex(i, j, k)];
}

void rad_grid::set_prolong(const std::vector<real> &data) {
	integer index = 0;
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = RAD_BW; i != RAD_NX - RAD_BW; ++i) {
			for (integer j = RAD_BW; j != RAD_NX - RAD_BW; ++j) {
				for (integer k = RAD_BW; k != RAD_NX - RAD_BW; ++k) {
					const integer iii = rindex(i, j, k);
					U[f][iii] = data[index];
					++index;
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_prolong(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub) {
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
		for (integer i = RAD_BW; i < RAD_NX - RAD_BW; i += 2) {
			for (integer j = RAD_BW; j < RAD_NX - RAD_BW; j += 2) {
				for (integer k = RAD_BW; k < RAD_NX - RAD_BW; k += 2) {
					const integer iii = rindex(i, j, k);
					real v = ZERO;
					for (integer x = 0; x != 2; ++x) {
						for (integer y = 0; y != 2; ++y) {
							for (integer z = 0; z != 2; ++z) {
								const integer jjj = iii + x * RAD_NX * RAD_NX + y * RAD_NX + z;
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

void rad_grid::set_restrict(const std::vector<real> &data, const geo::octant &octant) {
	integer index = 0;
	const integer i0 = octant.get_side(XDIM) * (INX / 2);
	const integer j0 = octant.get_side(YDIM) * (INX / 2);
	const integer k0 = octant.get_side(ZDIM) * (INX / 2);
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = RAD_BW; i != RAD_NX / 2; ++i) {
			for (integer j = RAD_BW; j != RAD_NX / 2; ++j) {
				for (integer k = RAD_BW; k != RAD_NX / 2; ++k) {
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
		for (auto &ci : full_set) {
			const auto &flags = amr_flags[ci];
			for (auto &dir : geo::direction::full_set()) {
				if (flags[dir]) {
					std::array<integer, NDIM> lb, ub;
					std::vector<real> data;
					get_boundary_size(lb, ub, dir, OUTER, INX, RAD_BW);
					for (integer dim = 0; dim != NDIM; ++dim) {
						lb[dim] = ((lb[dim] - RAD_BW)) + 2 * RAD_BW + ci.get_side(dim) * (INX);
						ub[dim] = ((ub[dim] - RAD_BW)) + 2 * RAD_BW + ci.get_side(dim) * (INX);
					}
					data = rad_grid_ptr->get_prolong(lb, ub);
					children[ci].send_rad_boundary(std::move(data), dir, rcycle);
				}
			}
		}
	}
}

using erad_init_action_type = node_server::erad_init_action;
HPX_REGISTER_ACTION(erad_init_action_type);

hpx::future<void> node_client::erad_init() const {
	return hpx::async<typename node_server::erad_init_action>(get_unmanaged_gid());
}

void node_server::erad_init() {
	std::array<hpx::future<void>, NCHILD> futs;
	int index = 0;
	if (is_refined) {
		for (auto &child : children) {
			futs[index++] = child.erad_init();
		}
	}
	grid_ptr->rad_init();
	if (is_refined) {
		hpx::wait_all(futs);
	}
}
