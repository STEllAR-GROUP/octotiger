#include "defs.hpp"
#include "rad_grid.hpp"
#include "grid.hpp"
#include "options.hpp"
#include "node_server.hpp"

extern options opts;

#ifdef RADIATION

//typedef node_server::send_rad_boundary_action send_rad_boundary_action_type;
//HPX_REGISTER_ACTION (send_rad_boundary_action_type);

typedef node_server::send_rad_bnd_boundary_action send_rad_bnd_boundary_action_type;
HPX_REGISTER_ACTION (send_rad_bnd_boundary_action_type);

//hpx::future<void> node_client::send_rad_boundary(std::vector<rad_type>&& data, const geo::octant& oct, const geo::dimension& dim) const {
//	return hpx::async<typename node_server::send_rad_boundary_action>(get_gid(), std::move(data), oct, dim);
//}

hpx::future<void> node_client::send_rad_boundary(std::vector<rad_type>&& data, const geo::face& f) const {
	return hpx::async<typename node_server::send_rad_bnd_boundary_action>(get_gid(), std::move(data), f);
}

void node_server::recv_rad_bnd_boundary(std::vector<rad_type>&& bdata, const geo::face& f) {
	sibling_rad_bnd_channels[f]->set_value(std::move(bdata));
}

//void node_server::recv_rad_boundary(std::vector<rad_type>&& bdata, const geo::octant& oct, const geo::dimension& dim) {
//	sibling_rad_channels[oct][dim]->set_value(std::move(bdata));
//}

//typedef node_server::send_rad_children_action send_rad_children_action_type;
//HPX_REGISTER_ACTION (send_rad_children_action_type);

//hpx::future<void> node_client::send_rad_children(std::vector<rad_type>&& data, const geo::octant& oct, const geo::octant& ioct) const {
//	return hpx::async<typename node_server::send_rad_children_action>(get_gid(), std::move(data), oct, ioct);
//}

real rad_grid::rad_imp_comoving(real& E, real& e, real rho, real dt) {
//	printf("%e %e\n", e, E);
	const integer max_iter = 100;
	const real E0 = E;
	const real e0 = e;
	real E1 = E0;
	real f = 1.0;
	real dE;
	integer i = 0;
	constexpr static auto c = LIGHTSPEED;
	do {
//		printf( "Error in rad_imp_comoving\n");
		const real dkp_de = dkappa_p_de(rho, e);
		const real dB_de = dB_p_de(rho, e);
		const real kp = kappa_p(rho, e);
		const real B = B_p(rho, e);
		f = (E - E0) + dt * c * kp * (E - 4.0 * M_PI / c * B);
		const real dfdE = 1.0 + dt * c * kp;
		const real dfde = dt * c * dkp_de * (E - 4.0 * M_PI / c * B) - dt * kp * 4.0 * M_PI * dB_de / c;
		dE = -f / (dfdE - dfde);
		const real w = 0.5;
		const real emin = std::min(e, E);
		dE = w * (E + dE) + (1.0 - w) * E1 - E;
		if (dE > 0.5 * emin) {
			dE = +0.5 * emin;
		} else if (dE < -0.5 * emin) {
			dE = -0.5 * emin;
		}
		E += dE;
		e -= dE;
		E1 = E;
		++i;
		if (i > max_iter) {
			printf("%i %e %e %e %e %e\n", int(i), E, e, E0, e0, f / std::max(E0, e0));
			abort();
		}
	} while (std::abs(f / std::max(E0, e0)) > 1.0e-9);
	return (E - E0)/dt;
}

void rad_grid::rad_imp(std::vector<real>& egas, std::vector<real>& tau, std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
		const std::vector<real>& rho, real dt) {
	const integer d = H_BW - R_BW;
	const real c = LIGHTSPEED;
	const real cinv = 1.0 / LIGHTSPEED;
	const real fgamma = grid::get_fgamma();
	for (integer i = R_BW; i != R_NX - R_BW; ++i) {
		for (integer j = R_BW; j != R_NX - R_BW; ++j) {
			for (integer k = R_BW; k != R_NX - R_BW; ++k) {
				const integer iiih = hindex(i + d, j + d, k + d);
				const integer iiir = rindex(i, j, k);

				real vx = sx[iiih] / rho[iiih];
				real vy = sy[iiih] / rho[iiih];
				real vz = sz[iiih] / rho[iiih];

				/* Compute e0 from dual energy formalism */
				real e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * rho[iiih];
				e0 -= 0.5 * vy * vy * rho[iiih];
				e0 -= 0.5 * vz * vz * rho[iiih];
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}

				/* Compute transformation parameters */
				const real v2 = vx * vx + vy * vy + vz * vz;
				const real kr = kappa_R(rho[iiih], e0);
				real coeff = c * kr * dt * 0.5;
				coeff = coeff * coeff / (1.0 + coeff);
				vx += coeff * U[fx_i][iiir] / rho[iiih];
				vy += coeff * U[fy_i][iiir] / rho[iiih];
				vz += coeff * U[fz_i][iiir] / rho[iiih];
				const real beta_x = vx * cinv;
				const real beta_y = vy * cinv;
				const real beta_z = vz * cinv;
				const real beta_2 = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z;

				/* Transform E and F from lab frame to comoving frame */

				real E0 = U[er_i][iiir];// * (1.0 + beta_2);
				real tmp1 = 0.0, tmp2 = 0.0;
				tmp1 -= 2.0 * beta_x * cinv * U[fx_i][iiir];
				tmp1 -= 2.0 * beta_y * cinv * U[fy_i][iiir];
				tmp1 -= 2.0 * beta_z * cinv * U[fz_i][iiir];
				const auto P = compute_p(U[er_i][iiir], U[fx_i][iiir], U[fy_i][iiir], U[fz_i][iiir]);
				tmp2 += 2.0 * beta_x * P[XDIM][XDIM] * beta_x;
				tmp2 += 2.0 * beta_y * P[YDIM][YDIM] * beta_y;
				tmp2 += 2.0 * beta_z * P[ZDIM][ZDIM] * beta_z;
				tmp2 += 4.0 * beta_x * P[XDIM][YDIM] * beta_y;
				tmp2 += 4.0 * beta_x * P[XDIM][ZDIM] * beta_z;
				tmp2 += 4.0 * beta_y * P[YDIM][ZDIM] * beta_z;
				E0 += tmp1 + tmp2;
				E0 = std::max(E0, 0.0);
			///	if( U[er_i][iiir] < 0.0 ) {
			//		printf( "1 %e %e %e %e %e %e %e %e \n", E0,  U[er_i][iiir] , tmp1, tmp2, beta_2, fEdd_xx[iiir], fEdd_yy[iiir], fEdd_zz[iiir]);
			//		abort();
			//	}
				real Fx0, Fy0, Fz0;
				Fx0 = U[fx_i][iiir];// * (1.0 + 2.0 * beta_2);
				Fy0 = U[fy_i][iiir]; // * (1.0 + 2.0 * beta_2);
				Fz0 = U[fz_i][iiir]; // * (1.0 + 2.0 * beta_2);

				Fx0 -= beta_x * c * U[er_i][iiir];
				Fx0 -= beta_y * c * P[XDIM][YDIM];
				Fx0 -= beta_z * c * P[XDIM][ZDIM];
				Fx0 -= beta_x * c * P[XDIM][XDIM];

				Fy0 -= beta_y * c * U[er_i][iiir];
				Fy0 -= beta_x * c * P[YDIM][XDIM];
				Fy0 -= beta_y * c * P[YDIM][YDIM];
				Fy0 -= beta_z * c * P[YDIM][ZDIM];

				Fz0 -= beta_z * c * U[er_i][iiir];
				Fz0 -= beta_x * c * P[ZDIM][XDIM];
				Fz0 -= beta_y * c * P[ZDIM][YDIM];
				Fz0 -= beta_z * c * P[ZDIM][ZDIM];

				real En, Fxn, Fyn, Fzn, en;
				real Enp1, Fxnp1, Fynp1, Fznp1, enp1;
				real E1, Fx1, Fy1, Fz1, e1;
				real E2, Fx2, Fy2, Fz2, e2;
				real de1, dE1, dFx1, dFy1, dFz1;
				real de2, dE2, dFx2, dFy2, dFz2;
				en = e0;
				En = E0;
				Fxn = Fx0;
				Fyn = Fy0;
				Fzn = Fz0;

				const real gam = 1.0 - std::sqrt(2.0) / 2.0;

				real this_E = En;
				real this_e = en;
				dE1 = rad_imp_comoving(this_E, this_e, rho[iiih], gam * dt);
				de1 = -dE1;
				real kR = kappa_R(rho[iiih], this_e);
				dFx1 = -(Fx0 * c * kR) / (1.0 + c * gam * dt * kR);
				dFy1 = -(Fy0 * c * kR) / (1.0 + c * gam * dt * kR);
				dFz1 = -(Fz0 * c * kR) / (1.0 + c * gam * dt * kR);

				this_E = En + (1.0 - 2.0 * gam) * dE1 * dt;
				this_e = en + (1.0 - 2.0 * gam) * de1 * dt;

				dE2 = rad_imp_comoving(this_E, this_e, rho[iiih], gam * dt);
				de2 = -dE1;
				kR = kappa_R(rho[iiih], this_e);
				dFx2 = -(Fx0 + (1.0 - 2.0 * gam) * dFx1 * dt) * (c * kR) / (1.0 + c * dt * kR);
				dFy2 = -(Fy0 + (1.0 - 2.0 * gam) * dFy1 * dt) * (c * kR) / (1.0 + c * dt * kR);
				dFz2 = -(Fz0 + (1.0 - 2.0 * gam) * dFz1 * dt) * (c * kR) / (1.0 + c * dt * kR);

				const real dE0_dt = (dE1 + dE2) * 0.5;
				const real de0_dt = (de1 + de2) * 0.5;
				const real dFx0_dt = (dFx1 + dFx2) * 0.5;
				const real dFy0_dt = (dFy1 + dFy2) * 0.5;
				const real dFz0_dt = (dFz1 + dFz2) * 0.5;
				e1 = e0 + de0_dt * dt;

				/* Transform time derivatives to lab frame */
		//		const real b2o2p1 = (1.0 + 0.5 * beta_2);
				const real b2o2p1 = 1.0;
				const real dE_dt = b2o2p1 * dE0_dt + (beta_x * dFx0_dt + beta_y * dFy0_dt + beta_z * dFz0_dt) * cinv;
				const real dFx_dt = b2o2p1 * dFx0_dt + beta_x * c * dE0_dt;
				const real dFy_dt = b2o2p1 * dFy0_dt + beta_y * c * dE0_dt;
				const real dFz_dt = b2o2p1 * dFz0_dt + beta_z * c * dE0_dt;

				/* Accumulate derivatives */
				U[er_i][iiir] += dE_dt * dt;
				U[fx_i][iiir] += dFx_dt * dt;
				U[fy_i][iiir] += dFy_dt * dt;
				U[fz_i][iiir] += dFz_dt * dt;

				egas[iiih] -= dE_dt * dt;
				sx[iiih] -= dFx_dt * dt * cinv * cinv;
				sy[iiih] -= dFy_dt * dt * cinv * cinv;
				sz[iiih] -= dFz_dt * dt * cinv * cinv;

				/* Find tau with dual energy formalism*/
				real e = egas[iiih];
				e -= 0.5 * sx[iiih] * sx[iiih] / rho[iiih];
				e -= 0.5 * sy[iiih] * sy[iiih] / rho[iiih];
				e -= 0.5 * sz[iiih] * sz[iiih] / rho[iiih];
				if (e < 0.1 * egas[iiih]) {
					e = e1;
				}
				if( U[er_i][iiir] < 0.0 )  {
					printf( "%e %e %e\n", E0, dE0_dt*dt, dE_dt*dt);
					abort();
				}
				tau[iiih] = std::pow(e, 1.0 / fgamma);
			//	if( U[er_i][iiir] < 0.0 ) {
			//		printf( "2 %e %e %e %e %e %e %e %e \n", E0,  U[er_i][iiir] , tmp1, tmp2, beta_2, fEdd_xx[iiir], fEdd_yy[iiir], fEdd_zz[iiir]);
			//		abort();
			//	}
			}
		}
	}
}
/*void node_server::recv_rad_children(std::vector<rad_type>&& bdata, const geo::octant& oct, const geo::octant& ioct) {
	child_rad_channels[ioct][oct]->set_value(std::move(bdata));
}*/

void rad_grid::get_output(std::array<std::vector<real>, NF + NGF + NRF + NPF>& v, integer i, integer j, integer k) const {
	const integer iii = rindex(i, j, k);
//	printf("%e\n", fEdd_xx[iii]);
//	v[NF + 0].push_back(fEdd_xx[iii]);
//	v[NF + 1].push_back(fEdd_xy[iii]);
//	v[NF + 2].push_back(fEdd_xz[iii]);
//	v[NF + 3].push_back(fEdd_yy[iii]);
//	v[NF + 4].push_back(fEdd_yz[iii]);
//	v[NF + 5].push_back(fEdd_zz[iii]);
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

real rad_grid::hydro_signal_speed(const std::vector<real>& egas, const std::vector<real>& tau, const std::vector<real>& sx, const std::vector<real>& sy,
		const std::vector<real>& sz, const std::vector<real>& rho) {
	real a = 0.0;
	const real fgamma = grid::get_fgamma();
	for (integer xi = R_BW; xi != R_NX - R_BW; ++xi) {
		for (integer yi = R_BW; yi != R_NX - R_BW; ++yi) {
			for (integer zi = R_BW; zi != R_NX - R_BW; ++zi) {
				const integer D = H_BW - R_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				real vx = sx[iiih] / rho[iiih];
				real vy = sy[iiih] / rho[iiih];
				real vz = sz[iiih] / rho[iiih];
				real e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * rho[iiih];
				e0 -= 0.5 * vy * vy * rho[iiih];
				e0 -= 0.5 * vz * vz * rho[iiih];
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}

				real this_a = (4.0 / 9.0) * U[er_i][iiir] / rho[iiih];
				this_a *= std::max(1.0 - std::exp(-kappa_R(rho[iiih], e0) * dx), 0.0);
				a = std::max(this_a, a);
			}
		}
	}
	return std::sqrt(a);
}


void node_server::compute_radiation(real dt) {
	if (my_location.level() == 0) {
		printf("Eddington\n");
	}

	rad_grid_ptr->set_dx(grid_ptr->get_dx());
	auto rgrid = rad_grid_ptr;

	const real min_dx = TWO * grid::get_scaling_factor() / real(INX << opts.max_level);
	const real c = LIGHTSPEED;
	const real max_dt = min_dx / c * 0.4 / 3.0;
	integer nsteps = std::max(int(std::ceil(dt / max_dt)), 1);

	const real this_dt = dt / real(nsteps);
	auto& egas = grid_ptr->get_field(egas_i);
	const auto& rho = grid_ptr->get_field(rho_i);
	auto& tau = grid_ptr->get_field(tau_i);
	auto& sx = grid_ptr->get_field(sx_i);
	auto& sy = grid_ptr->get_field(sy_i);
	auto& sz = grid_ptr->get_field(sz_i);
	if (my_location.level() == 0) {
		printf("Implicit 1\n");
	}
	rgrid->rad_imp(egas, tau, sx, sy, sz, rho, dt/2.0);
	if (my_location.level() == 0) {
		printf("Explicit\n");
	}
	rgrid->store();
	for (integer i = 0; i != nsteps; ++i) {
		rgrid->sanity_check();
		if (my_location.level() == 0) {
			printf("rad sub-step %i\n", int(i));
		}
		collect_radiation_boundaries().get();
		rgrid->compute_flux();
		rgrid->advance(this_dt, 1.0);
		collect_radiation_boundaries().get();
		rgrid->compute_flux();
		rgrid->advance(this_dt, 0.5);
	}
	if (my_location.level() == 0) {
		printf("Implicit 2\n");
	}
	rgrid->rad_imp(egas, tau, sx, sy, sz, rho, dt/2.0);
	collect_radiation_boundaries().get();
	if (my_location.level() == 0) {
		printf("Rad done\n");
	}
}


std::array<std::array<real,NDIM>,NDIM> rad_grid::compute_p( real E, real Fx, real Fy, real Fz) {
	constexpr real c = LIGHTSPEED;
	std::array<std::array<real,NDIM>,NDIM> P;
	const real f = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz) / (c*E);
	real nx, ny, nz;
	if (E > 0.0) {
		if (f > 0.0) {
			const real finv = 1.0 / (E * f);
			nx = Fx * finv;
			ny = Fy * finv;
			nz = Fz * finv;
		} else {
			nx = ny = nz = 0.0;
		}
		if (4.0 - 3 * f * f < 0.0) {
			printf("%e %e\n", f, 4.0 - 3 * f * f);
		}
		const real chi = (3.0 + 4.0 * f * f) / (5.0 + 2.0 * std::sqrt(4.0 - 3 * f * f));
		const real f1 = ((1.0 - chi) / 2.0);
		const real f2 = ((3.0 * chi - 1.0) / 2.0);
		P[XDIM][YDIM] = P[YDIM][XDIM] = f2 * nx * ny * E;
		P[XDIM][ZDIM] = P[ZDIM][XDIM] = f2 * nx * nz * E;
		P[ZDIM][YDIM] = P[YDIM][ZDIM] = f2 * ny * nz * E;
		P[XDIM][XDIM] = (f1 + f2 * nx * nx) * E;
		P[YDIM][YDIM] = (f1 + f2 * ny * ny) * E;
		P[ZDIM][ZDIM] = (f1 + f2 * nz * nz) * E;
	} else {
		for( integer d1 = 0; d1 != NDIM; ++d1) {
			for( integer d2 = 0; d2 != NDIM; ++d2) {
				P[d1][d2] = 0.0;
			}
		}
	}
	return P;
}
/*
void rad_grid::compute_fEdd() {
	const real c = LIGHTSPEED;
	for (integer xi = 0; xi != R_NX; ++xi) {
		for (integer yi = 0; yi != R_NX; ++yi) {
			for (integer zi = 0; zi != R_NX; ++zi) {
				const integer iii = rindex(xi, yi, zi);
				if (E[iii] > 0.0) {
					fEdd_xx[iii] /= E[iii];
					fEdd_xy[iii] /= E[iii];
					fEdd_xz[iii] /= E[iii];
					fEdd_yy[iii] /= E[iii];
					fEdd_yz[iii] /= E[iii];
					fEdd_zz[iii] /= E[iii];
				} else {
					//			printf( "%i %i %i %e\n", int(xi), int(yi), int(zi), E[iii]);
					fEdd_xx[iii] = fEdd_xy[iii] = fEdd_xz[iii] = fEdd_yy[iii] = fEdd_yz[iii] = fEdd_zz[iii] = 0.0;
				}
				const real er = U[er_i][iii];
				if (er != 0.0) {
					const real fx = U[fx_i][iii];
					const real fy = U[fy_i][iii];
					const real fz = U[fz_i][iii];
			//		printf("%e %e %e %e\n", er, fx, fy, fz);
					const real f = (std::sqrt(fx * fx + fy * fy + fz * fz)) / er;
					real nx, ny, nz;
					if (f > 0.0) {
						const real finv = 1.0 / (er * f);
						nx = fx * finv;
						ny = fy * finv;
						nz = fz * finv;
					} else {
						nx = ny = nz = 0.0;
					}
				//	printf("%e\n", f);
					fEdd_xy[iii] = nx * ny * f2;
					fEdd_xz[iii] = nx * nz * f2;
					fEdd_yz[iii] = ny * nz * f2;
					fEdd_xx[iii] = f1 + nx * nx * f2;
					fEdd_yy[iii] = f1 + ny * ny * f2;
					fEdd_zz[iii] = f1 + nz * nz * f2;
			//		if( f > 0.1 ) {
			//			printf( "%e %e %e %e %e\n", chi, f, f1, nx, f2);
			//		}
				}
				//	printf( "%e\n", fEdd_xx[iii]);
			}
		}
	}

}
*/
void rad_grid::initialize() {

/*	static std::once_flag flag;
	std::call_once(flag, [&]() {
		sphere_points = generate_sphere_points(3);
	});
*/
}

rad_grid_init::rad_grid_init() {
	rad_grid::initialize();
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
					return;
					//		printf( "%e\n", U[er_i][iiir] );
					//		abort();
				}
			}
		}
	}
}

std::size_t rad_grid::load(FILE * fp) {
//	printf( "LOADING\n");
	std::size_t cnt = 0;
	auto foo = std::fread;
	cnt += foo(&dx, sizeof(real), 1, fp) * sizeof(real);
	for (integer i = R_BW; i < R_NX - R_BW; ++i) {
		for (integer j = R_BW; j < R_NX - R_BW; ++j) {
			const integer iiir = rindex(i, j, R_BW);
			for (integer f = 0; f != NRF; ++f) {
				cnt += foo(&U[f][iiir], sizeof(real), INX, fp) * sizeof(real);
				for (integer k = R_BW; k < R_NX - R_BW; ++k) {
					const integer iiir = rindex(i, j, k);
					if (U[er_i][iiir] <= 0.0) {
						printf("!!!!!!!!!! %e %i %i %i\n", U[er_i][iiir], int(i), int(j), int(k));
					}
				}
			}
		}
	}
	return cnt;
}

std::size_t rad_grid::save(FILE * fp) const {
	std::size_t cnt = 0;
	auto foo = std::fwrite;
	cnt += foo(&dx, sizeof(real), 1, fp) * sizeof(real);
	for (integer i = R_BW; i < R_NX - R_BW; ++i) {
		for (integer j = R_BW; j < R_NX - R_BW; ++j) {
			const integer iiir = rindex(i, j, R_BW);
			for (integer f = 0; f != NRF; ++f) {
				cnt += foo(&U[f][iiir], sizeof(real), INX, fp) * sizeof(real);
			}
		}
	}
	return cnt;
}

void rad_grid::compute_flux() {
	real cx, cy, cz;
	const real c = LIGHTSPEED;
	std::vector<real> s[4];
	std::vector<real> fs[3];
	for (integer f = 0; f != NRF; ++f) {
		s[f].resize(R_N3);
	}
	for (integer d = 0; d != NDIM; ++d) {
		fs[d].resize(R_N3);
	}
//	auto& f = fEdd;
	for (integer i = 0; i != R_N3; ++i) {
		for (integer f = 0; f != NRF; ++f) {
			for (integer d = 0; d != NDIM; ++d) {
				flux[d][f][i] = 0.0;
			}
		}
	}

	const auto lambda_max = []( real mu, real er, real absf) {
		if( er > 0.0 ) {
			constexpr real c = LIGHTSPEED;
			const real f = absf / (c*er);
			const real tmp = std::sqrt(4.0-3.0*f*f);
			const real tmp2 = std::sqrt((2.0/3.0)*(4.0-3.0*f*f -tmp)+2*mu*mu*(2.0-f*f-tmp));
			return (tmp2 + std::abs(mu*f)) / tmp;
		} else {
			return 0.0;
		}
	};

	const integer D[3] = { DX, DY, DZ };
	for (integer d2 = 0; d2 != NDIM; ++d2) {
		for (integer f = 0; f != NRF; ++f) {
			for (integer i = DX; i != R_N3 - DX; ++i) {
				const real tmp0 = U[f][i];
				const real tmp1 = U[f][i + D[d2]] - tmp0;
				const real tmp2 = tmp0 - U[f][i - D[d2]];
				s[f][i] = minmod(tmp1, tmp2);
			}
		}
		for (integer i = 2 * DX; i != R_N3 - DX; ++i) {
			real f_p[3], f_m[3], absf_m = 0.0, absf_p = 0.0;
			const real er_m = U[er_i][i - D[d2]] + 0.5 * s[er_i][i - D[d2]];
			const real er_p = U[er_i][i] - 0.5 * s[er_i][i];
			for (integer d = 0; d != NDIM; ++d) {
				f_m[d] = U[fx_i + d][i - D[d2]] + 0.5 * s[fx_i + d][i - D[d2]];
				f_p[d] = U[fx_i + d][i] - 0.5 * s[fx_i + d][i];
				absf_m += f_m[d] * f_m[d];
				absf_p += f_p[d] * f_p[d];
			}
			absf_m = std::sqrt(absf_m);
			absf_p = std::sqrt(absf_p);
			const auto P_p = compute_p(er_p, f_p[0], f_p[1], f_p[2]);
			const auto P_m = compute_p(er_m, f_m[0], f_m[1], f_m[2]);
			real mu_m = 0.0;
			real mu_p = 0.0;
			if (absf_m > 0.0) {
				mu_m = f_m[d2] / absf_m;
			}
			if (absf_p > 0.0) {
				mu_p = f_p[d2] / absf_p;
			}
			const real a_m = lambda_max(mu_m, er_m, absf_m );
			const real a_p = lambda_max(mu_p, er_p, absf_p );
			const real a = std::max(a_m, a_p);
	//		real a = 1.0;
			flux[d2][er_i][i] += (f_p[d2] + f_m[d2]) * 0.5;
			for (integer d1 = 0; d1 != NDIM; ++d1) {
				flux[d2][fx_i + d1][i] += c * c * (P_p[d1][d2] + P_m[d1][d2]) * 0.5;
			}
			for (integer f = 0; f != NRF; ++f) {
				const real vp = U[f][i] - 0.5 * s[f][i];
				const real vm = U[f][i - D[d2]] + 0.5 * s[f][i - D[d2]];
				flux[d2][f][i] -= (vp - vm) * 0.5 * a;
			}
		}
	}
}

void rad_grid::advance(real dt, real beta) {
	const real l = dt / dx;
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
					iii1 = rindex(i, j, R_NX - 1 - k);
					iii0 = rindex(i, j, R_NX - 1 - R_BW);
					break;
				}
				for (integer f = 0; f != NRF; ++f) {
					U[f][iii1] = U[f][iii0];
			//		U[f][iii1] = 0.0;
				}
				//fEdd_xx[iii1] = fEdd_xx[iii0];
				//fEdd_yy[iii1] = fEdd_yy[iii0];
				//fEdd_zz[iii1] = fEdd_zz[iii0];
				//fEdd_xy[iii1] = fEdd_xy[iii0];
				//fEdd_xz[iii1] = fEdd_xz[iii0];
			//	fEdd_yz[iii1] = fEdd_yz[iii0];
			//	const real c = LIGHTSPEED;
			//	const real x = X[XDIM][iii1];
			//	const real y = X[YDIM][iii1];
			//	const real z = X[ZDIM][iii1];
			//	const real rinv = 1.0 / std::sqrt(x*x+y*y+z*z);
		//		const real nx = x / rinv;
		//		const real ny = y / rinv;
		//		const real nz = z / rinv;
				for (integer d = 0; d != NDIM; ++d) {
		//			U[fx_i][iii1] = nx * c * U[er_i][iii0];
		//			U[fy_i][iii1] = ny * c * U[er_i][iii0];
		//			U[fz_i][iii1] = nz * c * U[er_i][iii0];
				}
				switch (face) {
				case 0:
					U[fx_i][iii1] = std::min(U[fx_i][iii1], 0.0);
				//	U[fx_i][iii1] = -c * U[er_i][iii1] * std::sqrt(fEdd_xx[iii1]);
					break;
				case 1:
					U[fx_i][iii1] = std::max(U[fx_i][iii1], 0.0);
				//	U[fx_i][iii1] = +c * U[er_i][iii1] * std::sqrt(fEdd_xx[iii1]);
					break;
				case 2:
					U[fy_i][iii1] = std::min(U[fy_i][iii1], 0.0);
				//	U[fy_i][iii1] =  -c * U[er_i][iii1] * std::sqrt(fEdd_yy[iii1]);
					break;
				case 3:
					U[fy_i][iii1] = std::max(U[fy_i][iii1], 0.0);
				//	U[fy_i][iii1] = +c * U[er_i][iii1] * std::sqrt(fEdd_yy[iii1]);
					break;
				case 4:
					U[fz_i][iii1] = std::min(U[fz_i][iii1], 0.0);
			//		U[fz_i][iii1] =  -c * U[er_i][iii1] * std::sqrt(fEdd_zz[iii1]);
					break;
				case 5:
					U[fz_i][iii1] = std::max(U[fz_i][iii1], 0.0);
				//	U[fz_i][iii1] = +c * U[er_i][iii1] * std::sqrt(fEdd_zz[iii1]);
					break;
				}
			}
		}
	}
}

hpx::future<void> node_server::collect_radiation_boundaries() {

	std::vector<hpx::future<void>> futs;

	for (auto& dir : geo::direction::full_set()) {
		if (dir.is_face()) {
			if (!neighbors[dir].empty()) {
				auto bdata = rad_grid_ptr->get_boundary(dir);
				futs.push_back(neighbors[dir].send_rad_boundary(std::move(bdata), dir.to_face().flip()));
			}
		}
	}

	for (auto& dir : geo::direction::full_set()) {
		if (dir.is_face()) {
			if (!(neighbors[dir].empty())) {
				auto face = dir.to_face();
				auto tmp = sibling_rad_bnd_channels[face]->get_future().get();
				rad_grid_ptr->set_boundary(tmp, dir);
			}
		}
	}

	for (auto& face : geo::face::full_set()) {
		if (my_location.is_physical_boundary(face)) {
			rad_grid_ptr->set_physical_boundaries(face);
		}
	}

	return hpx::when_all(futs);

}

void rad_grid::initialize_erad(const std::vector<real> rho, const std::vector<real> tau) {
	const real fgamma = grid::get_fgamma();
	for (integer xi = R_BW; xi != R_NX - R_BW; ++xi) {
		for (integer yi = R_BW; yi != R_NX - R_BW; ++yi) {
			for (integer zi = R_BW; zi != R_NX - R_BW; ++zi) {
				const auto D = H_BW - R_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				U[er_i][iiir] = B_p(rho[iiih], std::pow(tau[iiih], fgamma));
				U[fx_i][iiir] = 0.0;
				U[fy_i][iiir] = 0.0;
				U[fz_i][iiir] = 0.0;
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

integer rad_grid::rindex(integer x, integer y, integer z) {
	return z + R_NX * (y + R_NX * x);
}

/*void rad_grid::compute_intensity(const geo::octant& oct) {
	const auto indexes = [](integer x, integer& inc, integer& b, integer& e) {
		if( x > 0 ) {
			inc = +1;
			b = R_BW;
			e = R_NX - R_BW;
		} else {
			inc = -1;
			b = R_NX - R_BW - 1;
			e = R_BW - 1;
		}
	};
	integer xinc, yinc, zinc, xb, yb, zb, xe, ye, ze;
	indexes(oct[XDIM], xinc, xb, xe);
	indexes(oct[YDIM], yinc, yb, ye);
	indexes(oct[ZDIM], zinc, zb, ze);
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		const auto& pt = sphere_points[spi];
		if (pt.get_octant() == oct) {
			for (integer xi = xb; xi != xe; xi += xinc) {
				for (integer yi = yb; yi != ye; yi += yinc) {
					for (integer zi = zb; zi != ze; zi += zinc) {
						const integer iii = rindex(xi, yi, zi);
						const integer iiix = iii - xinc * DX;
						const integer iiiy = iii - yinc * DY;
						const integer iiiz = iii - zinc * DZ;
						const real wx = std::abs(pt.nx);
						const real wy = std::abs(pt.ny);
						const real wz = std::abs(pt.nz);
						const real& bx = Beta_x[iii];
						const real& by = Beta_y[iii];
						const real& bz = Beta_z[iii];
				//		const real& bx = 0.0;
				//		const real& by = 0.0;
				//		const real& bz = 0.0;
						const real b2 = bx * bx + by * by + bz * bz;
						const real nuonu0 = (1.0 + pt.nx * bx + pt.ny * by + pt.nz * bz) * (1.0 + 0.5 * b2);
						const real j0 = J[iii] * nuonu0 * nuonu0 + sigma_s[iii] * U[er_i][iii] / nuonu0;
						const real k0 = (sigma_a[iii] + sigma_s[iii]) / nuonu0;
						I[spi][iii] = dx * j0 + wx * I[spi][iiix] + wy * I[spi][iiiy] + wz * I[spi][iiiz];
						I[spi][iii] /= dx * k0 + wx + wy + wz;
					}
				}
			}
		}
	}
}*/

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
	}
	/*for (integer d1 = 0; d1 != NDIM; ++d1) {
		for (integer d2 = d1; d2 != NDIM; ++d2) {
			for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
				for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
					for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
						(*fEdd[d1][d2])[rindex(i, j, k)] = data[iter];
						++iter;
					}
				}
			}
		}
	}*/PROF_END;
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
	/*for (integer d1 = 0; d1 != NDIM; ++d1) {
		for (integer d2 = d1; d2 != NDIM; ++d2) {
			auto& Ufield = (*fEdd[d1][d2]);
			for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
				for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
					for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
						data[iter] = Ufield[rindex(i, j, k)];
						++iter;
					}
				}
			}
		}
	}*/PROF_END;
	return data;
}
/*
void rad_grid::accumulate_intensity(const geo::octant& oct) {
	std::lock_guard < hpx::lcos::local::mutex > lock(Pmtx);
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		const auto& pt = sphere_points[spi];
		if (pt.get_octant() == oct) {
			for (integer xi = 0; xi != R_NX; ++xi) {
				for (integer yi = 0; yi != R_NX; ++yi) {
					for (integer zi = 0; zi != R_NX; ++zi) {
						const integer iii = rindex(xi, yi, zi);
						E[iii] += I[spi][iii] * pt.dA;
						fEdd_xx[iii] += I[spi][iii] * pt.nx * pt.nx * pt.dA;
						fEdd_xy[iii] += I[spi][iii] * pt.nx * pt.ny * pt.dA;
						fEdd_xz[iii] += I[spi][iii] * pt.nx * pt.nz * pt.dA;
						fEdd_yy[iii] += I[spi][iii] * pt.ny * pt.ny * pt.dA;
						fEdd_yz[iii] += I[spi][iii] * pt.ny * pt.nz * pt.dA;
						fEdd_zz[iii] += I[spi][iii] * pt.nz * pt.nz * pt.dA;
					}
				}
			}
		}
	}
}

void rad_grid::set_intensity(const std::vector<rad_type>& data, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
		const geo::octant& oct) {
	integer m = 0;
	const std::size_t box_size = (ub[XDIM] - lb[XDIM]) * (ub[YDIM] - lb[YDIM]) * (ub[ZDIM] - lb[ZDIM]);
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		const auto& pt = sphere_points[spi];
		if (pt.get_octant() == oct) {
			for (integer xi = lb[XDIM]; xi < ub[XDIM]; ++xi) {
				for (integer yi = lb[YDIM]; yi < ub[YDIM]; ++yi) {
					for (integer zi = lb[ZDIM]; zi < ub[ZDIM]; ++zi) {
						const integer iii = rindex(xi, yi, zi);
						I[spi][iii] = data[m];
						++m;
					}
				}
			}
		}
	}
}

void rad_grid::free_octant(const geo::octant& oct) {
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		const auto& pt = sphere_points[spi];
		if (pt.get_octant() == oct) {
			I[spi].resize(0);
			I[spi].reserve(0);
		}
	}
}

void rad_grid::alloc_octant(const geo::octant& oct) {
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		const auto& pt = sphere_points[spi];
		if (pt.get_octant() == oct) {
			I[spi].reserve(R_N3);
			I[spi].resize(R_N3);
		}
	}
}
*/

void rad_grid::set_field(rad_type v, integer f, integer i, integer j, integer k) {
	U[f][rindex(i, j, k)] = v;
}
/*
std::vector<rad_type> rad_grid::get_intensity(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant& oct) {
	std::vector<rad_type> data;
	const std::size_t box_size = (ub[XDIM] - lb[XDIM]) * (ub[YDIM] - lb[YDIM]) * (ub[ZDIM] - lb[ZDIM]);
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		const auto& pt = sphere_points[spi];
		if (pt.get_octant() == oct) {
			data.reserve(data.size() + box_size);
			for (integer xi = lb[XDIM]; xi < ub[XDIM]; ++xi) {
				for (integer yi = lb[YDIM]; yi < ub[YDIM]; ++yi) {
					for (integer zi = lb[ZDIM]; zi < ub[ZDIM]; ++zi) {
						const integer iii = rindex(xi, yi, zi);
						data.push_back(I[spi][iii]);
					}
				}
			}
		}
	}
	return data;
}

void rad_grid::set_prolong(const std::vector<real>& data, const geo::octant& ioct) {
	integer index = 0;
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		if (sphere_points[spi].get_octant() == ioct) {
			for (integer i = R_BW; i != R_NX - R_BW; ++i) {
				for (integer j = R_BW; j != R_NX - R_BW; ++j) {
					for (integer k = R_BW; k != R_NX - R_BW; ++k) {
						const integer iii = rindex(i, j, k);
						I[spi][iii] = data[index];
						++index;
					}
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant& ioct) {
	std::vector<real> data;
	integer size = NF;
	for (integer dim = 0; dim != NDIM; ++dim) {
		size *= (ub[dim] - lb[dim]);
	}
	auto lb0 = lb;
	auto ub0 = ub;
	for (integer d = 0; d != NDIM; ++d) {
		lb0[d] /= 2;
		ub0[d] /= 2;
	}

	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		if (sphere_points[spi].get_octant() == ioct) {
			for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
				for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
					for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
						const integer iii = rindex(i / 2, j / 2, k / 2);
						real value = I[spi][iii];
						data.push_back(value);
					}
				}
			}
		}
	}
	return data;
}

std::vector<real> rad_grid::get_restrict(const geo::octant& ioct) const {
	std::vector<real> data;
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		if (sphere_points[spi].get_octant() == ioct) {
			for (integer i = R_BW; i < R_NX - R_BW; i += 2) {
				for (integer j = R_BW; j < R_NX - R_BW; j += 2) {
					for (integer k = R_BW; k < R_NX - R_BW; k += 2) {
						const integer iii = rindex(i, j, k);
						real pt = ZERO;
						for (integer x = 0; x != 2; ++x) {
							for (integer y = 0; y != 2; ++y) {
								for (integer z = 0; z != 2; ++z) {
									const integer jjj = iii + x * R_NX * R_NX + y * R_NX + z;
									pt += I[spi][jjj];
								}
							}
						}
						pt /= real(NCHILD);
						data.push_back(pt);
					}
				}
			}
		}
	}
	return data;
}

void rad_grid::set_restrict(const std::vector<real>& data, const geo::octant& octant, const geo::octant& ioct) {
	integer index = 0;
	const integer i0 = octant.get_side(XDIM) * (INX / 2);
	const integer j0 = octant.get_side(YDIM) * (INX / 2);
	const integer k0 = octant.get_side(ZDIM) * (INX / 2);
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		if (sphere_points[spi].get_octant() == ioct) {
			for (integer i = R_BW; i != R_NX / 2; ++i) {
				for (integer j = R_BW; j != R_NX / 2; ++j) {
					for (integer k = R_BW; k != R_NX / 2; ++k) {
						const integer iii = rindex(i + i0, j + j0, k + k0);
						I[spi][iii] = data[index];
						++index;
						if (index > data.size()) {
							printf("rad_grid::set_restrict error %i %i\n", int(index), int(data.size()));
						}
					}
				}
			}
		}
	}

}
;
*/
#endif

