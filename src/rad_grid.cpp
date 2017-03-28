#include "defs.hpp"
#include "rad_grid.hpp"
#include "grid.hpp"
#include "options.hpp"
#include "node_server.hpp"
#include "physcon.hpp"

#include <iostream>

extern options opts;

#ifdef RADIATION

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

hpx::future<void> node_client::send_rad_boundary(std::vector<rad_type>&& data, const geo::direction& dir) const {
	return hpx::async<typename node_server::send_rad_boundary_action>(get_gid(), std::move(data), dir);
}

void node_server::recv_rad_boundary(std::vector<rad_type>&& bdata, const geo::direction& dir) {
	sibling_rad_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_rad_channels[dir].set_value(std::move(tmp));
}

typedef node_server::send_rad_children_action send_rad_children_action_type;
HPX_REGISTER_ACTION (send_rad_children_action_type);

void node_server::recv_rad_children(std::vector<real>&& data, const geo::octant& ci) {
	child_rad_channels[ci].set_value(std::move(data));
}

hpx::future<void> node_client::send_rad_children(std::vector<real>&& data, const geo::octant& ci) const {
	return hpx::async<typename node_server::send_rad_children_action>(get_unmanaged_gid(), std::move(data), ci);
}

real rad_grid::rad_imp_comoving(real& E, real& e, real rho, real mmw, real dt) {
//	const real gam = 1.0 - std::sqrt(2.0)/2.0;
//	printf("%e %e\n", e, E);
	const integer max_iter = 100;
	const real E0 = E;
	const real e0 = e;
	real E1 = E0;
	real e1 = e0;
	real f = 1.0;
	real dE;
	integer i = 0;
	const auto clight = physcon.c;
	real err;
	/*if( E < 0.0  ){
	 //		printf( "EFEF\n");
	 printf( "Error in %s on line %i\n", __FILE__, __LINE__);
	 abort();
	 //		abort();
	 * c=
	 }*/
	const real con1 = 4.0 * M_PI / clight;
	const real con2 = dt * clight;
	const real con3 = 4.0 * M_PI * dt;
	do {
		real dkp_de, dB_de, kp, B;
		rad_coupling_vars(rho, e, mmw, B, kp, dkp_de, dB_de);
		f = (E - E0) + con2 * kp * (E - con1 * B);
		const real den = 1.0 + kp * (con3 * dB_de + con2) - con2 * dkp_de * (E - con1 * B);
		dE = -f / den;
		const real dE0 = dE;
		err = std::abs(dE / E);
		dE = std::min(std::max(dE, -E * 0.9), +e * 0.9);
		E += dE;
		e -= dE;
		E1 = E;
		e1 = e;
		++i;
		if (i > max_iter || E < 0.0) {
			printf("%i E0 = %e E = %e e0 = %e e = %e dE = %e\n", int(i),  E0, E, e0, e, dE);
			if (i > max_iter + 100)
				abort();
		}
	} while (std::abs(err) > 1.0e-13);
	dE = E - E0;
	const real gam = 1.0 - 1.0 / std::sqrt(2.0);
	if (dE > 0.0) {
		dE = std::min(dE, e0 * gam);
	} else {
		dE = std::max(dE, -E0 * gam);
	}
	return dE / dt;
}

void rad_grid::rad_imp(std::vector<real>& egas, std::vector<real>& tau, std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
		const std::vector<real>& rho, real dt) {
	const integer d = H_BW - R_BW;
	const real clight = physcon.c;
	const real clightinv = 1.0 / clight;
	const real fgamma = grid::get_fgamma();
	for (integer i = R_BW; i != R_NX - R_BW; ++i) {
		for (integer j = R_BW; j != R_NX - R_BW; ++j) {
			for (integer k = R_BW; k != R_NX - R_BW; ++k) {
				const integer iiih = hindex(i + d, j + d, k + d);
				const integer iiir = rindex(i, j, k);
				const real den = rho[iiih];
				real vx = sx[iiih] / den;
				real vy = sy[iiih] / den;
				real vz = sz[iiih] / den;

				/* Compute e0 from dual energy formalism */
				real e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * den;
				e0 -= 0.5 * vy * vy * den;
				e0 -= 0.5 * vz * vz * den;
				if (opts.eos == WD) {
					e0 -= ztwd_energy(den);
				}
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}

				/* Compute transformation parameters */
				const real v2 = vx * vx + vy * vy + vz * vz;
				const real kr = kappa_R(den, e0, mmw[iiir]);
				real coeff = kr * dt * clight;
				coeff = coeff / (1.0 + coeff);
				vx += 0.5 * coeff * U[fx_i][iiir] / den * clightinv * clightinv;
				vy += 0.5 * coeff * U[fy_i][iiir] / den * clightinv * clightinv;
				vz += 0.5 * coeff * U[fz_i][iiir] / den * clightinv * clightinv;
				real beta_x = vx * clightinv;
				real beta_y = vy * clightinv;
				real beta_z = vz * clightinv;
				real beta_2 = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z;
				//real beta = std::sqrt(beta_2);
				//	const real f = 1.0 / std::sqrt(1.0 + beta*beta);
				//	beta_x *= f;
				//	beta_y *= f;
				//	beta_z *= f;
				//	beta_2 *= f * f;
				if (beta_x > 1.0) {
					printf("BETA = %e %e %e %e %e\n", std::sqrt(beta_2), coeff, kr, den, U[fx_i][iiir]);
				}
				//	beta_x = beta_y = beta_z = beta_2 = 0.0;
				/* Transform E and F from lab frame to comoving frame */

				real E0 = U[er_i][iiir]; // * (1.0 + beta_2);
				real tmp1 = 0.0, tmp2 = 0.0;
				tmp1 -= 2.0 * beta_x * clightinv * U[fx_i][iiir];
				tmp1 -= 2.0 * beta_y * clightinv * U[fy_i][iiir];
				tmp1 -= 2.0 * beta_z * clightinv * U[fz_i][iiir];
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
				Fx0 = U[fx_i][iiir];			// * (1.0 + 2.0 * beta_2);
				Fy0 = U[fy_i][iiir]; // * (1.0 + 2.0 * beta_2);
				Fz0 = U[fz_i][iiir]; // * (1.0 + 2.0 * beta_2);

				Fx0 -= beta_x * clight * U[er_i][iiir];
				Fx0 -= beta_y * clight * P[XDIM][YDIM];
				Fx0 -= beta_z * clight * P[XDIM][ZDIM];
				Fx0 -= beta_x * clight * P[XDIM][XDIM];

				Fy0 -= beta_y * clight * U[er_i][iiir];
				Fy0 -= beta_x * clight * P[YDIM][XDIM];
				Fy0 -= beta_y * clight * P[YDIM][YDIM];
				Fy0 -= beta_z * clight * P[YDIM][ZDIM];

				Fz0 -= beta_z * clight * U[er_i][iiir];
				Fz0 -= beta_x * clight * P[ZDIM][XDIM];
				Fz0 -= beta_y * clight * P[ZDIM][YDIM];
				Fz0 -= beta_z * clight * P[ZDIM][ZDIM];

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
				dE1 = rad_imp_comoving(this_E, this_e, den, mmw[iiir], gam * dt);
				de1 = -dE1;
				real kR = kappa_R(den, this_e, mmw[iiir]);
				dFx1 = -(Fx0 * clight * kR) / (1.0 + clight * gam * dt * kR);
				dFy1 = -(Fy0 * clight * kR) / (1.0 + clight * gam * dt * kR);
				dFz1 = -(Fz0 * clight * kR) / (1.0 + clight * gam * dt * kR);

				this_E = En + (1.0 - 2.0 * gam) * dE1 * dt;
				this_e = en + (1.0 - 2.0 * gam) * de1 * dt;
				//if( this_E < 0.0 ) {
				//	printf( "this_E %e %e %e\n", this_E, En, (1.0 - 2.0 * gam) * dE1 * dt);
				//	}
				dE2 = rad_imp_comoving(this_E, this_e, den, mmw[iiir], gam * dt);
				de2 = -dE2;
				kR = kappa_R(den, this_e, mmw[iiir]);
				dFx2 = -(Fx0 + (1.0 - 2.0 * gam) * dFx1 * dt) * (clight * kR) / (1.0 + clight * dt * gam * kR);
				dFy2 = -(Fy0 + (1.0 - 2.0 * gam) * dFy1 * dt) * (clight * kR) / (1.0 + clight * dt * gam * kR);
				dFz2 = -(Fz0 + (1.0 - 2.0 * gam) * dFz1 * dt) * (clight * kR) / (1.0 + clight * dt * gam * kR);

				const real dE0_dt = (dE1 + dE2) * 0.5;
				const real de0_dt = (de1 + de2) * 0.5;
				const real dFx0_dt = (dFx1 + dFx2) * 0.5;
				const real dFy0_dt = (dFy1 + dFy2) * 0.5;
				const real dFz0_dt = (dFz1 + dFz2) * 0.5;
				e1 = e0 + de0_dt * dt;
				if (e1 < 0.0) {
					abort_error()
					;
				}

				/* Transform time derivatives to lab frame */
				const real dE_dt = dE0_dt + (beta_x * dFx0_dt + beta_y * dFy0_dt + beta_z * dFz0_dt) * clightinv;
				const real dFx_dt = dFx0_dt;				// + beta_x * clight * dE0_dt;
				const real dFy_dt = dFy0_dt;				// + beta_y * clight * dE0_dt;
				const real dFz_dt = dFz0_dt;				// + beta_z * clight * dE0_dt;

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
				e -= 0.5 * sx[iiih] * sx[iiih] / den;
				e -= 0.5 * sy[iiih] * sy[iiih] / den;
				e -= 0.5 * sz[iiih] * sz[iiih] / den;
				if (opts.eos == WD) {
					e -= ztwd_energy(den);
				}
				if (e < 0.1 * egas[iiih]) {
					e = e1;
				}
				if (U[er_i][iiir] < 0.0) {
					printf("%e %e %e %e %e %e  %e !!!!!4!!!!!\n", U[er_i][iiir], E0, dE0_dt * dt, dE_dt * dt, dE1 * dt, dE2 * dt, e);
					abort();
				}
				tau[iiih] = std::pow(e, 1.0 / fgamma);
				if (tau[iiih] < 0.0) {
					abort_error()
					;
				}
				if (U[er_i][iiir] < 0.0) {
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
				if (opts.eos == WD) {
					e0 -= ztwd_energy(rho[iiih]);
				}
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}

				real this_a = (4.0 / 9.0) * U[er_i][iiir] / rho[iiih];
				printf( "%e %e %e %e\n",rho[iiih], e0, mmw[iiir],dx );
				const real cons = kappa_R(rho[iiih], e0, mmw[iiir]) * dx;
				if( cons < 32.0 ) {
					this_a *= std::max(1.0 - std::exp(-cons), 0.0);
				}
				a = std::max(this_a, a);
			}
		}
	}
	return std::sqrt(a);
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
	if (my_location.level() == 0) {
		printf("Eddington\n");
	}

	rad_grid_ptr->set_dx(grid_ptr->get_dx());
	auto rgrid = rad_grid_ptr;
	rad_grid_ptr->compute_mmw(grid_ptr->U);
	const real min_dx = TWO * grid::get_scaling_factor() / real(INX << opts.max_level);
	const real clight = physcon.c;
	const real max_dt = min_dx / clight * std::sqrt(3) / std::sqrt(2) / 6.0;
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
	rgrid->rad_imp(egas, tau, sx, sy, sz, rho, dt / 2.0);
	if (my_location.level() == 0) {
		printf("Explicit\n");
	}
	rgrid->store();
	for (integer i = 0; i != nsteps; ++i) {
		rgrid->sanity_check();
		if (my_location.level() == 0) {
			printf("rad sub-step %i of %i\n", int(i), int(nsteps));
		}
		all_rad_bounds();
		rgrid->compute_flux();
		exchange_rad_flux_corrections().get();
		rgrid->advance(this_dt, 1.0);
		all_rad_bounds();
		rgrid->compute_flux();
		exchange_rad_flux_corrections().get();
		rgrid->advance(this_dt, 0.5);
	}
	if (my_location.level() == 0) {
		printf("Implicit 2\n");
	}
	rgrid->sanity_check();
	rgrid->rad_imp(egas, tau, sx, sy, sz, rho, dt / 2.0);
	all_rad_bounds();
	if (my_location.level() == 0) {
		printf("Rad done\n");
	}
}

std::array<std::array<real, NDIM>, NDIM> rad_grid::compute_p(real E, real Fx, real Fy, real Fz) {
	const real clight = physcon.c;
	std::array<std::array<real, NDIM>, NDIM> P;
	real f = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz) / (clight * E);
	real nx, ny, nz;
	if (E > 0.0) {
		if (f > 0.0) {
			const real finv = 1.0 / (clight * E * f);
			nx = Fx * finv;
			ny = Fy * finv;
			nz = Fz * finv;
//			if(  std::abs(nx*nx+ny*ny+nz*nz - 1.0) > 1.0e-10)
//			printf( "%e\n", nx*nx+ny*ny+nz*nz);
		} else {
			nx = ny = nz = 0.0;
		}
		f = std::min(f, 1.0);
		if (4.0 - 3.0 * f * f < 0.0) {
			printf("%e %e\n", f, 4.0 - 3 * f * f);
			abort();
		}
		const real chi = (3.0 + 4.0 * f * f) / (5.0 + 2.0 * std::sqrt(4.0 - 3.0 * f * f));
		const real f1 = ((1.0 - chi) / 2.0);
		const real f2 = ((3.0 * chi - 1.0) / 2.0);
		P[XDIM][YDIM] = P[YDIM][XDIM] = f2 * nx * ny * E;
		P[XDIM][ZDIM] = P[ZDIM][XDIM] = f2 * nx * nz * E;
		P[ZDIM][YDIM] = P[YDIM][ZDIM] = f2 * ny * nz * E;
		P[XDIM][XDIM] = (f1 + f2 * nx * nx) * E;
		P[YDIM][YDIM] = (f1 + f2 * ny * ny) * E;
		P[ZDIM][ZDIM] = (f1 + f2 * nz * nz) * E;
	} else {
		for (integer d1 = 0; d1 != NDIM; ++d1) {
			for (integer d2 = 0; d2 != NDIM; ++d2) {
				P[d1][d2] = 0.0;
			}
		}
	}
	return P;
}
void rad_grid::initialize() {
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
					printf("%e\n", U[er_i][iiir]);
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

void rad_grid::compute_flux() {
	real cx, cy, cz;
	const real clight = physcon.c;
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
			const real clight = physcon.c;
			real f = absf / (clight*er);
			//		if( f > 1.0 ) {
			//			printf( "f > 1.0 ---- %e %e %e\n", f, er, absf);
			//	abort();
			//	}
			f = std::min(f,1.0);
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
				real vp, v0, vm;
				vp = U[f][i + D[d2]];
				v0 = U[f][i];
				vm = U[f][i - D[d2]];
				if (f != 0) {
					vp /= U[er_i][i + D[d2]];
					v0 /= U[er_i][i];
					vm /= U[er_i][i - D[d2]];
				}
				s[f][i] = minmod(vp - v0, v0 - vm);
			}
		}
		for (integer l = R_BW; l != R_NX - R_BW + (d2 == XDIM ? 1 : 0); ++l) {
			for (integer j = R_BW; j != R_NX - R_BW + (d2 == YDIM ? 1 : 0); ++j) {
				for (integer k = R_BW; k != R_NX - R_BW + (d2 == ZDIM ? 1 : 0); ++k) {
					integer i = rindex(l, j, k);
					real f_p[3], f_m[3], absf_m = 0.0, absf_p = 0.0;
					const real er_m = U[er_i][i - D[d2]] + 0.5 * s[er_i][i - D[d2]];
					const real er_p = U[er_i][i] - 0.5 * s[er_i][i];
					for (integer d = 0; d != NDIM; ++d) {
						f_m[d] = U[fx_i + d][i - D[d2]] / U[er_i][i - D[d2]] + 0.5 * s[fx_i + d][i - D[d2]];
						f_p[d] = U[fx_i + d][i] / U[er_i][i] - 0.5 * s[fx_i + d][i];
						f_m[d] *= er_m;
						f_p[d] *= er_p;
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
					const real a_m = lambda_max(mu_m, er_m, absf_m);
					const real a_p = lambda_max(mu_p, er_p, absf_p);
					const real a = std::max(a_m, a_p) * clight;
					//		real a = 1.0;
					//				const real tmp2 = std::abs((er_p * mu_p + er_m * mu_m) * 0.5 * clight);
					//		flux[d2][er_i][i] += std::max(std::min((f_p[d2] + f_m[d2]) * 0.5, +tmp2), -tmp2);
					flux[d2][er_i][i] = (f_p[d2] + f_m[d2]) * 0.5 - (er_p - er_m) * 0.5 * a;
					for (integer d1 = 0; d1 != NDIM; ++d1) {
						flux[d2][fx_i + d1][i] = clight * clight * (P_p[d1][d2] + P_m[d1][d2]) * 0.5 - (f_p[d1] - f_m[d1]) * 0.5 * a;
					}
					//for (integer f = 0; f != NRF; ++f) {
					//	const real vp = U[f][i] - 0.5 * s[f][i];
					//	const real vm = U[f][i - D[d2]] + 0.5 * s[f][i - D[d2]];
					//	flux[d2][f][i] -= (vp - vm) * 0.5 * a;
					//	}
				}
			}
		}
	}
}

void rad_grid::change_units(real m, real l, real t, real k) {
	const real l2 = l * l;
	const real t2 = t * t;
	const real t2inv = 1.0 / t2;
	const real tinv = 1.0 / t;
	const real l3 = l2 * l;
	const real l3inv = 1.0 / l3;
	for (integer i = 0; i != R_N3; ++i) {
		U[er_i][i] *= (m * l2 * t2inv) * l3inv;
		U[fx_i][i] *= tinv * (m * t2inv);
		U[fy_i][i] *= tinv * (m * t2inv);
		U[fz_i][i] *= tinv * (m * t2inv);
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
			if (this->nieces[f].size()) {
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
											lb[XDIM] = quadrant.get_side(0) * (INX / 2) + R_BW;
											lb[YDIM] = quadrant.get_side(1) * (INX / 2) + R_BW;
											lb[ZDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + R_BW;
											ub[XDIM] = lb[XDIM] + (INX / 2);
											ub[YDIM] = lb[YDIM] + (INX / 2);
											ub[ZDIM] = lb[ZDIM] + 1;
											break;
										}
										rad_grid_ptr->set_flux_restrict(fdata.get(), lb, ub, face_dim);
									}, "node_server::exchange_rad_flux_corrections::set_flux_restrict"
							));
				}
			}
		}
		return hpx::when_all(std::move(futs));
	}, "node_server::set_rad_flux_restrict"));
}

void rad_grid::set_flux_restrict(const std::vector<rad_type>& data, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
		const geo::dimension& dim) {
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

std::vector<rad_type> rad_grid::get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::dimension& dim) const {
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
	exchange_interlevel_rad_data();
	collect_radiation_bounds();
	send_rad_amr_bounds();
	// f.get();
}

hpx::future<void> node_server::exchange_interlevel_rad_data() {

	hpx::future<void> f = hpx::make_ready_future();
	integer ci = my_location.get_child_index();
//   printf( "--------------------%i %i %i\n", int(my_location.level()), int(ci), is_refined ? 1 : 0);

	if (is_refined) {
		//     std::vector<real> outflow(NRF, ZERO);
		for (auto const& ci : geo::octant::full_set()) {
			// 	printf( "%i\n", int(ci));
			auto data = child_rad_channels[ci].get_future().get();
			rad_grid_ptr->set_restrict(data, ci);
			// integer fi = 0;
			//  for (auto i = data.end() - NRF; i != data.end(); ++i) {
			//  outflow[fi] += *i;
			//       ++fi;
			// }
		}
		//rad_grid_ptr->set_outflows(std::move(outflow));
	}
	if (my_location.level() > 0) {
		auto data = rad_grid_ptr->get_restrict();
		f = parent.send_rad_children(std::move(data), ci);
	}
	return std::move(f);
}

void node_server::collect_radiation_bounds() {
	for (auto const& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			auto bdata = rad_grid_ptr->get_boundary(dir);
			neighbors[dir].send_rad_boundary(std::move(bdata), dir.flip());
		}
	}

	std::array<hpx::future<void>, geo::direction::count()> results;
	integer index = 0;
	for (auto const& dir : geo::direction::full_set()) {
		if (!(neighbors[dir].empty() && my_location.level() == 0)) {
			results[index++] = sibling_rad_channels[dir].get_future().then(hpx::util::annotated_function([this](hpx::future<sibling_rad_type> && f) -> void
			{
				auto&& tmp = f.get();
				rad_grid_ptr->set_boundary(tmp.data, tmp.direction );
			}, "node_server::collect_rad_bounds::set_rad_boundary"));
		}
	}
	wait_all_and_propagate_exceptions(std::move(results));

	for (auto& face : geo::face::full_set()) {
		if (my_location.is_physical_boundary(face)) {
			rad_grid_ptr->set_physical_boundaries(face);
		}
	}
}

void rad_grid::initialize_erad(const std::vector<real> rho, const std::vector<real> tau) {
	const real fgamma = grid::get_fgamma();
	for (integer xi = R_BW - 1; xi != R_NX - R_BW + 1; ++xi) {
		for (integer yi = R_BW - 1; yi != R_NX - R_BW + 1; ++yi) {
			for (integer zi = R_BW - 1; zi != R_NX - R_BW + 1; ++zi) {
				const auto D = H_BW - R_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);

				//			/************ CHECK THIS ********************/
				const real ei = std::pow(tau[iiih], fgamma);
				U[er_i][iiir] = B_p(rho[iiih], ei, mmw[iiir]) * (4.0 * M_PI / physcon.c);
				//	const real trad = std::pow(U[er_i][iiir] / (4.0*physcon.sigma/physcon.c), 0.25);
				//	if( rho[iiih] > 1.0e+4) {
				//		printf( "%e %e\n", physcon.sigma, physcon.c);
				//		printf( "%e %e %e %e %e\n", temperature(rho[iiih], ei, mmw[iiir]), trad, trad / temperature(rho[iiih], ei, mmw[iiir]), ei, U[er_i][iiir]);
				//	}
				//	printf( "%e %e %e\n", U[er_i][iiir], tau[iiih], mmw[iiir]);
			}
		}
	}
	for (integer xi = R_BW; xi != R_NX - R_BW; ++xi) {
		for (integer yi = R_BW; yi != R_NX - R_BW; ++yi) {
			for (integer zi = R_BW; zi != R_NX - R_BW; ++zi) {
				const auto D = H_BW - R_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				U[fx_i][iiir] = U[fy_i][iiir] = U[fz_i][iiir] = 0.0;
				/*	const real e = std::pow(tau[iiih], fgamma);
				 const real kR = kappa_R(rho[iiih], e, mmw[iiir]);
				 const real dEdx = (U[er_i][iiir + DX] - U[er_i][iiir - DX]) / (2.0 * dx);
				 const real dEdy = (U[er_i][iiir + DY] - U[er_i][iiir - DY]) / (2.0 * dx);
				 const real dEdz = (U[er_i][iiir + DZ] - U[er_i][iiir - DZ]) / (2.0 * dx);
				 //printf( "%e %e %e %e\n", kR, U[er_i][iiir + DX], U[er_i][iiir - DX], dx);
				 real Fx = -physcon.c * dEdx / kR;
				 real Fy = -physcon.c * dEdy / kR;
				 real Fz = -physcon.c * dEdz / kR;
				 real F = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
				 if (F > 0.0) {
				 real nx = Fx / F;
				 real ny = Fy / F;
				 real nz = Fz / F;
				 F = std::min(F, physcon.c * U[er_i][iiir]);
				 U[fx_i][iiir] = nx * F;
				 U[fy_i][iiir] = ny * F;
				 U[fz_i][iiir] = nz * F;
				 } else {
				 U[fx_i][iiir] = U[fy_i][iiir] = U[fz_i][iiir] = 0.0;
				 }*/
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
					if (index > data.size()) {
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
					const integer width = R_BW;
					get_boundary_size(lb, ub, dir, OUTER, INX, R_BW);
					for (integer dim = 0; dim != NDIM; ++dim) {
						lb[dim] = ((lb[dim] - R_BW)) + 2 * R_BW + ci.get_side(dim) * (INX);
						ub[dim] = ((ub[dim] - R_BW)) + 2 * R_BW + ci.get_side(dim) * (INX);
					}
					data = rad_grid_ptr->get_prolong(lb, ub);
					children[ci].send_rad_boundary(std::move(data), dir);
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
#endif

