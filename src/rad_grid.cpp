#include "defs.hpp"
#include "rad_grid.hpp"
#include "grid.hpp"
#include "options.hpp"
#include "node_server.hpp"

extern options opts;

#ifdef RADIATION

typedef node_server::send_rad_boundary_action send_rad_boundary_action_type;
HPX_REGISTER_ACTION (send_rad_boundary_action_type);

typedef node_server::send_rad_bnd_boundary_action send_rad_bnd_boundary_action_type;
HPX_REGISTER_ACTION (send_rad_bnd_boundary_action_type);

hpx::future<void> node_client::send_rad_boundary(std::vector<rad_type>&& data, const geo::octant& oct, const geo::dimension& dim) const {
	return hpx::async<typename node_server::send_rad_boundary_action>(get_gid(), std::move(data), oct, dim);
}

hpx::future<void> node_client::send_rad_boundary(std::vector<rad_type>&& data, const geo::face& f) const {
	return hpx::async<typename node_server::send_rad_bnd_boundary_action>(get_gid(), std::move(data), f);
}

void node_server::recv_rad_bnd_boundary(std::vector<rad_type>&& bdata, const geo::face& f) {
	sibling_rad_bnd_channels[f]->set_value(std::move(bdata));
}

void node_server::recv_rad_boundary(std::vector<rad_type>&& bdata, const geo::octant& oct, const geo::dimension& dim) {
	sibling_rad_channels[oct][dim]->set_value(std::move(bdata));
}

typedef node_server::send_rad_children_action send_rad_children_action_type;
HPX_REGISTER_ACTION (send_rad_children_action_type);

hpx::future<void> node_client::send_rad_children(std::vector<rad_type>&& data, const geo::octant& oct, const geo::octant& ioct) const {
	return hpx::async<typename node_server::send_rad_children_action>(get_gid(), std::move(data), oct, ioct);
}

void rad_grid::rad_imp_comoving(real& E, real& e, real rho, real dt) {
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

				/* Compute transformation parameters */
				const real vx = sx[iiih] / rho[iiih];
				const real vy = sy[iiih] / rho[iiih];
				const real vz = sz[iiih] / rho[iiih];
				const real v2 = vx * vx + vy * vy + vz * vz;
				const real beta_x = std::max(std::min(0.1, vx * cinv), -0.1);
				const real beta_y = std::max(std::min(0.1, vy * cinv), -0.1);
				const real beta_z = std::max(std::min(0.1, vz * cinv), -0.1);
				const real beta_2 = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z;

				/* Transform E and F from lab frame to comoving frame */

				real E0 = U[er_i][iiir] * (1.0 + beta_2);
				real tmp1 = 0.0, tmp2 = 0.0;
				tmp1 -= 2.0 * beta_x * cinv * U[fx_i][iiir];
				tmp1 -= 2.0 * beta_y * cinv * U[fy_i][iiir];
				tmp1 -= 2.0 * beta_z * cinv * U[fz_i][iiir];
				tmp2 += 2.0 * beta_x * fEdd_xx[iiir] * beta_x * U[er_i][iiir];
				tmp2 += 2.0 * beta_y * fEdd_yy[iiir] * beta_y * U[er_i][iiir];
				tmp2 += 2.0 * beta_z * fEdd_zz[iiir] * beta_z * U[er_i][iiir];
				tmp2 += 4.0 * beta_x * fEdd_xy[iiir] * beta_y * U[er_i][iiir];
				tmp2 += 4.0 * beta_x * fEdd_xz[iiir] * beta_z * U[er_i][iiir];
				tmp2 += 4.0 * beta_y * fEdd_yz[iiir] * beta_z * U[er_i][iiir];
				E0 += tmp1 + tmp2;
				if( E0 < 0.0 ) {
					printf( "1 %e %e %e %e %e %e %e %e \n", E0,  U[er_i][iiir] , tmp1, tmp2, beta_2, fEdd_xx[iiir], fEdd_yy[iiir], fEdd_zz[iiir]);
					abort();
				}
				real Fx0, Fy0, Fz0;
				Fx0 = U[fx_i][iiir] * (1.0 + 2.0 * beta_2);
				Fy0 = U[fy_i][iiir] * (1.0 + 2.0 * beta_2);
				Fz0 = U[fz_i][iiir] * (1.0 + 2.0 * beta_2);

				Fx0 -= beta_x * c * U[er_i][iiir];
				Fx0 -= beta_y * c * fEdd_xy[iiir] * U[er_i][iiir];
				Fx0 -= beta_z * c * fEdd_xz[iiir] * U[er_i][iiir];
				Fx0 -= beta_x * c * fEdd_xx[iiir] * U[er_i][iiir];

				Fy0 -= beta_y * c * U[er_i][iiir];
				Fy0 -= beta_x * c * fEdd_xy[iiir] * U[er_i][iiir];
				Fy0 -= beta_y * c * fEdd_yy[iiir] * U[er_i][iiir];
				Fy0 -= beta_z * c * fEdd_yz[iiir] * U[er_i][iiir];

				Fz0 -= beta_z * c * U[er_i][iiir];
				Fz0 -= beta_x * c * fEdd_xz[iiir] * U[er_i][iiir];
				Fz0 -= beta_y * c * fEdd_yz[iiir] * U[er_i][iiir];
				Fz0 -= beta_z * c * fEdd_zz[iiir] * U[er_i][iiir];


				/* Computer e0 from dual energy formalism */
				real e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * rho[iiih];
				e0 -= 0.5 * vy * vy * rho[iiih];
				e0 -= 0.5 * vz * vz * rho[iiih];
				const bool sw = false;
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}

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

				const real gam = 1.0 - std::sqrt(2.0)/2.0;

				real this_E = En;
				real this_e = en;
				rad_imp_comoving(this_E, this_e, rho[iiih], gam*dt);

				real kR = kappa_R(rho[iiih], this_e);
				de1 = (this_e - en) / (gam * dt);
				dE1 = (this_E - En) / (gam * dt);
				dFx1 = (-(Fx0 * (c * kR) / (1.0 + c * gam * dt * kR))) / (gam * dt);
				dFy1 = (-(Fy0 * (c * kR) / (1.0 + c * gam * dt * kR))) / (gam * dt);
				dFz1 = (-(Fz0 * (c * kR) / (1.0 + c * gam * dt * kR))) / (gam * dt);

				const real tmpE = (1.0 - 2.0 * gam) * dE1 * dt;
				const real tmpe = (1.0 - 2.0 * gam) * de1 * dt;
				const real tmpFx = (1.0 - 2.0 * gam) * dFx1 * dt;
				const real tmpFy = (1.0 - 2.0 * gam) * dFy1 * dt;
				const real tmpFz = (1.0 - 2.0 * gam) * dFz1 * dt;
				this_E = En + tmpE;
				this_e = en + tmpe;

				rad_imp_comoving(this_E, this_e, rho[iiih], gam * dt);
				kR = kappa_R(rho[iiih], this_e);
				de2 = (this_e - (en + tmpe)) / (gam * dt);
				dE2 = (this_E - (En + tmpE)) / (gam * dt);
				dFx2 = (-((Fx0 + tmpFx) * (c * kR) / (1.0 + c * dt * kR))) / (gam * dt);
				dFy2 = (-((Fy0 + tmpFy) * (c * kR) / (1.0 + c * dt * kR))) / (gam * dt);
				dFz2 = (-((Fz0 + tmpFz) * (c * kR) / (1.0 + c * dt * kR))) / (gam * dt);

				const real dE0_dt = (dE1 + dE2) * 0.5;
				const real de0_dt = (de1 + de2) * 0.5;
				const real dFx0_dt = (dFx1 + dFx2) * 0.5;
				const real dFy0_dt = (dFy1 + dFy2) * 0.5;
				const real dFz0_dt = (dFz1 + dFz2) * 0.5;


				/* Transform time derivatives to lab frame */
				const real b2o2p1 = (1.0 + 0.5 * beta_2);
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
				if( U[er_i][iiir] < 0.0 ) {
					printf( "2 %e %e %e %e %e %e %e %e \n", E0,  U[er_i][iiir] , tmp1, tmp2, beta_2, fEdd_xx[iiir], fEdd_yy[iiir], fEdd_zz[iiir]);
					abort();
				}
			}
		}
	}
}
void node_server::recv_rad_children(std::vector<rad_type>&& bdata, const geo::octant& oct, const geo::octant& ioct) {
	child_rad_channels[ioct][oct]->set_value(std::move(bdata));
}

void rad_grid::get_output(std::array<std::vector<real>, NF + NGF + NRADF + NPF>& v, integer i, integer j, integer k) const {
	const integer iii = rindex(i, j, k);
//	printf("%e\n", fEdd_xx[iii]);
	v[NF + 0].push_back(fEdd_xx[iii]);
	v[NF + 1].push_back(fEdd_xy[iii]);
	v[NF + 2].push_back(fEdd_xz[iii]);
	v[NF + 3].push_back(fEdd_yy[iii]);
	v[NF + 4].push_back(fEdd_yz[iii]);
	v[NF + 5].push_back(fEdd_zz[iii]);
	v[NF + 6].push_back(real(U[er_i][iii]));
	v[NF + 7].push_back(real(U[fx_i][iii]));
	v[NF + 8].push_back(real(U[fy_i][iii]));
	v[NF + 9].push_back(real(U[fz_i][iii]));

}

void rad_grid::set_dx(real _dx) {
	dx = _dx;
}

void node_server::compute_radiation(real dt, bool new_eddington) {
	if (my_location.level() == 0) {
		printf("Eddington\n");
	}

	rad_grid_ptr->set_dx(grid_ptr->get_dx());
	auto rgrid = rad_grid_ptr;
	if (new_eddington) {
		rgrid->set_emissivity(grid_ptr->get_field(rho_i), grid_ptr->get_field(sx_i), grid_ptr->get_field(sy_i), grid_ptr->get_field(sz_i),
				grid_ptr->get_field(tau_i));

		std::list<hpx::future<void>> oct_futs;
		for (auto& this_oct : geo::octant::full_set()) {
			oct_futs.push_back(hpx::async([&](const geo::octant& oct) {
				rgrid->alloc_octant(oct);
				geo::face f_out[NDIM];
				geo::face f_in[NDIM];
				std::list<hpx::future<void>> futs;
				for (integer d = 0; d != NDIM; ++d) {
					f_out[d] = oct.get_face(d);
					f_in[d] = f_out[d].flip();
				}

				/* get boundaries */
				for (integer d = 0; d != NDIM; ++d) {
					const auto& n = neighbors[f_in[d].to_direction()];
					if (!my_location.is_physical_boundary(f_in[d])) {
						auto f = sibling_rad_channels[oct][d]->get_future();
						std::array<integer, NDIM> lb, ub;
						get_boundary_size(lb, ub, f_in[d].to_direction(), OUTER, INX, R_BW, 1);
						rgrid->set_intensity(f.get(), lb, ub, oct);
					}
				}

				/* compute or get interior */
				if (!is_refined) {
					rgrid->compute_intensity(oct);
				} else {
					for (auto& ci : geo::octant::full_set()) {
						const auto& flags = amr_flags[ci];
						for (auto& dim : geo::dimension::full_set()) {
							auto dir = f_in[dim].to_direction();
							if (flags[dir] && !my_location.is_physical_boundary(f_in[dim])) {
								std::array<integer, NDIM> lb, ub;
								std::vector<real> data;
								get_boundary_size(lb, ub, dir, OUTER, INX, R_BW, 1);
								for (integer d = 0; d != NDIM; ++d) {
									lb[d] = ((lb[d] - R_BW)) + 2 * R_BW + ci.get_side(d) * (INX);
									ub[d] = ((ub[d] - R_BW)) + 2 * R_BW + ci.get_side(d) * (INX);
								}
								data = rgrid->get_prolong(lb, ub, oct);
								futs.push_back(children[ci].send_rad_boundary(std::move(data), oct, dim));
							}
						}
					}
					hpx::wait_all(futs.begin(), futs.end());
					futs.clear();
					for( integer ci = 0; ci != NCHILD; ++ci) {
						auto f = child_rad_channels[oct][ci]->get_future();
						auto fcont = f.then([=](hpx::future<std::vector<real>>&& fut) {
									rgrid->set_restrict(fut.get(), ci, oct);
								});
						futs.push_back(std::move(fcont));
					}
					hpx::wait_all(futs.begin(), futs.end());
					futs.clear();
				}

				/* send boundaries, restrict to parent, and send amr boundaries */
				if( !parent.empty() ) {
					auto f = parent.send_rad_children(rgrid->get_restrict(oct), my_location.get_child_index(), oct);
					futs.push_back(std::move(f));
				}
				for (integer d = 0; d != NDIM; ++d) {
					if (!my_location.is_physical_boundary(f_out[d])) {
						const auto& n = neighbors[f_out[d].to_direction()];
						if( !n.empty()) {
							std::array<integer, NDIM> lb, ub;
							get_boundary_size(lb, ub, f_out[d].to_direction(), INNER, INX, R_BW, 1);
							auto data = rgrid->get_intensity(lb, ub, oct);
							futs.push_back(n.send_rad_boundary(std::move(data), oct, d));
						}
					}
				}

				rgrid->accumulate_intensity(oct);

				hpx::wait_all(futs.begin(), futs.end());
				rgrid->free_octant(oct);

			}, this_oct));
		}
		hpx::wait_all(oct_futs.begin(), oct_futs.end());
		rgrid->compute_fEdd();
	}
	const real min_dx = TWO * grid::get_scaling_factor() / real(INX << opts.max_level);
	const real c = LIGHTSPEED;
	const real max_dt = min_dx / c * 0.4 / 3.0;
	integer nsteps = std::max(int(std::ceil(dt / max_dt)), 1);
	//if( nsteps % 2 == 1 ) {
	//	++nsteps;
//	}

//	if (my_location.level() == 0) {
//		printf("%i %e %e\n", int(nsteps), dt, max_dt);
//	}
//	nsteps *= 2;
	const real this_dt = dt / real(nsteps);
	auto& egas = grid_ptr->get_field(egas_i);
	const auto& rho = grid_ptr->get_field(rho_i);
	auto& tau = grid_ptr->get_field(tau_i);
	auto& sx = grid_ptr->get_field(sx_i);
	auto& sy = grid_ptr->get_field(sy_i);
	auto& sz = grid_ptr->get_field(sz_i);
	rgrid->store();
	for (integer i = 0; i != nsteps; ++i) {
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
		printf("Implicit\n");
	}
	rgrid->rad_imp(egas, tau, sx, sy, sz, rho, dt);
	if (my_location.level() == 0) {
		printf("Rad done\n");
	}
}

std::vector<sphere_point> rad_grid::sphere_points;

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
				//	printf( "%e\n", fEdd_xx[iii]);
			}
		}
	}

}

void rad_grid::initialize() {

	static std::once_flag flag;
	std::call_once(flag, [&]() {
		sphere_points = generate_sphere_points(3);
	});

}

rad_grid_init::rad_grid_init() {
	rad_grid::initialize();
}

void rad_grid::allocate() {
	fEdd[XDIM][XDIM] = &fEdd_xx;
	fEdd[YDIM][YDIM] = &fEdd_yy;
	fEdd[ZDIM][ZDIM] = &fEdd_zz;
	fEdd[XDIM][YDIM] = fEdd[YDIM][XDIM] = &fEdd_xy;
	fEdd[XDIM][ZDIM] = fEdd[ZDIM][XDIM] = &fEdd_xz;
	fEdd[YDIM][ZDIM] = fEdd[ZDIM][YDIM] = &fEdd_yz;
	rad_grid::dx = dx;
	J.resize(R_N3);
	sigma_a.resize(R_N3, 0.0);
	sigma_s.resize(R_N3, 0.0);
	I = std::vector<std::vector<rad_type> >(sphere_points.size(), std::vector<rad_type>(R_N3, 0.0));
	fEdd_xx.resize(R_N3, 0.0);
	fEdd_xy.resize(R_N3, 0.0);
	fEdd_xz.resize(R_N3, 0.0);
	fEdd_yy.resize(R_N3, 0.0);
	fEdd_yz.resize(R_N3, 0.0);
	fEdd_zz.resize(R_N3, 0.0);
	E.resize(R_N3, 0.0);
	Beta_x.resize(R_N3, 0.0);
	Beta_y.resize(R_N3, 0.0);
	Beta_z.resize(R_N3, 0.0);
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
	auto& f = fEdd;
	for (integer i = 0; i != R_N3; ++i) {
		for (integer f = 0; f != NRF; ++f) {
			for (integer d = 0; d != NDIM; ++d) {
				flux[d][f][i] = 0.0;
			}
		}
	}
	const integer D[3] = { DX, DY, DZ };
	for (integer d2 = 0; d2 != NDIM; ++d2) {
		for (integer f = 0; f != NRF; ++f) {
			for (integer i = DX; i != R_N3 - DX; ++i) {
				const real tmp0 = U[f][i];
				const real tmp1 = U[f][i + D[d2]] - tmp0;
				const real tmp2 = tmp0 - U[f][i - D[d2]];
				s[f][i] = minmod(tmp1, tmp2);
				//		s[f][i] = 0.0;
			}
		}
		for (integer d1 = 0; d1 != NDIM; ++d1) {
			for (integer i = DX; i != R_N3 - DX; ++i) {
				const real tmp0 = (*fEdd[d1][d2])[i];
				const real tmp1 = (*fEdd[d1][d2])[i + D[d2]] - tmp0;
				const real tmp2 = tmp0 - (*fEdd[d1][d2])[i - D[d2]];
				fs[d1][i] = minmod(tmp1, tmp2);
				//	fs[d1][i] = 0.0;
			}
		}
		for (integer i = 2 * DX; i != R_N3 - DX; ++i) {
			real cx;
			const real vp = U[fx_i + d2][i] - 0.5 * s[fx_i + d2][i];
			const real vm = U[fx_i + d2][i - D[d2]] + 0.5 * s[fx_i + d2][i - D[d2]];
			flux[d2][er_i][i] += (vp + vm) * 0.5;
			for (integer d1 = 0; d1 != NDIM; ++d1) {
				const real vp = U[er_i][i] - 0.5 * s[er_i][i];
				const real vm = U[er_i][i - D[d2]] + 0.5 * s[er_i][i - D[d2]];
				const real fp = (*fEdd[d1][d2])[i] - 0.5 * fs[d1][i];
				const real fm = (*fEdd[d1][d2])[i - D[d2]] + 0.5 * fs[d1][i - D[d2]];
				flux[d2][fx_i + d1][i] += c * c * (fp * vp + fm * vm) * 0.5;
			}
			cx = c * std::sqrt(std::max((*f[d2][d2])[i], (*f[d2][d2])[i - D[d2]]));
			for (integer f = 0; f != NRF; ++f) {
				const real vp = U[f][i] - 0.5 * s[f][i];
				const real vm = U[f][i - D[d2]] + 0.5 * s[f][i - D[d2]];
				flux[d2][f][i] -= (vp - vm) * 0.5 * cx;
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
				}
				fEdd_xx[iii1] = fEdd_xx[iii0];
				fEdd_yy[iii1] = fEdd_yy[iii0];
				fEdd_zz[iii1] = fEdd_zz[iii0];
				fEdd_xy[iii1] = fEdd_xy[iii0];
				fEdd_xz[iii1] = fEdd_xz[iii0];
				fEdd_yz[iii1] = fEdd_yz[iii0];
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

void rad_grid::compute_intensity(const geo::octant& oct) {
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
	}
	for (integer d1 = 0; d1 != NDIM; ++d1) {
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
	}PROF_END;
}

std::vector<real> rad_grid::get_boundary(const geo::direction& dir) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size;
	size = NRADF * get_boundary_size(lb, ub, dir, INNER, INX, R_BW);
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
	for (integer d1 = 0; d1 != NDIM; ++d1) {
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
	}PROF_END;
	return data;
}

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

void rad_grid::set_emissivity(const std::vector<real>& rho, const std::vector<real>& sx, const std::vector<real>& sy, const std::vector<real>& sz,
		const std::vector<real>& tau) {
	const real fgamma = grid::get_fgamma();
	for (integer xi = 0; xi != R_NX; ++xi) {
		for (integer yi = 0; yi != R_NX; ++yi) {
			for (integer zi = 0; zi != R_NX; ++zi) {
				const integer iii = rindex(xi, yi, zi);
				E[iii] = 0.0;
				fEdd_xx[iii] = fEdd_yy[iii] = fEdd_zz[iii] = 0.0;
				fEdd_xy[iii] = fEdd_yz[iii] = fEdd_xz[iii] = 0.0;
			}
		}
	}
	const real c = LIGHTSPEED;
	for (integer xi = 0; xi != R_NX; ++xi) {
		for (integer yi = 0; yi != R_NX; ++yi) {
			for (integer zi = 0; zi != R_NX; ++zi) {
				const auto D = H_BW - R_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				const real e = std::pow(tau[iiih], fgamma);
				//	sigma_a[iiir] = J[iiir] = rho[iiih];
				J[iiir] = kappa_p(rho[iiih], e) * B_p(rho[iiih], e) * (4.0*M_PI/c);
				sigma_a[iiir] = kappa_R(rho[iiih], e);
				sigma_s[iiir] = kappa_s(rho[iiih], e);
				Beta_x[iiir] = sx[iiih] / rho[iiih] / c;
				Beta_y[iiir] = sy[iiih] / rho[iiih] / c;
				Beta_z[iiir] = sz[iiih] / rho[iiih] / c;
			}
		}
	}
	I = std::vector<std::vector<rad_type> >(sphere_points.size(), std::vector<rad_type>(R_N3, 0.0));
}

void rad_grid::set_field(rad_type v, integer f, integer i, integer j, integer k) {
	U[f][rindex(i, j, k)] = v;
}

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

#endif

