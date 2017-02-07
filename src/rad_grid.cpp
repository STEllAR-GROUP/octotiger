#include "defs.hpp"
#include "rad_grid.hpp"
#include "grid.hpp"
#include "options.hpp"
#include "node_server.hpp"

extern options opts;

#define RADIATION

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

void rad_imp(real& e, real& E, const std::function<real(real)>& kp, const std::function<real(real)>& B, real dt) {
	const integer max_iter = 100;
	const real E0 = E;
	real E1 = E0;
	real f = 1.0;
	real dE;
	integer i = 0;
	constexpr static auto c = LIGHTSPEED;
	do {
		const real de = e * 0.001;
		const real dkp_de = (kp(e + de) - kp(e)) * 0.5 / de;
		const real dB_de = (B(e + de) - B(e)) * 0.5 / de;
		f = (E - E0) + dt * c * kp(e) * (E - 4.0 * M_PI / c * B(e));
		const real dfdE = 1.0 + dt * c * kp(e);
		const real dfde = dt * c * dkp_de * (E - 4.0 * M_PI / c * B(e)) - dt * kp(e) * 4.0 * M_PI * dB_de;
		dE = -f / (dfdE - dfde);
		const real w = 0.6;
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
			abort_error();
			abort();
		}
	} while (std::abs(f / E0) > 1.0e-9);
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

void node_server::compute_radiation(real dt) {
	dt = 1.0e-2;
	rad_grid_ptr->set_dx(grid_ptr->get_dx());
	auto rgrid = rad_grid_ptr;

	rgrid->set_emissivity(grid_ptr->get_field(rho_i), grid_ptr->get_field(sx_i), grid_ptr->get_field(sy_i), grid_ptr->get_field(sz_i),
			grid_ptr->get_field(tau_i));

	std::list<hpx::future<void>> oct_futs;
	for (auto& this_oct : geo::octant::full_set()) {
		oct_futs.push_back(hpx::async([&](const geo::octant& oct) {
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
					get_boundary_size(lb, ub, f_in[d].to_direction(), OUTER, INX, R_BW);
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
						if (flags[dir]) {
							std::array<integer, NDIM> lb, ub;
							std::vector<real> data;
							get_boundary_size(lb, ub, dir, OUTER, INX, R_BW);
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
			rgrid->accumulate_intensity(oct);

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
						get_boundary_size(lb, ub, f_out[d].to_direction(), INNER, INX, R_BW);
						auto data = rgrid->get_intensity(lb, ub, oct);
						futs.push_back(n.send_rad_boundary(std::move(data), oct, d));
					}
				}
			}
			hpx::wait_all(futs.begin(), futs.end());

		}, this_oct));
	}
	hpx::wait_all(oct_futs.begin(), oct_futs.end());
	rgrid->compute_fEdd();

	const real min_dx = TWO * grid::get_scaling_factor() / real(INX << opts.max_level);
	const real c = LIGHTSPEED;
	const real max_dt = min_dx / c;
	const integer nsteps = std::min(int(std::ceil(dt / max_dt)), 1);
	const real this_dt = dt / real(nsteps);
	for (integer i = 0; i != nsteps; ++i) {
		if (my_location.level() == 0) {
			printf("%e %e %lli\n", dt, this_dt, nsteps);
		}
		collect_radiation_boundaries().get();
		rgrid->compute_flux();
		rgrid->advance(this_dt);
	}
	collect_radiation_boundaries().get();
}

std::vector<sphere_point> rad_grid::sphere_points;

void rad_grid::compute_fEdd() {
	const real c = LIGHTSPEED;
	auto& Pxx = fEdd_xx;
	auto& Pxy = fEdd_xy;
	auto& Pxz = fEdd_xz;
	auto& Pyy = fEdd_yy;
	auto& Pyz = fEdd_yz;
	auto& Pzz = fEdd_zz;
	const auto Pxx0 = fEdd_xx;
	const auto Pxy0 = fEdd_xy;
	const auto Pxz0 = fEdd_xz;
	const auto Pyy0 = fEdd_yy;
	const auto Pyz0 = fEdd_yz;
	const auto Pzz0 = fEdd_zz;
	for (integer xi = 0; xi != R_NX; ++xi) {
		for (integer yi = 0; yi != R_NX; ++yi) {
			for (integer zi = 0; zi != R_NX; ++zi) {
				const integer iii = rindex(xi, yi, zi);
				Pxx[iii] += (Fx[iii] * vx[iii] + vx[iii] * Fx[iii]) / (c * c) + vx[iii] * vx[iii] * E[iii] / (c * c);
				Pxy[iii] += (Fx[iii] * vy[iii] + vx[iii] * Fy[iii]) / (c * c) + vx[iii] * vy[iii] * E[iii] / (c * c);
				Pxz[iii] += (Fx[iii] * vz[iii] + vx[iii] * Fz[iii]) / (c * c) + vx[iii] * vz[iii] * E[iii] / (c * c);
				Pyy[iii] += (Fy[iii] * vy[iii] + vy[iii] * Fy[iii]) / (c * c) + vy[iii] * vy[iii] * E[iii] / (c * c);
				Pyz[iii] += (Fy[iii] * vz[iii] + vy[iii] * Fz[iii]) / (c * c) + vy[iii] * vz[iii] * E[iii] / (c * c);
				Pzz[iii] += (Fz[iii] * vz[iii] + vz[iii] * Fz[iii]) / (c * c) + vz[iii] * vz[iii] * E[iii] / (c * c);
				const real vPx = vx[iii] * Pxx0[iii] + vy[iii] * Pxy0[iii] + vz[iii] * Pxz0[iii];
				const real vPy = vx[iii] * Pxy0[iii] + vy[iii] * Pyy0[iii] + vz[iii] * Pyz0[iii];
				const real vPz = vx[iii] * Pxz0[iii] + vy[iii] * Pyz0[iii] + vz[iii] * Pzz0[iii];
				Pxx[iii] += vx[iii] * vPx / (c * c);
				Pyy[iii] += vy[iii] * vPy / (c * c);
				Pzz[iii] += vz[iii] * vPz / (c * c);
				Pxy[iii] += 0.5 * vx[iii] * vPy / (c * c);
				Pxz[iii] += 0.5 * vx[iii] * vPz / (c * c);
				Pyz[iii] += 0.5 * vy[iii] * vPz / (c * c);
				Pxy[iii] += 0.5 * vy[iii] * vPx / (c * c);
				Pxz[iii] += 0.5 * vz[iii] * vPx / (c * c);
				Pyz[iii] += 0.5 * vz[iii] * vPy / (c * c);

				if (E[iii] > 0.0) {
					fEdd_xx[iii] /= E[iii];
					fEdd_xy[iii] /= E[iii];
					fEdd_xz[iii] /= E[iii];
					fEdd_yy[iii] /= E[iii];
					fEdd_yz[iii] /= E[iii];
					fEdd_zz[iii] /= E[iii];
				} else {
					fEdd_xx[iii] = fEdd_xy[iii] = fEdd_xz[iii] = fEdd_yy[iii] = fEdd_yz[iii] = fEdd_zz[iii] = 0.0;
				}
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
	I = std::vector<std::vector<rad_type> >(sphere_points.size(), std::vector<rad_type>(R_N3, 0.0));
	fEdd_xx.resize(R_N3, 0.0);
	fEdd_xy.resize(R_N3, 0.0);
	fEdd_xz.resize(R_N3, 0.0);
	fEdd_yy.resize(R_N3, 0.0);
	fEdd_yz.resize(R_N3, 0.0);
	fEdd_zz.resize(R_N3, 0.0);
	E.resize(R_N3, 0.0);
	Fx.resize(R_N3, 0.0);
	Fy.resize(R_N3, 0.0);
	Fz.resize(R_N3, 0.0);
	vx.resize(R_N3, 0.0);
	vy.resize(R_N3, 0.0);
	vz.resize(R_N3, 0.0);
	for (integer f = 0; f != NRF; ++f) {
		U[f].resize(R_N3);
		fx[f].resize(R_N3);
		fy[f].resize(R_N3);
		fz[f].resize(R_N3);
	}
}

void rad_grid::compute_flux() {
	real cx, cy, cz;
	const real c = LIGHTSPEED;

	auto& f = fEdd;
	/*	for (integer i = 0; i != R_N3; ++i) {
	 for (integer d1 = 0; d1 != 3; ++d1) {
	 for (integer d2 = d1; d2 != 3; ++d2) {
	 (*f[d1][d2])[i] = 0.0;
	 }
	 }
	 for (integer d1 = 0; d1 != 3; ++d1) {
	 (*f[d1][d1])[i] = 1.0 / 3.0;
	 }
	 }*/
	for (integer i = DX; i != R_N3; ++i) {
		for (integer f = 0; f != NRF; ++f) {
			fx[f][i] = fy[f][i] = fz[f][i] = 0.0;
		}
	}
	for (integer i = DX; i != R_N3; ++i) {
		cx = c * std::sqrt(std::max((*f[0][0])[i], (*f[0][0])[i - DX]));
		cy = c * std::sqrt(std::max((*f[1][1])[i], (*f[1][1])[i - DY]));
		cz = c * std::sqrt(std::max((*f[2][2])[i], (*f[2][2])[i - DZ]));
		fx[er_i][i] += (U[fx_i][i] + U[fx_i][i - DX]) * 0.5;
		fy[er_i][i] += (U[fy_i][i] + U[fy_i][i - DY]) * 0.5;
		fz[er_i][i] += (U[fz_i][i] + U[fz_i][i - DZ]) * 0.5;
		for (integer d = 0; d != NDIM; ++d) {
			fx[fx_i + d][i] += c * c * ((*f[0][d])[i] * U[er_i][i] + (*f[0][d])[i - DX] * U[er_i][i - DX]) * 0.5;
			fy[fx_i + d][i] += c * c * ((*f[1][d])[i] * U[er_i][i] + (*f[1][d])[i - DY] * U[er_i][i - DY]) * 0.5;
			fz[fx_i + d][i] += c * c * ((*f[2][d])[i] * U[er_i][i] + (*f[2][d])[i - DZ] * U[er_i][i - DZ]) * 0.5;
		}

		for (integer f = 0; f != NRF; ++f) {
			fx[f][i] -= (U[f][i] - U[f][i - DX]) * 0.5 * cx;
			fy[f][i] -= (U[f][i] - U[f][i - DY]) * 0.5 * cy;
			fz[f][i] -= (U[f][i] - U[f][i - DZ]) * 0.5 * cz;
		}
	}
}

void rad_grid::advance(real dt) {
	const real l = dt / dx;
	for (integer f = 0; f != NRF; ++f) {
		for (integer xi = R_BW; xi != R_NX - R_BW; ++xi) {
			for (integer yi = R_BW; yi != R_NX - R_BW; ++yi) {
				for (integer zi = R_BW; zi != R_NX - R_BW; ++zi) {
					const integer iii = rindex(xi, yi, zi);
					U[f][iii] -= l * (fx[f][iii + DX] - fx[f][iii]);
					U[f][iii] -= l * (fy[f][iii + DY] - fy[f][iii]);
					U[f][iii] -= l * (fz[f][iii + DZ] - fz[f][iii]);
				}
			}
		}
	}
}

void rad_grid::set_physical_boundaries(geo::face f) {
	for (integer i = 0; i != R_NX; ++i) {
		for (integer j = 0; j != R_NX; ++j) {
			for (integer k = 0; k != R_BW; ++k) {
				integer iii1, iii0;
				switch (f) {
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
				switch (f) {
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
			b = 1;
			e = R_NX;
		} else {
			inc = -1;
			b = R_NX - 2;
			e = -1;
		}
	};
	integer xinc, yinc, zinc, xb, yb, zb, xe, ye, ze;
	indexes(oct[XDIM], xinc, xb, xe);
	indexes(oct[YDIM], yinc, yb, ye);
	indexes(oct[ZDIM], zinc, zb, ze);
//	printf("%i %i %i\n", int(xinc), int(yinc), int(zinc));
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		const auto& pt = sphere_points[spi];
		//	printf( "%i %i\n", int(pt.get_octant()), int(oct) );
		if (pt.get_octant() == oct) {
			for (integer xi = xb; xi != xe; xi += xinc) {
				for (integer yi = yb; yi != ye; yi += yinc) {
					for (integer zi = zb; zi != ze; zi += zinc) {
						const integer iii = rindex(xi, yi, zi);
						const integer iiix = rindex(xi - xinc, yi, zi);
						const integer iiiy = rindex(xi, yi - yinc, zi);
						const integer iiiz = rindex(xi, yi, zi - zinc);
						rad_type I_m1 = 0.0;
						rad_type j0 = 0.0;
						rad_type sigma_a0 = 0.0;
						I_m1 += pt.wx * I[spi][iiix];
						I_m1 += pt.wy * I[spi][iiiy];
						I_m1 += pt.wz * I[spi][iiiz];
						j0 += pt.wx * J[iiix];
						j0 += pt.wy * J[iiiy];
						j0 += pt.wz * J[iiiz];
						j0 = 0.5 * (j0 + J[iii]);
						sigma_a0 += pt.wx * sigma_a[iiix];
						sigma_a0 += pt.wy * sigma_a[iiiy];
						sigma_a0 += pt.wz * sigma_a[iiiz];
						sigma_a0 = 0.5 * (sigma_a0 + sigma_a[iii]);
						I[spi][iii] = I_m1 * std::exp(-sigma_a0 * dx) + j0;
					}
				}
			}
		}
	}
}

void rad_grid::set_boundary(const std::vector<real>& data, const geo::direction& dir) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, dir, OUTER, INX, 1);
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
	size = NRADF * get_boundary_size(lb, ub, dir, INNER, INX, 1);
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
						Fx[iii] += I[spi][iii] * pt.nx * pt.dA;
						Fy[iii] += I[spi][iii] * pt.nx * pt.dA;
						Fz[iii] += I[spi][iii] * pt.nx * pt.dA;
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
			I[spi].reserve(0);
		}
	}
}

void rad_grid::set_emissivity(const std::vector<real>& rho, const std::vector<real>& sx, const std::vector<real>& sy, const std::vector<real>& sz,
		const std::vector<real>& tau) {
	for (integer xi = 0; xi != R_NX; ++xi) {
		for (integer yi = 0; yi != R_NX; ++yi) {
			for (integer zi = 0; zi != R_NX; ++zi) {
				const auto D = H_BW - R_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				J[iiir] = rho[iiih];
				sigma_a[iiir] = 10000.0;
				vx[iiir] = sx[iiih] / rho[iiih];
				vy[iiir] = sy[iiih] / rho[iiih];
				vz[iiir] = sz[iiih] / rho[iiih];
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

