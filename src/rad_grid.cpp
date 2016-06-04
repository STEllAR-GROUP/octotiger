#include "defs.hpp"
#include "rad_grid.hpp"
#include "grid.hpp"
#include "node_server.hpp"

#define RADIATION

#ifdef RADIATION

#define NPHI 256

typedef node_server::send_rad_boundary_action send_rad_boundary_action_type;
HPX_REGISTER_ACTION(send_rad_boundary_action_type);

hpx::future<void> node_client::send_rad_boundary(std::vector<rad_type>&& data, const geo::octant& oct, const geo::dimension& dim) const {
	return hpx::async<typename node_server::send_rad_boundary_action>(get_gid(), std::move(data), oct, dim);
}

void node_server::recv_rad_boundary(std::vector<rad_type>&& bdata, const geo::octant& oct, const geo::dimension& dim) {
	sibling_rad_channels[oct][dim]->set_value(std::move(bdata));
}

typedef node_server::send_rad_children_action send_rad_children_action_type;
HPX_REGISTER_ACTION(send_rad_children_action_type);

hpx::future<void> node_client::send_rad_children(std::vector<rad_type>&& data, const geo::octant& oct, const geo::octant& ioct) const {
	return hpx::async<typename node_server::send_rad_children_action>(get_gid(), std::move(data), oct, ioct);
}

void node_server::recv_rad_children(std::vector<rad_type>&& bdata, const geo::octant& oct, const geo::octant& ioct) {
	child_rad_channels[ioct][oct]->set_value(std::move(bdata));
}

void node_server::compute_radiation() {
	auto rgrid = std::make_shared < rad_grid > (grid_ptr->get_dx());

	rgrid->set_emissivity(grid_ptr->get_field(rho_i), grid_ptr->get_field(tau_i));

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
				const auto& n = neighbors[f_in[d].get_direction()];
				if (!my_location.is_physical_boundary(f_in[d])) {
					auto f = sibling_rad_channels[oct][d]->get_future();
					std::array<integer, NDIM> lb, ub;
					get_boundary_size(lb, ub, f_in[d].get_direction(), OUTER, R_BW);
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
						auto dir = f_in[dim].get_direction();
						if (flags[dir]) {
							std::array<integer, NDIM> lb, ub;
							std::vector<real> data;
							get_boundary_size(lb, ub, dir, OUTER, R_BW);
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
					const auto& n = neighbors[f_out[d].get_direction()];
					if( !n.empty()) {
						std::array<integer, NDIM> lb, ub;
						get_boundary_size(lb, ub, f_out[d].get_direction(), INNER, R_BW);
						auto data = rgrid->get_intensity(lb, ub, oct);
						futs.push_back(n.send_rad_boundary(std::move(data), oct, d));
					}
				}
			}
			hpx::wait_all(futs.begin(), futs.end());

		}, this_oct));
	}
	hpx::wait_all(oct_futs.begin(), oct_futs.end());
	if (!is_refined) {
		grid_ptr->set_field(rgrid->get_P(XDIM, XDIM), pxx_i);
		grid_ptr->set_field(rgrid->get_P(XDIM, YDIM), pxy_i);
		grid_ptr->set_field(rgrid->get_P(XDIM, ZDIM), pxz_i);
		grid_ptr->set_field(rgrid->get_P(YDIM, YDIM), pyy_i);
		grid_ptr->set_field(rgrid->get_P(YDIM, ZDIM), pyz_i);
		grid_ptr->set_field(rgrid->get_P(ZDIM, ZDIM), pzz_i);
	}
}

std::vector<sphere_point> rad_grid::sphere_points;

void rad_grid::initialize() {

	static std::once_flag flag;
	std::call_once(flag, [&]() {
		sphere_points = generate_sphere_points(4);
	});

}

std::vector<real> rad_grid::get_P(const geo::dimension& dim1, const geo::dimension& dim2) const {
	const auto& p = *(P[dim1][dim2]);
	const auto D = H_BW - R_BW;
	std::vector<real> return_p(H_N3);
	for (integer xi = R_BW; xi != R_NX - R_BW; ++xi) {
		for (integer yi = R_BW; yi != R_NX - R_BW; ++yi) {
			for (integer zi = R_BW; zi != R_NX - R_BW; ++zi) {
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi+D,yi+D,zi+D);
				return_p[iiih] = p[iiir] / E[iiir];
			}
		}
	}
	return return_p;
}

rad_grid_init::rad_grid_init() {
	rad_grid::initialize();
}

rad_grid::rad_grid(real dx) {
	P[XDIM][XDIM] = &Pxx;
	P[YDIM][YDIM] = &Pyy;
	P[ZDIM][ZDIM] = &Pzz;
	P[XDIM][YDIM] = P[YDIM][XDIM] = &Pxy;
	P[XDIM][ZDIM] = P[ZDIM][XDIM] = &Pxz;
	P[YDIM][ZDIM] = P[ZDIM][YDIM] = &Pyz;
	rad_grid::dx = dx;
	J.resize(R_N3);
	sigma_a.resize(R_N3, 0.0);
	I = std::vector<std::vector<rad_type> >(sphere_points.size(), std::vector<rad_type>(R_N3, 0.0));
	Pxx.resize(R_N3, 0.0);
	Pxy.resize(R_N3, 0.0);
	Pxz.resize(R_N3, 0.0);
	Pyy.resize(R_N3, 0.0);
	Pyz.resize(R_N3, 0.0);
	Pzz.resize(R_N3, 0.0);
	E.resize(R_N3, 0.0);
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
	std::lock_guard < hpx::mutex > lock(Pmtx);
	for (integer spi = 0; spi != sphere_points.size(); ++spi) {
		const auto& pt = sphere_points[spi];
		if (pt.get_octant() == oct) {
			for (integer xi = R_BW; xi != R_NX - R_BW; ++xi) {
				for (integer yi = R_BW; yi != R_NX - R_BW; ++yi) {
					for (integer zi = R_BW; zi != R_NX - R_BW; ++zi) {
						const integer iii = rindex(xi, yi, zi);
						E[iii] += I[spi][iii] * pt.dA;
						Pxx[iii] += I[spi][iii] * pt.nx * pt.nx * pt.dA;
						Pxy[iii] += I[spi][iii] * pt.nx * pt.ny * pt.dA;
						Pxz[iii] += I[spi][iii] * pt.nx * pt.nz * pt.dA;
						Pyy[iii] += I[spi][iii] * pt.ny * pt.ny * pt.dA;
						Pyz[iii] += I[spi][iii] * pt.ny * pt.nz * pt.dA;
						Pzz[iii] += I[spi][iii] * pt.nz * pt.nz * pt.dA;
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

void rad_grid::set_emissivity(const std::vector<real>& rho, const std::vector<real>& tau) {
	for (integer xi = 0; xi != R_NX; ++xi) {
		for (integer yi = 0; yi != R_NX; ++yi) {
			for (integer zi = 0; zi != R_NX; ++zi) {
				const auto D = H_BW - R_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi+D,yi+D,zi+D);
				J[iiir] = rho[iiih];
			}
		}
	}
	I = std::vector<std::vector<rad_type> >(sphere_points.size(), std::vector<rad_type>(R_N3, 0.0));
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

#endif
