#include "defs.hpp"
#include "rad_grid.hpp"
#include "grid.hpp"
#include "node_server.hpp"

#define RADIATION

#ifdef RADIATION

#define NPHI 1024

void node_server::compute_radiation() {

	auto rgrid = std::make_shared < rad_grid > (grid_ptr->get_dx());
	rgrid->set_emissivity(grid_ptr->get_field(rho_i), grid_ptr->get_field(tau_i));
	hpx::future<void> fut = hpx::make_ready_future();

	std::list<hpx::future<void>> oct_futs;

	for (auto& this_oct : geo::octant::full_set()) {
		oct_futs.push_back(hpx::async([&](const geo::octant& oct) {
			geo::face f_out[NDIM];
			geo::face f_in[NDIM];
			hpx::future<std::vector<real>> data_fut[NDIM];
			for (integer d = 0; d != NDIM; ++d) {
				f_out[d] = oct.get_face(d);
				f_in[d] = f_out[d].flip();
				if (!my_location.is_physical_boundary(f_in[d])) {
					data_fut[d] = sibling_rad_channels[oct][d]->get_future();
				}
			}
			for (integer d = 0; d != NDIM; ++d) {
				if (!my_location.is_physical_boundary(f_in[d])) {
					std::array<integer, NDIM> lb, ub;
					get_boundary_size(lb, ub, f_in[d].to_direction(), OUTER, R_BW);
					rgrid->set_intensity(data_fut[d].get(), lb, ub, oct);
				}
			}
			rgrid->compute_intensity(oct);
	//		printf( "%i\n", int(oct));
			std::list<hpx::future<void>> futs;
			for (integer d = 0; d != NDIM; ++d) {
				if (!my_location.is_physical_boundary(f_out[d])) {
					std::array<integer, NDIM> lb, ub;
					get_boundary_size(lb, ub, f_out[d].to_direction(), INNER, R_BW);
					auto data = rgrid->get_intensity(lb, ub, oct);
					auto f = neighbors[f_out[d].to_direction()].send_rad_boundary(std::move(data), oct, d);
					futs.push_back(std::move(f));
				}
			}
			hpx::wait_all(futs.begin(), futs.end());
		}, this_oct));
	}
	hpx::wait_all(oct_futs.begin(), oct_futs.end());
	grid_ptr->set_field(rgrid->get_P(XDIM, XDIM), pxx_i);
	grid_ptr->set_field(rgrid->get_P(XDIM, YDIM), pxy_i);
	grid_ptr->set_field(rgrid->get_P(XDIM, ZDIM), pxz_i);
	grid_ptr->set_field(rgrid->get_P(YDIM, YDIM), pyy_i);
	grid_ptr->set_field(rgrid->get_P(YDIM, ZDIM), pyz_i);
	grid_ptr->set_field(rgrid->get_P(ZDIM, ZDIM), pzz_i);
}

std::vector<rad_grid::sphere_point> rad_grid::sphere_points;

geo::octant rad_grid::sphere_point::get_octant() const {
	geo::octant octant;
	const integer ix = nx > 0.0 ? 1 : 0;
	const integer iy = ny > 0.0 ? 1 : 0;
	const integer iz = nz > 0.0 ? 1 : 0;
	return geo::octant(4 * iz + 2 * iy + ix);

}

std::vector<rad_grid::sphere_point> rad_grid::generate_sphere_points(int n_theta) {
	std::vector<sphere_point> points;
	const int n_points = n_theta;
	std::vector<rad_grid::sphere_point> pts;
	const real c0 = std::sqrt(5.0) / 2.0 + 2.5;
	const unsigned N = n_points;
	real phi = 0.0;
	for (unsigned i = 0; i != N; ++i) {
		const unsigned k = i + 1;
		const real h = -1.0 + 2.0 * (k - 1.0) / (N - 1.0);
		const real theta = std::acos(h);
		const real h2 = h * h;
		if (h2 != 1.0) {
			const real dphi = c0 / std::sqrt(real(N)) / std::sqrt(1.0 - h2);
			phi += dphi;
		}
		sphere_point pt;
		pt.nx = std::cos(phi) * std::sin(theta);
		pt.ny = std::sin(phi) * std::sin(theta);
		pt.nz = std::cos(theta);
		pt.dA = 4.0 * M_PI / n_theta;
		pt.dl = std::abs(pt.nx) + std::abs(pt.ny) + std::abs(pt.nz);
		pt.wx = std::abs(pt.nx) / pt.dl;
		pt.wy = std::abs(pt.ny) / pt.dl;
		pt.wz = std::abs(pt.nz) / pt.dl;
		points.push_back(pt);
//		printf("%e %e %e\n", double(h), pt.nx, pt.ny, pt.nz);
	}

	/*	double dtheta = M_PI / double(n_theta);
	 int ntot = 0;
	 int even = 1;
	 int iter = 0;
	 double area_sum = 0.0;
	 std::vector<sphere_point> points;
	 for (double theta = dtheta / 2.0; theta < M_PI + 1.0e-10; theta += dtheta) {
	 printf( "\n\n%e\n", theta);
	 //		const int n_phi = std::max(int(2.0 * sin(theta) * n_theta + 0.5), 1);
	 const int n_phi = 2.0 * n_theta;
	 const double dphi = 2.0 * M_PI / double(n_phi);
	 const double theta_max = std::min(M_PI, theta + dtheta / 2.0);
	 const double theta_min = std::max(0.0, theta - dtheta / 2.0);
	 const double area = -dphi * (cos(theta_max) - cos(theta_min));
	 for (double phi = even * dphi / 2.0; phi < 2.0 * M_PI - dphi / 4.0; phi += dphi) {
	 area_sum += area;
	 sphere_point pt;
	 pt.nx = cos(phi) * sin(theta);
	 pt.ny = sin(phi) * sin(theta);
	 pt.nz = cos(theta);
	 pt.dA = area;
	 pt.dl = std::abs(pt.nx) + std::abs(pt.ny) + std::abs(pt.nz);
	 pt.wx = std::abs(pt.nx) / pt.dl;
	 pt.wy = std::abs(pt.ny) / pt.dl;
	 pt.wz = std::abs(pt.nz) / pt.dl;
	 points.push_back(pt);
	 printf( "%e %e %e | %e \n", pt.nx, pt.ny, pt.nz, area);
	 }
	 ntot += n_phi;
	 printf("%i %e %e\n", n_phi, theta, area);
	 even = even ^ 1;
	 ++iter;
	 }
	 printf("tot = %i %e\n", ntot, area_sum);*/
	return points;
}

void rad_grid::initialize() {

	static std::once_flag flag;
	std::call_once(flag, [&]() {
		sphere_points = generate_sphere_points(NPHI);
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

#endif
