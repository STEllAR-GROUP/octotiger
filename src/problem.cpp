//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/eos.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/lane_emden.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/problem.hpp"

#include <hpx/include/lcos.hpp>

#include <array>
#include <cmath>
#include <vector>

constexpr integer spc_ac_i = spc_i;
constexpr integer spc_ae_i = spc_i + 1;
constexpr integer spc_dc_i = spc_i + 2;
constexpr integer spc_de_i = spc_i + 3;
constexpr integer spc_vac_i = spc_i + 4;

// TODO (daissgr) mutex....
// namespace hpx {
// using mutex = hpx::lcos::local::spinlock;
// }
const Real ssr0 = 1.0 / 3.0;

init_func_type problem = nullptr;
analytic_func_type analytic = nullptr;
refine_test_type refine_test_function = refine_test;

bool radiation_test_refine(integer level, integer max_level, Real x, Real y, Real z, std::vector<Real> U,
		std::array<std::vector<Real>, NDIM> const &dudx) {
	return level < max_level;
	// return refine_blast(level, max_level, x, y, z, U, dudx);
	//
	// bool rc = false;
	// Real den_floor = 1.0e-1;
	// integer test_level = max_level;
	// for (integer this_test_level = test_level; this_test_level >= 1; --this_test_level) {
	// 	if (U[rho_i] > den_floor) {
	// 		rc = rc || (level < this_test_level);
	// 	}
	// 	if (rc) {
	// 		break;
	// 	}
	// 	den_floor /= 8.0;
	// }
	// return rc;

}

std::vector<Real> radiation_test_problem(Real x, Real y, Real z, Real dx) {
//	return blast_wave(x,y,z,dx);

	std::vector<Real> u(opts().n_fields + NRF, Real(0));
	x -= 0.0e11;
	y -= 0.0e11;
	z -= 0.0e11;
	Real r = std::max(2.0 * dx, 0.50);
	Real eint;
	if (x < 0) {
		u[rho_i] = 1.0;
		eint = 1;
		u[opts().n_fields] = 1;
		u[opts().n_fields + 1] = 0.999999;
	} else {
		u[opts().n_fields] = 1e-10;
		u[opts().n_fields + 1] = 0.999999e-10;
		u[rho_i] = 1.0;
		eint = 1;
	}
	u[tau_i] = POWER(eint * u[rho_i], 1.0 / grid::get_fgamma());
//	u[sx_i] = 0.0; //u[rho_i] / 10.0;
	const Real fgamma = grid::get_fgamma();
	u[egas_i] = POWER(u[tau_i], fgamma);
	const Real rhoinv = INVERSE(u[rho_i]);
	u[egas_i] += u[sx_i] * u[sx_i] * rhoinv / 2.0;
	u[egas_i] += u[sy_i] * u[sy_i] * rhoinv / 2.0;
	u[egas_i] += u[sz_i] * u[sz_i] * rhoinv / 2.0;
	u[spc_ac_i] = u[rho_i];
	return u;
}

std::vector<Real> radiation_diffusion_test_problem(Real x, Real y, Real z, Real dx) {
	return radiation_diffusion_analytic(x, y, z, 0);
}

std::vector<Real> radiation_coupling_test_problem(Real x, Real y, Real z, Real dx) {
	std::vector<Real> u(opts().n_fields + NRF, Real(0));
	Real eint;
	u[rho_i] = 1.0;
	u[spc_i] = u[rho_i];

	specie_state_t<> species;
	species[0] = u[rho_i];
	Real mmw;
	Real X;
	Real Z;
	mean_ion_weight(species, mmw, X, Z);

	const double er = (1.0e-10);
	double T = pow(er / (4.0 * physcon().sigma / physcon().c), 0.25);
	T *= 10.0;
	double Pgas = u[rho_i] * T * physcon().kb / (physcon().mh * mmw);
	const Real fgamma = grid::get_fgamma();
	double ei = (1.0 / (fgamma - 1.0)) * Pgas;
	u[tau_i] = POWER(ei, 1.0 / grid::get_fgamma());
	u[egas_i] = POWER(u[tau_i], fgamma);
	double fx, fy, fz;
	fx = fy = fz = 0.0;
	u[opts().n_fields + 0] = er;
	u[opts().n_fields + 1] = fx;
	u[opts().n_fields + 2] = fy;
	u[opts().n_fields + 3] = fz;
//	if( er!= 0.0)
	//printf( "--->%e\n",er);
	return u;

}

std::vector<Real> radiation_diffusion_analytic(Real x, Real y, Real z, Real t) {
//	printf( "%e\n", t);
	//t += 100;
	std::vector<Real> u(opts().n_fields + NRF, Real(0));
	x -= 0.0e11;
	y -= 0.0e11;
	z -= 0.0e11;
	Real eint;
	u[rho_i] = 1.0;
	u[spc_i] = u[rho_i];

	x /= 1e5;
	y /= 1e5;
	z /= 1e5;

	specie_state_t<> species;
	species[0] = u[rho_i];
	Real mmw;
	Real X;
	Real Z;
	mean_ion_weight(species, mmw, X, Z);

	//const double r0 = 0.1;
	const double r2 = x * x + y * y + z * z;
	const double r = sqrt(r2);
	const double D0 = 1.0 / 3.0 * physcon().c / (1e2);
	const double er = 1.0e-6 * std::max(pow(t + 1.0, -1.5) * exp(-r2 / (4.0 * D0 * (t + 1.0))), 1e-10);

//	const Real A = 4.0 * dt * kap_p * sigma * pow(mmw[iiih] * mh * (fgamma - 1.) / (kb * rho[iiih]), 4.0);
//	const Real B = (1.0 + clight * dt * kap_p);
//	const Real C = -(1.0 + clight * dt * kap_p) * e0 - U[er_i][iiir] * dt * clight * kap_p;


	double T = pow(er / (4.0 * physcon().sigma / physcon().c), 0.25);
	double Pgas = u[rho_i] * T * physcon().kb / (physcon().mh * mmw);
	const Real fgamma = grid::get_fgamma();
	double ei = (1.0 / (fgamma - 1.0)) * Pgas;
//printf( "1. %e\n", 4.0 * physcon().sigma * pow(physcon().mh * mmw * (fgamma - 1.) / (physcon().kb * u[rho_i]), 4.0) );
	u[tau_i] = POWER(ei, 1.0 / grid::get_fgamma());
	u[egas_i] = POWER(u[tau_i], fgamma);
	const double derdr = -0.5 * (r) / (1 + t) * er / D0;
	double fx, fy, fz;
	double vx = 1e-3 * physcon().c;
	if (r == 0.0) {
		fx = fy = fz = 0.0;
	} else {
		fx = -derdr * x / r * D0;// + vx * er;
		fy = -derdr * y / r * D0;
		fz = -derdr * z / r * D0;
	}
	//u[egas_i] += 0.5 * vx * vx * u[rho_i];
	//u[sx_i] = u[rho_i] * vx;
	double nx = fx;
	double ny = fy;
	double nz = fz;
	double ninv = 1.0 / sqrt(nx * nx + ny * ny + nz * nz);
	nx *= ninv;
	ny *= ninv;
	nz *= ninv;
//	fx = copysign(std::min(fabs(fx),fabs(0.999*nx*er*physcon().c)), nx);
//	fy = copysign(std::min(fabs(fy),fabs(0.999*ny*er*physcon().c)), ny);
//	fz = copysign(std::min(fabs(fz),fabs(0.999*nz*er*physcon().c)), nz);
//	printf( "%e\n", sqrt(fx*fx+fy*fy+fz*fz)/er);
	u[opts().n_fields + 0] = er;
	u[opts().n_fields + 1] = fx;
	u[opts().n_fields + 2] = fy;
	u[opts().n_fields + 3] = fz;
//	if( er!= 0.0)
	//printf( "--->%e\n",er);
	return u;
}

bool refine_sod(integer level, integer max_level, Real x, Real y, Real z, std::vector<Real> const &U,
		std::array<std::vector<Real>, NDIM> const &dudx) {
	for (integer i = 0; i != NDIM; ++i) {
		if (std::abs(dudx[i][rho_i] / U[rho_i]) >= 0.1) {
			return level < max_level;
		}
	}
	return false;
}

bool refine_blast(integer level, integer max_level, Real x, Real y, Real z, std::vector<Real> const &U,
		std::array<std::vector<Real>, NDIM> const &dudx) {
	bool rc = false;
	if (level < 1) {
		rc = true;
	}
	if (!rc) {
		for (integer i = 0; i != NDIM; ++i) {
			if (std::abs(dudx[i][rho_i]) > 0.1) {
				rc = rc || (level < max_level);
			}
			if (std::abs(dudx[i][tau_i]) > 0.1) {
				rc = rc || (level < max_level);
			}
		}
	}
	return rc;
}

bool refine_test_center(integer level, integer max_level, Real x, Real y, Real z, std::vector<Real> const &U,
		std::array<std::vector<Real>, NDIM> const &dudx) {
	if (x * x + y * y + z * z < ssr0) {
		return level < max_level;
	}
	return false;
}

bool refine_test(integer level, integer max_level, Real x, Real y, Real z, std::vector<Real> const &U,
		std::array<std::vector<Real>, NDIM> const &dudx) {
	bool rc = false;
	Real dx = (opts().xscale / INX) / Real(1 << level);
	if (level < max_level / 2) {
		return std::sqrt(x * x + y * y + z * z) < 10.0 * dx;
	}
	int test_level = max_level;
	bool enuf_core = U[spc_ac_i] + U[spc_dc_i] > 0.25 * U[rho_i];
	bool majority_accretor = U[spc_ae_i] + U[spc_ac_i] > 0.5 * U[rho_i];
	bool majority_donor = U[spc_de_i] + U[spc_dc_i] > 0.5 * U[rho_i];
	if (opts().core_refine) {
		if (!enuf_core) {
			test_level -= 1;
		}
	}
	if (!majority_donor) {
		test_level -= opts().donor_refine;
	}
	if (!majority_accretor) {
		test_level -= opts().accretor_refine;
	}
	const auto grad_rho = opts().grad_rho_refine;
	if( grad_rho > 0.0 ) {
		test_level--;
	}
	Real den_floor = opts().refinement_floor;
	for (integer this_test_level = test_level; this_test_level >= 1; --this_test_level) {
		if (U[rho_i] > den_floor) {
			rc = rc || (level < this_test_level);
		}
		if (rc) {
			break;
		}
		den_floor /= 8.0;
	}
	if(!rc && grad_rho > 0.0 && level < max_level) {
		for( int dim = 0; dim < NDIM; dim++) {
			if( std::abs(dudx[dim][rho_i])/U[rho_i] > grad_rho && U[rho_i] > 1000*den_floor) {
				
				rc = true;
			}
		}
	}

	return rc;
}

bool refine_test_moving_star(integer level, integer max_level, Real x, Real y, Real z, std::vector<Real> const &U,
		std::array<std::vector<Real>, NDIM> const &dudx) {
	bool rc = false;
	Real den_floor = opts().refinement_floor;
	integer test_level = max_level;
	if (x > 0.0 && opts().rotating_star_amr) {
		test_level--;
	}
	for (integer this_test_level = test_level; this_test_level >= 1; --this_test_level) {
		if (U[rho_i] > den_floor) {
			rc = rc || (level < this_test_level);
		}
		if (rc) {
			break;
		}
		den_floor /= 8.0;
	}
	return rc;

}

bool refine_test_marshak(integer level, integer max_level, Real x, Real y, Real z, std::vector<Real> const &U,
		std::array<std::vector<Real>, NDIM> const &dudx) {
	if (level >= max_level) {
		return false;
	} else {
		return true;
	}

}

bool refine_test_unigrid(integer level, integer max_level, Real x, Real y, Real z, std::vector<Real> const &U,
		std::array<std::vector<Real>, NDIM> const &dudx) {
	if (level >= max_level) {
		return false;
	} else {
		return true;
	}

}

void set_refine_test(const refine_test_type &rt) {
	if (opts().unigrid) {
		refine_test_function = refine_test_unigrid;
	} else {
		refine_test_function = rt;
	}
}

refine_test_type get_refine_test() {
	return refine_test_function;
}

void set_problem(const init_func_type &p) {
	problem = p;
}

init_func_type get_problem() {
	return problem;
}

void set_analytic(const init_func_type &p) {
	analytic = p;
}

init_func_type get_analytic() {
	return analytic;
}

/*
 std::vector<Real> null_problem(Real x, Real y, Real z, Real dx) {
 std::vector<Real> u(opts().n_fields, Real(0));
 return u;
 }*/

const Real dxs = 0.0;
const Real dys = -0.0;

std::vector<Real> double_solid_sphere_analytic_phi(Real x0, Real y0, Real z0) {
	std::vector<Real> u(4, Real(0));
	auto u1 = solid_sphere_analytic_phi(x0, y0, z0, dxs);
	auto u2 = solid_sphere_analytic_phi(x0, y0, z0, dys);
	for (integer f = 0; f != 4; ++f) {
		u[f] = u1[f] + u2[f];
	}
	return u;
}

std::vector<Real> solid_sphere_analytic_phi(Real x, Real y, Real z, Real xshift) {
	std::vector<Real> g(4);
	x -= opts().solid_sphere_xcenter;
	y -= opts().solid_sphere_ycenter;
	z -= opts().solid_sphere_zcenter;
	const Real r0 = opts().solid_sphere_radius;
	const Real M = opts().solid_sphere_mass;

	const Real r = std::sqrt(x * x + y * y + z * z);
	const Real r3 = r * r * r;
	const Real Menc = M * std::pow(std::min(r / r0, 1.0), 3);
	if (r < r0) {
		g[phi_i] = -M * (3.0 * r0 * r0 - r * r) / (2.0 * r0 * r0 * r0);
	} else {
		g[phi_i] = -M / r;
	}
	g[gx_i] = -Menc * x / r3;
	g[gy_i] = -Menc * y / r3;
	g[gz_i] = -Menc * z / r3;
	return g;
}

std::vector<Real> double_solid_sphere(Real x0, Real y0, Real z0, Real dx) {
	std::vector<Real> u(opts().n_fields, Real(0));
	auto u1 = solid_sphere(x0, y0, z0, dx, dxs);
	auto u2 = solid_sphere(x0, y0, z0, dx, dys);
	for (integer f = 0; f != opts().n_fields; ++f) {
		u[f] = u1[f] + u2[f];
	}
	return u;
}

std::vector<Real> solid_sphere(Real x0, Real y0, Real z0, Real dx, Real xshift) {
	const integer N = 25;
	const Real r0 = opts().solid_sphere_radius;
	const Real M = opts().solid_sphere_mass;
	const Real V = 4.0 / 3.0 * M_PI * r0 * r0 * r0;
	const Real drho = M / Real(N * N * N) / V;
	std::vector<Real> u(opts().n_fields, Real(0));
	x0 -= opts().solid_sphere_xcenter;
	y0 -= opts().solid_sphere_ycenter;
	z0 -= opts().solid_sphere_zcenter;

	const auto mm = [](Real a, Real b) {
		if (a * b < ZERO) {
			return ZERO;
		} else if (a > ZERO) {
			return std::min(a, b);
		} else {
			return std::max(a, b);
		}
	};
	const Real xmax = std::max(std::abs(x0 + dx / 2.0), std::abs(x0 - dx / 2.0));
	const Real ymax = std::max(std::abs(y0 + dx / 2.0), std::abs(y0 - dx / 2.0));
	const Real zmax = std::max(std::abs(z0 + dx / 2.0), std::abs(z0 - dx / 2.0));
	const Real xmin = mm(x0 + dx / 2.0, x0 - dx / 2.0);
	const Real ymin = mm(y0 + dx / 2.0, y0 - dx / 2.0);
	const Real zmin = mm(z0 + dx / 2.0, z0 - dx / 2.0);
	if (xmax * xmax + ymax * ymax + zmax * zmax <= r0 * r0) {
		u[rho_i] += drho * N * N * N;
	} else if (xmin * xmin + ymin * ymin + zmin * zmin <= r0 * r0) {
		x0 -= dx / 2.0;
		y0 -= dx / 2.0;
		z0 -= dx / 2.0;
		const Real d = dx / N;
		x0 += d / 2.0;
		y0 += d / 2.0;
		z0 += d / 2.0;
		const Real r2 = r0 * r0;
		for (integer i = 0; i < N; ++i) {
			const Real x2 = std::pow(x0 + Real(i) * d, 2);
			for (integer j = 0; j < N; ++j) {
				const Real y2 = std::pow(y0 + Real(j) * d, 2);
#pragma GCC ivdep
				for (integer k = 0; k < N; ++k) {
					const Real z2 = std::pow(z0 + Real(k) * d, 2);
					if (x2 + y2 + z2 < r2) {
						u[rho_i] += drho;
					}
				}
			}
		}
	}
	u[rho_i] = u[spc_i] = std::max(u[rho_i], opts().solid_sphere_rho_min);
	return u;
}

std::vector<Real> star(Real x, Real y, Real z, Real) {
	const Real fgamma = grid::get_fgamma();
	const Real rho_out = opts().star_rho_out;
	std::vector<Real> u(opts().n_fields, Real(0));
	if (opts().eos == WD) {
		const Real r = std::sqrt(x * x + y * y + z * z);
		static struct_eos eos(1.0, 1.0);
		physcon().A = eos.A;
		physcon().B = eos.B();
		normalize_constants();
		const Real rho = std::max(eos.density_at(r, 0.01), rho_out);
		const Real ei = eos.energy(rho);
		u[rho_i] = rho;
		u[egas_i] = ei;
		u[tau_i] = std::pow(std::max(ei - ztwd_energy(rho), 0.0), 1.0 / fgamma);
		u[spc_i] = rho;
		return u;
	} else {

		const Real xshift = opts().star_xcenter;
		const Real yshift = opts().star_ycenter;
		const Real zshift = opts().star_zcenter;
		const Real rho_c = opts().star_rho_center;
		const Real alpha = opts().star_alpha;
		const Real rmax = opts().star_rmax;
		const Real dr = opts().star_dr;
		const Real n = opts().star_n;

		x -= xshift;
		y -= yshift;
		z -= zshift;

		const Real r = std::sqrt(x * x + y * y + z * z);
		Real theta;
		//const Real n = Real(1) / (fgamma - Real(1));
		//const Real rho_min = 1.0e-10;
		const Real theta_min = std::pow(rho_out / rho_c, Real(1) / n);
		const auto c0 = Real(4) * Real(M_PI) * std::pow(alpha, 2) * std::pow(rho_c, (n - Real(1)) / n) / (n + Real(1));
		if (r <= rmax) {
			theta = lane_emden(r / alpha, dr / alpha, n);
			theta = std::max(theta, theta_min);
			u[rho_i] = rho_c * std::pow(theta, n);
		} else {
			theta = theta_min;
			u[rho_i] = rho_out;
		}
//		u[rho_i] = std::max(rho_c * std::pow(theta, n), rho_out);
		u[spc_i] = u[rho_i];
		const Real p = std::pow(rho_c * std::pow(theta, n), (Real(1) + Real(1)/n)) * c0;
		if (opts().eos == IPR) {
			//printf("p(%e) = %e, rho = %e\n", r, p, u[rho_i]);
			u[egas_i] = std::max(opts().star_egas_out, find_ei_rad_gas(p, u[rho_i], opts().atomic_mass[0] / (opts().atomic_number[0] + 1.0), fgamma, u[tau_i])); // makes sure the calculated pressure will be as the polytropic one
			//printf("egas(%e) = %e, epoly=%e\n", r, u[egas_i], n * p);
		} else {
			// u[egas_i] = std::max(opts().star_egas_out, p * n);
			u[egas_i] = std::max(opts().star_egas_out, p / (fgamma - 1.0));
			u[tau_i] = std::pow(u[egas_i], 1.0 / fgamma);
			//u[tau_i] = std::pow(u[egas_i], (n / (Real(1) + n)));
		}

		/*		const Real r = std::sqrt(x * x + y * y + z * z);
		 static struct_eos eos(0.0040083, 0.33593, 3.0, 1.5, 0.1808, 2.0);
		 const Real rho = std::max(eos.density_at(r,0.01),1.0e-10);
		 const Real ei = eos.pressure(rho) / (fgamma - 1.0);
		 u[rho_i] = rho;
		 u[egas_i] = ei;
		 u[tau_i] = std::pow(u[egas_i], (Real(1) / Real(fgamma)));
		 if( rho > eos.dC() ) {
		 u[spc_i+0] = rho;
		 } else  {
		 u[spc_i+1] = rho;
		 }*/
		return u;
	}
}

std::vector<Real> moving_star(Real x, Real y, Real z, Real dx) {
	const Real vx = opts().moving_star_xvelocity;
	const Real vy = opts().moving_star_yvelocity;
	const Real vz = opts().moving_star_zvelocity;
	const Real rho_out = opts().star_rho_out;
	auto u = star(x, y, z, dx);
	if (u[rho_i] > rho_out) {
		u[sx_i] = u[rho_i] * vx;
		u[sy_i] = u[rho_i] * vy;
		u[sz_i] = u[rho_i] * vz;
		u[egas_i] += (u[sx_i] * u[sx_i] + u[sy_i] * u[sy_i] + u[sz_i] * u[sz_i]) / u[rho_i] / 2.0;
		u[spc_i + 1] = ZERO;
	} else {
		u[spc_i] = ZERO;
		u[spc_i + 1] = u[rho_i];
	}
	return u;
}

std::vector<Real> moving_star_analytic(Real x, Real y, Real z, Real t) {
	const Real vx = opts().moving_star_xvelocity;
	const Real vy = opts().moving_star_yvelocity;
	const Real vz = opts().moving_star_zvelocity;
	const Real omega = grid::get_omega();
	const Real x0 = x;
	const Real y0 = y;
	x = x0 * cos(omega * t) - y0 * sin(omega * t);
	y = y0 * cos(omega * t) + x0 * sin(omega * t);
	x -= vx * t;
	y -= vy * t;
	z -= vz * t;
	auto u = star(x, y, z, 0.0);
	u[sx_i] = u[rho_i] * vx;
	u[sy_i] = u[rho_i] * vy;
	u[sz_i] = u[rho_i] * vz;
	u[egas_i] += (u[sx_i] * u[sx_i] + u[sy_i] * u[sy_i] + u[sz_i] * u[sz_i]) / u[rho_i] / 2.0;
	return u;
}

std::vector<Real> equal_mass_binary(Real x, Real y, Real z, Real) {

	const Real rmax = 1.0 / 3.0;
	const Real dr = rmax / 128.0;
	const integer don_i = spc_ac_i;
	const integer acc_i = spc_dc_i;
	const Real fgamma = grid::get_fgamma();

	Real theta;
	Real alpha = 1.0 / 15.0;
	const Real n = Real(1) / (fgamma - Real(1));
	const Real rho_min = 1.0e-12;
	std::vector<Real> u(opts().n_fields, Real(0));
	const Real d = 1.0 / 2.0;
	Real x1 = x - d;
	Real x2 = x + d;
	Real y1 = y;
	Real y2 = y;
	Real z1 = z;
	Real z2 = z;

	const Real r1 = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1) / alpha;
	const Real r2 = std::sqrt(x2 * x2 + y2 * y2 + z2 * z2) / alpha;

	const Real theta_min = std::pow(rho_min, Real(1) / n);
	const auto c0 = Real(4) * Real(M_PI) * alpha * alpha / (n + Real(1));

	if (r1 <= rmax || r2 <= rmax) {
		Real r = std::min(r1, r2);
		theta = lane_emden(r, dr, n);
		theta = std::max(theta, theta_min);
	} else {
		theta = theta_min;
	}
	u[rho_i] = std::pow(theta, n);
	u[egas_i] = std::pow(theta, fgamma * n) * c0 / (fgamma - Real(1));
	u[egas_i] = std::max(u[egas_i], ei_floor);
	u[tau_i] = std::pow(u[egas_i], (Real(1) / Real(fgamma)));
	u[sx_i] = -DEFAULT_OMEGA * y * u[rho_i];
	u[sy_i] = +DEFAULT_OMEGA * x * u[rho_i];
	u[egas_i] += HALF * DEFAULT_OMEGA * DEFAULT_OMEGA * (x * x + y * y) * u[rho_i];
	if (x < ZERO) {
		u[acc_i] = u[rho_i];
		u[don_i] = ZERO;
	} else {
		u[don_i] = u[rho_i];
		u[acc_i] = ZERO;
	}
	return u;
}
