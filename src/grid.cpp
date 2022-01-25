//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/test_problems/exact_sod.hpp"

#include <fenv.h>

#include "octotiger/diagnostics.hpp"
#include "octotiger/future.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/problem.hpp"
#include "octotiger/profiler.hpp"
#include "octotiger/io/silo.hpp"
#include "octotiger/taylor.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/test_problems/amr/amr.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct.hpp"
#include "octotiger/unitiger/hydro_impl/flux.hpp"

#include <hpx/include/runtime.hpp>
#include <hpx/collectives/broadcast_direct.hpp>
#include <hpx/synchronization/once.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <unordered_map>

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"
//#include "octotiger/unitiger/hydro_impl/hydro_cuda_interface.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"

#if !defined(HPX_COMPUTE_DEVICE_CODE)

std::vector<int> grid::field_bw;
std::vector<int> grid::energy_bw;
std::unordered_map<std::string, int> grid::str_to_index_hydro;
std::unordered_map<std::string, int> grid::str_to_index_gravity;
std::unordered_map<int, std::string> grid::index_to_str_hydro;
std::unordered_map<int, std::string> grid::index_to_str_gravity;
double grid::idle_rate;

bool grid::is_hydro_field(const std::string &str) {
	return str_to_index_hydro.find(str) != str_to_index_hydro.end();
}

std::vector<std::pair<std::string, real>> grid::get_outflows() const {
	std::vector<std::pair<std::string, real>> rc;
	rc.reserve(str_to_index_hydro.size());
	for (auto i = str_to_index_hydro.begin(); i != str_to_index_hydro.end(); ++i) {
		rc.push_back(std::make_pair(i->first, U_out[i->second]));
	}
	return std::move(rc);
}

void grid::set_outflows(std::vector<std::pair<std::string, real>> &&u) {
	for (const auto &p : u) {
		U_out[str_to_index_hydro[p.first]] = p.second;
	}
}

void grid::set_outflow(std::pair<std::string, real> &&p) {
	U_out[str_to_index_hydro[p.first]] = p.second;
	U_out[rho_i] = 0.0;
	for (integer s = 0; s < opts().n_species; s++) {
		U_out[rho_i] += U_out[spc_i + s];
	}
}

void grid::static_init() {
	field_bw.resize(opts().n_fields, 3);
	energy_bw.resize(opts().n_fields, 0);
	energy_bw[egas_i] = 1;
	for (int dim = 0; dim < NDIM; dim++) {
		field_bw[lx_i + dim] = 2;
	}

	str_to_index_hydro[std::string("egas")] = egas_i;
	str_to_index_hydro[std::string("tau")] = tau_i;
	for (integer s = 0; s < opts().n_species; s++) {
		str_to_index_hydro[std::string("rho_") + std::to_string(s + 1)] = spc_i + s;
	}
	str_to_index_hydro[std::string("sx")] = sx_i;
	str_to_index_hydro[std::string("sy")] = sy_i;
	str_to_index_hydro[std::string("sz")] = sz_i;
	str_to_index_hydro[std::string("pot")] = pot_i;
	str_to_index_hydro[std::string("lx")] = lx_i;
	str_to_index_hydro[std::string("ly")] = ly_i;
	str_to_index_hydro[std::string("lz")] = lz_i;
	str_to_index_gravity[std::string("gx")] = gx_i;
	str_to_index_gravity[std::string("gy")] = gy_i;
	str_to_index_gravity[std::string("gz")] = gz_i;
	for (const auto &s : str_to_index_hydro) {
		index_to_str_hydro[s.second] = s.first;
	}
	for (const auto &s : str_to_index_gravity) {
		index_to_str_gravity[s.second] = s.first;
	}
	if (opts().radiation) {
		rad_grid::static_init();
	}
}

std::vector<std::string> grid::get_field_names() {
	std::vector<std::string> rc = get_hydro_field_names();
	if (opts().gravity) {
		for (auto i : str_to_index_gravity) {
			rc.push_back(i.first);
		}
	}
	if (opts().radiation) {
		const auto rnames = rad_grid::get_field_names();
		for (auto &n : rnames) {
			rc.push_back(n);
		}
	}
	if (opts().idle_rates) {
		rc.push_back("locality");
		rc.push_back("idle_rate");
	}
//	rc.push_back("roche_lobe");
	return rc;
}

std::vector<std::string> grid::get_hydro_field_names() {
	std::vector<std::string> rc;
//	if (opts().hydro) {
	for (auto i : str_to_index_hydro) {
		rc.push_back(i.first);
	}
//	}
	return rc;
}

void grid::set(const std::string name, real *data, int version) {
	PROFILE();
	auto iter = str_to_index_hydro.find(name);
	real unit = convert_hydro_units(iter->second);

	if (iter != str_to_index_hydro.end()) {
		int f = iter->second;
		int jjj = 0;

		/* Correct for bugfix across versions */
		if (version == 100 && f >= sx_i && f <= sz_i) {
			unit /= opts().code_to_s;
		}
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = hindex(k + H_BW, j + H_BW, i + H_BW);
					U[f][iii] = data[jjj] / unit;
					jjj++;
				}
			}
		}
	} else if (opts().radiation) {
		rad_grid_ptr->set(name, data);
	}

}

void grid::rho_from_species() {
	for (integer iii = 0; iii < H_N3; iii++) {
		U[rho_i][iii] = 0.0;
		for (integer s = 0; s < opts().n_species; s++) {
			U[rho_i][iii] += U[spc_i + s][iii];
		}
	}
}

real grid::convert_hydro_units(int i) {
	real val = 1.0;
	if (opts().problem != MARSHAK) {
		const real cm = opts().code_to_cm;
		//print( "%e\n", cm);
		const real s = opts().code_to_s;
		const real g = opts().code_to_g;
		if (i >= spc_i && i <= spc_i + opts().n_species) {
			val *= g / (cm * cm * cm);
		} else if (i >= sx_i && i <= sz_i) {
			val *= g / (s * cm * cm);
		} else if (i == egas_i || i == pot_i) {
			val *= g / (s * s * cm);
		} else if ((i >= lx_i && i <= lz_i)) {
			val *= g / (s * cm);
		} else if (i == tau_i) {
			val *= POWER(g / (s * s * cm), 1.0 / fgamma);
		} else {
			print("Asked to convert units for unknown field %i\n", i);
			abort();
		}
	}
	return val;
}

real grid::convert_gravity_units(int i) {
	real val = 1.0;
	const real cm = opts().code_to_cm;
	const real s = opts().code_to_s;
	const real g = opts().code_to_g;
	if (i == phi_i) {
		val *= cm * cm / s / s;
	} else {
		val *= cm / s / s;
	}
	return val;
}

std::vector<grid::roche_type> grid::get_roche_lobe() const {
	int jjj = 0;
	std::vector<grid::roche_type> this_s(INX * INX * INX);
	for (int i = 0; i < INX; i++) {
		for (int j = 0; j < INX; j++) {
			for (int k = 0; k < INX; k++) {
				const int iii0 = h0index(k, j, i);
				this_s[jjj] = roche_lobe[iii0];
				jjj++;
			}
		}
	}
	return std::move(this_s);
}

std::string grid::hydro_units_name(const std::string &nm) {
	int f = str_to_index_hydro[nm];
	if (f >= spc_i && f <= spc_i + opts().n_species) {
		return "g / cm^3";
	} else if (f >= sx_i && f <= sz_i) {
		return "g / (cm s)^2 ";
	} else if (f == egas_i || (f >= lx_i && f <= lz_i)) {
		return "g / (cm s^2)";
	} else if (f == tau_i) {
		return "(g / cm)^(3/5) / s^(6/5)";
	}
	return "<unknown>";
}

std::string grid::gravity_units_name(const std::string &nm) {
	int f = str_to_index_gravity[nm];
	if (f == phi_i) {
		return "cm^2 / s^2";
	} else {
		return "cm / s^2";
	}
}

std::vector<silo_var_t> grid::var_data() const {
	std::vector<silo_var_t> s;
	real unit;
//	if (opts().hydro) {
	const auto &x0 = opts().silo_offset_x;
	const auto &y0 = opts().silo_offset_y;
	const auto &z0 = opts().silo_offset_z;
	for (auto l : str_to_index_hydro) {
		unit = convert_hydro_units(l.second);
		const int f = l.second;

		std::string this_name = l.first;
		int jjj = 0;
		silo_var_t this_s(this_name);
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = hindex(k + H_BW - x0, j + H_BW - y0, i + H_BW - z0);
					this_s(jjj) = U[f][iii] * unit;
					this_s.set_range(this_s(jjj));
					jjj++;
				}
			}
		}
		s.push_back(std::move(this_s));
	}
//	}

	if (opts().gravity) {
		for (auto l : str_to_index_gravity) {
			unit = convert_gravity_units(l.second);
			const int f = l.second;
			std::string this_name = l.first;
			int jjj = 0;
			silo_var_t this_s(this_name);
			for (int i = 0; i < INX; i++) {
				for (int j = 0; j < INX; j++) {
					for (int k = 0; k < INX; k++) {
						const int iii = gindex(k, j, i);
						this_s(jjj) = G[iii][f] * unit;
						this_s.set_range(this_s(jjj));
						jjj++;
					}
				}
			}
			s.push_back(std::move(this_s));
		}
	}
	if (opts().radiation) {
		auto rad = rad_grid_ptr->var_data();
		for (auto &r : rad) {
			s.push_back(std::move(r));
		}
	}

	if (opts().idle_rates) {

		const int id = hpx::get_locality_id();
		{
			int jjj = 0;
			silo_var_t this_s("locality");
			for (int i = 0; i < INX; i++) {
				for (int j = 0; j < INX; j++) {
					for (int k = 0; k < INX; k++) {
						this_s(jjj) = id;
						this_s.set_range(this_s(jjj));
						jjj++;
					}
				}
			}
			s.push_back(std::move(this_s));
		}
		{

			int jjj = 0;
			silo_var_t this_s("idle_rate");
			for (int i = 0; i < INX; i++) {
				for (int j = 0; j < INX; j++) {
					for (int k = 0; k < INX; k++) {
						this_s(jjj) = idle_rate;
						this_s.set_range(idle_rate);
						jjj++;
					}
				}
			}
			s.push_back(std::move(this_s));
		}
	}

//	{
//
//		int jjj = 0;
//		silo_var_t this_s("roche_lobe");
//		for (int i = 0; i < INX; i++) {
//			for (int j = 0; j < INX; j++) {
//				for (int k = 0; k < INX; k++) {
//					this_s(jjj) = 	roche_lobe[h0index(i,j,k)];
//					this_s.set_range(this_s(jjj));
//					jjj++;
//				}
//			}
//		}
//		s.push_back(std::move(this_s));
//	}
	return std::move(s);
}

void grid::set_idle_rate() {
	std::string counter_name = "/threads{" + std::to_string(hpx::get_locality_id()) + "/total}/idle-rate";
	hpx::performance_counters::performance_counter count(counter_name);
	idle_rate = count.get_value<double>().get();
	count.reset();
}

// MSVC needs this variable to be in the global namespace
constexpr integer nspec = 2;
diagnostics_t grid::diagnostics(const diagnostics_t &diags) {
	PROFILE();
	diagnostics_t rc;
	if (opts().disable_diagnostics) {
		return rc;
	}
	const real dV = dx * dx * dx;
	real x, y, z;
	integer iii, iiig;

	if (opts().problem != DWD) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				for (integer l = H_BW; l != H_NX - H_BW; ++l) {
					const integer iii = hindex(j, k, l);
					const integer iiig = gindex(j - H_BW, k - H_BW, l - H_BW);
					real ek = ZERO;
					ek += HALF * pow(U[sx_i][iii], 2) * INVERSE(U[rho_i][iii]);
					ek += HALF * pow(U[sy_i][iii], 2) * INVERSE(U[rho_i][iii]);
					ek += HALF * pow(U[sz_i][iii], 2) * INVERSE(U[rho_i][iii]);
					real ei;
					if (opts().eos == WD) {
						ei = U[egas_i][iii] - ek - ztwd_energy(U[rho_i][iii]);
					} else {
						ei = U[egas_i][iii] - ek;
					}
					real et = U[egas_i][iii];
					if (ei < de_switch2 * et) {
						ei = POWER(U[tau_i][iii], fgamma);
					}
					real p = (fgamma - 1.0) * ei;
					if (opts().eos == WD) {
						p += ztwd_pressure(U[rho_i][iii]);
					}
					if (opts().gravity) {
						rc.virial += (2.0 * ek + 0.5 * U[rho_i][iii] * G[iiig][phi_i] + 3.0 * p) * (dx * dx * dx);
						rc.virial_norm += (2.0 * ek - 0.5 * U[rho_i][iii] * G[iiig][phi_i] + 3.0 * p) * (dx * dx * dx);
					} else {
						rc.virial_norm = 1.0;
					}
					for (integer f = 0; f != opts().n_fields; ++f) {
						rc.grid_sum[f] += U[f][iii] * dV;
					}
					rc.grid_sum[egas_i] += 0.5 * U[pot_i][iii] * dV;
					rc.lsum[0] += U[lx_i][iii] * dV - (X[YDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sy_i][iii]) * dV;
					rc.lsum[1] -= U[ly_i][iii] * dV - (X[XDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sx_i][iii]) * dV;
					rc.lsum[2] += U[lz_i][iii] * dV - (X[XDIM][iii] * U[sy_i][iii] - X[YDIM][iii] * U[sx_i][iii]) * dV;
				}
			}
		}
		for (integer f = 0; f != opts().n_fields; ++f) {
			rc.grid_out[f] += U_out[f];
		}
		rc.grid_out[egas_i] += U_out[pot_i];
		return rc;
	}

	const auto is_loc = [this, diags](integer j, integer k, integer l) {
		const integer iii = hindex(j, k, l);
		const real ax = X[XDIM][iii] - diags.com[0][XDIM];
		const real ay = X[YDIM][iii] - diags.com[0][YDIM];
		const real az = X[ZDIM][iii] - diags.com[0][ZDIM];

		const real bx = diags.com[1][XDIM] - diags.com[0][XDIM];
		const real by = diags.com[1][YDIM] - diags.com[0][YDIM];
		const real bz = diags.com[1][ZDIM] - diags.com[0][ZDIM];

		const real aa = (ax * ax + ay * ay + az * az);
		const real bb = (bx * bx + by * by + bz * bz);
		const real ab = (ax * bx + ay * by + az * bz);

		const real d2bb = aa * bb - ab * ab;
		if ((d2bb < dx * dx * bb * 3.0 / 4.0)) {
			if ((ab < bb) && (ab > 0.0)) {
				return 2;
			} else if (ab <= 0.0) {
				return 1;
			} else {
				return 3;
			}
		} else {
			return 0;
		}
	};

	constexpr integer spc_ac_i = spc_i;
	constexpr integer spc_ae_i = spc_i + 1;
	constexpr integer spc_dc_i = spc_i + 2;
	constexpr integer spc_de_i = spc_i + 3;
	constexpr integer spc_vac_i = spc_i + 4;

	const auto in_star = [&](integer j, integer k, integer l) {
		if (opts().problem != DWD) {
			return integer(0);
		}
		const integer iii = hindex(j, k, l);
		const integer iiig = gindex(j - H_BW, k - H_BW, l - H_BW);
		integer rc = 0;
		const real x = X[XDIM][iii] - diags.grid_com[0];
		const real y = X[YDIM][iii] - diags.grid_com[1];
		const real z = X[ZDIM][iii];
		real ax = G[iiig][gx_i] + x * diags.omega * diags.omega;
		real ay = G[iiig][gy_i] + y * diags.omega * diags.omega;
		real az = G[iiig][gz_i];
		real nx, ny, nz;
		const real a = SQRT(ax * ax + ay * ay + az * az);
		if (a > 0.0) {
			nx = ax / a;
			ny = ay / a;
			nz = az / a;
			space_vector dX[nspec];
			real g[nspec] = { 0.0, 0.0 };
			for (integer s = 0; s != nspec; ++s) {
				dX[s][XDIM] = x - diags.com[s][XDIM];
				dX[s][YDIM] = y - diags.com[s][YDIM];
				dX[s][ZDIM] = z - diags.com[s][ZDIM];
			}
			const real x0 = std::sqrt(std::pow(dX[0][XDIM], 2) + std::pow(dX[0][YDIM], 2) + std::pow(dX[0][ZDIM], 2));
			const real x1 = std::sqrt(std::pow(dX[1][XDIM], 2) + std::pow(dX[1][YDIM], 2) + std::pow(dX[1][ZDIM], 2));
			if (x1 > 0.25 * diags.rL[1] && x0 < 0.25 * diags.rL[0] && diags.stage > 1) {
				rc = +1;
			} else if (x0 > 0.25 * diags.rL[0] && x1 < 0.25 * diags.rL[1] && diags.stage > 1) {
				rc = -1;
			} else if (x0 < 0.25 * diags.rL[0] && x1 < 0.25 * diags.rL[1] && diags.stage > 1) {
				rc = x0 < x1 ? +1 : -1;
			} else {
				for (integer s = 0; s != nspec; ++s) {
					const real this_x = s == 0 ? x0 : x1;
					if (this_x == 0.0) {
						rc = 99;
						return rc;
					}
					g[s] += ax * dX[s][XDIM] * INVERSE(this_x);
					g[s] += ay * dX[s][YDIM] * INVERSE(this_x);
					g[s] += az * dX[s][ZDIM] * INVERSE(this_x);
				}
				if (g[0] <= 0.0 && g[1] > 0.0) {
					rc = +1;
				} else if (g[0] > 0.0 && g[1] <= 0.0) {
					rc = -1;
				} else if (g[0] <= 0.0 && g[1] <= 0.0) {
					if (std::abs(g[0]) > std::abs(g[1])) {
						rc = +1;
					} else {
						rc = -1;
					}
				}
			}
		}
		return rc;
	};
	roche_lobe.resize(INX * INX * INX);
	for (integer j = H_BW; j != H_NX - H_BW; ++j) {
		for (integer k = H_BW; k != H_NX - H_BW; ++k) {
			for (integer l = H_BW; l != H_NX - H_BW; ++l) {
				iii = hindex(j, k, l);
				iiig = gindex(j - H_BW, k - H_BW, l - H_BW);
				x = X[XDIM][iii];
				y = X[YDIM][iii];
				z = X[ZDIM][iii];
				const real o2 = diags.omega * diags.omega;
				const real rhoinv = 1.0 * INVERSE(U[rho_i][iii]);
				const real vx = U[sx_i][iii] * INVERSE(U[rho_i][iii]);
				const real vy = U[sy_i][iii] * INVERSE(U[rho_i][iii]);
				const real vz = U[sz_i][iii] * INVERSE(U[rho_i][iii]);
				std::array<real, nspec> rho;
				integer star;
				if (diags.stage < 2) {
					rho = { U[spc_ac_i][iii], U[spc_dc_i][iii] };
				} else {
					star = in_star(j, k, l);
					if (star == +1) {
						rho = { U[rho_i][iii], 0.0 };
					} else if (star == -1) {
						rho = { 0.0, U[rho_i][iii] };
					} else if (star != 99) {
						rho = { 0.0, 0.0 };
					} else {
						rc.failed = true;
						return rc;
					}
				}
				if (diags.stage > 1) {
					const real R2 = x * x + y * y;
					const real phi_g = G[iiig][phi_i];
					if (diags.omega < 0.0) {
						rc.failed = true;
						return rc;
					}
					const safe_real phi_r = -0.5 * POWER(diags.omega, 2) * R2;
					const safe_real phi_eff = phi_g + phi_r;
					const safe_real rho0 = U[rho_i][iii];
					const auto ekin = (pow(U[sx_i][iii], 2) + pow(U[sy_i][iii], 2) + pow(U[sz_i][iii], 2)) / 2.0 / U[rho_i][iii] * dV;
					if (ekin / U[rho_i][iii] / dV + phi_g > 0.0) {
						rc.munbound1 += U[rho_i][iii] * dx * dx * dx;
					}
					if (ekin / U[rho_i][iii] / dV + phi_eff > 0.0) {
						rc.munbound2 += U[rho_i][iii] * dx * dx * dx;
					}
					integer i;
					if (rho[1] > 0.5 * rho0) {
						i = 1;
					} else if (rho[0] > 0.5 * rho0) {
						i = 0;
					} else {
						i = -1;
					}
					if (i != -1) {
						const real dX[NDIM] = { (x - diags.com[i][XDIM]), (y - diags.com[i][YDIM]), (z - diags.com[i][ZDIM]) };
						rc.js[i] += dX[0] * U[sy_i][iii] * dV;
						rc.js[i] -= dX[1] * U[sx_i][iii] * dV;
						rc.lz2[i] += x * U[sy_i][iii] * dV;
						rc.lz2[i] -= y * U[sx_i][iii] * dV;
						rc.lz1[i] += U[lz_i][iii] * dV;
						rc.Ts[i] += dX[0] * G[iiig][gy_i] * dV * rho0;
						rc.Ts[i] -= dX[1] * G[iiig][gx_i] * dV * rho0;
						rc.g[i][0] += G[iiig][gx_i] * dV * rho0;
						rc.g[i][1] += G[iiig][gy_i] * dV * rho0;
						rc.g[i][2] += G[iiig][gz_i] * dV * rho0;
						safe_real eint;
						if (opts().eos == WD) {
							eint = U[egas_i][iii] * dV - ekin - ztwd_energy(rho0) * dV;
						} else {
							eint = U[egas_i][iii] * dV - ekin;
						}
						const auto epot = 0.5 * U[pot_i][iii] * dV;
						if (eint < de_switch2 * U[egas_i][iii] * dV) {
							eint = POWER(U[tau_i][iii], fgamma) * dV;
						}
						rc.ekin[i] += ekin;
						rc.epot[i] += epot;
						rc.eint[i] += eint;
						const real r = SQRT(dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2]);
						for (integer n = 0; n != NDIM; ++n) {
							for (integer m = 0; m <= n; ++m) {
								rc.mom[i](n, m) += 3.0 * dX[n] * dX[m] * rho0 * dV;
							}
							rc.mom[i](n, n) -= r * r * rho0 * dV;
						}
						rc.z_moment[i] += (dX[0] * dX[0] + dX[1] * dX[1]) * rho0 * dV;
						if (phi_eff < diags.l1_phi) {
							rc.roche_vol[i] += dV;
						}
						if (U[rho_i][iii] > 10.0 * rho_floor) {
							rc.stellar_vol[i] += dV;
						}
						rc.rho_max[i] = std::max(rc.rho_max[i], safe_real(rho0));
					}

//					auto &rl = roche_lobe[h0index(j - H_BW, k - H_BW, l - H_BW)];
//
//					auto lmin23 = std::min(diags.l2_phi, diags.l3_phi);
//					auto lmax23 = std::max(diags.l2_phi, diags.l3_phi);
//
//					if (i != -1) {
//						rl = i == 0 ? -1 : +1;
//						const integer s = rl * INVERSE(std::abs(rl));
//
//						if (phi_eff > diags.l1_phi) {
//							rl += s;
//						}
//						if (phi_eff > lmin23) {
//							rl += s;
//						}
//						if (phi_eff > lmax23) {
//							rl += s;
//						}
//					} else {
//						rl = 0;
//					}

					auto loc = is_loc(j, k, l);
					if (loc == 2) {
						rc.l1_phi = std::max(phi_eff, rc.l1_phi);
					} else if (loc == 1) {
						rc.l2_phi = std::max(phi_eff, rc.l2_phi);
					} else if (loc == 3) {
						rc.l3_phi = std::max(phi_eff, rc.l3_phi);
					}
					const integer iii = hindex(j, k, l);
					real ek = ZERO;
					ek += HALF * pow(U[sx_i][iii], 2) * INVERSE(U[rho_i][iii]);
					ek += HALF * pow(U[sy_i][iii], 2) * INVERSE(U[rho_i][iii]);
					ek += HALF * pow(U[sz_i][iii], 2) * INVERSE(U[rho_i][iii]);
					real ei;
					if (opts().eos == WD) {
						ei = U[egas_i][iii] - ek - ztwd_energy(U[rho_i][iii]);
					} else {
						ei = U[egas_i][iii] - ek;
					}
					real et = U[egas_i][iii];
					if (ei < de_switch2 * et) {
						ei = POWER(U[tau_i][iii], fgamma);
					}
					real p = (fgamma - 1.0) * ei;
					if (opts().eos == WD) {
						p += ztwd_pressure(U[rho_i][iii]);
					}
					if (opts().problem == DWD) {
						rc.virial += (2.0 * ek + 0.5 * U[rho_i][iii] * G[iiig][phi_i] + 3.0 * p) * (dx * dx * dx);
						rc.virial_norm += (2.0 * ek - 0.5 * U[rho_i][iii] * G[iiig][phi_i] + 3.0 * p) * (dx * dx * dx);
					}
					for (integer f = 0; f != opts().n_fields; ++f) {
						rc.grid_sum[f] += U[f][iii] * dV;
					}
					rc.grid_sum[egas_i] += 0.5 * U[pot_i][iii] * dV;
					safe_real lz = (X[XDIM][iii] * U[sy_i][iii] - X[YDIM][iii] * U[sx_i][iii]) * dV;
					rc.lsum[0] += U[lx_i][iii] * dV - (X[YDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sy_i][iii]) * dV;
					rc.lsum[1] -= U[ly_i][iii] * dV - (X[XDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sx_i][iii]) * dV;
					rc.lsum[2] += U[lz_i][iii] * dV - lz;
					const auto nonvac = (1.0 - U[spc_i + opts().n_species - 1][iii] / U[rho_i][iii]);
					rc.nonvacj += lz * nonvac;
					rc.nonvacjlz == U[lz_i][iii] * nonvac * dV;
				}

				for (integer s = 0; s != nspec; ++s) {
					rc.m[s] += rho[s] * dV;
					rc.com[s][XDIM] += x * rho[s] * dV;
					rc.com[s][YDIM] += y * rho[s] * dV;
					rc.com[s][ZDIM] += z * rho[s] * dV;
					rc.com_dot[s][XDIM] += vx * rho[s] * dV;
					rc.com_dot[s][YDIM] += vy * rho[s] * dV;
					rc.com_dot[s][ZDIM] += vz * rho[s] * dV;
				}
			}
		}
	}
	for (integer s = 0; s != nspec; ++s) {
		if (rc.m[s] >= std::numeric_limits<double>::min()) {
			const auto tmp = INVERSE(rc.m[s]);
			rc.com[s][XDIM] *= tmp;
			rc.com[s][YDIM] *= tmp;
			rc.com[s][ZDIM] *= tmp;
			rc.com_dot[s][XDIM] *= tmp;
			rc.com_dot[s][YDIM] *= tmp;
			rc.com_dot[s][ZDIM] *= tmp;
		}
	}
	for (integer f = 0; f != opts().n_fields; ++f) {
		rc.grid_out[f] += U_out[f];
	}
	rc.grid_out[egas_i] += U_out[pot_i];

	return rc;
}

hpx::lcos::local::spinlock grid::omega_mtx;
real grid::omega = ZERO;
real grid::scaling_factor = 1.0;

integer grid::min_level = 0;
integer grid::max_level = 0;

space_vector grid::get_cell_center(integer i, integer j, integer k) {
	const integer iii0 = hindex(H_BW, H_BW, H_BW);
	space_vector c;
	c[XDIM] = X[XDIM][iii0] + (i) * dx;
	c[YDIM] = X[XDIM][iii0] + (j) * dx;
	c[ZDIM] = X[XDIM][iii0] + (k) * dx;
	return c;
}

std::vector<real> grid::get_prolong(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub) {
	PROFILE();
	std::vector<real> data;

	integer size = opts().n_fields;
	for (integer dim = 0; dim != NDIM; ++dim) {
		size *= (ub[dim] - lb[dim]);
	}
	data.reserve(size);

	for (integer f = 0; f < opts().n_fields; f++) {
		const auto &u = U[f];
		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
			const real x = (i % 2) ? +1 : -1;
			for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
				const real y = (j % 2) ? +1 : -1;
				for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
					const integer iii = hindex(i / 2, j / 2, k / 2);
					const real z = (k % 2) ? +1 : -1;
					const auto u0 = u[iii];
					real value = u0;
					value += (9. / 64.) * minmod(u[iii + x * H_DNX] - u0, u0 - u[iii - x * H_DNX]);
					value += (9. / 64.) * minmod(u[iii + y * H_DNY] - u0, u0 - u[iii - y * H_DNY]);
					value += (9. / 64.) * minmod(u[iii + z * H_DNZ] - u0, u0 - u[iii - z * H_DNZ]);
					value += (3. / 64.) * minmod(u[iii + x * H_DNX + y * H_DNY] - u0, u0 - u[iii - x * H_DNX - y * H_DNY]);
					value += (3. / 64.) * minmod(u[iii + x * H_DNX + z * H_DNZ] - u0, u0 - u[iii - x * H_DNX - z * H_DNZ]);
					value += (3. / 64.) * minmod(u[iii + y * H_DNY + z * H_DNZ] - u0, u0 - u[iii - y * H_DNY - z * H_DNZ]);
					value += (1. / 64.) * minmod(u[iii + x * H_DNX + y * H_DNY + z * H_DNZ] - u0, u0 - u[iii - x * H_DNX - y * H_DNY - z * H_DNZ]);
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

std::vector<real> grid::get_restrict() const {
	PROFILE();
	integer Size = opts().n_fields * INX * INX * INX / NCHILD + opts().n_fields;
	std::vector<real> data;
	data.reserve(Size);
	for (integer field = 0; field != opts().n_fields; ++field) {
		for (integer i = H_BW; i < H_NX - H_BW; i += 2) {
			for (integer j = H_BW; j < H_NX - H_BW; j += 2) {
				for (integer k = H_BW; k < H_NX - H_BW; k += 2) {
					const integer iii = hindex(i, j, k);
					real pt = ZERO;
					for (integer x = 0; x != 2; ++x) {
						for (integer y = 0; y != 2; ++y) {
							for (integer z = 0; z != 2; ++z) {
								const integer jjj = iii + x * H_DNX + y * H_DNY + z * H_DNZ;
								pt += U[field][jjj];
							}
						}
					}
					pt /= real(NCHILD);
					data.push_back(pt);
				}
			}
		}
	}
	for (integer field = 0; field != opts().n_fields; ++field) {
		data.push_back(U_out[field]);
	}
	return data;
}

void grid::set_restrict(const std::vector<real> &data, const geo::octant &octant) {
	PROFILE();
	integer index = 0;
	const integer i0 = octant.get_side(XDIM) * (INX / 2);
	const integer j0 = octant.get_side(YDIM) * (INX / 2);
	const integer k0 = octant.get_side(ZDIM) * (INX / 2);
	for (integer field = 0; field != opts().n_fields; ++field) {
		for (integer i = H_BW; i != H_NX / 2; ++i) {
			for (integer j = H_BW; j != H_NX / 2; ++j) {
				for (integer k = H_BW; k != H_NX / 2; ++k) {
					const integer iii = (i + i0) * H_DNX + (j + j0) * H_DNY + (k + k0) * H_DNZ;
					auto &v = U[field][iii];
					v = data[index];
					++index;
				}
			}
		}
	}
}

void grid::set_hydro_boundary(const std::vector<real> &data, const geo::direction &dir, bool energy_only) {
	PROFILE();
	std::array<integer, NDIM> lb, ub;
	integer iter = 0;
	const auto &bw = energy_only ? energy_bw : field_bw;

	for (integer field = 0; field != opts().n_fields; ++field) {
		get_boundary_size(lb, ub, dir, OUTER, INX, H_BW, bw[field]);
		auto &Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					Ufield[hindex(i, j, k)] = data[iter];
					++iter;
				}
			}
		}
	}
}

std::vector<real> grid::get_hydro_boundary(const geo::direction &dir, bool energy_only) {
	PROFILE();

	const auto &bw = energy_only ? energy_bw : field_bw;
	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size = 0;

	for (integer field = 0; field != opts().n_fields; ++field) {
		size += get_boundary_size(lb, ub, dir, INNER, INX, H_BW, bw[field]);
	}
	data.resize(size);
	integer iter = 0;
	for (integer field = 0; field != opts().n_fields; ++field) {
		get_boundary_size(lb, ub, dir, INNER, INX, H_BW, bw[field]);
		auto &Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					data[iter] = Ufield[hindex(i, j, k)];
					++iter;
				}
			}
		}
	}
	return data;

}

line_of_centers_t grid::line_of_centers(const std::pair<space_vector, space_vector> &line) {

	line_of_centers_t loc;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i - H_BW, j - H_BW, k - H_BW);
				const auto R2 = std::sqrt(X[XDIM][iii] * X[XDIM][iii] + X[YDIM][iii] * X[YDIM][iii]);
				const real dV = dx * dx * dx;
				/*		for (integer d = 0; d != NDIM; ++d) {
				 loc.core1_s[d] += U[sx_i + d][iii] * U[spc_ac_i][iii]
				 / U[rho_i][iii]* dV;
				 loc.core2_s[d] += U[sx_i + d][iii] * U[spc_dc_i][iii]
				 / U[rho_i][iii]* dV;
				 }
				 loc.core1 += U[spc_ac_i][iii] * dV;
				 loc.core2 += U[spc_dc_i][iii] * dV;*/
				space_vector a = line.first;
				const space_vector &o = 0.0;
				space_vector b;
				real bb = 0.0;
				real ab = 0.0;
				for (integer d = 0; d != NDIM; ++d) {
					//		a[d] -= o[d];
					b[d] = X[d][iii] - o[d];
				}
				for (integer d = 0; d != NDIM; ++d) {
					bb += b[d] * b[d];
					ab += a[d] * b[d];
				}
				const real d = std::sqrt(std::max(bb - ab * ab, 0.0));
				real p = ab;
				std::vector<real> data(opts().n_fields + NGF);
				if (d < std::sqrt(3.0) * dx / 2.0) {
					for (integer ui = 0; ui != opts().n_fields; ++ui) {
						data[ui] = U[ui][iii];
					}
					for (integer gi = 0; gi != NGF; ++gi) {
						data[opts().n_fields + gi] = G[iiig][gi];
					}
					loc.resize(loc.size() + 1);
					loc[loc.size() - 1].first = p;
					loc[loc.size() - 1].second = std::move(data);
				}
			}
		}
	}
	return loc;
}

std::pair<std::vector<real>, std::vector<real>> grid::diagnostic_error() const {

	std::pair<std::vector<real>, std::vector<real>> e;
	const real dV = dx * dx * dx;
	if (opts().problem == SOLID_SPHERE) {
		e.first.resize(8, ZERO);
		e.second.resize(8, ZERO);
	}
	for (integer i = 0; i != G_NX; ++i) {
		for (integer j = 0; j != G_NX; ++j) {
			for (integer k = 0; k != G_NX; ++k) {
				const integer iii = gindex(i, j, k);
				const integer bdif = H_BW;
				const integer iiih = hindex(i + bdif, j + bdif, k + bdif);
				const real x = X[XDIM][iiih];
				const real y = X[YDIM][iiih];
				const real z = X[ZDIM][iiih];
				if (opts().problem == SOLID_SPHERE) {
					const auto a = solid_sphere_analytic_phi(x, y, z, 0.25);
					std::vector<real> n(4);
					n[phi_i] = G[iii][phi_i];
					n[gx_i] = G[iii][gx_i];
					n[gy_i] = G[iii][gy_i];
					n[gz_i] = G[iii][gz_i];
					const real rho = U[rho_i][iiih];
					for (integer l = 0; l != 4; ++l) {
						e.first[l] += std::abs(a[l] - n[l]) * dV * rho;
						e.first[4 + l] += std::abs(a[l]) * dV * rho;
						e.second[l] += sqr((a[l] - n[l]) * rho) * dV;
						e.second[4 + l] += sqr(a[l] * rho) * dV;
					}
				}
			}
		}
	}
//	print("%e\n", e[0]);

	return e;
}

real& grid::get_omega() {
	return omega;
}

void grid::velocity_inc(const space_vector &dv) {

	for (integer iii = 0; iii != H_N3; ++iii) {
		const real rho = U[rho_i][iii];
		if (rho != ZERO) {
			const real rhoinv = ONE / rho;
			safe_real &sx = U[sx_i][iii];
			safe_real &sy = U[sy_i][iii];
			safe_real &sz = U[sz_i][iii];
			safe_real &egas = U[egas_i][iii];
			egas -= HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
			sx += dv[XDIM] * rho;
			sy += dv[YDIM] * rho;
			sz += dv[ZDIM] * rho;
			egas += HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
		}
	}

}

std::vector<real> grid::get_flux_restrict(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub, const geo::dimension &dim) const {
	PROFILE();
	std::vector<real> data;
	integer size = 1;
	for (auto &dim : geo::dimension::full_set()) {
		size *= (ub[dim] - lb[dim]);
	}
	size /= (NCHILD / 2);
	size *= opts().n_fields;
	data.reserve(size);
	const integer stride1 = (dim == XDIM) ? (INX + 1) : (INX + 1) * (INX + 1);
	const integer stride2 = (dim == ZDIM) ? (INX + 1) : 1;
	for (integer field = 0; field != opts().n_fields; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; i += 2) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; j += 2) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; k += 2) {
					const integer i00 = findex(i, j, k);
					const integer i10 = i00 + stride1;
					const integer i01 = i00 + stride2;
					const integer i11 = i00 + stride1 + stride2;
					real value = ZERO;
					value += F[dim][field][i00];
					value += F[dim][field][i10];
					value += F[dim][field][i01];
					value += F[dim][field][i11];
					value /= real(4);
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

void grid::set_flux_restrict(const std::vector<real> &data, const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub,
		const geo::dimension &dim) {
	PROFILE();
	integer index = 0;
	for (integer field = 0; field != opts().n_fields; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					const integer iii = findex(i, j, k);
					F[dim][field][iii] = data[index];
					++index;
				}
			}
		}
	}
}

void grid::set_prolong(const std::vector<real> &data, std::vector<real> &&outflows) {
	PROFILE();
	integer index = 0;
	U_out = std::move(outflows);
	for (integer field = 0; field != opts().n_fields; ++field) {
		for (integer i = H_BW; i != H_NX - H_BW; ++i) {
			for (integer j = H_BW; j != H_NX - H_BW; ++j) {
				for (integer k = H_BW; k != H_NX - H_BW; ++k) {
					const integer iii = hindex(i, j, k);
					auto &value = U[field][iii];
					value = data[index];
					++index;
				}
			}
		}
	}
	for (int i = H_BW; i < H_NX - H_BW; i += 2) {
		for (int j = H_BW; j < H_NX - H_BW; j += 2) {
			for (int k = H_BW; k < H_NX - H_BW; k += 2) {
				double zx = 0.0, zy = 0.0, zz = 0.0;
				for (int i1 = 0; i1 < 2; i1++) {
					for (int j1 = 0; j1 < 2; j1++) {
						for (int k1 = 0; k1 < 2; k1++) {
							const int iii = hindex(i + i1, j + j1, k + k1);
							U[lx_i][iii] -= X[YDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sy_i][iii];
							U[ly_i][iii] += X[XDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sx_i][iii];
							U[lz_i][iii] -= X[XDIM][iii] * U[sy_i][iii] - X[YDIM][iii] * U[sx_i][iii];
						}
					}
				}
				for (int i1 = 0; i1 < 2; i1++) {
					for (int j1 = 0; j1 < 2; j1++) {
						for (int k1 = 0; k1 < 2; k1++) {
							const int iii = hindex(i + i1, j + j1, k + k1);
							zx += U[lx_i][iii] / 8.0;
							zy += U[ly_i][iii] / 8.0;
							zz += U[lz_i][iii] / 8.0;
						}
					}
				}
				for (int i1 = 0; i1 < 2; i1++) {
					for (int j1 = 0; j1 < 2; j1++) {
						for (int k1 = 0; k1 < 2; k1++) {
							const int iii = hindex(i + i1, j + j1, k + k1);
							U[lx_i][iii] = zx;
							U[ly_i][iii] = zy;
							U[lz_i][iii] = zz;
						}
					}
				}

				for (int i1 = 0; i1 < 2; i1++) {
					for (int j1 = 0; j1 < 2; j1++) {
						for (int k1 = 0; k1 < 2; k1++) {
							const int iii = hindex(i + i1, j + j1, k + k1);
							U[lx_i][iii] += X[YDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sy_i][iii];
							U[ly_i][iii] -= X[XDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sx_i][iii];
							U[lz_i][iii] += X[XDIM][iii] * U[sy_i][iii] - X[YDIM][iii] * U[sx_i][iii];
						}
					}
				}
			}
		}
	}

}

std::pair<std::vector<real>, std::vector<real> > grid::field_range() const {

	std::pair<std::vector<real>, std::vector<real> > minmax;
	minmax.first.resize(opts().n_fields);
	minmax.second.resize(opts().n_fields);
	for (integer field = 0; field != opts().n_fields; ++field) {
		minmax.first[field] = +std::numeric_limits<real>::max();
		minmax.second[field] = -std::numeric_limits<real>::max();
	}
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				for (integer field = 0; field != opts().n_fields; ++field) {
					minmax.first[field] = std::min(minmax.first[field], (double) U[field][iii]);
					minmax.second[field] = std::max(minmax.second[field], (double) U[field][iii]);
				}
			}
		}
	}
	return minmax;
}

void grid::change_units(real m, real l, real t, real k) {
	const real l2 = l * l;
	const real t2 = t * t;
	const real t2inv = 1.0 / t2;
	const real tinv = 1.0 / t;
	const real l3 = l2 * l;
	const real l3inv = 1.0 / l3;
	xmin[XDIM] *= l;
	xmin[YDIM] *= l;
	xmin[ZDIM] *= l;
	dx *= l;
	if (dx > 1.0e+12)
		print("++++++!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1+++++++++++++++++++++++++++++++++++++ %e %e\n", dx, dx * l);
	for (integer i = 0; i != H_N3; ++i) {
		U[rho_i][i] *= m * l3inv;
		for (integer si = 0; si != opts().n_species; ++si) {
			U[spc_i + si][i] *= m * l3inv;
		}
		U[egas_i][i] *= (m * l2 * t2inv) * l3inv;
		U[tau_i][i] *= std::pow(m * l2 * t2inv * l3inv, 1.0 / fgamma);
		U[pot_i][i] *= (m * l2 * t2inv) * l3inv;
		U[sx_i][i] *= (m * l * tinv) * l3inv;
		U[sy_i][i] *= (m * l * tinv) * l3inv;
		U[sz_i][i] *= (m * l * tinv) * l3inv;
		U[lx_i][i] *= (m * l2 * tinv) * l3inv;
		U[ly_i][i] *= (m * l2 * tinv) * l3inv;
		U[lz_i][i] *= (m * l2 * tinv) * l3inv;
		X[XDIM][i] *= l;
		X[YDIM][i] *= l;
		X[ZDIM][i] *= l;
//		if (std::abs(X[XDIM][i]) > 1.0e+12) {
//			print("!!!!!!!!!!!! %e !!!!!!!!!!!!!!!!\n", std::abs(X[XDIM][i]));
//		}
	}
	for (integer i = 0; i != INX * INX * INX; ++i) {
		G[i][phi_i] *= l2 * t2inv;
		G[i][gx_i] *= l2 * tinv;
		G[i][gy_i] *= l2 * tinv;
		G[i][gz_i] *= l2 * tinv;
	}
	if (opts().radiation) {
		rad_grid_ptr->change_units(m, l, t, k);
	}
}

HPX_PLAIN_ACTION(grid::set_omega, set_omega_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION (set_omega_action);
HPX_REGISTER_BROADCAST_ACTION (set_omega_action);

void grid::set_omega(real omega, bool bcast) {
	if (bcast) {
		if (hpx::get_locality_id() == 0 && options::all_localities.size() > 1) {
			std::vector<hpx::id_type> remotes;
			remotes.reserve(options::all_localities.size() - 1);
			for (hpx::id_type const &id : options::all_localities) {
				if (id != hpx::find_here()) {
					remotes.push_back(id);
				}
			}
			if (remotes.size() > 0) {
				hpx::lcos::broadcast < set_omega_action > (remotes, omega, false).get();
			}
		}
	}
	std::unique_lock<hpx::lcos::local::spinlock> l(grid::omega_mtx, std::try_to_lock);
// if someone else has the lock, it's fine, we just return and have it set
// by the other thread
	if (!l)
		return;
	grid::omega = omega;
}

real grid::roche_volume(const std::pair<space_vector, space_vector> &axis, const std::pair<real, real> &l1, real cx, bool donor) const {

	const real dV = dx * dx * dx;
	real V = 0.0;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer D = 0 - H_BW;
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				real x0 = X[XDIM][iii];
				real x = x0 - cx;
				real y = X[YDIM][iii];
				real z = X[ZDIM][iii];
				const real R = std::sqrt(x0 * x0 + y * y);
				real phi_eff = G[iiig][phi_i] - 0.5 * sqr(omega * R);
				//	real factor = axis.first[0] == l1.first ? 0.5 : 1.0;
				if ((x0 <= l1.first && !donor) || (x0 >= l1.first && donor)) {
					if (phi_eff <= l1.second) {
						const real fx = G[iiig][gx_i] + x0 * sqr(omega);
						const real fy = G[iiig][gy_i] + y * sqr(omega);
						const real fz = G[iiig][gz_i];
						real g = x * fx + y * fy + z * fz;
						if (g <= 0.0) {
							V += dV;
						}
					}
				}
			}
		}
	}
	return V;
}

std::vector<real> grid::frac_volumes() const {

	std::vector<real> V(opts().n_species, 0.0);
	const real dV = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				for (integer si = 0; si != opts().n_species; ++si) {
					if (U[spc_i + si][iii] > 1.0e-5) {
						V[si] += (U[spc_i + si][iii] / U[rho_i][iii]) * dV;
					}
				}
			}
		}
	}
//	print( "%e", V[0]);

	return V;
}

bool grid::is_in_star(const std::pair<space_vector, space_vector> &axis, const std::pair<real, real> &l1, integer frac, integer iii, real rho_cut) const {
	bool use = false;
	if (frac == 0) {
		use = true;
	} else {
		if (U[rho_i][iii] < rho_cut) {
			use = false;
		} else {
			space_vector a = axis.first;
			const space_vector &o = axis.second;
			space_vector b;
			real ab = 0.0;
			for (integer d = 0; d != NDIM; ++d) {
				b[d] = X[d][iii] - o[d];
			}
			for (integer d = 0; d != NDIM; ++d) {
				ab += a[d] * b[d];
			}
			real p = ab;
			if (p < l1.first && frac == +1) {
				use = true;
			} else if (p >= l1.first && frac == -1) {
				use = true;
			}
		}
	}
	return use;
}

real grid::z_moments(const std::pair<space_vector, space_vector> &axis, const std::pair<real, real> &l1, integer frac, real rho_cut) const {

	real mom = 0.0;
	const real dV = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				if (is_in_star(axis, l1, frac, iii, rho_cut)) {
					mom += (sqr(X[XDIM][iii]) + sqr(dx) / 6.0) * U[rho_i][iii] * dV;
					mom += (sqr(X[YDIM][iii]) + sqr(dx) / 6.0) * U[rho_i][iii] * dV;
				}
			}
		}
	}
	return mom;
}

std::vector<real> grid::conserved_sums(space_vector &com, space_vector &com_dot, const std::pair<space_vector, space_vector> &axis,
		const std::pair<real, real> &l1, integer frac, real rho_cut) const {

	std::vector<real> sum(opts().n_fields, ZERO);
	com[0] = com[1] = com[2] = 0.0;
	com_dot[0] = com_dot[1] = com_dot[2] = 0.0;
	const real dV = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				if (is_in_star(axis, l1, frac, iii, rho_cut)) {
					com[0] += X[XDIM][iii] * U[rho_i][iii] * dV;
					com[1] += X[YDIM][iii] * U[rho_i][iii] * dV;
					com[2] += X[ZDIM][iii] * U[rho_i][iii] * dV;
					com_dot[0] += U[sx_i][iii] * dV;
					com_dot[1] += U[sy_i][iii] * dV;
					com_dot[2] += U[sz_i][iii] * dV;
					for (integer field = 0; field != opts().n_fields; ++field) {
						sum[field] += U[field][iii] * dV;
					}
					if (opts().gravity) {
						sum[egas_i] += U[pot_i][iii] * HALF * dV;
					}
				}
			}
		}
	}
	if (sum[rho_i] > 0.0) {
		for (integer d = 0; d != NDIM; ++d) {
			com[d] /= sum[rho_i];
			com_dot[d] /= sum[rho_i];
		}
	}
	return sum;
}

std::vector<real> grid::gforce_sum(bool torque) const {

	std::vector<real> sum(NDIM, ZERO);
	const real dV = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const auto D = 0 - H_BW;
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				const real &rho = U[rho_i][iii];
				const real x = X[XDIM][iii];
				const real y = X[YDIM][iii];
				const real z = X[ZDIM][iii];
				const real fx = rho * G[iiig][gx_i] * dV;
				const real fy = rho * G[iiig][gy_i] * dV;
				const real fz = rho * G[iiig][gz_i] * dV;
				if (!torque) {
					sum[XDIM] += fx;
					sum[YDIM] += fy;
					sum[ZDIM] += fz;
				} else {
					sum[XDIM] -= z * fy - y * fz;
					sum[YDIM] += z * fx - x * fz;
					sum[ZDIM] -= y * fx - x * fy;
				}
			}
		}
	}
	return sum;
}

std::vector<real> grid::l_sums() const {

	std::vector<real> sum(NDIM);
	const real dV = dx * dx * dx;
	std::fill(sum.begin(), sum.end(), ZERO);
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				sum[XDIM] += X[YDIM][iii] * U[sz_i][iii] * dV;
				sum[XDIM] -= X[ZDIM][iii] * U[sy_i][iii] * dV;

				sum[YDIM] -= X[XDIM][iii] * U[sz_i][iii] * dV;
				sum[YDIM] += X[ZDIM][iii] * U[sx_i][iii] * dV;

				sum[ZDIM] += X[XDIM][iii] * U[sy_i][iii] * dV;
				sum[ZDIM] -= X[YDIM][iii] * U[sx_i][iii] * dV;

			}
		}
	}
	return sum;
}

bool grid::refine_me(integer lev, integer last_ngrids) const {
	PROFILE();

	auto test = get_refine_test();
	if (lev < min_level) {

		return true;
	}
	bool rc = false;
	std::vector<real> state(opts().n_fields);
	std::array<std::vector<real>, NDIM> dud;
	std::vector<real> &dudx = dud[0];
	std::vector<real> &dudy = dud[1];
	std::vector<real> &dudz = dud[2];
	dudx.resize(opts().n_fields);
	dudy.resize(opts().n_fields);
	dudz.resize(opts().n_fields);
	for (integer i = H_BW - REFINE_BW; i != H_NX - H_BW + REFINE_BW; ++i) {
		for (integer j = H_BW - REFINE_BW; j != H_NX - H_BW + REFINE_BW; ++j) {
			for (integer k = H_BW - REFINE_BW; k != H_NX - H_BW + REFINE_BW; ++k) {
				int cnt = 0;
				if (i < H_BW || i >= H_NX - H_BW) {
					++cnt;
				}
				if (j < H_BW || j >= H_NX - H_BW) {
					++cnt;
				}
				if (k < H_BW || k >= H_NX - H_BW) {
					++cnt;
				}
				if (cnt > 1) {
					continue;
				}
				const integer iii = hindex(i, j, k);
				for (integer i = 0; i != opts().n_fields; ++i) {
					state[i] = U[i][iii];
					dudx[i] = (U[i][iii + H_DNX] - U[i][iii - H_DNX]) / 2.0;
					dudy[i] = (U[i][iii + H_DNY] - U[i][iii - H_DNY]) / 2.0;
					dudz[i] = (U[i][iii + H_DNZ] - U[i][iii - H_DNZ]) / 2.0;
				}
				if (test(lev, max_level, X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii], state, dud)) {
					rc = true;
					break;
				}

			}
			if (rc) {
				break;
			}
		}
		if (rc) {
			break;
		}
	}
	return rc;
}

void grid::rho_mult(real f0, real f1) {
	for (integer i = 0; i != H_NX; ++i) {
		for (integer j = 0; j != H_NX; ++j) {
			for (integer k = 0; k != H_NX; ++k) {

				constexpr integer spc_ac_i = spc_i;
				constexpr integer spc_ae_i = spc_i + 1;
				constexpr integer spc_dc_i = spc_i + 2;
				constexpr integer spc_de_i = spc_i + 3;

				U[spc_ac_i][hindex(i, j, k)] *= f0;
				U[spc_dc_i][hindex(i, j, k)] *= f1;
				U[spc_ae_i][hindex(i, j, k)] *= f0;
				U[spc_de_i][hindex(i, j, k)] *= f1;
				U[rho_i][hindex(i, j, k)] = 0.0;
				for (integer si = 0; si != opts().n_species; ++si) {
					U[rho_i][hindex(i, j, k)] += U[spc_i + si][hindex(i, j, k)];
				}
			}
		}
	}

}

void grid::rho_move(real x) {
	real w = x / dx;
	U0 = U;

	w = std::max(-0.5, std::min(0.5, w));
	for (integer i = 1; i != H_NX - 1; ++i) {
		for (integer j = 1; j != H_NX - 1; ++j) {
			for (integer k = 1; k != H_NX - 1; ++k) {
				for (integer si = spc_i; si != opts().n_species + spc_i; ++si) {
					U[si][hindex(i, j, k)] += w * U0[si][hindex(i + 1, j, k)];
					U[si][hindex(i, j, k)] -= w * U0[si][hindex(i - 1, j, k)];
					U[si][hindex(i, j, k)] = std::max((double) U[si][hindex(i, j, k)], 0.0);
				}
				U[rho_i][hindex(i, j, k)] = 0.0;
				for (integer si = 0; si != opts().n_species; ++si) {
					U[rho_i][hindex(i, j, k)] += U[spc_i + si][hindex(i, j, k)];
				}
				U[rho_i][hindex(i, j, k)] = std::max((double) U[rho_i][hindex(i, j, k)], rho_floor);
			}
		}
	}
}
/*
 space_vector& grid::center_of_mass_value(integer i, integer j, integer k) {
 return com[0][gindex(i, j, k)];
 }

 const space_vector& grid::center_of_mass_value(integer i, integer j, integer k) const {
 return com[0][gindex(i, j, k)];
 }*/

space_vector grid::center_of_mass() const {
	auto &M = *M_ptr;
	auto &mon = *mon_ptr;

	space_vector this_com;
	this_com[0] = this_com[1] = this_com[2] = ZERO;
	real m = ZERO;
	auto &com0 = *(com_ptr)[0];
	for (integer i = 0; i != INX + 0; ++i) {
		for (integer j = 0; j != INX + 0; ++j) {
			for (integer k = 0; k != INX + 0; ++k) {
				const integer iii = gindex(i, j, k);
				const real this_m = is_leaf ? mon[iii] : M[iii]();
				for (auto &dim : geo::dimension::full_set()) {
					this_com[dim] += this_m * com0[iii][dim];
				}
				m += this_m;
			}
		}
	}
	if (m != ZERO) {
		for (auto &dim : geo::dimension::full_set()) {
			this_com[dim] /= m;
		}
	}
//	print( "kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk %e %e %e\n", this_com[0], this_com[1], this_com[2] );
	return this_com;
}

grid::grid(real _dx, std::array<real, NDIM> _xmin) :
		is_coarse(H_N3), has_coarse(H_N3), Ushad(opts().n_fields), U(opts().n_fields), U0(opts().n_fields), dUdt(opts().n_fields), F(NDIM), X(NDIM), G(NGF), is_root(
				false), is_leaf(true) {
	dx = _dx;
	xmin = _xmin;
	allocate();

}

real grid::fgamma = 5.0 / 3.0;

void grid::set_coordinates() {

	for (integer i = 0; i != H_NX; ++i) {
		for (integer j = 0; j != H_NX; ++j) {
			for (integer k = 0; k != H_NX; ++k) {
				const integer iii = hindex(i, j, k);
				X[XDIM][iii] = (real(i - H_BW) + HALF) * dx + xmin[XDIM];
				X[YDIM][iii] = (real(j - H_BW) + HALF) * dx + xmin[YDIM];
				X[ZDIM][iii] = (real(k - H_BW) + HALF) * dx + xmin[ZDIM];
			}
		}
	}
}

std::vector<std::pair<std::string, std::string>> grid::get_scalar_expressions() {
	std::vector<std::pair<std::string, std::string>> rc;
	std::string rho;
	for (integer i = 0; i < opts().n_species; i++) {
		rho += "rho_" + std::to_string(i + 1) + " + ";
	}
	rho += '0';
	rc.push_back(std::make_pair(std::string("rho"), std::move(rho)));
	rc.push_back(std::make_pair(std::string("vx"), hpx::util::format("sx / rho + {:e} * coord(quadmesh)[1]", omega)));
	rc.push_back(std::make_pair(std::string("vy"), hpx::util::format("sy / rho - {:e} * coord(quadmesh)[0]", omega)));
	rc.push_back(std::make_pair(std::string("vz"), std::string("sz / rho")));
	rc.push_back(std::make_pair(std::string("zx"), "lx - coord(quadmesh)[1]*sz + coord(quadmesh)[2]*sy"));
	rc.push_back(std::make_pair(std::string("zy"), "ly + coord(quadmesh)[0]*sz - coord(quadmesh)[2]*sx"));
	rc.push_back(std::make_pair(std::string("zz"), "lz - coord(quadmesh)[0]*sy + coord(quadmesh)[1]*sx"));

	std::string n;
	std::string X = "(";
	std::string Z = "(";

	for (integer i = 0; i < opts().n_species; i++) {
		const real mu = opts().atomic_mass[i] / (opts().atomic_number[i] + 1.);
		n += hpx::util::format("rho_{} / {:e} + ", int(i + 1), mu * physcon().mh * opts().code_to_g);
		X += hpx::util::format("{:e} * rho_{} + ", opts().X[i], i + 1);
		Z += hpx::util::format("{:e} * rho_{} + ", opts().Z[i], i + 1);
	}
	n += '0';
	X += "0) / rho";
	Z += "0) / rho";
	rc.push_back(std::make_pair(std::string("sigma_T"), std::string("(1 + X) * 0.2 * T * T / ((T * T + 2.7e+11 * rho) * (1 + (T / 4.5e+8)^0.86))")));
	rc.push_back(std::make_pair(std::string("sigma_xf"), std::string("4e+25*(1+X)*(Z+0.001)*rho*(T^(-3.5))")));
	rc.push_back(std::make_pair(std::string("mfp"), std::string("1 / kappa_R")));
	if (opts().problem == MARSHAK) {
		rc.push_back(std::make_pair(std::string("kappa_R"), std::string("rho")));
		rc.push_back(std::make_pair(std::string("kappa_P"), std::string("rho")));
	} else {
		rc.push_back(std::make_pair(std::string("kappa_R"), std::string("rho * (sigma_xf + sigma_T)")));
		rc.push_back(std::make_pair(std::string("kappa_P"), std::string("rho * 30.262 * sigma_xf")));
	}
	rc.push_back(std::make_pair(std::string("n"), std::move(n)));
	rc.push_back(std::make_pair(std::string("X"), std::move(X)));
	rc.push_back(std::make_pair(std::string("Y"), std::string("1.0 - X - Z")));
	rc.push_back(std::make_pair(std::string("Z"), std::move(Z)));
	rc.push_back(std::make_pair(std::string("etot_dual"), std::string("ei + ek")));
	rc.push_back(std::make_pair(std::string("ek"), std::string("(sx*sx+sy*sy+sz*sz)/2.0/rho")));
	const auto kb = physcon().kb * std::pow(opts().code_to_cm / opts().code_to_s, 2) * opts().code_to_g;
	rc.push_back(std::make_pair(std::string("phi"), std::string("pot/rho")));
	rc.push_back(
			std::make_pair(std::string("B_p"), hpx::util::format("{:e} * T^4", physcon().sigma / M_PI * opts().code_to_g * std::pow(opts().code_to_cm, 3))));
	if (opts().eos == WD) {
		rc.push_back(std::make_pair(std::string("A"), "6.00228e+22"));
		rc.push_back(std::make_pair(std::string("B"), "(2 * 9.81011e+5)"));
		rc.push_back(std::make_pair(std::string("x"), "(rho/B)^(1.0/3.0)"));
		rc.push_back(std::make_pair(std::string("Pdeg"), "if( gt(x, 0.001), A*(x*(2.0*x*x-3.0)*sqrt(x*x+1.0)+3.0*ln(x+sqrt(x*x+1))), 1.6*A*x^5)"));
		rc.push_back(std::make_pair(std::string("hdeg"), "8.0*A/B*(sqrt(x*x+1)-1)"));
		rc.push_back(std::make_pair(std::string("Edeg"), "if( gt(x, 0.001), rho*hdeg - Pdeg, 2.4*A*x^5"));
	}
	if (opts().problem == MARSHAK) {
		rc.push_back(std::make_pair(std::string("T"), std::string("(ei/rho)^(1.0/3.0)")));
	} else if (opts().eos != WD) {
		rc.push_back(std::make_pair(std::string("ei"), hpx::util::format("if( gt(egas-ek,{:e}*egas), egas-ek, tau^{:e})", opts().dual_energy_sw1, fgamma)));
		rc.push_back(std::make_pair(std::string("P"), hpx::util::format("{:e} * ei", (fgamma - 1.0))));
		rc.push_back(std::make_pair(std::string("T"), hpx::util::format("{:e} * ei / n", 1.0 / (kb / (fgamma - 1.0)))));
	} else {
		rc.push_back(
				std::make_pair(std::string("ei"),
						hpx::util::format("if( gt(egas-ek-Edeg,{:e}*egas), egas-ek-Edeg, tau^{:e})", opts().dual_energy_sw1, fgamma)));
		rc.push_back(std::make_pair(std::string("P"), hpx::util::format("Pdeg + {:e} * ei", (fgamma - 1.0))));
		rc.push_back(std::make_pair(std::string("T"), hpx::util::format("{:e} * ei / n", 1.0 / (kb / (fgamma - 1.0)))));
	}
	return std::move(rc);
}

std::vector<std::pair<std::string, std::string>> grid::get_vector_expressions() {
	std::vector<std::pair<std::string, std::string>> rc;
	rc.push_back(std::make_pair(std::string("s"), std::string("{sx,sy,sz}")));
	rc.push_back(std::make_pair(std::string("v"), std::string("{vx,vy,vz}")));
	return std::move(rc);
}

analytic_t grid::compute_analytic(real t) {
	analytic_t a;
	if (opts().hydro) {
		a = analytic_t(opts().n_fields);
	} else {
		a = analytic_t(opts().n_fields + NRF);
	}
	const auto func = get_analytic();
	const real dv = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i)
		for (integer j = H_BW; j != H_NX - H_BW; ++j)
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				auto A = func(X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii], t);
				const auto nrho = U[rho_i][iii];
				for (int M = 2; M <= INX; M *= 2) {
					auto last_rho = A[rho_i];
					if (last_rho == nrho) {
						break;
					}
					for (int f = 0; f < opts().n_fields; f++) {
						A[f] = 0.0;
					}
					for (int i0 = 0; i0 < M; i0++) {
						const auto x = X[XDIM][iii] + ((real(i0) + 0.5) / real(M) - 0.5) * dx;
						for (int j0 = 0; j0 < M; j0++) {
							const auto y = X[YDIM][iii] + ((real(j0) + 0.5) / real(M) - 0.5) * dx;
							for (int k0 = 0; k0 < M; k0++) {
								const auto z = X[ZDIM][iii] + ((real(k0) + 0.5) / real(M) - 0.5) * dx;
								const auto a = func(x, y, z, t);
								for (int f0 = 0; f0 < opts().n_fields; f0++) {
									A[f0] += a[f0] / (M * M * M);
								}
							}
						}
					}
					const auto this_rho = A[rho_i];
					const auto err = std::abs(std::abs((nrho - this_rho) / (nrho - last_rho)) - 1.0);
					if (M > INX) {
						print("%i %e\n", M, err);
					}
					if (err < 0.1) {
						break;
					}
				}
				for (integer field = 0; field != opts().n_fields; ++field) {
					real dif = std::abs(A[field] - U[field][iii]);
					a.l1[field] += dif * dv;
					a.l2[field] += dif * dif * dv;
					a.linf[field] = std::max(dif, a.linf[field]);
					U[field][iii] = A[field];
				}
				if (opts().radiation) {
					for (integer field = opts().n_fields; field != opts().n_fields + NRF; ++field) {
						auto tmp = rad_grid_ptr->get_field(field - opts().n_fields, i - H_BW + R_BW, j - H_BW + R_BW, k - H_BW + R_BW);
						real dif = std::abs(A[field] - tmp);
						a.l1[field] += dif * dv;
						a.l2[field] += dif * dif * dv;
						rad_grid_ptr->set_field(A[field], field - opts().n_fields, i - H_BW + R_BW, j - H_BW + R_BW, k - H_BW + R_BW);
					}
				}
				if (opts().problem == SOLID_SPHERE) {
					const auto a = solid_sphere_analytic_phi(X[0][iii], X[1][iii], X[2][iii], 0.25);
					for (int f = 0; f < 4; f++) {
						G[gindex(i - H_BW, j - H_BW, k - H_BW)][f] = a[f];
					}
					U[pot_i][hindex(i, j, k)] = a[0] * U[rho_i][hindex(i, j, k)];
				}
			}
	return a;
}

void grid::allocate() {

	if (opts().radiation) {
		rad_grid_ptr = std::make_shared<rad_grid>();
		rad_grid_ptr->set_dx(dx);
	}
	U_out0 = std::vector<real>(opts().n_fields, ZERO);
	U_out = std::vector<real>(opts().n_fields, ZERO);
	dphi_dt = std::vector<real>(INX * INX * INX);
	G.resize(G_N3);
	for (integer dim = 0; dim != NDIM; ++dim) {
		X[dim].resize(H_N3);
	}

	for (integer field = 0; field != opts().n_fields; ++field) {
		U0[field].resize(INX * INX * INX);

		U[field].resize(H_N3, 0.0);
		Ushad[field].resize(HS_N3, 1.0);
		dUdt[field].resize(INX * INX * INX);
		for (integer dim = 0; dim != NDIM; ++dim) {
			F[dim][field].resize(F_N3);
		}
	}
	L.resize(G_N3);
	L_c.resize(G_N3);
	integer nlevel = 0;
	com_ptr.resize(2);

	set_coordinates();

#ifdef OCTOTIGER_HAVE_GRAV_PAR
	L_mtx.reset(new hpx::lcos::local::spinlock);
#endif

}

grid::grid() :
		is_coarse(H_N3), has_coarse(H_N3), Ushad(opts().n_fields), U(opts().n_fields), U0(opts().n_fields), dUdt(opts().n_fields), F(NDIM), X(NDIM), G(NGF), dphi_dt(
				H_N3), is_root(false), is_leaf(true), U_out(opts().n_fields, ZERO), U_out0(opts().n_fields, ZERO) {
//	allocate();
}

grid::grid(const init_func_type &init_func, real _dx, std::array<real, NDIM> _xmin) :
		is_coarse(H_N3), has_coarse(H_N3), Ushad(opts().n_fields), U(opts().n_fields), U0(opts().n_fields), dUdt(opts().n_fields), F(NDIM), X(NDIM), G(NGF), is_root(
				false), is_leaf(true), U_out(opts().n_fields, ZERO), U_out0(opts().n_fields, ZERO), dphi_dt(H_N3) {

	dx = _dx;
	xmin = _xmin;
	allocate();
	for (integer i = 0; i != H_NX; ++i) {
		for (integer j = 0; j != H_NX; ++j) {
			for (integer k = 0; k != H_NX; ++k) {
				const integer iii = hindex(i, j, k);
				if (init_func != nullptr) {
					std::vector<real> this_u = init_func(X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii], dx);
					for (integer field = 0; field != opts().n_fields; ++field) {
						U[field][iii] = this_u[field];
					}
				} else {
					print("No problem specified\n");
					abort();
				}
			}
		}
	}
	init_z_field();
	if (opts().radiation) {
		if (init_func != nullptr) {
			rad_init();
		}
	}
	if (opts().gravity) {
		for (integer i = 0; i != G_N3; ++i) {
			for (integer field = 0; field != NGF; ++field) {
				G[i][field] = 0.0;
			}
		}
	}
}

void grid::init_z_field() {
	for (int j = H_BW; j < H_NX - H_BW; j++) {
		for (int k = H_BW; k < H_NX - H_BW; k++) {
			for (int l = H_BW; l < H_NX - H_BW; l++) {
				const int i = hindex(j, k, l);
				auto dsx_dy = U[sx_i][i + H_DNY] - U[sx_i][i - H_DNY];
				auto dsx_dz = U[sx_i][i + H_DNZ] - U[sx_i][i - H_DNZ];

				auto dsy_dx = U[sy_i][i + H_DNX] - U[sy_i][i - H_DNX];
				auto dsy_dz = U[sy_i][i + H_DNZ] - U[sy_i][i - H_DNZ];

				auto dsz_dy = U[sz_i][i + H_DNY] - U[sz_i][i - H_DNY];
				auto dsz_dx = U[sz_i][i + H_DNX] - U[sz_i][i - H_DNX];

				U[lx_i][i] = 0.5 * (dx / 12.0) * (dsz_dy - dsy_dz);
				U[ly_i][i] = 0.5 * (dx / 12.0) * (dsx_dz - dsz_dx);
				U[lz_i][i] = 0.5 * (dx / 12.0) * (dsy_dx - dsx_dy);
				U[lx_i][i] += X[YDIM][i] * U[sz_i][i] - X[ZDIM][i] * U[sy_i][i];
				U[ly_i][i] -= X[XDIM][i] * U[sz_i][i] - X[ZDIM][i] * U[sx_i][i];
				U[lz_i][i] += X[XDIM][i] * U[sy_i][i] - X[YDIM][i] * U[sx_i][i];
			}
		}
	}
}
void grid::rad_init() {
	rad_grid_ptr->set_dx(dx);
	rad_grid_ptr->compute_mmw(U);
	rad_grid_ptr->initialize_erad(U[rho_i], U[tau_i]);
}

timestep_t grid::compute_fluxes() {
	PROFILE();
	static hpx::lcos::local::once_flag flag;
	hpx::lcos::local::call_once(flag, [this]() {
		physics<NDIM>::set_fgamma(fgamma);
		if (opts().eos == WD) {
//			print("%e %e\n", physcon().A, physcon().B);
			physics<NDIM>::set_degenerate_eos(physcon().A, physcon().B);
		}
		physics<NDIM>::set_dual_energy_switches(opts().dual_energy_sw1, opts().dual_energy_sw2);
	});

	/******************************/
//	hydro.set_low_order();
	/******************************/
	hydro.use_experiment(opts().experiment);
	if (opts().correct_am_hydro) {
		hydro.use_angmom_correction(sx_i);
	}
	if (opts().cdisc_detect) {
		hydro.use_disc_detect(rho_i);
		for (int i = spc_i; i < spc_i + opts().n_species; i++) {
			hydro.use_disc_detect(i);
		}
	}
	hydro.use_smooth_recon(pot_i);
    
  const interaction_host_kernel_type host_type = opts().hydro_host_kernel_type;
  const interaction_device_kernel_type device_type = opts().hydro_device_kernel_type;
  const size_t device_queue_length = opts().cuda_buffer_capacity;
  return launch_hydro_kernels(hydro, U, X, omega, F, host_type, device_type, device_queue_length);

}

real grid::compute_positivity_speed_limit() const {
	double max_lambda = 0.0;
	for (integer i = 0; i < INX; ++i) {
		for (integer j = 0; j < INX; ++j) {
			for (integer k = 0; k < INX; ++k) {
				double drho_dt = 0.0;
				double dtau_dt = 0.0;
				drho_dt -= (F[0][rho_i][findex(i + 1, j, k)] - F[0][rho_i][findex(i, j, k)]) / dx;
				drho_dt -= (F[1][rho_i][findex(i, j + 1, k)] - F[1][rho_i][findex(i, j, k)]) / dx;
				drho_dt -= (F[2][rho_i][findex(i, j, k + 1)] - F[2][rho_i][findex(i, j, k)]) / dx;
				dtau_dt -= (F[0][tau_i][findex(i + 1, j, k)] - F[0][tau_i][findex(i, j, k)]) / dx;
				dtau_dt -= (F[1][tau_i][findex(i, j + 1, k)] - F[1][tau_i][findex(i, j, k)]) / dx;
				dtau_dt -= (F[2][tau_i][findex(i, j, k + 1)] - F[2][tau_i][findex(i, j, k)]) / dx;
				max_lambda = std::max(max_lambda, -drho_dt * dx / U[rho_i][hindex(i + H_BW, j + H_BW, k + H_BW)]);
				max_lambda = std::max(max_lambda, -dtau_dt * dx / U[tau_i][hindex(i + H_BW, j + H_BW, k + H_BW)]);
			}
		}
	}
	return max_lambda / opts().dt_max;
}

void grid::set_min_level(integer l) {
	min_level = l;
}

void grid::set_max_level(integer l) {
	max_level = l;
}

void grid::store() {
	PROFILE();
	for (integer field = 0; field != opts().n_fields; ++field) {
#pragma GCC ivdep
		for (integer i = 0; i != INX; ++i) {
			for (integer j = 0; j != INX; ++j) {
				for (integer k = 0; k != INX; ++k) {
					U0[field][h0index(i, j, k)] = U[field][hindex(i + H_BW, j + H_BW, k + H_BW)];
				}
			}
		}
	}
	U_out0 = U_out;
}

void grid::restore() {
	for (integer field = 0; field != opts().n_fields; ++field) {
#pragma GCC ivdep
		for (integer i = 0; i != INX; ++i) {
			for (integer j = 0; j != INX; ++j) {
				for (integer k = 0; k != INX; ++k) {
					U[field][h0index(i, j, k)] = U0[field][hindex(i + H_BW, j + H_BW, k + H_BW)];
				}
			}
		}
	}
	U_out = U_out0;
}

void grid::set_physical_boundaries(const geo::face &face, real t) {
	PROFILE();
	const auto dim = face.get_dimension();
	const auto side = face.get_side();
	const integer dni = dim == XDIM ? H_DNY : H_DNX;
	const integer dnj = dim == ZDIM ? H_DNY : H_DNZ;
	const integer dnk = H_DN[dim];
	const integer klb = side == geo::MINUS ? 0 : H_NX - H_BW;
	const integer kub = side == geo::MINUS ? H_BW : H_NX;
	const integer ilb = 0;
	const integer iub = H_NX;
	const integer jlb = 0;
	const integer jub = H_NX;

	if (opts().problem == AMR_TEST) {
		for (integer k = klb; k != kub; ++k) {
			for (integer j = jlb; j != jub; ++j) {
				for (integer i = ilb; i != iub; ++i) {
					const auto iii = hindex(i, j, k);
					const auto u = amr_test(X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii], dx);
					for (int f = 0; f < opts().n_fields; f++) {
						U[f][iii] = u[f];
					}
				}
			}
		}
//	} else if (opts().problem == SOD) {
//		for (integer k = klb; k != kub; ++k) {
//			for (integer j = jlb; j != jub; ++j) {
//				for (integer i = ilb; i != iub; ++i) {
//					const integer iii = i * dni + j * dnj + k * dnk;
//					for (integer f = 0; f != opts().n_fields; ++f) {
//						U[f][iii] = 0.0;
//					}
//					sod_state_t s;
//					//			real x = (X[XDIM][iii] + X[YDIM][iii] + X[ZDIM][iii]) / std::sqrt(3.0);
//					real x = X[XDIM][iii];
//					real y = X[YDIM][iii];
//					real z = X[ZDIM][iii];
//					exact_sod(&s, &sod_init, x, t);
//					U[rho_i][iii] = s.rho;
//					U[egas_i][iii] = s.p / (fgamma - 1.0);
////					U[sx_i][iii] = s.rho * s.v / std::sqrt(3.0);
////					U[sy_i][iii] = s.rho * s.v / std::sqrt(3.0);
////					U[sz_i][iii] = s.rho * s.v / std::sqrt(3.0);
//					U[sx_i][iii] = s.rho * s.v;
//					U[sy_i][iii] = 0.0;
//					U[sz_i][iii] = 0.0;
//					U[lx_i][iii] = +y * U[sz_i][iii] - z * U[sy_i][iii];
//					U[ly_i][iii] = -x * U[sz_i][iii] + z * U[sx_i][iii];
//					U[lz_i][iii] = +x * U[sy_i][iii] - y * U[sx_i][iii];
//					U[tau_i][iii] = std::pow(U[egas_i][iii], 1.0 / fgamma);
//					U[egas_i][iii] += s.rho * s.v * s.v / 2.0;
//					U[spc_i][iii] = s.rho;
//					integer k0 = side == geo::MINUS ? H_BW : H_NX - H_BW - 1;
//				}
//			}
//		}
	} else {
		for (integer field = 0; field != opts().n_fields; ++field) {
			for (integer k = klb; k != kub; ++k) {
				for (integer j = jlb; j != jub; ++j) {
					for (integer i = ilb; i != iub; ++i) {
						integer k0;
						if (opts().reflect_bc) {
							k0 = side == geo::MINUS ? (2 * H_BW - k - 1) : (2 * (H_NX - H_BW) - k - 1);
						} else {
							k0 = side == geo::MINUS ? H_BW : H_NX - H_BW - 1;
						}
						const integer iii0 = i * dni + j * dnj + k0 * dnk;
						const real value = U[field][iii0];
						const integer iii = i * dni + j * dnj + k * dnk;
						safe_real &ref = U[field][iii];
						real x = X[XDIM][iii];
						real y = X[YDIM][iii];
						real z = X[ZDIM][iii];
						real x0 = X[XDIM][iii0];
						real y0 = X[YDIM][iii0];
						real z0 = X[ZDIM][iii0];
						if (field == sx_i + dim) {
							real s0;
							if (field == sx_i) {
								s0 = -omega * X[YDIM][iii] * U[rho_i][iii];
							} else if (field == sy_i) {
								s0 = +omega * X[XDIM][iii] * U[rho_i][iii];
							} else {
								s0 = ZERO;
							}
							if (opts().reflect_bc) {
								ref = -value;
							} else if (opts().inflow_bc) {
								ref = value;
							} else {
								if (opts().problem != AMR_TEST) {
									const real before = value;
									if (side == geo::MINUS) {
										ref = s0 + std::min(value - s0, ZERO);
									} else {
										ref = s0 + std::max(value - s0, ZERO);
									}
									const real after = ref;
									assert(rho_i < field);
									assert(egas_i < field);
									real this_rho = U[rho_i][iii];
									if (this_rho != ZERO) {
										U[egas_i][iii] += HALF * (after * after - before * before) / this_rho;
									}
								}
							}
//						} else if (field == rho_i) {
//							ref = std::max(rho_floor,value);
						} else if (field == lx_i) {
							ref = +value;
							U[lx_i][iii] += +y * U[sz_i][iii] - z * U[sy_i][iii];
							U[lx_i][iii] -= +y0 * U[sz_i][iii0] - z0 * U[sy_i][iii0];
						} else if (field == ly_i) {
							ref = +value;
							U[ly_i][iii] += -x * U[sz_i][iii] + z * U[sx_i][iii];
							U[ly_i][iii] -= -x0 * U[sz_i][iii0] + z0 * U[sx_i][iii0];
						} else if (field == lz_i) {
							ref = +value;
							U[lz_i][iii] += +x * U[sy_i][iii] - y * U[sx_i][iii];
							U[lz_i][iii] -= +x0 * U[sy_i][iii0] - y0 * U[sx_i][iii0];
						} else {
							ref = +value;
						}
					}
				}
			}
		}
	}
}

void grid::compute_sources(real t, real rotational_time) {
	PROFILE();
	auto &src = dUdt;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				const integer iii = hindex(i, j, k);
				const integer iiif = findex(i - H_BW, j - H_BW, k - H_BW);
				const integer iiig = gindex(i - H_BW, j - H_BW, k - H_BW);
				for (integer field = 0; field != opts().n_fields; ++field) {
					src[field][iii0] = ZERO;
				}
				const real rho = U[rho_i][iii];
				if (opts().gravity) {
					src[sx_i][iii0] += rho * G[iiig][gx_i];
					src[sy_i][iii0] += rho * G[iiig][gy_i];
					src[sz_i][iii0] += rho * G[iiig][gz_i];
				}
				if (opts().gravity) {
					src[egas_i][iii0] -= omega * X[YDIM][iii] * rho * G[iiig][gx_i];
					src[egas_i][iii0] += omega * X[XDIM][iii] * rho * G[iiig][gy_i];
				}
				if (opts().driving_rate != 0.0) {
					const real period_len = 2.0 * M_PI / grid::omega;
					if (opts().driving_time > rotational_time / (2.0 * M_PI)) {
						const real ff = -opts().driving_rate / period_len;
						///	print("%e %e %e\n", ff, opts().driving_rate, period_len);
						const real rho = U[rho_i][iii];
						const real sx = U[sx_i][iii];
						const real sy = U[sy_i][iii];
						const real x = X[XDIM][iii];
						const real y = X[YDIM][iii];
						const real R = std::sqrt(x * x + y * y);
						const real lz = (x * sy - y * sx);
						const real dsx = -y / R / R * lz * ff;
						const real dsy = +x / R / R * lz * ff;
						src[sx_i][iii0] += dsx;
						src[sy_i][iii0] += dsy;
						src[egas_i][iii0] += (sx * dsx + sy * dsy) / rho;
					}
				}
				if (opts().entropy_driving_rate != 0.0) {

					constexpr integer spc_ac_i = spc_i;
					constexpr integer spc_ae_i = spc_i + 1;

					const real period_len = 2.0 * M_PI / grid::omega;
					if (opts().entropy_driving_time > rotational_time / (2.0 * M_PI)) {

						constexpr integer spc_ac_i = spc_i;
						constexpr integer spc_ae_i = spc_i + 1;

						real ff = +opts().entropy_driving_rate / period_len;
						ff *= (U[spc_ac_i][iii] + U[spc_ae_i][iii]) / U[rho_i][iii];
						real ek = ZERO;
						ek += HALF * pow(U[sx_i][iii], 2) / U[rho_i][iii];
						ek += HALF * pow(U[sy_i][iii], 2) / U[rho_i][iii];
						ek += HALF * pow(U[sz_i][iii], 2) / U[rho_i][iii];
						real ei;
						if (opts().eos == WD) {
							ei = U[egas_i][iii] - ek - ztwd_energy(U[rho_i][iii]);
						} else {
							ei = U[egas_i][iii] - ek;
						}
						real et = U[egas_i][iii];
						real tau;
						if (ei < de_switch2 * et) {
							tau = U[tau_i][iii];
						} else {
							tau = std::pow(ei, 1.0 / fgamma);
						}
						ei = std::pow(tau, fgamma);
						const real dtau = ff * tau;
						const real dei = dtau * ei / tau * fgamma;
						src[tau_i][iii0] += dtau;
						src[egas_i][iii0] += dei;
					}
				}
				src[lx_i][iii0] += X[YDIM][iii] * src[sz_i][iii0] - X[ZDIM][iii] * src[sy_i][iii0];
				src[ly_i][iii0] -= X[XDIM][iii] * src[sz_i][iii0] - X[ZDIM][iii] * src[sx_i][iii0];
				src[lz_i][iii0] += X[XDIM][iii] * src[sy_i][iii0] - X[YDIM][iii] * src[sx_i][iii0];
				src[sx_i][iii0] += omega * U[sy_i][iii];
				src[sy_i][iii0] -= omega * U[sx_i][iii];
				src[lx_i][iii0] += omega * U[ly_i][iii];
				src[ly_i][iii0] -= omega * U[lx_i][iii];
			}
		}
	}
}

void grid::compute_dudt() {
	PROFILE();
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer field = 0; field != opts().n_fields; ++field) {
#pragma GCC ivdep
				for (integer k = H_BW; k != H_NX - H_BW; ++k) {
					const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
					const integer iiif = findex(i - H_BW, j - H_BW, k - H_BW);
					dUdt[field][iii0] -= (F[XDIM][field][iiif + F_DNX] - F[XDIM][field][iiif]) / dx;
					dUdt[field][iii0] -= (F[YDIM][field][iiif + F_DNY] - F[YDIM][field][iiif]) / dx;
					dUdt[field][iii0] -= (F[ZDIM][field][iiif + F_DNZ] - F[ZDIM][field][iiif]) / dx;
				}
			}
			if (opts().gravity) {

#pragma GCC ivdep
				for (integer k = H_BW; k != H_NX - H_BW; ++k) {
					const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
					dUdt[egas_i][iii0] += dUdt[pot_i][iii0];
					dUdt[pot_i][iii0] = ZERO;
				}
			}
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				const integer iiig = gindex(i - H_BW, j - H_BW, k - H_BW);
				if (opts().gravity) {
					dUdt[egas_i][iii0] -= (dUdt[rho_i][iii0] * G[iiig][phi_i]) * HALF;
				}
			}
		}
	}
//	solve_gravity(DRHODT);
}

void grid::egas_to_etot() {
	PROFILE();
	if (opts().gravity) {

		for (integer i = H_BW; i != H_NX - H_BW; ++i) {
			for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
				for (integer k = H_BW; k != H_NX - H_BW; ++k) {
					const integer iii = hindex(i, j, k);
					U[egas_i][iii] += U[pot_i][iii] * HALF;
				}
			}
		}
	}
}

void grid::etot_to_egas() {
	PROFILE();
	if (opts().gravity) {

		for (integer i = H_BW; i != H_NX - H_BW; ++i) {
			for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
				for (integer k = H_BW; k != H_NX - H_BW; ++k) {
					const integer iii = hindex(i, j, k);
					U[egas_i][iii] -= U[pot_i][iii] * HALF;
				}
			}
		}
	}
}

void grid::next_u(integer rk, real t, real dt) {
	PROFILE();
	if (!opts().hydro) {
		return;
	}
//	return;

	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				const integer iii = hindex(i, j, k);
				dUdt[egas_i][iii0] += (dphi_dt[iii0] * U[rho_i][iii]) * HALF;
			}
		}
	}

	std::vector<real> du_out(opts().n_fields, ZERO);

	std::vector<real> ds(NDIM, ZERO);
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				for (integer field = 0; field != opts().n_fields; ++field) {
					const real u1 = U[field][iii] + dUdt[field][iii0] * dt;
					const real u0 = U0[field][h0index(i - H_BW, j - H_BW, k - H_BW)];
					U[field][iii] = (ONE - rk_beta[rk]) * u0 + rk_beta[rk] * u1;
				}
			}
		}
	}

	du_out[sx_i] += omega * U_out[sy_i] * dt;
	du_out[sy_i] -= omega * U_out[sx_i] * dt;

	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
#pragma GCC ivdep
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			const real dx2 = sqr(dx);
			const integer iii_p0 = findex(INX, i - H_BW, j - H_BW);
			const integer jjj_p0 = findex(j - H_BW, INX, i - H_BW);
			const integer kkk_p0 = findex(i - H_BW, j - H_BW, INX);
			const integer iii_m0 = findex(0, i - H_BW, j - H_BW);
			const integer jjj_m0 = findex(j - H_BW, 0, i - H_BW);
			const integer kkk_m0 = findex(i - H_BW, j - H_BW, 0);
			const integer iii_p = H_DNX * (H_NX - H_BW) + H_DNY * i + H_DNZ * j;
			const integer jjj_p = H_DNY * (H_NX - H_BW) + H_DNZ * i + H_DNX * j;
			const integer kkk_p = H_DNZ * (H_NX - H_BW) + H_DNX * i + H_DNY * j;
			const integer iii_m = H_DNX * (H_BW) + H_DNY * i + H_DNZ * j;
			const integer jjj_m = H_DNY * (H_BW) + H_DNZ * i + H_DNX * j;
			const integer kkk_m = H_DNZ * (H_BW) + H_DNX * i + H_DNY * j;
			std::vector<real> du(opts().n_fields);
			for (integer field = 0; field != opts().n_fields; ++field) {
				du[field] = ZERO;
				if (X[XDIM][iii_p] > scaling_factor) {
					du[field] += (F[XDIM][field][iii_p0]) * dx2;
				}
				if (X[YDIM][jjj_p] > scaling_factor) {
					du[field] += (F[YDIM][field][jjj_p0]) * dx2;
				}
				if (X[ZDIM][kkk_p] > scaling_factor) {
					du[field] += (F[ZDIM][field][kkk_p0]) * dx2;
				}
				if (X[XDIM][iii_m] < -scaling_factor + dx) {
					du[field] += (-F[XDIM][field][iii_m0]) * dx2;
				}
				if (X[YDIM][jjj_m] < -scaling_factor + dx) {
					du[field] += (-F[YDIM][field][jjj_m0]) * dx2;
				}
				if (X[ZDIM][kkk_m] < -scaling_factor + dx) {
					du[field] += (-F[ZDIM][field][kkk_m0]) * dx2;
				}
			}
			for (integer field = 0; field != opts().n_fields; ++field) {
				du_out[field] += du[field] * dt;
			}
		}
	}
//#pragma GCC ivdep
	for (integer field = 0; field != opts().n_fields; ++field) {
		const real out1 = U_out[field] + du_out[field];
		const real out0 = U_out0[field];
		U_out[field] = (ONE - rk_beta[rk]) * out0 + rk_beta[rk] * out1;
	}
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				if (opts().tau_floor > 0.0) {
					U[tau_i][iii] = std::max(U[tau_i][iii], opts().tau_floor);
				} else if (U[tau_i][iii] < ZERO) {
					print("Tau is negative- %e %i %i %i  %e %e %e\n", real(U[tau_i][iii]), int(i), int(j), int(k), (double) X[XDIM][iii],
							(double) X[YDIM][iii], (double) X[ZDIM][iii]);
					print("Use tau_floor option\n");
					abort();
				}
				if (opts().rho_floor > 0.0) {
					double x;
					x = 0.0;
					for (int s = 0; s < opts().n_species; s++) {
						U[spc_i + s][iii] = std::max(U[spc_i + s][iii], 0.0);
						x += U[spc_i + s][iii];
					}
					if (x != 0.0) {
						for (int s = 0; s < opts().n_species; s++) {
							U[spc_i + s][iii] /= x;
						}
					} else {
						U[spc_i + opts().n_species - 1][iii] = 1.0;
					}
					if (U[rho_i][iii] < opts().rho_floor) {
						x = 1.0 - std::max(U[rho_i][iii], 0.0) / opts().rho_floor;
						U[rho_i][iii] = opts().rho_floor;
						U[tau_i][iii] += x * (opts().tau_floor - U[tau_i][iii]);
						U[egas_i][iii] += x * (std::pow(opts().tau_floor, fgamma) - U[egas_i][iii]);
						U[sx_i][iii] -= x * U[sx_i][iii];
						U[sy_i][iii] -= x * U[sy_i][iii];
						U[sz_i][iii] -= x * U[sz_i][iii];

					}
					for (int s = 0; s < opts().n_species; s++) {
						U[spc_i + s][iii] *= U[rho_i][iii];
					}

				} else if (U[rho_i][iii] <= ZERO) {
					print("Rho is non-positive - %e %i %i %i %e %e %e\n", real(U[rho_i][iii]), int(i), int(j), int(k), real(X[XDIM][iii]), real(X[YDIM][iii]),
							real(X[ZDIM][iii]));
					print("Use rho_floor option\n");
					abort();
				}
			}
		}
	}
}

void grid::dual_energy_update() {
	PROFILE();

//	bool in_bnd;

	physics<NDIM>::post_process<INX>(U, X, dx);

	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);

				double rho_tot = 0.0;
				for (int s = 0; s < opts().n_species; s++) {
					rho_tot += U[spc_i + s][iii];
				}
				U[rho_i][iii] = rho_tot;

			}
		}
	}
}

std::pair<real, real> grid::virial() const {

//	bool in_bnd;
	std::pair<real, real> v;
	v.first = v.second = 0.0;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				real ek = ZERO;
				ek += HALF * pow(U[sx_i][iii], 2) / U[rho_i][iii];
				ek += HALF * pow(U[sy_i][iii], 2) / U[rho_i][iii];
				ek += HALF * pow(U[sz_i][iii], 2) / U[rho_i][iii];
				real ei;
				if (opts().eos == WD) {
					ei = U[egas_i][iii] - ek - ztwd_energy(U[rho_i][iii]);
				} else {
					ei = U[egas_i][iii] - ek;
				}
				real et = U[egas_i][iii];
				if (ei < de_switch2 * et) {
					ei = std::pow(U[tau_i][iii], fgamma);
				}
				real p = (fgamma - 1.0) * ei;
				if (opts().eos == WD) {
					p += ztwd_pressure(U[rho_i][iii]);
				}
				v.first += (2.0 * ek + 0.5 * U[pot_i][iii] + 3.0 * p) * (dx * dx * dx);
				v.second += (2.0 * ek - 0.5 * U[pot_i][iii] + 3.0 * p) * (dx * dx * dx);
			}
		}
	}
	return v;
}

std::vector<real> grid::conserved_outflows() const {
	auto Uret = U_out;
	if (opts().gravity) {
		Uret[egas_i] += Uret[pot_i];
	}
	return Uret;
}
#endif
