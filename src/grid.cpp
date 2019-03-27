#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/test_problems/sod/exact_sod.hpp"

#include "octotiger/diagnostics.hpp"
#include "octotiger/future.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/problem.hpp"
#include "octotiger/profiler.hpp"
#include "octotiger/real.hpp"
#include "octotiger/silo.hpp"
#include "octotiger/taylor.hpp"

#include <hpx/include/runtime.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <hpx/util/format.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <unordered_map>

std::unordered_map<std::string, int> grid::str_to_index_hydro;
std::unordered_map<std::string, int> grid::str_to_index_gravity;
std::unordered_map<int, std::string> grid::index_to_str_hydro;
std::unordered_map<int, std::string> grid::index_to_str_gravity;

bool grid::is_hydro_field(const std::string& str) {
	return str_to_index_hydro.find(str) != str_to_index_hydro.end();
}

std::vector<std::pair<std::string, real>> grid::get_outflows() const {
	std::vector<std::pair<std::string, real>> rc;
	for (auto i = str_to_index_hydro.begin(); i != str_to_index_hydro.end(); i++) {
		rc.push_back(std::make_pair(i->first, U_out[i->second]));
	}
	return std::move(rc);
}

void grid::set_outflows(std::vector<std::pair<std::string, real>>&& u) {
	for (const auto& p : u) {
		U_out[str_to_index_hydro[p.first]] = p.second;
	}
}

void grid::set_outflow(std::pair<std::string, real>&& p) {
	U_out[str_to_index_hydro[p.first]] = p.second;
}

void grid::static_init() {
	str_to_index_hydro[std::string("egas")] = egas_i;
	str_to_index_hydro[std::string("tau")] = tau_i;
	for (integer s = 0; s < opts().n_species; s++) {
		str_to_index_hydro[std::string("rho_") + std::to_string(s + 1)] = spc_i + s;
	}
	str_to_index_hydro[std::string("sx")] = sx_i;
	str_to_index_hydro[std::string("sy")] = sy_i;
	str_to_index_hydro[std::string("sz")] = sz_i;
	str_to_index_hydro[std::string("zx")] = zx_i;
	str_to_index_hydro[std::string("zy")] = zy_i;
	str_to_index_hydro[std::string("zz")] = zz_i;
	str_to_index_gravity[std::string("phi")] = phi_i;
	str_to_index_gravity[std::string("gx")] = gx_i;
	str_to_index_gravity[std::string("gy")] = gy_i;
	str_to_index_gravity[std::string("gz")] = gz_i;
	for (const auto& s : str_to_index_hydro) {
		index_to_str_hydro[s.second] = s.first;
	}
	for (const auto& s : str_to_index_gravity) {
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
		for (auto& n : rnames) {
			rc.push_back(n);
		}
	}
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

void grid::set(const std::string name, real* data, int version) {
	auto iter = str_to_index_hydro.find(name);
	real unit = convert_hydro_units(iter->second);

	if (iter != str_to_index_hydro.end()) {
		int f = iter->second;
		int jjj = 0;

		/* Correct for bugfix across versions */				
		if( version == 100 && f >= sx_i && f <= sz_i ) {
			unit *= opts().code_to_s;
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
		const real s = opts().code_to_s;
		const real g = opts().code_to_g;
		if (i >= spc_i && i <= spc_i + opts().n_species) {
			val *= g / (cm * cm * cm);
		} else if (i >= sx_i && i <= sz_i) {
			val *= g / (s * cm * cm);
		} else if (i == egas_i || (i >= zx_i && i <= zz_i)) {
			val *= g / (s * s * cm);
		} else if (i == tau_i) {
			val *= POWER(g / (s * s * cm), 1.0 / fgamma);
		} else {
			printf("Asked to convert units for unknown field %i\n", i);
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

std::string grid::hydro_units_name(const std::string& nm) {
	int f = str_to_index_hydro[nm];
	if (f >= spc_i && f <= spc_i + opts().n_species) {
		return "g / cm^3";
	} else if (f >= sx_i && f <= sz_i) {
		return "g / (cm s)^2 ";
	} else if (f == egas_i || (f >= zx_i && f <= zz_i)) {
		return "g / (cm s^2)";
	} else if (f == tau_i) {
		return "(g / cm)^(3/5) / s^(6/5)";
	}
	return "<unknown>";
}

std::string grid::gravity_units_name(const std::string& nm) {
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
	for (auto l : str_to_index_hydro) {
		unit = convert_hydro_units(l.second);
		const int f = l.second;


		std::string this_name = l.first;
		int jjj = 0;
		silo_var_t this_s(this_name);
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = hindex(k + H_BW, j + H_BW, i + H_BW);
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
		for (auto& r : rad) {
			s.push_back(std::move(r));
		}
	}
	return std::move(s);
}

// MSVC needs this variable to be in the global namespace
constexpr integer nspec = 2;
diagnostics_t grid::diagnostics(const diagnostics_t& diags) {
	diagnostics_t rc;
	if( opts().disable_diagnostics) {
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
					rc.grid_sum[zx_i] += (X[YDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sy_i][iii]) * dV;
					rc.grid_sum[zy_i] -= (X[XDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sx_i][iii]) * dV;
					rc.grid_sum[zz_i] += (X[XDIM][iii] * U[sy_i][iii] - X[YDIM][iii] * U[sx_i][iii]) * dV;
					rc.lsum[0] += (X[YDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sy_i][iii]) * dV;
					rc.lsum[1] -= (X[XDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sx_i][iii]) * dV;
					rc.lsum[2] += (X[XDIM][iii] * U[sy_i][iii] - X[YDIM][iii] * U[sx_i][iii]) * dV;
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
		const integer iii = hindex(j,k,l);
		const real ax = X[XDIM][iii] - diags.com[0][XDIM];
		const real ay = X[YDIM][iii] - diags.com[0][YDIM];
		const real az = X[ZDIM][iii] - diags.com[0][ZDIM];

		const real bx = diags.com[1][XDIM] - diags.com[0][XDIM];
		const real by = diags.com[1][YDIM] - diags.com[0][YDIM];
		const real bz = diags.com[1][ZDIM] - diags.com[0][ZDIM];

		const real aa = (ax*ax+ay*ay+az*az);
		const real bb = (bx*bx+by*by+bz*bz);
		const real ab = (ax*bx+ay*by+az*bz);

		const real d2bb = aa * bb - ab * ab;
		if( (d2bb < dx * dx * bb * 3.0 / 4.0) ) {
			if( (ab < bb) && (ab > 0.0) ) {
				return 2;
			} else if( ab <= 0.0 ) {
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
		if( opts().problem != DWD) {
			return integer(0);
		}
		const integer iii = hindex(j,k,l);
		const integer iiig = gindex(j-H_BW,k-H_BW,l-H_BW);
		integer rc = 0;
		const real x = X[XDIM][iii];
		const real y = X[YDIM][iii];
		const real z = X[ZDIM][iii];
		real ax = G[iiig][gx_i] + x * diags.omega * diags.omega;
		real ay = G[iiig][gy_i] + y * diags.omega * diags.omega;
		real az = G[iiig][gz_i];
		real nx, ny, nz;
		const real a = SQRT(ax * ax + ay * ay + az * az);
		if( a > 0.0 ) {
			nx = ax / a;
			ny = ay / a;
			nz = az / a;
			space_vector dX[nspec];
			real g[nspec] = {0.0,0.0};
			for( integer s = 0; s != nspec; ++s) {
				dX[s][XDIM] = x - diags.com[s][XDIM];
				dX[s][YDIM] = y - diags.com[s][YDIM];
				dX[s][ZDIM] = z - diags.com[s][ZDIM];
			}
			const real x0 = std::pow(dX[0][XDIM],2)+std::pow(dX[0][YDIM],2)+std::pow(dX[0][ZDIM],2);
			const real x1 = std::pow(dX[1][XDIM],2)+std::pow(dX[1][YDIM],2)+std::pow(dX[1][ZDIM],2);
			if( x1 >= 0.25 * diags.rL[1] && x0 < 0.25 * diags.rL[0] && diags.stage > 3) {
				rc = +1;
			} else if(x0 >= 0.25 * diags.rL[0] && x1 < 0.25 * diags.rL[1] && diags.stage > 3) {
				rc = -1;
			} else if(x0 < 0.25 * diags.rL[0] && x1 < 0.25 * diags.rL[1] && diags.stage > 3) {
				rc = x0 < x1 ? +1 : -1;
			} else {
				for( integer s = 0; s != nspec; ++s) {
					const real this_x = s == 0 ? x0 : x1;
					if( this_x == 0.0 ) {
						rc = 99;
						return rc;
					}
					g[s] += ax * dX[s][XDIM] *INVERSE( this_x);
					g[s] += ay * dX[s][YDIM] *INVERSE( this_x);
					g[s] += az * dX[s][ZDIM] *INVERSE( this_x);
				}
				if( g[0] <= 0.0 && g[1] > 0.0) {
					rc = +1;
				} else if( g[0] > 0.0 && g[1] <= 0.0 ) {
					rc = -1;
				} else if( g[0] <= 0.0 && g[1] <= 0.0 ) {
					if( std::abs(g[0]) > std::abs(g[1]) ) {
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
				if (diags.stage <= 2) {
					rho =
					{	U[spc_ac_i][iii], U[spc_dc_i][iii]};
				} else {
					star = in_star(j,k,l);
					if( star == +1 ) {
						rho = {U[rho_i][iii], 0.0};
					} else if( star == -1 ) {
						rho = {0.0, U[rho_i][iii]};
					} else if( star != 99 ){
						rho = {0.0, 0.0};
					} else {
						rc.failed = true;
						return rc;
					}
				}
				if (diags.stage > 1) {
					const real R2 = x * x + y * y;
					const real phi_g = G[iiig][phi_i];
					if( diags.omega < 0.0 ) {
						rc.failed = true;
						return rc;
					}
					const real phi_r = -0.5 * POWER(diags.omega, 2) * R2;
					const real phi_eff = phi_g + phi_r;
					const real rho0 = U[rho_i][iii];
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
						rc.gt[i] += dX[0] * G[iiig][gy_i] * dV * rho0;
						rc.gt[i] -= dX[1] * G[iiig][gx_i] * dV * rho0;
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
					}

					rc.rho_max[i] = std::max(rc.rho_max[i], rho0);
					auto& rl = roche_lobe[h0index(j - H_BW, k - H_BW, l - H_BW)];

					auto lmin23 = std::min(diags.l2_phi, diags.l3_phi);
					auto lmax23 = std::max(diags.l2_phi, diags.l3_phi);

					if (i != -1) {
						rl = i == 0 ? -1 : +1;
						const integer s = rl * INVERSE(std::abs(rl));

						if (phi_eff > diags.l1_phi) {
							rl += s;
						}
						if (phi_eff > lmin23) {
							rl += s;
						}
						if (phi_eff > lmax23) {
							rl += s;
						}
					} else {
						rl = 0;
					}

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
					rc.grid_sum[zx_i] += (X[YDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sy_i][iii]) * dV;
					rc.grid_sum[zy_i] -= (X[XDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sx_i][iii]) * dV;
					rc.grid_sum[zz_i] += (X[XDIM][iii] * U[sy_i][iii] - X[YDIM][iii] * U[sx_i][iii]) * dV;
					rc.lsum[0] += (X[YDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sy_i][iii]) * dV;
					rc.lsum[1] -= (X[XDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sx_i][iii]) * dV;
					rc.lsum[2] += (X[XDIM][iii] * U[sy_i][iii] - X[YDIM][iii] * U[sx_i][iii]) * dV;
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
		if (rc.m[s] > 0.0) {
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

std::vector<real> grid::get_flux_check(const geo::face& f) {
	std::vector<real> flux;
	const integer dim = f.get_dimension();
	int i, j, k;
	int& i1 = dim == XDIM ? j : i;
	int& i2 = dim == ZDIM ? j : k;
	int& n = dim == XDIM ? i : (dim == YDIM ? j : k);
	if (f.get_side() == geo::MINUS) {
		n = 0;
	} else {
		n = INX;
	}
	for (integer f = 0; f != opts().n_fields; ++f) {
		for (i1 = 0; i1 != INX; ++i1) {
			for (i2 = 0; i2 != INX; ++i2) {
				flux.push_back(F[dim][f][findex(i, j, k)]);
			}
		}
	}
	return flux;
}

void grid::set_flux_check(const std::vector<real>& data, const geo::face& f) {
	const integer dim = f.get_dimension();
	integer index = 0;
	int i, j, k;
	int& i1 = dim == XDIM ? j : i;
	int& i2 = dim == ZDIM ? j : k;
	int& n = dim == XDIM ? i : (dim == YDIM ? j : k);
	if (f.get_side() == geo::MINUS) {
		n = 0;
	} else {
		n = INX;
	}
	for (integer f = 0; f != opts().n_fields; ++f) {
		for (i1 = 0; i1 != INX; ++i1) {
			for (i2 = 0; i2 != INX; ++i2) {
				const real a = F[dim][f][findex(i, j, k)];
				const real b = data[index];
				++index;
				if (a != b) {
					printf("Flux error\n");
					//		printf("%e %e %i\n", a, b, f);
//					abort();
				}
			}
		}
	}
}

hpx::lcos::local::spinlock grid::omega_mtx;
real grid::omega = ZERO;
real grid::scaling_factor = 1.0;

integer grid::max_level = 0;

struct tls_data_t {
	std::vector<std::vector<real>> v;
	std::vector<std::vector<std::vector<real>>> dvdx;
	std::vector<std::vector<std::vector<real>>> dudx;
	std::vector<std::vector<std::vector<real>>> uf;
	std::vector<std::vector<real>> zz;
};

#if !defined(_MSC_VER)

#include <boost/thread/tss.hpp>

class tls_t {
private:
	pthread_key_t key;
public:
	static void cleanup(void* ptr) {
		tls_data_t* _ptr = (tls_data_t*) ptr;
		delete _ptr;
	}
	tls_t() {
		pthread_key_create(&key, cleanup);
	}
	tls_data_t* get_ptr() {
		tls_data_t* ptr = (tls_data_t*) pthread_getspecific(key);
		if (ptr == nullptr) {
			ptr = new tls_data_t;
			ptr->v.resize(opts().n_fields, std::vector<real>(H_N3));
			ptr->zz.resize(NDIM, std::vector<real>(H_N3));
			ptr->dvdx.resize(NDIM, std::vector<std::vector<real>>(opts().n_fields, std::vector<real>(H_N3)));
			ptr->dudx.resize(NDIM, std::vector<std::vector<real>>(opts().n_fields, std::vector<real>(H_N3)));
			ptr->uf.resize(NFACE, std::vector<std::vector<real>>(opts().n_fields, std::vector<real>(H_N3)));
			pthread_setspecific(key, ptr);
		}
		return ptr;
	}
};

#else
#include <hpx/util/thread_specific_ptr.hpp>

class tls_t
{
private:
	struct tls_data_tag {};
	static hpx::util::thread_specific_ptr<tls_data_t, tls_data_tag> data;

public:
//	 static void cleanup(void* ptr)
//	 {
//		 tls_data_t* _ptr = (tls_data_t*) ptr;
//		 delete _ptr;
//	 }

	tls_data_t* get_ptr()
	{
		tls_data_t* ptr = data.get();
		if (ptr == nullptr) {
			ptr = new tls_data_t;
			ptr->v.resize(opts().n_fields, std::vector < real > (H_N3));
			ptr->zz.resize(NDIM, std::vector < real > (H_N3));
			ptr->dvdx.resize(NDIM, std::vector < std::vector < real >> (opts().n_fields, std::vector < real > (H_N3)));
			ptr->dudx.resize(NDIM, std::vector < std::vector < real >> (opts().n_fields, std::vector < real > (H_N3)));
			ptr->uf.resize(NFACE, std::vector < std::vector < real >> (opts().n_fields, std::vector < real > (H_N3)));
			data.reset(ptr);
		}
		return ptr;
	}
};

hpx::util::thread_specific_ptr<tls_data_t, tls_t::tls_data_tag> tls_t::data;

#endif

static tls_t tls;

std::vector<std::vector<real>>& TLS_V() {
	return tls.get_ptr()->v;
}

static std::vector<std::vector<std::vector<real>>>& TLS_dVdx() {
	return tls.get_ptr()->dvdx;
}

static std::vector<std::vector<std::vector<real>>>& TLS_dUdx() {
	return tls.get_ptr()->dudx;
}

static std::vector<std::vector<real>>& TLS_zz() {
	return tls.get_ptr()->zz;
}

static std::vector<std::vector<std::vector<real>>>& TLS_Uf() {
	return tls.get_ptr()->uf;
}

space_vector grid::get_cell_center(integer i, integer j, integer k) {
	const integer iii0 = hindex(H_BW, H_BW, H_BW);
	space_vector c;
	c[XDIM] = X[XDIM][iii0] + (i) * dx;
	c[YDIM] = X[XDIM][iii0] + (j) * dx;
	c[ZDIM] = X[XDIM][iii0] + (k) * dx;
	return c;
}

void grid::set_hydro_boundary(const std::vector<real>& data, const geo::direction& dir, integer width, bool etot_only) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	if (!etot_only) {
		get_boundary_size(lb, ub, dir, OUTER, INX, width);
	} else {
		get_boundary_size(lb, ub, dir, OUTER, INX, width);
	}
	integer iter = 0;

	for (integer field = 0; field != opts().n_fields; ++field) {
		auto& Ufield = U[field];
		if (!etot_only || (etot_only && field == egas_i)) {
			for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
				for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
					for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
						Ufield[hindex(i, j, k)] = data[iter];
						++iter;
					}
				}
			}
		}
	}PROF_END;
}

std::vector<real> grid::get_hydro_boundary(const geo::direction& dir, integer width, bool etot_only) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size;
	if (!etot_only) {
		size = opts().n_fields * get_boundary_size(lb, ub, dir, INNER, INX, width);
	} else {
		size = get_boundary_size(lb, ub, dir, INNER, INX, width);
	}
	data.resize(size);
	integer iter = 0;

	for (integer field = 0; field != opts().n_fields; ++field) {
		auto& Ufield = U[field];
		if (!etot_only || (etot_only && field == egas_i)) {
			for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
				for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
					for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
						data[iter] = Ufield[hindex(i, j, k)];
						++iter;
					}
				}
			}
		}
	}PROF_END;
	return data;

}

line_of_centers_t grid::line_of_centers(const std::pair<space_vector, space_vector>& line) {
	PROF_BEGIN;
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
				const space_vector& o = 0.0;
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
	}PROF_END;
	return loc;
}

std::pair<std::vector<real>, std::vector<real>> grid::diagnostic_error() const {
	PROF_BEGIN;
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
//	printf("%e\n", e[0]);
	PROF_END;
	return e;
}

real& grid::get_omega() {
	return omega;
}

void grid::velocity_inc(const space_vector& dv) {

	for (integer iii = 0; iii != H_N3; ++iii) {
		const real rho = U[rho_i][iii];
		if (rho != ZERO) {
			const real rhoinv = ONE / rho;
			real& sx = U[sx_i][iii];
			real& sy = U[sy_i][iii];
			real& sz = U[sz_i][iii];
			real& egas = U[egas_i][iii];
			egas -= HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
			sx += dv[XDIM] * rho;
			sy += dv[YDIM] * rho;
			sz += dv[ZDIM] * rho;
			egas += HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
		}
	}

}

OCTOTIGER_FORCEINLINE real minmod(real a, real b) {
//	return (std::copysign(HALF, a) + std::copysign(HALF, b)) * std::min(std::abs(a), std::abs(b));
	bool a_is_neg = a < 0;
	bool b_is_neg = b < 0;
	if (a_is_neg != b_is_neg)
		return ZERO;

	real val = std::min(std::abs(a), std::abs(b));
	return a_is_neg ? -val : val;
}

OCTOTIGER_FORCEINLINE real minmod_theta(real a, real b, real c, real theta) {
	return minmod(theta * minmod(a, b), c);
}

OCTOTIGER_FORCEINLINE real minmod_theta(real a, real b, real theta = 1.0) {
	return minmod(theta * minmod(a, b), HALF * (a + b));
}

std::vector<real> grid::get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
		const geo::dimension& dim) const {
	PROF_BEGIN;
	std::vector<real> data;
	integer size = 1;
	for (auto& dim : geo::dimension::full_set()) {
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
					const real f = dx / TWO;
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
					value /= real(4);
					data.push_back(value);
				}
			}
		}
	}PROF_END;
	return data;
}

void grid::set_flux_restrict(const std::vector<real>& data, const std::array<integer, NDIM>& lb,
		const std::array<integer, NDIM>& ub, const geo::dimension& dim) {
	PROF_BEGIN;
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
	}PROF_END;
}

void grid::set_prolong(const std::vector<real>& data, std::vector<real>&& outflows) {
	PROF_BEGIN;
	integer index = 0;
	U_out = std::move(outflows);
	for (integer field = 0; field != opts().n_fields; ++field) {
		for (integer i = H_BW; i != H_NX - H_BW; ++i) {
			for (integer j = H_BW; j != H_NX - H_BW; ++j) {
				for (integer k = H_BW; k != H_NX - H_BW; ++k) {
					const integer iii = hindex(i, j, k);
					auto& value = U[field][iii];
					value = data[index];
					++index;
				}
			}
		}
	}PROF_END;
}

std::vector<real> grid::get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, bool etot_only) {
	PROF_BEGIN;
	auto& dUdx = TLS_dUdx();
	auto& tmpz = TLS_zz();
	std::vector<real> data;

	integer size = opts().n_fields;
	for (integer dim = 0; dim != NDIM; ++dim) {
		size *= (ub[dim] - lb[dim]);
	}
	data.reserve(size);
	auto lb0 = lb;
	auto ub0 = ub;
	for (integer d = 0; d != NDIM; ++d) {
		lb0[d] /= 2;
		ub0[d] = (ub[d] - 1) / 2 + 1;
	}
	compute_primitives(lb0, ub0, etot_only);
	compute_primitive_slopes(1.0, lb0, ub0, etot_only);
	compute_conserved_slopes(lb0, ub0, etot_only);

	if (!etot_only) {
// #if !defined(HPX_HAVE_DATAPAR)
		for (integer i = lb0[XDIM]; i != ub0[XDIM]; ++i) {
			for (integer j = lb0[YDIM]; j != ub0[YDIM]; ++j) {
#pragma GCC ivdep
				for (integer k = lb0[ZDIM]; k != ub0[ZDIM]; ++k) {
					const integer iii = hindex(i, j, k);
					tmpz[XDIM][iii] = U[zx_i][iii];
					tmpz[YDIM][iii] = U[zy_i][iii];
					tmpz[ZDIM][iii] = U[zz_i][iii];
				}
			}
		}
// #else
// #endif
	}

	for (integer field = 0; field != opts().n_fields; ++field) {
		if (!etot_only || (etot_only && field == egas_i)) {
			for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
				const real xsgn = (i % 2) ? +1 : -1;
				for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
					const real ysgn = (j % 2) ? +1 : -1;
#pragma GCC ivdep
					for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
						const integer iii = hindex(i / 2, j / 2, k / 2);
						const real zsgn = (k % 2) ? +1 : -1;
						real value = U[field][iii];
						value += xsgn * dUdx[XDIM][field][iii] * 0.25;
						value += ysgn * dUdx[YDIM][field][iii] * 0.25;
						value += zsgn * dUdx[ZDIM][field][iii] * 0.25;
						if (field == sx_i) {
							U[zy_i][iii] -= 0.25 * zsgn * value * dx / 8.0;
							U[zz_i][iii] += 0.25 * ysgn * value * dx / 8.0;
						} else if (field == sy_i) {
							U[zx_i][iii] += 0.25 * zsgn * value * dx / 8.0;
							U[zz_i][iii] -= 0.25 * xsgn * value * dx / 8.0;
						} else if (field == sz_i) {
							U[zx_i][iii] -= 0.25 * ysgn * value * dx / 8.0;
							U[zy_i][iii] += 0.25 * xsgn * value * dx / 8.0;
						}
						data.push_back(value);
					}
				}
			}
		}
	}

	if (!etot_only) {
		for (integer i = lb0[XDIM]; i != ub0[XDIM]; ++i) {
			for (integer j = lb0[YDIM]; j != ub0[YDIM]; ++j) {
#pragma GCC ivdep
				for (integer k = lb0[ZDIM]; k != ub0[ZDIM]; ++k) {
					const integer iii = hindex(i, j, k);
					U[zx_i][iii] = tmpz[XDIM][iii];
					U[zy_i][iii] = tmpz[YDIM][iii];
					U[zz_i][iii] = tmpz[ZDIM][iii];
				}
			}
		}
	}

	PROF_END;
	return data;
}

std::vector<real> grid::get_restrict() const {
	PROF_BEGIN;
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
								if (field == zx_i) {
									pt += X[YDIM][jjj] * U[sz_i][jjj];
									pt -= X[ZDIM][jjj] * U[sy_i][jjj];
								} else if (field == zy_i) {
									pt -= X[XDIM][jjj] * U[sz_i][jjj];
									pt += X[ZDIM][jjj] * U[sx_i][jjj];
								} else if (field == zz_i) {
									pt += X[XDIM][jjj] * U[sy_i][jjj];
									pt -= X[YDIM][jjj] * U[sx_i][jjj];
								}
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
	}PROF_END;
	return data;
}

void grid::set_restrict(const std::vector<real>& data, const geo::octant& octant) {
	PROF_BEGIN;
	integer index = 0;
	const integer i0 = octant.get_side(XDIM) * (INX / 2);
	const integer j0 = octant.get_side(YDIM) * (INX / 2);
	const integer k0 = octant.get_side(ZDIM) * (INX / 2);
	for (integer field = 0; field != opts().n_fields; ++field) {
		for (integer i = H_BW; i != H_NX / 2; ++i) {
			for (integer j = H_BW; j != H_NX / 2; ++j) {
				for (integer k = H_BW; k != H_NX / 2; ++k) {
					const integer iii = (i + i0) * H_DNX + (j + j0) * H_DNY + (k + k0) * H_DNZ;
					auto& v = U[field][iii];
					v = data[index];
					if (field == zx_i) {
						v -= X[YDIM][iii] * U[sz_i][iii];
						v += X[ZDIM][iii] * U[sy_i][iii];
					} else if (field == zy_i) {
						v += X[XDIM][iii] * U[sz_i][iii];
						v -= X[ZDIM][iii] * U[sx_i][iii];
					} else if (field == zz_i) {
						v -= X[XDIM][iii] * U[sy_i][iii];
						v += X[YDIM][iii] * U[sx_i][iii];
					}
					++index;
				}
			}
		}
	}PROF_END;
}

std::pair<std::vector<real>, std::vector<real> > grid::field_range() const {
	PROF_BEGIN;
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
					minmax.first[field] = std::min(minmax.first[field], U[field][iii]);
					minmax.second[field] = std::max(minmax.second[field], U[field][iii]);
				}
			}
		}
	}PROF_END;
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
		printf("++++++!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1+++++++++++++++++++++++++++++++++++++ %e %e\n", dx, dx * l);
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
		U[zx_i][i] *= (m * l2 * tinv) * l3inv;
		U[zy_i][i] *= (m * l2 * tinv) * l3inv;
		U[zz_i][i] *= (m * l2 * tinv) * l3inv;
		X[XDIM][i] *= l;
		X[YDIM][i] *= l;
		X[ZDIM][i] *= l;
//		if (std::abs(X[XDIM][i]) > 1.0e+12) {
//			printf("!!!!!!!!!!!! %e !!!!!!!!!!!!!!!!\n", std::abs(X[XDIM][i]));
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
			for (hpx::id_type const& id : options::all_localities) {
				if (id != hpx::find_here()) {
					remotes.push_back(id);
				}
			}
			if (remotes.size() > 0) {
				hpx::lcos::broadcast < set_omega_action > (remotes, omega, false).get();
			}
		}
	}
	std::unique_lock < hpx::lcos::local::spinlock > l(grid::omega_mtx, std::try_to_lock);
// if someone else has the lock, it's fine, we just return and have it set
// by the other thread
	if (!l)
		return;
	grid::omega = omega;
}

real grid::roche_volume(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, real cx,
		bool donor) const {
	PROF_BEGIN;
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
	}PROF_END;
	return V;
}

std::vector<real> grid::frac_volumes() const {
	PROF_BEGIN;
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
//	printf( "%e", V[0]);
	PROF_END;
	return V;
}

bool grid::is_in_star(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, integer frac,
		integer iii, real rho_cut) const {
	bool use = false;
	if (frac == 0) {
		use = true;
	} else {
		if (U[rho_i][iii] < rho_cut) {
			use = false;
		} else {
			space_vector a = axis.first;
			const space_vector& o = axis.second;
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

real grid::z_moments(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, integer frac,
		real rho_cut) const {
	PROF_BEGIN;
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
	}PROF_END;
	return mom;
}

std::vector<real> grid::conserved_sums(space_vector& com, space_vector& com_dot,
		const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, integer frac, real rho_cut) const {
	PROF_BEGIN;
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
					sum[zx_i] += X[YDIM][iii] * U[sz_i][iii] * dV;
					sum[zx_i] -= X[ZDIM][iii] * U[sy_i][iii] * dV;
					sum[zy_i] -= X[XDIM][iii] * U[sz_i][iii] * dV;
					sum[zy_i] += X[ZDIM][iii] * U[sx_i][iii] * dV;
					sum[zz_i] += X[XDIM][iii] * U[sy_i][iii] * dV;
					sum[zz_i] -= X[YDIM][iii] * U[sx_i][iii] * dV;
				}
			}
		}
	}
	if (sum[rho_i] > 0.0) {
		for (integer d = 0; d != NDIM; ++d) {
			com[d] /= sum[rho_i];
			com_dot[d] /= sum[rho_i];
		}
	}PROF_END;
	return sum;
}

std::vector<real> grid::gforce_sum(bool torque) const {
	PROF_BEGIN;
	std::vector<real> sum(NDIM, ZERO);
	const real dV = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const auto D = 0 - H_BW;
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				const real& rho = U[rho_i][iii];
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
	}PROF_END;
	return sum;
}

std::vector<real> grid::l_sums() const {
	PROF_BEGIN;
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
	}PROF_END;
	return sum;
}

bool grid::refine_me(integer lev, integer last_ngrids) const {
	PROF_BEGIN;
	auto test = get_refine_test();
	if (lev < 1) {
		PROF_END;
		return true;
	}
	bool rc = false;
	std::vector<real> state(opts().n_fields);
	std::array<std::vector<real>, NDIM> dud;
	std::vector<real>& dudx = dud[0];
	std::vector<real>& dudy = dud[1];
	std::vector<real>& dudz = dud[2];
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
	}PROF_END;
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
					U[si][hindex(i, j, k)] = std::max(U[si][hindex(i, j, k)], 0.0);
				}
				U[rho_i][hindex(i, j, k)] = 0.0;
				for (integer si = 0; si != opts().n_species; ++si) {
					U[rho_i][hindex(i, j, k)] += U[spc_i + si][hindex(i, j, k)];
				}
				U[rho_i][hindex(i, j, k)] = std::max(U[rho_i][hindex(i, j, k)], rho_floor);
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
	auto& M = *M_ptr;
	auto& mon = *mon_ptr;
	PROF_BEGIN;
	space_vector this_com;
	this_com[0] = this_com[1] = this_com[2] = ZERO;
	real m = ZERO;
	auto& com0 = *(com_ptr)[0];
	for (integer i = 0; i != INX + 0; ++i) {
		for (integer j = 0; j != INX + 0; ++j) {
			for (integer k = 0; k != INX + 0; ++k) {
				const integer iii = gindex(i, j, k);
				const real this_m = is_leaf ? mon[iii] : M[iii]();
				for (auto& dim : geo::dimension::full_set()) {
					this_com[dim] += this_m * com0[iii][dim];
				}
				m += this_m;
			}
		}
	}
	if (m != ZERO) {
		for (auto& dim : geo::dimension::full_set()) {
			this_com[dim] /= m;
		}
	}PROF_END;
//	printf( "kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk %e %e %e\n", this_com[0], this_com[1], this_com[2] );
	return this_com;
}

grid::grid(real _dx, std::array<real, NDIM> _xmin) :
		U(opts().n_fields), U0(opts().n_fields), dUdt(opts().n_fields), F(NDIM), X(NDIM), G(NGF), is_root(false), is_leaf(true) {
	dx = _dx;
	xmin = _xmin;
	allocate();

}

void grid::compute_primitives(const std::array<integer, NDIM> lb, const std::array<integer, NDIM> ub, bool etot_only) const {
	PROF_BEGIN;
	auto& V = TLS_V();
	if (!etot_only) {
		for (integer i = lb[XDIM] - 1; i != ub[XDIM] + 1; ++i) {
			for (integer j = lb[YDIM] - 1; j != ub[YDIM] + 1; ++j) {
#pragma GCC ivdep
				for (integer k = lb[ZDIM] - 1; k != ub[ZDIM] + 1; ++k) {
					const integer iii = hindex(i, j, k);
					V[rho_i][iii] = U[rho_i][iii];
					V[tau_i][iii] = U[tau_i][iii];
					const real rho = V[rho_i][iii];
					if (rho <= 0.0) {
						printf("%i %i %i %e\n", int(i), int(j), int(k), rho);
						abort_error()
						;
					}
					const real rhoinv = 1.0 / rho;

					if (opts().eos == WD) {
						V[egas_i][iii] = (U[egas_i][iii] - ztwd_energy(U[rho_i][iii]));	// * rhoinv;
					} else {
						V[egas_i][iii] = (U[egas_i][iii]);	// * rhoinv;
					}
					for (integer si = 0; si != opts().n_species; ++si) {
						V[spc_i + si][iii] = U[spc_i + si][iii] * rhoinv;
					}
					if (opts().gravity) {
						V[pot_i][iii] = U[pot_i][iii] * rhoinv;
					}
					for (integer d = 0; d != NDIM; ++d) {
						auto& v = V[sx_i + d][iii];
						v = U[sx_i + d][iii] * rhoinv;
						V[egas_i][iii] -= 0.5 * v /** v;*/* U[sx_i + d][iii];
						V[zx_i + d][iii] = U[zx_i + d][iii] * rhoinv;
					}

					V[sx_i][iii] += X[YDIM][iii] * omega;
					V[sy_i][iii] -= X[XDIM][iii] * omega;
					V[zz_i][iii] -= sqr(dx) * omega / 6.0;
				}
			}
		}
	} else {
		for (integer i = lb[XDIM] - 1; i != ub[XDIM] + 1; ++i) {
			for (integer j = lb[YDIM] - 1; j != ub[YDIM] + 1; ++j) {
#pragma GCC ivdep
				for (integer k = lb[ZDIM] - 1; k != ub[ZDIM] + 1; ++k) {
					const integer iii = hindex(i, j, k);
					V[rho_i][iii] = U[rho_i][iii];
					const real rhoinv = 1.0 / V[rho_i][iii];
					if (opts().eos == WD) {
						V[egas_i][iii] = (U[egas_i][iii] - ztwd_energy(U[rho_i][iii]));	// * rhoinv;
					} else {
						V[egas_i][iii] = (U[egas_i][iii]);	// * rhoinv;
					}
					for (integer d = 0; d != NDIM; ++d) {
						auto& v = V[sx_i + d][iii];
						v = U[sx_i + d][iii] * rhoinv;
						V[egas_i][iii] -= 0.5 * v /** v;*/* U[sx_i + d][iii];
						V[zx_i + d][iii] = U[zx_i + d][iii] * rhoinv;
					}
					V[sx_i][iii] += X[YDIM][iii] * omega;
					V[sy_i][iii] -= X[XDIM][iii] * omega;
					V[zz_i][iii] -= sqr(dx) * omega / 6.0;
				}
			}
		}
	}PROF_END;
}

void grid::compute_primitive_slopes(real theta, const std::array<integer, NDIM> lb, const std::array<integer, NDIM> ub,
		bool etot_only) {
	PROF_BEGIN;
	auto& dVdx = TLS_dVdx();
	auto& V = TLS_V();
	for (integer f = 0; f != opts().n_fields; ++f) {
		if (etot_only && (f == tau_i || f == pot_i || (f >= spc_i && f < spc_i + opts().n_species))) {
			continue;
		}
		const auto& v = V[f];
		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
				for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
					const integer iii = hindex(i, j, k);
					if (f != pot_i) {
						const auto v0 = v[iii];
						dVdx[XDIM][f][iii] = minmod_theta(v[iii + H_DNX] - v0, v0 - v[iii - H_DNX], theta);
						dVdx[YDIM][f][iii] = minmod_theta(v[iii + H_DNY] - v0, v0 - v[iii - H_DNY], theta);
						dVdx[ZDIM][f][iii] = minmod_theta(v[iii + H_DNZ] - v0, v0 - v[iii - H_DNZ], theta);
					} else {
						dVdx[XDIM][f][iii] = (v[iii + H_DNX] - v[iii - H_DNX]) * 0.5;
						dVdx[YDIM][f][iii] = (v[iii + H_DNY] - v[iii - H_DNY]) * 0.5;
						dVdx[ZDIM][f][iii] = (v[iii + H_DNZ] - v[iii - H_DNZ]) * 0.5;
					}
				}
			}
		}
	}
	for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
			for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
				const integer iii = hindex(i, j, k);
				real dV_sym[3][3];
				real dV_ant[3][3];
				for (integer d0 = 0; d0 != NDIM; ++d0) {
					for (integer d1 = 0; d1 != NDIM; ++d1) {
						dV_sym[d1][d0] = (dVdx[d0][sx_i + d1][iii] + dVdx[d1][sx_i + d0][iii]) / 2.0;
						dV_ant[d1][d0] = 0.0;
					}
				}
				dV_ant[XDIM][YDIM] = +6.0 * V[zz_i][iii] / dx;
				dV_ant[XDIM][ZDIM] = -6.0 * V[zy_i][iii] / dx;
				dV_ant[YDIM][ZDIM] = +6.0 * V[zx_i][iii] / dx;
				dV_ant[YDIM][XDIM] = -dV_ant[XDIM][YDIM];
				dV_ant[ZDIM][XDIM] = -dV_ant[XDIM][ZDIM];
				dV_ant[ZDIM][YDIM] = -dV_ant[YDIM][ZDIM];
				for (integer d0 = 0; d0 != NDIM; ++d0) {
					for (integer d1 = 0; d1 != NDIM; ++d1) {
						const real tmp = dV_sym[d0][d1] + dV_ant[d0][d1];
						dVdx[d0][sx_i + d1][iii] = minmod(tmp, 2.0 / theta * dVdx[d0][sx_i + d1][iii]);
					}
				}
			}
		}
	}

	PROF_END;
}

void grid::compute_conserved_slopes(const std::array<integer, NDIM> lb, const std::array<integer, NDIM> ub, bool etot_only) {
	PROF_BEGIN;
	auto& dVdx = TLS_dVdx();
	auto& dUdx = TLS_dUdx();
	auto& V = TLS_V();
	const real theta = 1.0;
	if (!etot_only) {
		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
				for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
					const integer iii = hindex(i, j, k);
					V[sx_i][iii] -= X[YDIM][iii] * omega;
					V[sy_i][iii] += X[XDIM][iii] * omega;
					V[zz_i][iii] += sqr(dx) * omega / 6.0;
					dVdx[YDIM][sx_i][iii] -= dx * omega;
					dVdx[XDIM][sy_i][iii] += dx * omega;
				}
			}
		}
		for (integer d0 = 0; d0 != NDIM; ++d0) {
			auto& dV = dVdx[d0];
			auto& dU = dUdx[d0];
			for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
				for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
					for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
						const integer iii = hindex(i, j, k);
						dU[rho_i][iii] = 0.0;
						for (integer si = 0; si != opts().n_species; ++si) {
							dU[spc_i + si][iii] = V[spc_i + si][iii] * dV[rho_i][iii] + dV[spc_i + si][iii] * V[rho_i][iii];
							dU[rho_i][iii] += dU[spc_i + si][iii];
						}
						if (opts().gravity) {
							dU[pot_i][iii] = V[pot_i][iii] * dV[rho_i][iii] + dV[pot_i][iii] * V[rho_i][iii];
						}
						//						dU[egas_i][iii] = V[egas_i][iii] * dV[rho_i][iii] + dV[egas_i][iii] * V[rho_i][iii];
						dU[egas_i][iii] = dV[egas_i][iii];
						for (integer d1 = 0; d1 != NDIM; ++d1) {
							dU[sx_i + d1][iii] = V[sx_i + d1][iii] * dV[rho_i][iii] + dV[sx_i + d1][iii] * V[rho_i][iii];
							dU[egas_i][iii] += V[rho_i][iii] * (V[sx_i + d1][iii] * dV[sx_i + d1][iii]);
							dU[egas_i][iii] += dV[rho_i][iii] * 0.5 * sqr(V[sx_i + d1][iii]);
							dU[zx_i + d1][iii] = V[zx_i + d1][iii] * dV[rho_i][iii];	// + dV[zx_i + d1][iii] * V[rho_i][iii];
						}
						if (opts().eos == WD) {
							dU[egas_i][iii] += ztwd_enthalpy(V[rho_i][iii]) * dV[rho_i][iii];
						}
						dU[tau_i][iii] = dV[tau_i][iii];
					}
				}
			}
		}
	} else {
		for (integer d0 = 0; d0 != NDIM; ++d0) {
			auto& dV = dVdx[d0];
			auto& dU = dUdx[d0];
			for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
				for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
					for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
						const integer iii = hindex(i, j, k);
						//						dU[egas_i][iii] = V[egas_i][iii] * dV[rho_i][iii] + dV[egas_i][iii] * V[rho_i][iii];
						dU[egas_i][iii] = dV[egas_i][iii];
						for (integer d1 = 0; d1 != NDIM; ++d1) {
							dU[egas_i][iii] += V[rho_i][iii] * (V[sx_i + d1][iii] * dV[sx_i + d1][iii]);
							dU[egas_i][iii] += dV[rho_i][iii] * 0.5 * sqr(V[sx_i + d1][iii]);
						}
					}
				}
			}
		}
	}

	PROF_END;
}

real grid::fgamma = 5.0 / 3.0;

void grid::set_coordinates() {
	PROF_BEGIN;
	for (integer i = 0; i != H_NX; ++i) {
		for (integer j = 0; j != H_NX; ++j) {
			for (integer k = 0; k != H_NX; ++k) {
				const integer iii = hindex(i, j, k);
				X[XDIM][iii] = (real(i - H_BW) + HALF) * dx + xmin[XDIM];
				X[YDIM][iii] = (real(j - H_BW) + HALF) * dx + xmin[YDIM];
				X[ZDIM][iii] = (real(k - H_BW) + HALF) * dx + xmin[ZDIM];
			}
		}
	}PROF_END;
}

std::vector<std::pair<std::string, std::string>> grid::get_scalar_expressions() {
	std::vector<std::pair<std::string, std::string>> rc;
	std::string rho;
	for (integer i = 0; i < opts().n_species; i++) {
		rho += "rho_" + std::to_string(i + 1) + " + ";
	}
	rho += '0';
	rc.push_back(std::make_pair(std::string("rho"), std::move(rho)));
	rc.push_back(std::make_pair(std::string("vx"), hpx::util::format("s_x / rho + {:e} * mesh(quadmesh)[0]", omega)));
	rc.push_back(std::make_pair(std::string("vy"), hpx::util::format("s_y / rho - {:e} * mesh(quadmesh)[1]", omega)));
	rc.push_back(std::make_pair(std::string("vz"), std::string("s_z / rho")));

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
	rc.push_back(
			std::make_pair(std::string("sigma_T"),
					std::string("(1 + X) * 0.2 * T * T / ((T * T + 2.7e+11 * rho) * (1 + (T / 4.5e+8)^0.86))")));
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
	rc.push_back(std::make_pair(std::string("ek"), std::string("(sx*sx+sy*sy+sz*sz)/2.0/rho")));
	rc.push_back(
			std::make_pair(std::string("ei"), hpx::util::format("if( gt(egas-ek,0.001*egas), egas-ek, tau^{:e})", fgamma)));
	const auto kb = physcon().kb * std::pow(opts().code_to_cm / opts().code_to_s, 2) * opts().code_to_g;
	if (opts().problem == MARSHAK) {
		rc.push_back(std::make_pair(std::string("T"), std::string("(ei/rho)^(1.0/3.0)")));
	} else {
		rc.push_back(std::make_pair(std::string("T"), hpx::util::format("{:e} * ei / n", 1.0 / kb / (fgamma - 1.0))));
	}
	rc.push_back(std::make_pair(std::string("P"), hpx::util::format("{:e} * ei", (fgamma - 1.0))));
	rc.push_back(
			std::make_pair(std::string("B_p"),
					hpx::util::format("{:e} * T^4",
							physcon().sigma / M_PI * opts().code_to_g * std::pow(opts().code_to_cm, 3))));
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
				const auto A = func(X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii], t);
				for (integer field = 0; field != opts().n_fields; ++field) {
					real dif = std::abs(A[field] - U[field][iii]);
					a.l1[field] += dif * dv;
					a.l2[field] += dif * dif * dv;
					a.l1a[field] += std::abs(A[field]) * dv;
					a.l2a[field] += A[field] * A[field] * dv;
					U[field][iii] = A[field];
				}
				if (opts().radiation) {
					for (integer field = opts().n_fields; field != opts().n_fields + NRF; ++field) {
						auto tmp = rad_grid_ptr->get_field(field - opts().n_fields, i - H_BW + R_BW, j - H_BW + R_BW,
								k - H_BW + R_BW);
						real dif = std::abs(A[field] - tmp);
						a.l1[field] += dif * dv;
						a.l2[field] += dif * dif * dv;
						a.l1a[field] += std::abs(A[field]) * dv;
						a.l2a[field] += A[field] * A[field] * dv;
						rad_grid_ptr->set_field(A[field], field - opts().n_fields, i - H_BW + R_BW, j - H_BW + R_BW,
								k - H_BW + R_BW);
					}
				}
				if (opts().problem == SOLID_SPHERE) {
					const auto a = solid_sphere_analytic_phi(X[0][iii], X[1][iii], X[2][iii], 0.25);
					for (int f = 0; f < 4; f++) {
						G[gindex(i - H_BW, j - H_BW, k - H_BW)][f] = a[f];
					}
				}
			}
	return a;
}

void grid::allocate() {
	PROF_BEGIN;
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

	PROF_END;
}

grid::grid() :
		U(opts().n_fields), U0(opts().n_fields), dUdt(opts().n_fields), F(NDIM), X(NDIM), G(NGF), is_root(false), is_leaf(true), U_out(
				opts().n_fields, ZERO), U_out0(opts().n_fields, ZERO), dphi_dt(H_N3) {
//	allocate();
}

grid::grid(const init_func_type& init_func, real _dx, std::array<real, NDIM> _xmin) :
		U(opts().n_fields), U0(opts().n_fields), dUdt(opts().n_fields), F(NDIM), X(NDIM), G(NGF), is_root(false), is_leaf(true), U_out(
				opts().n_fields, ZERO), U_out0(opts().n_fields, ZERO), dphi_dt(H_N3) {
	PROF_BEGIN;
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
					printf("No problem specified\n");
					abort();
				}
			}
		}
	}
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
	}PROF_END;
}

void grid::rad_init() {
	rad_grid_ptr->set_dx(dx);
	rad_grid_ptr->compute_mmw(U);
	rad_grid_ptr->initialize_erad(U[rho_i], U[tau_i]);
}

inline void limit_slope(real& ql, real q0, real& qr) {
	const real tmp1 = qr - ql;
	const real tmp2 = qr + ql;

	if (bool(qr < q0) != bool(q0 < ql)) {
		qr = ql = q0;
		return;
	}
	const real tmp3 = sqr(tmp1) * SIXTH;
	const real tmp4 = tmp1 * (q0 - 0.5 * tmp2);
	if (tmp4 > tmp3) {
		ql = 3.0 * q0 - 2.0 * qr;
	} else if (-tmp3 > tmp4) {
		qr = 3.0 * q0 - 2.0 * ql;
	}
}
;

static inline real vanleer(real a, real b) {
	if (a == 0.0 || b == 0.0) {
		return 0.0;
	} else {
		const real absa = std::abs(a);
		const real absb = std::abs(b);
		return (absa * b + absb * a) / (absa + absb);
	}
}

void grid::reconstruct() {

	PROF_BEGIN;
	auto& Uf = TLS_Uf();
	auto& dUdx = TLS_dUdx();
	auto& dVdx = TLS_dVdx();
	auto& V = TLS_V();

	auto& slpx = dUdx[XDIM];
	auto& slpy = dUdx[YDIM];
	auto& slpz = dUdx[ZDIM];

	compute_primitives();

	for (integer field = 0; field != opts().n_fields; ++field) {
		if (field >= zx_i && field <= zz_i) {
			continue;
		}
		// TODO: Remove these unused variables
		//real theta_x, theta_y, theta_z;
		std::vector<real> const& Vfield = V[field];
#pragma GCC ivdep
		for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
			if (field == sy_i || field == sz_i) {
				slpx[field][iii] = vanleer(Vfield[iii + H_DNX] - Vfield[iii], Vfield[iii] - Vfield[iii - H_DNX]);
			} else {
				slpx[field][iii] = minmod_theta(Vfield[iii + H_DNX] - Vfield[iii], Vfield[iii] - Vfield[iii - H_DNX], 2.0);
			}
			if (field == sx_i || field == sz_i) {
				slpy[field][iii] = vanleer(Vfield[iii + H_DNY] - Vfield[iii], Vfield[iii] - Vfield[iii - H_DNY]);
			} else {
				slpy[field][iii] = minmod_theta(Vfield[iii + H_DNY] - Vfield[iii], Vfield[iii] - Vfield[iii - H_DNY], 2.0);
			}
			if (field == sx_i || field == sy_i) {
				slpz[field][iii] = vanleer(Vfield[iii + H_DNZ] - Vfield[iii], Vfield[iii] - Vfield[iii - H_DNZ]);
			} else {
				slpz[field][iii] = minmod_theta(Vfield[iii + H_DNZ] - Vfield[iii], Vfield[iii] - Vfield[iii - H_DNZ], 2.0);
			}
		}
	}

//#pragma GCC ivdep
	auto step1 = [&](real& lhs, real const& rhs) {lhs += 6.0 * rhs / dx;};
	auto step2 = [&](real& lhs, real const& rhs) {lhs -= 6.0 * rhs / dx;};
	auto minmod_step = [](real& lhs, real const& r1, real const& r2, real const& r3)
	{
		lhs = minmod(lhs, vanleer(r1 - r2, r2 - r3));
	};

	for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
		inplace_average(slpx[sy_i][iii], slpy[sx_i][iii]);
		inplace_average(slpx[sz_i][iii], slpz[sx_i][iii]);
		inplace_average(slpy[sz_i][iii], slpz[sy_i][iii]);

		step1(slpx[sy_i][iii], V[zz_i][iii]);
		step1(slpy[sz_i][iii], V[zx_i][iii]);
		step1(slpz[sx_i][iii], V[zy_i][iii]);

		step2(slpy[sx_i][iii], V[zz_i][iii]);
		step2(slpz[sy_i][iii], V[zx_i][iii]);
		step2(slpx[sz_i][iii], V[zy_i][iii]);

		minmod_step(slpx[sy_i][iii], V[sy_i][iii + H_DNX], V[sy_i][iii], V[sy_i][iii - H_DNX]);
		minmod_step(slpx[sz_i][iii], V[sz_i][iii + H_DNX], V[sz_i][iii], V[sz_i][iii - H_DNX]);
		minmod_step(slpy[sx_i][iii], V[sx_i][iii + H_DNY], V[sx_i][iii], V[sx_i][iii - H_DNY]);
		minmod_step(slpy[sz_i][iii], V[sz_i][iii + H_DNY], V[sz_i][iii], V[sz_i][iii - H_DNY]);
		minmod_step(slpz[sx_i][iii], V[sx_i][iii + H_DNZ], V[sx_i][iii], V[sx_i][iii - H_DNZ]);
		minmod_step(slpz[sy_i][iii], V[sy_i][iii + H_DNZ], V[sy_i][iii], V[sy_i][iii - H_DNZ]);

		const real zx_lim = +(slpy[sz_i][iii] - slpz[sy_i][iii]) / 12.0;
		const real zy_lim = -(slpx[sz_i][iii] - slpz[sx_i][iii]) / 12.0;
		const real zz_lim = +(slpx[sy_i][iii] - slpy[sx_i][iii]) / 12.0;

		const real Vzxi = V[zx_i][iii] - zx_lim * dx;
		const real Vzyi = V[zy_i][iii] - zy_lim * dx;
		const real Vzzi = V[zz_i][iii] - zz_lim * dx;

		for (int face = 0; face != NFACE; ++face) {
			Uf[face][zx_i][iii] = Vzxi;
			Uf[face][zy_i][iii] = Vzyi;
			Uf[face][zz_i][iii] = Vzzi;
		}
	}
	for (integer field = 0; field != opts().n_fields; ++field) {
		std::vector<real>& Vfield = V[field];

		std::vector<real>& UfFXPfield = Uf[FXP][field];
		std::vector<real>& UfFXMfield = Uf[FXM][field];
		std::vector<real> const& slpxfield = slpx[field];

		if (field >= zx_i && field <= zz_i) {
			continue;
		}
		if (!(field == sy_i || field == sz_i)) {
#pragma GCC ivdep
			for (integer iii = 0; iii != H_N3 - H_NX * H_NX; ++iii) {
				UfFXPfield[iii] = UfFXMfield[iii + H_DNX] = average(Vfield[iii + H_DNX], Vfield[iii]);
			}
#pragma GCC ivdep
			for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
				const real& sx = slpxfield[iii];
				UfFXPfield[iii] += (-(slpxfield[iii + H_DNX] - sx) / 3.0) * HALF;
				UfFXMfield[iii] += ((slpxfield[iii - H_DNX] - sx) / 3.0) * HALF;
				limit_slope(UfFXMfield[iii], Vfield[iii], UfFXPfield[iii]);
			}
		} else {
#pragma GCC ivdep
			for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
				const real& u0 = Vfield[iii];
				const real slpxfield_half = slpxfield[iii] * HALF;
				UfFXPfield[iii] = u0 + slpxfield_half;
				UfFXMfield[iii] = u0 - slpxfield_half;
			}
		}

		std::vector<real>& UfFYPfield = Uf[FYP][field];
		std::vector<real>& UfFYMfield = Uf[FYM][field];
		std::vector<real> const& slpyfield = slpy[field];

		if (!(field == sx_i || field == sz_i)) {
#pragma GCC ivdep
			for (integer iii = 0; iii != H_N3 - H_NX * H_NX; ++iii) {
				UfFYPfield[iii] = UfFYMfield[iii + H_DNY] = average(Vfield[iii + H_DNY], Vfield[iii]);
			}
#pragma GCC ivdep
			for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
				const real& sy = slpyfield[iii];
				UfFYPfield[iii] += (-(slpyfield[iii + H_DNY] - sy) / 3.0) * HALF;
				UfFYMfield[iii] += ((slpyfield[iii - H_DNY] - sy) / 3.0) * HALF;
				limit_slope(UfFYMfield[iii], Vfield[iii], UfFYPfield[iii]);
			}
		} else {
#pragma GCC ivdep
			for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
				const real& u0 = Vfield[iii];
				const real slpyfield_half = slpyfield[iii] * HALF;
				UfFYPfield[iii] = u0 + slpyfield_half;
				UfFYMfield[iii] = u0 - slpyfield_half;
			}
		}

		std::vector<real>& UfFZPfield = Uf[FZP][field];
		std::vector<real>& UfFZMfield = Uf[FZM][field];
		std::vector<real> const& slpzfield = slpz[field];

		if (!(field == sx_i || field == sy_i)) {
#pragma GCC ivdep
			for (integer iii = 0; iii != H_N3 - H_NX * H_NX; ++iii) {
				UfFZPfield[iii] = UfFZMfield[iii + H_DNZ] = average(Vfield[iii + H_DNZ], Vfield[iii]);
			}
#pragma GCC ivdep
			for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
				const real& sz = slpzfield[iii];
				UfFZPfield[iii] += (-(slpzfield[iii + H_DNZ] - sz) / 3.0) * HALF;
				UfFZMfield[iii] += ((slpzfield[iii - H_DNZ] - sz) / 3.0) * HALF;
				limit_slope(UfFZMfield[iii], Vfield[iii], UfFZPfield[iii]);
			}
		} else {
#pragma GCC ivdep
			for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
				const real& u0 = Vfield[iii];
				const real slpzfield_half = slpzfield[iii] * HALF;
				UfFZPfield[iii] = u0 + slpzfield_half;
				UfFZMfield[iii] = u0 - slpzfield_half;
			}
		}
	}

	for (integer iii = 0; iii != H_N3; ++iii) {
#pragma GCC ivdep
		for (integer face = 0; face != NFACE; ++face) {
			real w = 0.0;
			std::vector<std::vector<real> >& Ufface = Uf[face];
			for (integer si = 0; si != opts().n_species; ++si) {
				w += Ufface[spc_i + si][iii];
			}
			if (w > ZERO) {
				for (integer si = 0; si != opts().n_species; ++si) {
					Ufface[spc_i + si][iii] /= w;
				}
			}
		}
	}

	if (opts().gravity) {
//#pragma GCC ivdep
		std::vector<real>& UfFXMpot_i = Uf[FXM][pot_i];
		std::vector<real>& UfFYMpot_i = Uf[FYM][pot_i];
		std::vector<real>& UfFZMpot_i = Uf[FZM][pot_i];

		std::vector<real>& UfFXPpot_i = Uf[FXP][pot_i];
		std::vector<real>& UfFYPpot_i = Uf[FYP][pot_i];
		std::vector<real>& UfFZPpot_i = Uf[FZP][pot_i];

		for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
			const real phi_x = HALF * (UfFXMpot_i[iii] + UfFXPpot_i[iii - H_DNX]);
			const real phi_y = HALF * (UfFYMpot_i[iii] + UfFYPpot_i[iii - H_DNY]);
			const real phi_z = HALF * (UfFZMpot_i[iii] + UfFZPpot_i[iii - H_DNZ]);
			UfFXMpot_i[iii] = phi_x;
			UfFYMpot_i[iii] = phi_y;
			UfFZMpot_i[iii] = phi_z;
			UfFXPpot_i[iii - H_DNX] = phi_x;
			UfFYPpot_i[iii - H_DNY] = phi_y;
			UfFZPpot_i[iii - H_DNZ] = phi_z;
		}
	}
	for (integer field = 0; field != opts().n_fields; ++field) {
		if (field != rho_i && field != tau_i && field != egas_i) {
#pragma GCC ivdep
			for (integer face = 0; face != NFACE; ++face) {
				std::vector<real>& Uffacefield = Uf[face][field];
				std::vector<real> const& Uffacerho_i = Uf[face][rho_i];
				for (integer iii = 0; iii != H_N3; ++iii) {
					Uffacefield[iii] *= Uffacerho_i[iii];
				}
			}
		}
	}

	for (integer i = H_BW - 1; i != H_NX - H_BW + 1; ++i) {
		for (integer j = H_BW - 1; j != H_NX - H_BW + 1; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW - 1; k != H_NX - H_BW + 1; ++k) {
				const integer iii = hindex(i, j, k);
				for (integer face = 0; face != NFACE; ++face) {
					std::vector<std::vector<real> >& Ufface = Uf[face];
					real const Uffacerho_iii = Ufface[rho_i][iii];

					real x0 = ZERO;
					real y0 = ZERO;
					if (face == FXP) {
						x0 = +HALF * dx;
					} else if (face == FXM) {
						x0 = -HALF * dx;
					} else if (face == FYP) {
						y0 = +HALF * dx;
					} else if (face == FYM) {
						y0 = -HALF * dx;
					}

					Ufface[sx_i][iii] -= omega * (X[YDIM][iii] + y0) * Uffacerho_iii;
					Ufface[sy_i][iii] += omega * (X[XDIM][iii] + x0) * Uffacerho_iii;
					Ufface[zz_i][iii] += sqr(dx) * omega * Uffacerho_iii / 6.0;
					Ufface[egas_i][iii] += HALF * sqr(Ufface[sx_i][iii]) / Uffacerho_iii;
					Ufface[egas_i][iii] += HALF * sqr(Ufface[sy_i][iii]) / Uffacerho_iii;
					Ufface[egas_i][iii] += HALF * sqr(Ufface[sz_i][iii]) / Uffacerho_iii;
					if (opts().eos == WD) {
						Ufface[egas_i][iii] += ztwd_energy(Uffacerho_iii);
					}
				}
			}
		}
	}PROF_END;
}

real grid::compute_fluxes() {
	if (opts().radiation) {
		rad_grid_ptr->set_dx(dx);
	}PROF_BEGIN;
	const auto& Uf = TLS_Uf();
	real max_lambda = ZERO;
	hydro_state_t<std::vector<real>> ur, ul, f;
	std::vector<space_vector> x;

	const integer line_sz = H_NX - 2 * H_BW + 1;
	for (integer field = 0; field != opts().n_fields; ++field) {
		ur[field].resize(line_sz);
		ul[field].resize(line_sz);
		f[field].resize(line_sz);
	}
	x.resize(line_sz);

	for (integer dim = 0; dim != NDIM; ++dim) {

		const integer dx_i = dim;
		const integer dy_i = (dim == XDIM ? YDIM : XDIM);
		const integer dz_i = (dim == ZDIM ? YDIM : ZDIM);
		const integer face_p = 2 * dim + 1;
		const integer face_m = 2 * dim;
		for (integer k = H_BW; k != H_NX - H_BW; ++k) {
			for (integer j = H_BW; j != H_NX - H_BW; ++j) {
				for (integer i = H_BW; i != H_NX - H_BW + 1; ++i) {
					const integer i0 = H_DN[dx_i] * i + H_DN[dy_i] * j + H_DN[dz_i] * k;
					const integer im = i0 - H_DN[dx_i];
					for (integer field = 0; field != opts().n_fields; ++field) {
						ur[field][i - H_BW] = Uf[face_m][field][i0];
						ul[field][i - H_BW] = Uf[face_p][field][im];
					}
					for (integer d = 0; d != NDIM; ++d) {
						x[i - H_BW][d] = (X[d][i0] + X[d][im]) * HALF;
					}
				}
				const real this_max_lambda = roe_fluxes(f, ul, ur, x, omega, dim, dx);
				max_lambda = std::max(max_lambda, this_max_lambda);
				for (integer field = 0; field != opts().n_fields; ++field) {
					for (integer i = H_BW; i != H_NX - H_BW + 1; ++i) {
						const integer i0 = F_DN[dx_i] * (i - H_BW) + F_DN[dy_i] * (j - H_BW) + F_DN[dz_i] * (k - H_BW);
						F[dim][field][i0] = f[field][i - H_BW];
					}
				}
				for (integer i = H_BW; i != H_NX - H_BW + 1; ++i) {
					real rho_tot = 0.0;
					const integer i0 = F_DN[dx_i] * (i - H_BW) + F_DN[dy_i] * (j - H_BW) + F_DN[dz_i] * (k - H_BW);
					for (integer field = spc_i; field != spc_i + opts().n_species; ++field) {
						rho_tot += F[dim][field][i0];
					}
					F[dim][rho_i][i0] = rho_tot;
				}
			}
		}
	}
	/*
	 if (false && opts().radiation) {
	 const auto& egas = get_field(egas_i);
	 const auto& rho = get_field(rho_i);
	 const auto& tau = get_field(tau_i);
	 const auto& sx = get_field(sx_i);
	 const auto& sy = get_field(sy_i);
	 const auto& sz = get_field(sz_i);
	 const real b = rad_grid_ptr->hydro_signal_speed(egas, tau, sx, sy, sz, rho);
	 max_lambda += b;
	 }	PROF_END;
	 */

//	return physcon.c;
//	printf( "%e %e %e\n", omega, dx, (2./15.) / max_lambda);
	if (opts().hard_dt < 0.0) {
		return max_lambda;
	} else {
		return dx / opts().hard_dt;
	}
	if (opts().radiation && !opts().hydro) {
		return physcon().c / 10.0;
	}
	return max_lambda;
}

void grid::set_max_level(integer l) {
	max_level = l;
}

void grid::store() {
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

void grid::set_physical_boundaries(const geo::face& face, real t) {
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

	if (opts().problem == SOD) {
		for (integer k = klb; k != kub; ++k) {
			for (integer j = jlb; j != jub; ++j) {
				for (integer i = ilb; i != iub; ++i) {
					const integer iii = i * dni + j * dnj + k * dnk;
					for (integer f = 0; f != opts().n_fields; ++f) {
						U[f][iii] = 0.0;
					}
					sod_state_t s;
					real x = (X[XDIM][iii] + X[YDIM][iii] + X[ZDIM][iii]) / std::sqrt(3.0);
					exact_sod(&s, &sod_init, x, t);
					U[rho_i][iii] = s.rho;
					U[egas_i][iii] = s.p / (fgamma - 1.0);
					U[sx_i][iii] = s.rho * s.v / std::sqrt(3.0);
					U[sy_i][iii] = s.rho * s.v / std::sqrt(3.0);
					U[sz_i][iii] = s.rho * s.v / std::sqrt(3.0);
					U[tau_i][iii] = std::pow(U[egas_i][iii], 1.0 / fgamma);
					U[egas_i][iii] += s.rho * s.v * s.v / 2.0;
					U[spc_i][iii] = s.rho;
					integer k0 = side == geo::MINUS ? H_BW : H_NX - H_BW - 1;
				}
			}
		}
	} else {
		for (integer field = 0; field != opts().n_fields; ++field) {
			for (integer k = klb; k != kub; ++k) {
				for (integer j = jlb; j != jub; ++j) {
					for (integer i = ilb; i != iub; ++i) {
						integer k0;
						switch (boundary_types[face]) {
						case REFLECT:
							k0 = side == geo::MINUS ? (2 * H_BW - k - 1) : (2 * (H_NX - H_BW) - k - 1);
							break;
						case OUTFLOW:
							k0 = side == geo::MINUS ? H_BW : H_NX - H_BW - 1;
							break;
						default:
							k0 = -1;
							assert(false);
							abort();
						}
						const real value = U[field][i * dni + j * dnj + k0 * dnk];
						const integer iii = i * dni + j * dnj + k * dnk;
						real& ref = U[field][iii];
						if (field == sx_i + dim) {
							real s0;
							if (field == sx_i) {
								s0 = -omega * X[YDIM][iii] * U[rho_i][iii];
							} else if (field == sy_i) {
								s0 = +omega * X[XDIM][iii] * U[rho_i][iii];
							} else {
								s0 = ZERO;
							}
							switch (boundary_types[face]) {
							case REFLECT:
								ref = -value;
								break;
							case OUTFLOW:
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
								break;
							}
//						} else if (field == rho_i) {
//							ref = std::max(rho_floor,value);
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
	PROF_BEGIN;
	auto& src = dUdt;
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
				src[zx_i][iii0] = (-(F[YDIM][sz_i][iiif + F_DNY] + F[YDIM][sz_i][iiif])
						+ (F[ZDIM][sy_i][iiif + F_DNZ] + F[ZDIM][sy_i][iiif])) * HALF;
				src[zy_i][iii0] = (+(F[XDIM][sz_i][iiif + F_DNX] + F[XDIM][sz_i][iiif])
						- (F[ZDIM][sx_i][iiif + F_DNZ] + F[ZDIM][sx_i][iiif])) * HALF;
				src[zz_i][iii0] = (-(F[XDIM][sy_i][iiif + F_DNX] + F[XDIM][sy_i][iiif])
						+ (F[YDIM][sx_i][iiif + F_DNY] + F[YDIM][sx_i][iiif])) * HALF;
				if (opts().gravity) {
					src[sx_i][iii0] += rho * G[iiig][gx_i];
					src[sy_i][iii0] += rho * G[iiig][gy_i];
					src[sz_i][iii0] += rho * G[iiig][gz_i];
				}
				src[sx_i][iii0] += omega * U[sy_i][iii];
				src[sy_i][iii0] -= omega * U[sx_i][iii];
				if (opts().gravity) {
					src[egas_i][iii0] -= omega * X[YDIM][iii] * rho * G[iiig][gx_i];
					src[egas_i][iii0] += omega * X[XDIM][iii] * rho * G[iiig][gy_i];
				}
				if (opts().driving_rate != 0.0) {
					const real period_len = 2.0 * M_PI / grid::omega;
					if (opts().driving_time > rotational_time / (2.0 * M_PI)) {
						const real ff = -opts().driving_rate / period_len;
						///	printf("%e %e %e\n", ff, opts().driving_rate, period_len);
						const real rho = U[rho_i][iii];
						const real sx = U[sx_i][iii];
						const real sy = U[sy_i][iii];
						const real zz = U[zz_i][iii];
						const real x = X[XDIM][iii];
						const real y = X[YDIM][iii];
						const real R = std::sqrt(x * x + y * y);
						const real lz = (x * sy - y * sx);
						const real dsx = -y / R / R * lz * ff;
						const real dsy = +x / R / R * lz * ff;
						src[sx_i][iii0] += dsx;
						src[sy_i][iii0] += dsy;
						src[egas_i][iii0] += (sx * dsx + sy * dsy) / rho;
						src[zz_i][iii0] += ff * zz;
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
			}
		}
	}PROF_END;
}

void grid::compute_dudt() {
	PROF_BEGIN;
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
	}PROF_END;
//	solve_gravity(DRHODT);
}

void grid::egas_to_etot() {
	PROF_BEGIN;
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
	}PROF_END;
}

void grid::etot_to_egas() {
	PROF_BEGIN;
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
	}PROF_END;
}

void grid::next_u(integer rk, real t, real dt) {
	if (!opts().hydro) {
		return;
	}
//	return;

	PROF_BEGIN;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				const integer iii = hindex(i, j, k);
				dUdt[egas_i][iii0] += (dphi_dt[iii0] * U[rho_i][iii]) * HALF;
				dUdt[zx_i][iii0] -= omega * X[ZDIM][iii] * U[sx_i][iii];
				dUdt[zy_i][iii0] -= omega * X[ZDIM][iii] * U[sy_i][iii];
				dUdt[zz_i][iii0] += omega * (X[XDIM][iii] * U[sx_i][iii] + X[YDIM][iii] * U[sy_i][iii]);
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
			if (X[XDIM][iii_p] > scaling_factor) {
				const real xp = X[XDIM][iii_p] - HALF * dx;
				du[zx_i] += (X[YDIM][iii_p] * F[XDIM][sz_i][iii_p0]) * dx2;
				du[zx_i] -= (X[ZDIM][iii_p] * F[XDIM][sy_i][iii_p0]) * dx2;
				du[zy_i] -= (xp * F[XDIM][sz_i][iii_p0]) * dx2;
				du[zy_i] += (X[ZDIM][iii_p] * F[XDIM][sx_i][iii_p0]) * dx2;
				du[zz_i] += (xp * F[XDIM][sy_i][iii_p0]) * dx2;
				du[zz_i] -= (X[YDIM][iii_p] * F[XDIM][sx_i][iii_p0]) * dx2;
			}
			if (X[YDIM][jjj_p] > scaling_factor) {
				const real yp = X[YDIM][jjj_p] - HALF * dx;
				du[zx_i] += (yp * F[YDIM][sz_i][jjj_p0]) * dx2;
				du[zx_i] -= (X[ZDIM][jjj_p] * F[YDIM][sy_i][jjj_p0]) * dx2;
				du[zy_i] -= (X[XDIM][jjj_p] * F[YDIM][sz_i][jjj_p0]) * dx2;
				du[zy_i] += (X[ZDIM][jjj_p] * F[YDIM][sx_i][jjj_p0]) * dx2;
				du[zz_i] += (X[XDIM][jjj_p] * F[YDIM][sy_i][jjj_p0]) * dx2;
				du[zz_i] -= (yp * F[YDIM][sx_i][jjj_p0]) * dx2;
			}
			if (X[ZDIM][kkk_p] > scaling_factor) {
				const real zp = X[ZDIM][kkk_p] - HALF * dx;
				du[zx_i] -= (zp * F[ZDIM][sy_i][kkk_p0]) * dx2;
				du[zx_i] += (X[YDIM][kkk_p] * F[ZDIM][sz_i][kkk_p0]) * dx2;
				du[zy_i] += (zp * F[ZDIM][sx_i][kkk_p0]) * dx2;
				du[zy_i] -= (X[XDIM][kkk_p] * F[ZDIM][sz_i][kkk_p0]) * dx2;
				du[zz_i] += (X[XDIM][kkk_p] * F[ZDIM][sy_i][kkk_p0]) * dx2;
				du[zz_i] -= (X[YDIM][kkk_p] * F[ZDIM][sx_i][kkk_p0]) * dx2;
			}

			if (X[XDIM][iii_m] < -scaling_factor + dx) {
				const real xm = X[XDIM][iii_m] - HALF * dx;
				du[zx_i] += (-X[YDIM][iii_m] * F[XDIM][sz_i][iii_m0]) * dx2;
				du[zx_i] -= (-X[ZDIM][iii_m] * F[XDIM][sy_i][iii_m0]) * dx2;
				du[zy_i] -= (-xm * F[XDIM][sz_i][iii_m0]) * dx2;
				du[zy_i] += (-X[ZDIM][iii_m] * F[XDIM][sx_i][iii_m0]) * dx2;
				du[zz_i] += (-xm * F[XDIM][sy_i][iii_m0]) * dx2;
				du[zz_i] -= (-X[YDIM][iii_m] * F[XDIM][sx_i][iii_m0]) * dx2;
			}
			if (X[YDIM][jjj_m] < -scaling_factor + dx) {
				const real ym = X[YDIM][jjj_m] - HALF * dx;
				du[zx_i] -= (-X[ZDIM][jjj_m] * F[YDIM][sy_i][jjj_m0]) * dx2;
				du[zx_i] += (-ym * F[YDIM][sz_i][jjj_m0]) * dx2;
				du[zy_i] -= (-X[XDIM][jjj_m] * F[YDIM][sz_i][jjj_m0]) * dx2;
				du[zy_i] += (-X[ZDIM][jjj_m] * F[YDIM][sx_i][jjj_m0]) * dx2;
				du[zz_i] += (-X[XDIM][jjj_m] * F[YDIM][sy_i][jjj_m0]) * dx2;
				du[zz_i] -= (-ym * F[YDIM][sx_i][jjj_m0]) * dx2;
			}
			if (X[ZDIM][kkk_m] < -scaling_factor + dx) {
				const real zm = X[ZDIM][kkk_m] - HALF * dx;
				du[zx_i] -= (-zm * F[ZDIM][sy_i][kkk_m0]) * dx2;
				du[zx_i] += (-X[YDIM][kkk_m] * F[ZDIM][sz_i][kkk_m0]) * dx2;
				du[zy_i] += (-zm * F[ZDIM][sx_i][kkk_m0]) * dx2;
				du[zy_i] -= (-X[XDIM][kkk_m] * F[ZDIM][sz_i][kkk_m0]) * dx2;
				du[zz_i] += (-X[XDIM][kkk_m] * F[ZDIM][sy_i][kkk_m0]) * dx2;
				du[zz_i] -= (-X[YDIM][kkk_m] * F[ZDIM][sx_i][kkk_m0]) * dx2;
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

				if (U[tau_i][iii] < ZERO) {
					printf("Tau is negative- %e %i %i %i  %e %e %e\n", double(U[tau_i][iii]), int(i), int(j), int(k),
							X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii]);
					//	abort();
				} else if (U[rho_i][iii] <= ZERO) {
					printf("Rho is non-positive - %e %i %i %i %e %e %e\n", double(U[rho_i][iii]), int(i), int(j), int(k), double(X[XDIM][iii]),double(X[YDIM][iii]),double(X[ZDIM][iii]));
					abort();
				}
			}
		}
	}PROF_END;
}

void grid::dual_energy_update() {
	PROF_BEGIN;
//	bool in_bnd;
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
				et = std::max(et, U[egas_i][iii + H_DNX]);
				et = std::max(et, U[egas_i][iii - H_DNX]);
				et = std::max(et, U[egas_i][iii + H_DNY]);
				et = std::max(et, U[egas_i][iii - H_DNY]);
				et = std::max(et, U[egas_i][iii + H_DNZ]);
				et = std::max(et, U[egas_i][iii - H_DNZ]);
				if (ei > de_switch1 * et) {
					U[tau_i][iii] = std::pow(ei, ONE / fgamma);
				}
			}
		}
	}PROF_END;
}

std::pair<real, real> grid::virial() const {
	PROF_BEGIN;
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
	}PROF_END;
	return v;
}

std::vector<real> grid::conserved_outflows() const {
	auto Uret = U_out;
	if (opts().gravity) {
		Uret[egas_i] += Uret[pot_i];
	}
	return Uret;
}
