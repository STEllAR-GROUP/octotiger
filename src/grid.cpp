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
#include "octotiger/io/silo.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/problem.hpp"
#include "octotiger/profiler.hpp"
#include "octotiger/taylor.hpp"
#include "octotiger/test_problems/amr/amr.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/hydro_impl/flux.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct.hpp"

#include <hpx/collectives/broadcast_direct.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/synchronization/once.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <unordered_map>

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"
// #include "octotiger/unitiger/hydro_impl/hydro_cuda_interface.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"

#if !defined(HPX_COMPUTE_DEVICE_CODE)

std::vector<int> grid::field_bw;
std::vector<int> grid::energy_bw;
std::unordered_map<std::string, int> grid::str_to_index_hydro;
std::unordered_map<std::string, int> grid::str_to_index_gravity;
std::unordered_map<int, std::string> grid::index_to_str_hydro;
std::unordered_map<int, std::string> grid::index_to_str_gravity;
double grid::idle_rate;

bool grid::is_hydro_field(const std::string& str) {
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

void grid::set_outflows(std::vector<std::pair<std::string, real>>&& u) {
    for (const auto& p : u) {
        U_out[str_to_index_hydro[p.first]] = p.second;
    }
}

void grid::set_outflow(std::pair<std::string, real>&& p) {
    U_out[str_to_index_hydro[p.first]] = p.second;
    U_out[rho_i] = ZERO;
    for (integer s = 0; s < opts().n_species; s++) {
        U_out[rho_i] += U_out[spc_i + s];
    }
}

void grid::static_init() {
    field_bw.resize(opts().n_fields, 3);
    energy_bw.resize(opts().n_fields, 0);
    energy_bw[egas_i] = 1;
    for (integer dim = 0; dim < NDIM; dim++) {
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
    for (const auto& s : str_to_index_hydro) {
        index_to_str_hydro[s.second] = s.first;
    }
    for (const auto& s : str_to_index_gravity) {
        index_to_str_gravity[s.second] = s.first;
    }
}

std::vector<std::string> grid::get_field_names() {
    std::vector<std::string> rc = get_hydro_field_names();
    if (opts().gravity) {
        for (auto i : str_to_index_gravity) {
            rc.push_back(i.first);
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

void grid::set(const std::string name, real* data, int version) {
    PROFILE();
    auto iter = str_to_index_hydro.find(name);
    real unit = convert_hydro_units(iter->second);

    if (iter != str_to_index_hydro.end()) {
        integer f = iter->second;

        /* Correct for bugfix across versions */
        if (version == 100 && f >= sx_i && f <= sz_i) {
            unit /= opts().code_to_s;
        }
        integer jjj = 0;
        for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
            U[f][iii] = data[jjj] / unit;
            jjj++;
        }
    }
}

void grid::rho_from_species() {
    for (integer iii = 0; iii < extBox.volume(); iii++) {
        U[rho_i][iii] = ZERO;
        for (integer s = 0; s < opts().n_species; s++) {
            U[rho_i][iii] += U[spc_i + s][iii];
        }
    }
}

real grid::convert_hydro_units(int i) {
    real val = ONE;
    if (opts().problem != MARSHAK) {
        real const cm = opts().code_to_cm;
        // printf( "%e\n", cm);
        real const s = opts().code_to_s;
        real const g = opts().code_to_g;
        if (i >= spc_i && i <= spc_i + opts().n_species) {
            val *= g / (cm * cm * cm);
        } else if (i >= sx_i && i <= sz_i) {
            val *= g / (s * cm * cm);
        } else if (i == egas_i || i == pot_i) {
            val *= g / (s * s * cm);
        } else if ((i >= lx_i && i <= lz_i)) {
            val *= g / (s * cm);
        } else if (i == tau_i) {
            if (opts().eos != IPR) {    // for IPR eos, tau is actually the temperature
                val *= POWER(g / (s * s * cm), ONE / fgamma);
            }
        } else {
            printf("Asked to convert units for unknown field %i\n", i);
            abort();
        }
    }
    return val;
}

real grid::convert_gravity_units(int i) {
    real val = ONE;
    real const cm = opts().code_to_cm;
    real const s = opts().code_to_s;
    real const g = opts().code_to_g;
    if (i == phi_i) {
        val *= cm * cm / s / s;
    } else {
        val *= cm / s / s;
    }
    return val;
}

std::vector<grid::roche_type> grid::get_roche_lobe() const {
    std::vector<grid::roche_type> this_s(intBox.volume());
    for (integer iii = 0; iii < intBox.volume(); iii++) {
        this_s[iii] = roche_lobe[iii];
    }
    return std::move(this_s);
}

std::string grid::hydro_units_name(const std::string& nm) {
    integer f = str_to_index_hydro[nm];
    if (f >= spc_i && f <= spc_i + opts().n_species) {
        return "g / cm^3";
    } else if (f >= sx_i && f <= sz_i) {
        return "g / (cm s)^2 ";
    } else if (f == egas_i || (f >= lx_i && f <= lz_i)) {
        return "g / (cm s^2)";
    } else if (f == tau_i) {
        if (opts().eos == IPR) {    // for IPR eos, tau is actually the temperature
            return "cm / cm";
        } else {
            return "(g / cm)^(3/5) / s^(6/5)";
        }
    }
    return "<unknown>";
}

std::string grid::gravity_units_name(const std::string& nm) {
    integer f = str_to_index_gravity[nm];
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
    const auto& x0 = opts().silo_offset_x;
    const auto& y0 = opts().silo_offset_y;
    const auto& z0 = opts().silo_offset_z;
    for (auto l : str_to_index_hydro) {
        unit = convert_hydro_units(l.second);
        integer const f = l.second;

        std::string this_name = l.first;
        integer jjj = 0;
        silo_var_t this_s(this_name);
        for (integer iii = 0; iii < extBox.volume(); iii++) {
            this_s(iii) = U[f][iii] * unit;
            this_s.set_range(this_s(iii));
            if (fabs(this_s(iii)) > 1e100) {
                printf("%e\n", this_s(iii));
            }
        }
        s.push_back(std::move(this_s));
    }
    //	}

    if (opts().gravity) {
        for (auto l : str_to_index_gravity) {
            unit = convert_gravity_units(l.second);
            integer const f = l.second;
            std::string this_name = l.first;
            silo_var_t this_s(this_name);
            for (integer iii = 0; iii < extBox.volume(); iii++) {
                this_s(iii) = G[iii][f] * unit;
                this_s.set_range(this_s(iii));
            }
            s.push_back(std::move(this_s));
        }
    }
    if (opts().idle_rates) {
        integer const id = hpx::get_locality_id();
        {
            silo_var_t this_s("locality");
            for (integer iii = 0; iii < extBox.volume(); iii++) {
                this_s(iii) = id;
                this_s.set_range(this_s(iii));
            }
            s.push_back(std::move(this_s));
        }
        {
            silo_var_t this_s("idle_rate");
            for (integer iii = 0; iii < extBox.volume(); iii++) {
                this_s(iii) = idle_rate;
                this_s.set_range(idle_rate);
            }
            s.push_back(std::move(this_s));
        }
    }

    //	{
    //
    //		integer jjj = 0;
    //		silo_var_t this_s("roche_lobe");
    //		for (integer i = 0; i < INX; i++) {
    //			for (integer j = 0; j < INX; j++) {
    //				for (integer k = 0; k < INX; k++) {
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
    std::string counter_name =
        "/threads{" + std::to_string(hpx::get_locality_id()) + "/total}/idle-rate";
    hpx::performance_counters::performance_counter count(counter_name);
    idle_rate = count.get_value<double>().get();
    count.reset();
}

// MSVC needs this variable to be in the global namespace
constexpr integer nspec = 2;
diagnostics_t grid::diagnostics(const diagnostics_t& diags) {
    PROFILE();
    diagnostics_t rc;
    if (opts().disable_diagnostics) {
        return rc;
    }
    real const dV = pow(dx, NDIM);
    real x, y, z;
    if (opts().problem != DWD) {
        integer iiig = 0;
        for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii, ++iiig) {
            real dyz2 = ZERO;
            for (integer d = 1; d < NDIM; d++) {
                dyz2 += sqr(X[d][iii]);
            }
            if (dyz2 < sqr(dx)) {
                rc.xline.push_back(std::make_pair(X[XDIM][iii], std::vector<real>()));
                for (integer fi = 0; fi < opts().n_fields; fi++) {
                    rc.xline.back().second.push_back(U[fi][iii]);
                }
            }
            real ek = ZERO;
            real const irho = ONE / U[rho_i][iii];
            for (integer d = 0; d < NDIM; d++) {
                ek += irho * sqr(U[sx_i + d][iii]);
            }
            ek *= HALF;
            real ei;
            if (opts().eos == WD) {
                ei = U[egas_i][iii] - ek - ztwd_energy(U[rho_i][iii]);
            } else {
                ei = U[egas_i][iii] - ek;
            }
            real et = U[egas_i][iii];
            real p;
            if (opts().eos == IPR) {
                ei = std::max(opts().ipr_eint_floor, ei);
                specie_state_t<real> spc;
                real mmw_loc, X_loc, Z_loc;
                for (integer si = 0; si != opts().n_species; ++si) {
                    spc[si] = U[spc_i + si][iii];
                }
                mean_ion_weight(spc, mmw_loc, X_loc, Z_loc);
                p = ipr_pressure(U[tau_i][iii], U[rho_i][iii], mmw_loc);
            } else {
                if (ei < de_switch2 * et) {
                    ei = POWER(U[tau_i][iii], fgamma);
                }
                p = (fgamma - ONE) * ei;
                if (opts().eos == WD) {
                    p += ztwd_pressure(U[rho_i][iii]);
                }
            }
            if (opts().gravity) {
                rc.virial +=
                    (2.0 * ek + 0.5 * U[rho_i][iii] * G[iiig][phi_i] + 3.0 * p) * (dx * dx * dx);
                rc.virial_norm +=
                    (2.0 * ek - 0.5 * U[rho_i][iii] * G[iiig][phi_i] + 3.0 * p) * (dx * dx * dx);
            } else {
                rc.virial_norm = ONE;
            }
            for (integer f = 0; f != opts().n_fields; ++f) {
                rc.grid_sum[f] += U[f][iii] * dV;
            }
            rc.grid_sum[egas_i] += 0.5 * U[pot_i][iii] * dV;
        }
        for (integer f = 0; f != opts().n_fields; ++f) {
            rc.grid_out[f] += U_out[f];
        }
        rc.grid_out[egas_i] += U_out[pot_i];
        return rc;
    }

    const auto is_loc = [this, diags](auto const& iii) {
        real aa = ZERO, bb = ZERO, ab = ZERO;
        for (integer d = 0; d < NDIM; d++) {
            real const& x0 = diags.com[0][d];
            real const a = X[d][iii] - x0;
            real const b = diags.com[1][d] - x0;
            aa += a * a;
            ab += a * b;
            bb += b * b;
        }
        real const d2bb = aa * bb - ab * ab;
        if ((d2bb < dx * dx * bb * 3.0 / 4.0)) {
            if ((ab < bb) && (ab > ZERO)) {
                return 2;
            } else if (ab <= ZERO) {
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

    const auto in_star = [&](auto const& iii, integer iiig) {
        if (opts().problem != DWD) {
            return int(0);
        }
        int rc = 0;
        std::array<real, NDIM> ax, n, x;
        for (integer d = 0; d < NDIM; d++) {
            x[d] = X[iii][d];
            ax[d] = G[iiig][gx_i + d];
        }
        if constexpr (NDIM >= 2) {
            x[XDIM] -= diags.grid_com[XDIM];
            x[YDIM] -= diags.grid_com[YDIM];
            ax[XDIM] += x[XDIM] * sqr(diags.omega);
            ax[YDIM] += x[YDIM] * sqr(diags.omega);
        }
        real a = ZERO;
        for (integer d = 0; d < NDIM; d++) {
            a += sqr(ax[d]);
        }
        if (a > ZERO) {
            a = sqrt(a);
            real const ia + ONE / a;
            for (integer d = 0; d < NDIM; d++) {
                n[d] = ax[d] * ia;
            }
            space_vector dX[nspec];
            real g[nspec] = {ZERO, ZERO};
            for (integer s = 0; s != nspec; ++s) {
                for (integer d = 0; d < NDIM; d++) {
                    dX[s][d] = x[d] - diags.com[s][d];
                }
            }
            real x0 = ZERO, x1 = ZERO;
            for (integer d = 0; d < NDIM; d++) {
                x0 += sqr(dX[0][d]);
                x1 += sqr(dX[1][d]);
            }
            x0 = sqrt(x0);
            x1 = sqrt(x1);
            if (x1 > 0.25 * diags.rL[1] && x0 < 0.25 * diags.rL[0] && diags.stage > 1) {
                rc = +1;
            } else if (x0 > 0.25 * diags.rL[0] && x1 < 0.25 * diags.rL[1] && diags.stage > 1) {
                rc = -1;
            } else if (x0 < 0.25 * diags.rL[0] && x1 < 0.25 * diags.rL[1] && diags.stage > 1) {
                rc = x0 < x1 ? +1 : -1;
            } else {
                for (integer s = 0; s != nspec; ++s) {
                    real const this_x = s == 0 ? x0 : x1;
                    if (this_x == ZERO) {
                        rc = 99;
                        return rc;
                    }
                    for (integer d = 0; d < NDIM; d++) {
                        g[s] += ax[d] * dX[s][d] * INVERSE(this_x);
                    }
                }
                if (g[0] <= ZERO && g[1] > ZERO) {
                    rc = +1;
                } else if (g[0] > ZERO && g[1] <= ZERO) {
                    rc = -1;
                } else if (g[0] <= ZERO && g[1] <= ZERO) {
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
    integer iiig = 0;
    roche_lobe.resize(intBox.volume());
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii, ++iiig) {
        real const irho = ONE / U[rho_i][iii];
        std::array<real, NDIM> v, x;
        for (integer d = 0; d < NDIM; d++) {
            x[d] = X[iii][d];
            v[d] = U[sx_i + d][iii] * irho;
        }
        real const o2 = diags.omega * diags.omega;
        std::array<real, nspec> rho;
        integer star;
        if (diags.stage < 2) {
            rho = {U[spc_ac_i][iii], U[spc_dc_i][iii]};
        } else {
            star = in_star(iii, iiig);
            if (star == +1) {
                rho = {U[rho_i][iii], ZERO};
            } else if (star == -1) {
                rho = {ZERO, U[rho_i][iii]};
            } else if (star != 99) {
                rho = {ZERO, ZERO};
            } else {
                rc.failed = true;
                return rc;
            }
        }
        if (diags.stage > 1 && NDIM >= 2) {
            real const R2 = sqr(x[0]) + sqr(x[1]);
            real const phi_g = G[iiig][phi_i];
            if (diags.omega < ZERO) {
                rc.failed = true;
                return rc;
            }
            const safe_real phi_r = -0.5 * POWER(diags.omega, 2) * R2;
            const safe_real phi_eff = phi_g + phi_r;
            const safe_real rho0 = U[rho_i][iii];
            real ekin = ZERO;
            for (integer d = 0; d < NDIM; d++) {
                kin += irho * sqr(U[sx_i + d][iii]);
            }
            kin *= HALF;
            if (ekin / U[rho_i][iii] / dV + phi_g > ZERO) {
                rc.munbound1 += U[rho_i][iii] * dV;
            }
            if (ekin / U[rho_i][iii] / dV + phi_eff > ZERO) {
                rc.munbound2 += U[rho_i][iii] * dV;
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
                real const dX[NDIM] = {
                    (x - diags.com[i][XDIM]), (y - diags.com[i][YDIM]), (z - diags.com[i][ZDIM])};
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
                if (opts().eos == IPR) {
                    eint = std::max(opts().ipr_eint_floor * dV, eint);
                } else if (eint < de_switch2 * U[egas_i][iii] * dV) {
                    eint = POWER(U[tau_i][iii], fgamma) * dV;
                }
                rc.ekin[i] += ekin;
                rc.epot[i] += epot;
                rc.eint[i] += eint;
                real const r = SQRT(dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2]);
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
                if (U[rho_i][iii] > real(10) * opts().scf_rho_floor) {
                    rc.stellar_vol[i] += dV;
                }
                rc.rho_max[i] = std::max(rc.rho_max[i], safe_real(rho0));
            }

            //					auto &rl = roche_lobe[h0index(j - H_BW, k - H_BW, l -
            // H_BW)];
            //
            //					auto lmin23 = std::min(diags.l2_phi, diags.l3_phi);
            //					auto lmax23 = std::max(diags.l2_phi, diags.l3_phi);
            //
            //					if (i != -1) {
            //						rl = i == 0 ? -1 : +1;
            //						integer const s = rl * INVERSE(std::abs(rl));
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

            auto loc = is_loc(iii);
            if (loc == 2) {
                rc.l1_phi = std::max(phi_eff, rc.l1_phi);
            } else if (loc == 1) {
                rc.l2_phi = std::max(phi_eff, rc.l2_phi);
            } else if (loc == 3) {
                rc.l3_phi = std::max(phi_eff, rc.l3_phi);
            }
            real ek = ZERO;
            real const irho = ONE / U[rho_i][iii];
            for (integer d = 0; d < NDIM; d++) {
                ek += irho * sqr(U[sx_i + d][iii]);
            }
            ek *= HALF;
            real ei;
            if (opts().eos == WD) {
                ei = U[egas_i][iii] - ek - ztwd_energy(U[rho_i][iii]);
            } else {
                ei = U[egas_i][iii] - ek;
            }
            real et = U[egas_i][iii];
            real p;
            if (ei < de_switch2 * et) {
                ei = POWER(U[tau_i][iii], fgamma);
            }
            p = (fgamma - ONE) * ei;
            if (opts().eos == WD) {
                p += ztwd_pressure(U[rho_i][iii]);
            }
            if (opts().problem == DWD) {
                rc.virial += (2.0 * ek + 0.5 * U[rho_i][iii] * G[iiig][phi_i] + 3.0 * p) * dV;
                rc.virial_norm += (2.0 * ek - 0.5 * U[rho_i][iii] * G[iiig][phi_i] + 3.0 * p) * dV;
            }
            for (integer f = 0; f != opts().n_fields; ++f) {
                rc.grid_sum[f] += U[f][iii] * dV;
            }
            rc.grid_sum[egas_i] += 0.5 * U[pot_i][iii] * dV;
            safe_real lz = (X[XDIM][iii] * U[sy_i][iii] - X[YDIM][iii] * U[sx_i][iii]) * dV;
            rc.lsum[0] += U[lx_i][iii] * dV -
                (X[YDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sy_i][iii]) * dV;
            rc.lsum[1] -= U[ly_i][iii] * dV -
                (X[XDIM][iii] * U[sz_i][iii] - X[ZDIM][iii] * U[sx_i][iii]) * dV;
            rc.lsum[2] += U[lz_i][iii] * dV - lz;
            const auto nonvac = (ONE - U[spc_i + opts().n_species - 1][iii] / U[rho_i][iii]);
            rc.nonvacj += lz * nonvac;
            rc.nonvacjlz += U[lz_i][iii] * nonvac * dV;
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

hpx::spinlock grid::omega_mtx;
real grid::omega = ZERO;
real grid::scaling_factor = ONE;

integer grid::min_level = 0;
integer grid::max_level = 0;

space_vector grid::get_cell_center(integer i, integer j, integer k) {
    integer const iii0 = hindex(H_BW, H_BW, H_BW);
    space_vector c;
    c[XDIM] = X[XDIM][iii0] + (i) *dx;
    c[YDIM] = X[XDIM][iii0] + (j) *dx;
    c[ZDIM] = X[XDIM][iii0] + (k) *dx;
    return c;
}

// std::vector<real> grid::get_prolong(const std::array<integer, NDIM> &lb, const
// std::array<integer, NDIM> &ub) { 	PROFILE(); 	std::vector<real> data;
//
//	integer size = opts().n_fields;
//	for (integer dim = 0; dim != NDIM; ++dim) {
//		size *= (ub[dim] - lb[dim]);
//	}
//	data.reserve(size);
//
//	for (integer f = 0; f < opts().n_fields; f++) {
//		const auto &u = U[f];
//		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
//			real const x = (i % 2) ? +1 : -1;
//			for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
//				real const y = (j % 2) ? +1 : -1;
//				for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
//					integer const iii = hindex(i / 2, j / 2, k / 2);
//					real const z = (k % 2) ? +1 : -1;
//					const auto u0 = u[iii];
//					real value = u0;
//					value += (9. / 64.) * minmod(u[iii + x * H_DNX] - u0, u0 - u[iii - x * H_DNX]);
//					value += (9. / 64.) * minmod(u[iii + y * H_DNY] - u0, u0 - u[iii - y * H_DNY]);
//					value += (9. / 64.) * minmod(u[iii + z * H_DNZ] - u0, u0 - u[iii - z * H_DNZ]);
//					value += (3. / 64.) * minmod(u[iii + x * H_DNX + y * H_DNY] - u0, u0 - u[iii - x
//* H_DNX - y * H_DNY]); 					value += (3. / 64.) * minmod(u[iii + x * H_DNX + z *
// H_DNZ] - u0, u0 - u[iii - x * H_DNX - z * H_DNZ]); 					value += (3. / 64.) *
// minmod(u[iii + y * H_DNY + z * H_DNZ] - u0, u0 - u[iii - y * H_DNY - z * H_DNZ]);
// value += (1. / 64.)
//							* minmod(u[iii + x * H_DNX + y * H_DNY + z * H_DNZ] - u0,
//									u0 - u[iii - x * H_DNX - y * H_DNY - z * H_DNZ]);
//					data.push_back(value);
//				}
//			}
//		}
//	}
//	return data;
// }

std::vector<real> grid::get_prolong(
    const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub) {
    PROFILE();
    std::vector<real> data;

    integer size = opts().n_fields;
    for (integer dim = 0; dim != NDIM; ++dim) {
        size *= (ub[dim] - lb[dim]);
    }
    data.reserve(size);

    for (integer f = 0; f < opts().n_fields; f++) {
        const auto& u = U[f];
        std::array<integer, NDIM> w;
        w[0] = real(0.25);
        for (integer d = 0; d + 1 < NDIM; d++) {
            w[d + 1] = w[d] * real(0.25);
        }
        for (BoxIterator iiif(Box<NDIM>(lb, ub), extBox2); !iiif.end(); ++iiif) {
            std::array<integer, NDIM> x;
            integer iiic = 0;
            for (integer d = 0; d < NDIM; d++) {
                iiic += H_DN[d] * iiif[d];
                x[d] = (iiif[d] & 1) ? +ONE : -ONE;
            }
            const auto u0 = u[iiic];
            real value = u0;
            for (integer d1 = 0; d1 < NDIM; d1++) {
                integer const di1 = x[d1] * H_DN[d1];
                value += w[NDIM - 1] * minmod(u[iiic + di1] - u0, u0 - u[iiic - di1]);
                if constexpr (NDIM > 1) {
                    for (integer d2 = d1 + 1; d2 < NDIM; d2++) {
                        integer const di2 = di1 + x[d2] * H_DN[d2];
                        value += w[NDIM - 2] * minmod(u[iiic + di2] - u0, u0 - u[iiic - di2]);
                        if constexpr (NDIM > 2) {
                            for (integer d3 = d2 + 1; d3 < NDIM; d3++) {
                                integer const di3 = di2 + x[d3] * H_DN[d3];
                                value +=
                                    w[NDIM - 3] * minmod(u[iiic + di3] - u0, u0 - u[iiic - di3]);
                            }
                        }
                    }
                }
            }
            data.push_back(value);
        }
    }
    return data;
}

std::vector<real> grid::get_restrict() const {
    PROFILE();
    integer Size = opts().n_fields * (intBox.volume() / NCHILD + 1);
    std::vector<real> data;
    data.reserve(Size);
    for (integer field = 0; field != opts().n_fields; ++field) {
        for (BoxIterator iii(intBox, extBox); !iii.end(); iii += 2) {
            real pt = ZERO;
            for (integer x = 0; x < NCHILD; ++x) {
                std::bitset<NDIM> bits(x);
                integer jjj = iii;
                for (integer d = 0; d < NDIM; d++) {
                    jjj += bits[d] * H_DN[d];
                }
                pt += U[field][jjj];
            }
            pt /= real(NCHILD);
            data.push_back(pt);
        }
    }
    for (integer field = 0; field != opts().n_fields; ++field) {
        data.push_back(U_out[field]);
    }
    return data;
}

void grid::set_restrict(const std::vector<real>& data, const geo::octant& octant) {
    PROFILE();
    constexpr integer INXo2 = INX / 2;
    std::array<integer, NDIM> lb, ub;
    lb.fill(H_BW);
    ub.fill(H_BW + INXo2);
    for (integer d = 0; d < NDIM; d++) {
        lb[d] += octant[d] * INXo2;
        ub[d] += octant[d] * INXo2;
    }
    Box<NDIM> box(lb, ub);
    for (integer field = 0; field != opts().n_fields; ++field) {
        integer jjj = 0;
        for (BoxIterator iii(box, extBox); iii.end(); ++iii) {
            U[field][iii] = data[jjj++];
        }
    }
}

void grid::set_hydro_boundary(
    const std::vector<real>& data, const geo::direction& dir, bool energy_only) {
    PROFILE();
    std::array<integer, NDIM> lb, ub;
    auto ptr = data.begin();
    const auto& bw = energy_only ? energy_bw : field_bw;

    for (integer field = 0; field != opts().n_fields; ++field) {
        get_boundary_size(lb, ub, dir, OUTER, INX, H_BW, bw[field]);
        Box<NDIM> box(lb, ub);
        auto& Ufield = U[field];
        for (BoxIterator iii(box, extBox); !iii.end(); ++iii) {
            Ufield[iii] = *ptr++;
        }
    }
}

std::vector<real> grid::get_hydro_boundary(const geo::direction& dir, bool energy_only) {
    PROFILE();

    const auto& bw = energy_only ? energy_bw : field_bw;
    std::array<integer, NDIM> lb, ub;
    std::vector<real> data;
    integer size = 0;

    for (integer field = 0; field != opts().n_fields; ++field) {
        size += get_boundary_size(lb, ub, dir, INNER, INX, H_BW, bw[field]);
    }
    data.reserve(size);
    integer iter = 0;
    for (integer field = 0; field != opts().n_fields; ++field) {
        get_boundary_size(lb, ub, dir, INNER, INX, H_BW, bw[field]);
        Box<NDIM> box(lb, ub);
        auto& Ufield = U[field];
        for (BoxIterator iii(box, extBox); !iii.end(); ++iii) {
            data.push_back(Ufield[iii]);
        }
    }
    return data;
}

line_of_centers_t grid::line_of_centers(const std::pair<space_vector, space_vector>& line) {
    line_of_centers_t loc;
    integer iiig = 0;
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii, ++iiig) {
        const auto R2 = std::sqrt(X[XDIM][iii] * X[XDIM][iii] + X[YDIM][iii] * X[YDIM][iii]);
        real const dV = dx * dx * dx;
        /*		for (integer d = 0; d != NDIM; ++d) {
         loc.core1_s[d] += U[sx_i + d][iii] * U[spc_ac_i][iii]
         / U[rho_i][iii]* dV;
         loc.core2_s[d] += U[sx_i + d][iii] * U[spc_dc_i][iii]
         / U[rho_i][iii]* dV;
         }
         loc.core1 += U[spc_ac_i][iii] * dV;
         loc.core2 += U[spc_dc_i][iii] * dV;*/
        space_vector a = line.first;
        const space_vector& o = ZERO;
        space_vector b;
        real bb = ZERO;
        real ab = ZERO;
        for (integer d = 0; d != NDIM; ++d) {
            //		a[d] -= o[d];
            b[d] = X[d][iii] - o[d];
        }
        for (integer d = 0; d != NDIM; ++d) {
            bb += b[d] * b[d];
            ab += a[d] * b[d];
        }
        real const d = std::sqrt(std::max(bb - ab * ab, ZERO));
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
    return loc;
}

std::pair<std::vector<real>, std::vector<real>> grid::diagnostic_error() const {
    std::pair<std::vector<real>, std::vector<real>> e;
    real const dV = dx * dx * dx;
    if (opts().problem == SOLID_SPHERE) {
        e.first.resize(8, ZERO);
        e.second.resize(8, ZERO);
    }
    integer iii = 0;
    for (BoxIterator iiih(intBox, extBox); !iiih.end(); ++iii, ++iiih) {
        real const x = X[XDIM][iiih];
        real const y = X[YDIM][iiih];
        real const z = X[ZDIM][iiih];
        if (opts().problem == SOLID_SPHERE) {
            const auto a = solid_sphere_analytic_phi(x, y, z, 0.25);
            std::vector<real> n(NDIM + 1);
            n[phi_i] = G[iii][phi_i];
            n[gx_i] = G[iii][gx_i];
            n[gy_i] = G[iii][gy_i];
            n[gz_i] = G[iii][gz_i];
            real const rho = U[rho_i][iiih];
            for (integer l = 0; l != 4; ++l) {
                e.first[l] += std::abs(a[l] - n[l]) * dV * rho;
                e.first[4 + l] += std::abs(a[l]) * dV * rho;
                e.second[l] += sqr((a[l] - n[l]) * rho) * dV;
                e.second[4 + l] += sqr(a[l] * rho) * dV;
            }
        }
    }
    //	printf("%e\n", e[0]);

    return e;
}

real& grid::get_omega() {
    return omega;
}

void grid::velocity_inc(const space_vector& dv) {
    for (integer iii = 0; iii != extBox.volume(); ++iii) {
        real const rho = U[rho_i][iii];
        if (rho != ZERO) {
            real const rhoinv = ONE / rho;
            safe_real& sx = U[sx_i][iii];
            safe_real& sy = U[sy_i][iii];
            safe_real& sz = U[sz_i][iii];
            safe_real& egas = U[egas_i][iii];
            egas -= HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
            sx += dv[XDIM] * rho;
            sy += dv[YDIM] * rho;
            sz += dv[ZDIM] * rho;
            egas += HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
        }
    }
}

void grid::energy_adj() {
    for (integer iii = 0; iii != extBox.volume(); ++iii) {
        real const rho = U[rho_i][iii];
        if (rho != ZERO) {
            real const rhoinv = ONE / rho;
            safe_real sx = U[sx_i][iii];
            safe_real sy = U[sy_i][iii];
            safe_real sz = U[sz_i][iii];
            safe_real& egas = U[egas_i][iii];
            safe_real& tau = U[tau_i][iii];
            egas -= HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
            egas = std::max(opts().ipr_eint_floor, egas);

            specie_state_t<real> spc;
            real mmw_loc, X_loc, Z_loc;
            for (integer si = 0; si != opts().n_species; ++si) {
                spc[si] = U[spc_i + si][iii];
            }
            mean_ion_weight(spc, mmw_loc, X_loc, Z_loc);
            safe_real p_SCF =
                egas * (fgamma - ONE);    // assuming polytropic + ideal eos for the SCF
            egas = std::max(opts().star_egas_out,
                find_ei_rad_gas(
                    p_SCF, rho, mmw_loc, fgamma, tau));    // update both egas and tau
                                                           //       sx += dv[XDIM] * rho;
                                                           //      sy += dv[YDIM] * rho;
                                                           //       sz += dv[ZDIM] * rho;
            egas += HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
        }
    }
}

std::vector<real> grid::get_flux_restrict(
    std::array<integer, NDIM> lb, std::array<integer, NDIM> ub, const geo::dimension& dim) const {
    PROFILE();
    constexpr int NVERTEX = 1 << (NDIM - 1);
    constexpr real weight = ONE / NVERTEX;
    std::vector<real> data;
    integer size = 1;
    for (auto& dim : geo::dimension::full_set()) {
        size *= (ub[dim] - lb[dim]);
    }
    size /= (NCHILD / 2);
    size *= opts().n_fields;
    data.reserve(size);
    Box<NDIM> const coarseBox(lb, ub);
    ub.fill(2);
    ub[dim] = 1;
    Box<NDIM> faceBox(ub);
    for (integer field = 0; field != opts().n_fields; ++field) {
        for (BoxIterator jc(coarseBox, extBoxFlux); !jc.end(); jc += 2) {
            auto const fintBox = faceBox.shift(jc);
            real value = ZERO;
            for (BoxIterator jf(fintBox, extBoxFlux); !jf.end(); ++jf) {
                value += weight * F[dim][field][jf];
            }
            data.push_back(value);
        }
    }
    return data;
}

void grid::set_flux_restrict(const std::vector<real>& data, const std::array<integer, NDIM>& lb,
    const std::array<integer, NDIM>& ub, const geo::dimension& dim) {
    PROFILE();
    integer index = 0;
    Box<NDIM> const box(lb, ub);
    auto ptr = data.begin();
    for (integer field = 0; field != opts().n_fields; ++field) {
        for (BoxIterator iii(box, extBoxFlux); !iii.end(); ++iii) {
            F[dim][field][iii] = *ptr++;
        }
    }
}

void grid::set_prolong(const std::vector<real>& data, std::vector<real>&& outflows) {
    PROFILE();
    integer index = 0;
    U_out = std::move(outflows);
    auto ptr = data.begin();
    for (integer field = 0; field != opts().n_fields; ++field) {
        for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
            U[field][iii] = *ptr++;
        }
    }
}

std::pair<std::vector<real>, std::vector<real>> grid::field_range() const {
    std::pair<std::vector<real>, std::vector<real>> minmax;
    minmax.first.resize(opts().n_fields);
    minmax.second.resize(opts().n_fields);
    for (integer field = 0; field != opts().n_fields; ++field) {
        minmax.first[field] = +std::numeric_limits<real>::max();
        minmax.second[field] = -std::numeric_limits<real>::max();
    }
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
        for (integer field = 0; field != opts().n_fields; ++field) {
            minmax.first[field] = std::min(minmax.first[field], (double) U[field][iii]);
            minmax.second[field] = std::max(minmax.second[field], (double) U[field][iii]);
        }
    }
    return minmax;
}

void grid::change_units(real m, real l, real t, real k) {
    real const l2 = l * l;
    real const t2 = t * t;
    real const t2inv = ONE / t2;
    real const tinv = ONE / t;
    real const l3 = l2 * l;
    real const l3inv = ONE / l3;
    xmin[XDIM] *= l;
    xmin[YDIM] *= l;
    xmin[ZDIM] *= l;
    dx *= l;
    if (dx > 1.0e+12)
        printf("++++++!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1+++++++++++++++++++++++++++++++++++++ "
               "%e %e\n",
            dx, dx * l);
    for (integer i = 0; i != extBox.volume(); ++i) {
        U[rho_i][i] *= m * l3inv;
        for (integer si = 0; si != opts().n_species; ++si) {
            U[spc_i + si][i] *= m * l3inv;
        }
        U[egas_i][i] *= (m * l2 * t2inv) * l3inv;
        if (opts().eos != IPR) {    // for IPR eos, tau is the temperature
            U[tau_i][i] *= std::pow(m * l2 * t2inv * l3inv, ONE / fgamma);
        }
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
        //			printf("!!!!!!!!!!!! %e !!!!!!!!!!!!!!!!\n", std::abs(X[XDIM][i]));
        //		}
    }
    for (integer i = 0; i != intBox.volume(); ++i) {
        G[i][phi_i] *= l2 * t2inv;
        G[i][gx_i] *= l2 * tinv;
        G[i][gy_i] *= l2 * tinv;
        G[i][gz_i] *= l2 * tinv;
    }
}

HPX_PLAIN_ACTION(grid::set_omega, set_omega_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(set_omega_action);
HPX_REGISTER_BROADCAST_ACTION(set_omega_action);

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
                hpx::lcos::broadcast<set_omega_action>(remotes, omega, false).get();
            }
        }
    }
    std::unique_lock<hpx::spinlock> l(grid::omega_mtx, std::try_to_lock);
    // if someone else has the lock, it's fine, we just return and have it set
    // by the other thread
    if (!l) return;
    grid::omega = omega;
}

real grid::roche_volume(const std::pair<space_vector, space_vector>& axis,
    const std::pair<real, real>& l1, real cx, bool donor) const {
    real const dV = dx * dx * dx;
    real V = ZERO;
    integer iiig = 0;
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii, ++iiig) {
        real x0 = X[XDIM][iii];
        real x = x0 - cx;
        real y = X[YDIM][iii];
        real z = X[ZDIM][iii];
        real const R = std::sqrt(x0 * x0 + y * y);
        real phi_eff = G[iiig][phi_i] - 0.5 * sqr(omega * R);
        //	real factor = axis.first[0] == l1.first ? 0.5 : 1.0;
        if ((x0 <= l1.first && !donor) || (x0 >= l1.first && donor)) {
            if (phi_eff <= l1.second) {
                real const fx = G[iiig][gx_i] + x0 * sqr(omega);
                real const fy = G[iiig][gy_i] + y * sqr(omega);
                real const fz = G[iiig][gz_i];
                real g = x * fx + y * fy + z * fz;
                if (g <= ZERO) {
                    V += dV;
                }
            }
        }
    }
    return V;
}

std::vector<real> grid::frac_volumes() const {
    std::vector<real> V(opts().n_species, ZERO);
    real const dV = dx * dx * dx;
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
        for (integer si = 0; si != opts().n_species; ++si) {
            if (U[spc_i + si][iii] > 1.0e-5) {
                V[si] += (U[spc_i + si][iii] / U[rho_i][iii]) * dV;
            }
        }
    }
    //	printf( "%e", V[0]);

    return V;
}

bool grid::is_in_star(const std::pair<space_vector, space_vector>& axis,
    const std::pair<real, real>& l1, integer frac, integer iii, real rho_cut) const {
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
            real ab = ZERO;
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

real grid::z_moments(const std::pair<space_vector, space_vector>& axis,
    const std::pair<real, real>& l1, integer frac, real rho_cut) const {
    real mom = ZERO;
    real const dV = dx * dx * dx;
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
        if (is_in_star(axis, l1, frac, iii, rho_cut)) {
            mom += (sqr(X[XDIM][iii]) + sqr(dx) / 6.0) * U[rho_i][iii] * dV;
            mom += (sqr(X[YDIM][iii]) + sqr(dx) / 6.0) * U[rho_i][iii] * dV;
        }
    }
    return mom;
}

std::vector<real> grid::conserved_sums(space_vector& com, space_vector& com_dot,
    const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1,
    integer frac, real rho_cut) const {
    std::vector<real> sum(opts().n_fields, ZERO);
    com[0] = com[1] = com[2] = ZERO;
    com_dot[0] = com_dot[1] = com_dot[2] = ZERO;
    real const dV = dx * dx * dx;
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
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
    if (sum[rho_i] > ZERO) {
        for (integer d = 0; d != NDIM; ++d) {
            com[d] /= sum[rho_i];
            com_dot[d] /= sum[rho_i];
        }
    }
    return sum;
}

std::vector<real> grid::gforce_sum(bool torque) const {
    std::vector<real> sum(NDIM, ZERO);
    real const dV = dx * dx * dx;
    integer iiig = 0;
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii, ++iiig) {
        real const& rho = U[rho_i][iii];
        real const x = X[XDIM][iii];
        real const y = X[YDIM][iii];
        real const z = X[ZDIM][iii];
        real const fx = rho * G[iiig][gx_i] * dV;
        real const fy = rho * G[iiig][gy_i] * dV;
        real const fz = rho * G[iiig][gz_i] * dV;
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
    return sum;
}

std::vector<real> grid::l_sums() const {
    std::vector<real> sum(NDIM);
    real const dV = dx * dx * dx;
    std::fill(sum.begin(), sum.end(), ZERO);
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
        sum[XDIM] += X[YDIM][iii] * U[sz_i][iii] * dV;
        sum[XDIM] -= X[ZDIM][iii] * U[sy_i][iii] * dV;
        sum[YDIM] -= X[XDIM][iii] * U[sz_i][iii] * dV;
        sum[YDIM] += X[ZDIM][iii] * U[sx_i][iii] * dV;
        sum[ZDIM] += X[XDIM][iii] * U[sy_i][iii] * dV;
        sum[ZDIM] -= X[YDIM][iii] * U[sx_i][iii] * dV;
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
    std::array<std::vector<real>, NDIM> dudx;
    dudx.fill(std::vector<real>(opts().n_fields));
    static_assert(REFINE_BW <= H_BW);
    for (BoxIterator iii(intBoxRefine, extBox); !iii.end(); ++iii) {
        integer cnt = 0;
        for (integer d = 0; d < NDIM; d++) {
            if (iii[d] < H_BW || iii[d] >= H_NX - H_BW) cnt++;
            if (cnt > 1) break;
        }
        if (cnt > 1) continue;
        for (integer field = 0; field != opts().n_fields; ++field) {
            state[field] = U[field][iii];
            for (integer d = 0; d < NDIM; d++) {
                dudx[d][field] = (U[field][iii + H_DN[d]] - U[field][iii - H_DN[d]]) / 2.0;
            }
        }
        if (test(lev, max_level, X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii], state, dudx)) {
            rc = true;
            break;
        }
    }
    return rc;
}

void grid::rho_mult(real f0, real f1) {
    for (integer iii = 0; iii != extBox.volume(); ++iii) {
        constexpr integer spc_ac_i = spc_i;
        constexpr integer spc_ae_i = spc_i + 1;
        constexpr integer spc_dc_i = spc_i + 2;
        constexpr integer spc_de_i = spc_i + 3;
        U[spc_ac_i][iii] *= f0;
        U[spc_dc_i][iii] *= f1;
        U[spc_ae_i][iii] *= f0;
        U[spc_de_i][iii] *= f1;
        U[rho_i][iii] = ZERO;
        for (integer si = 0; si != opts().n_species; ++si) {
            U[rho_i][iii] += U[spc_i + si][iii];
        }
    }
}

void grid::rho_move(real x) {
    real w = x / dx;
    U0 = U;

    w = std::max(-0.5, std::min(0.5, w));
    for (BoxIterator iii(extBox.shrink(1), extBox); !iii.end(); ++iii) {
        for (integer si = spc_i; si != opts().n_species + spc_i; ++si) {
            U[si][iii] += w * U0[si][iii + H_DN[0]];
            U[si][iii] -= w * U0[si][iii - H_DN[0]];
            U[si][iii] = std::max((double) U[si][iii], ZERO);
        }
        U[rho_i][iii] = ZERO;
        for (integer si = 0; si != opts().n_species; ++si) {
            U[rho_i][iii] += U[spc_i + si][iii];
        }
        U[rho_i][iii] = std::max((double) U[rho_i][iii], opts().scf_rho_floor);
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

    space_vector this_com;
    this_com.fill(ZERO);
    real m = ZERO;
    auto& com0 = *(com_ptr)[0];
    for (integer iii = 0; iii != intBox.volume(); ++iii) {
        real const this_m = is_leaf ? mon[iii] : M[iii]();
        for (integer d = 0; d < NDIM; d++) {
            this_com[d] += this_m * com0[iii][d];
        }
        m += this_m;
    }
    if (m != ZERO) {
        for (integer d = 0; d < NDIM; d++) {
            this_com[d] /= m;
        }
    }
    return this_com;
}

grid::grid(real _dx, std::array<real, NDIM> _xmin)
  : is_coarse(extBox.volume())
  , has_coarse(extBox.volume())
  , Ushad(opts().n_fields)
  , U(opts().n_fields)
  , U0(opts().n_fields)
  , dUdt(opts().n_fields)
  , F(NDIM)
  , X(3)
  , G(NGF)
  , is_root(false)
  , is_leaf(true) {
    dx = _dx;
    xmin = _xmin;
    allocate();
}

real grid::fgamma = 5.0 / 3.0;

void grid::set_coordinates() {
    for (BoxIterator i(extBox); !i.end(); ++i) {
        for (integer d = 0; d < NDIM; d++) {
            X[d][i] = (real(i[d] - H_BW) + HALF) * dx + xmin[d];
        }
        for (integer d = NDIM; d < 3; d++) {
            X[d][i] = ZERO;
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
    rc.push_back(std::make_pair(
        std::string("vx"), hpx::util::format("sx / rho + {:e} * coord(quadmesh)[1]", omega)));
    rc.push_back(std::make_pair(
        std::string("vy"), hpx::util::format("sy / rho - {:e} * coord(quadmesh)[0]", omega)));
    rc.push_back(std::make_pair(std::string("vz"), std::string("sz / rho")));
    rc.push_back(
        std::make_pair(std::string("zx"), "lx - coord(quadmesh)[1]*sz + coord(quadmesh)[2]*sy"));
    rc.push_back(
        std::make_pair(std::string("zy"), "ly + coord(quadmesh)[0]*sz - coord(quadmesh)[2]*sx"));
    rc.push_back(
        std::make_pair(std::string("zz"), "lz - coord(quadmesh)[0]*sy + coord(quadmesh)[1]*sx"));

    std::string n;
    std::string X = "(";
    std::string Z = "(";

    for (integer i = 0; i < opts().n_species; i++) {
        real const mu = opts().atomic_mass[i] / (opts().atomic_number[i] + 1.);
        n +=
            hpx::util::format("rho_{} / {:e} + ", int(i + 1), mu * physcon().mh * opts().code_to_g);
        X += hpx::util::format("{:e} * rho_{} + ", opts().X[i], i + 1);
        Z += hpx::util::format("{:e} * rho_{} + ", opts().Z[i], i + 1);
    }
    n += '0';
    X += "0) / rho";
    Z += "0) / rho";
    rc.push_back(std::make_pair(std::string("sigma_T"),
        std::string(
            "(1 + X) * 0.2 * T * T / ((T * T + 2.7e+11 * rho) * (1 + (T / 4.5e+8)^0.86))")));
    rc.push_back(std::make_pair(
        std::string("sigma_xf"), std::string("4e+25*(1+X)*(Z+0.001)*rho*(T^(-3.5))")));
    rc.push_back(std::make_pair(std::string("mfp"), std::string("1 / kappa_R")));
    if (opts().problem == MARSHAK) {
        rc.push_back(std::make_pair(std::string("kappa_R"), std::string("rho")));
        rc.push_back(std::make_pair(std::string("kappa_P"), std::string("rho")));
    } else {
        rc.push_back(
            std::make_pair(std::string("kappa_R"), std::string("rho * (sigma_xf + sigma_T)")));
        rc.push_back(
            std::make_pair(std::string("kappa_P"), std::string("rho * 30.262 * sigma_xf")));
    }
    rc.push_back(std::make_pair(std::string("n"), std::move(n)));
    rc.push_back(std::make_pair(std::string("X"), std::move(X)));
    rc.push_back(std::make_pair(std::string("Y"), std::string("1.0 - X - Z")));
    rc.push_back(std::make_pair(std::string("Z"), std::move(Z)));
    rc.push_back(std::make_pair(std::string("etot_dual"), std::string("ei + ek")));
    rc.push_back(std::make_pair(std::string("ek"), std::string("(sx*sx+sy*sy+sz*sz)/2.0/rho")));
    const auto kb =
        physcon().kb * std::pow(opts().code_to_cm / opts().code_to_s, 2) * opts().code_to_g;
    rc.push_back(std::make_pair(std::string("phi"), std::string("pot/rho")));
    rc.push_back(std::make_pair(std::string("B_p"),
        hpx::util::format("{:e} * T^4",
            physcon().sigma / M_PI * opts().code_to_g * std::pow(opts().code_to_cm, 3))));
    if (opts().eos == WD) {
        rc.push_back(std::make_pair(std::string("A"), "6.00228e+22"));
        rc.push_back(std::make_pair(std::string("B"), "(2 * 9.81011e+5)"));
        rc.push_back(std::make_pair(std::string("x"), "(rho/B)^(1.0/3.0)"));
        rc.push_back(std::make_pair(std::string("Pdeg"),
            "if( gt(x, 0.001), A*(x*(2.0*x*x-3.0)*sqrt(x*x+1.0)+3.0*ln(x+sqrt(x*x+1))), "
            "1.6*A*x^5)"));
        rc.push_back(std::make_pair(std::string("hdeg"), "8.0*A/B*(sqrt(x*x+1)-1)"));
        rc.push_back(
            std::make_pair(std::string("Edeg"), "if( gt(x, 0.001), rho*hdeg - Pdeg, 2.4*A*x^5"));
    }
    if (opts().problem == MARSHAK) {
        rc.push_back(std::make_pair(std::string("T"), std::string("(ei/rho)^(1.0/3.0)")));
    } else if (opts().eos == IPR) {
        rc.push_back(std::make_pair(std::string("T"), std::string("tau")));
        rc.push_back(std::make_pair(std::string("P"),
            hpx::util::format("n * {:e} * tau + {:e} * tau^4", kb,
                (4.0 * physcon().sigma * opts().code_to_g / std::pow(opts().code_to_s, 3)) /
                    (3.0 * physcon().c * opts().code_to_cm / opts().code_to_s))));
        rc.push_back(std::make_pair(std::string("ei"),
            hpx::util::format("max(egas-ek,{:e})",
                opts().ipr_eint_floor * opts().code_to_g / std::pow(opts().code_to_s, 2) /
                    opts().code_to_cm)));
    } else if (opts().eos != WD) {
        rc.push_back(std::make_pair(std::string("ei"),
            hpx::util::format(
                "if( gt(egas-ek,{:e}*egas), egas-ek, tau^{:e})", opts().dual_energy_sw1, fgamma)));
        rc.push_back(
            std::make_pair(std::string("P"), hpx::util::format("{:e} * ei", (fgamma - ONE))));
        rc.push_back(std::make_pair(
            std::string("T"), hpx::util::format("{:e} * ei / n", ONE / (kb / (fgamma - ONE)))));
    } else {
        rc.push_back(std::make_pair(std::string("ei"),
            hpx::util::format("if( gt(egas-ek-Edeg,{:e}*egas), egas-ek-Edeg, tau^{:e})",
                opts().dual_energy_sw1, fgamma)));
        rc.push_back(std::make_pair(
            std::string("P"), hpx::util::format("Pdeg + {:e} * ei", (fgamma - ONE))));
        rc.push_back(std::make_pair(
            std::string("T"), hpx::util::format("{:e} * ei / n", ONE / (kb / (fgamma - ONE)))));
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
    real const dv = dx * dx * dx;
    integer ig = 0;
    for (BoxIterator i(intBox, extBox); !i.end(); ++i, ++ig) {
        auto A = func(X[XDIM][i], X[YDIM][i], X[ZDIM][i], t);
        const auto nrho = U[rho_i][i];
        for (integer M = 2; M <= INX; M *= 2) {
            auto last_rho = A[rho_i];
            if (last_rho == nrho) {
                break;
            }
            for (integer f = 0; f < opts().n_fields; f++) {
                A[f] = ZERO;
            }
            std::array<real, 3> x{};
            for (BoxIterator i0{Box<NDIM>(M)}; !i0.end(); ++i0) {
                for (integer d = 0; d < NDIM; d++) {
                    x[d] = X[d][i] + ((real(i0[d]) + 0.5) / real(M) - 0.5) * dx;
                }
                const auto a = func(x[0], x[1], x[2], t);
                for (integer f0 = 0; f0 < opts().n_fields; f0++) {
                    A[f0] += a[f0] * pow(M, -NDIM);
                }
            }
            const auto this_rho = A[rho_i];
            real const err = std::abs(std::abs((nrho - this_rho) / (nrho - last_rho)) - ONE);
            if (M > INX) {
                printf("%lli %e\n", M, err);
            }
            if (err < 0.1) {
                break;
            }
        }
        for (integer field = 0; field != opts().n_fields; ++field) {
            real dif = std::abs(A[field] - U[field][i]);
            a.l1[field] += dif * dv;
            a.l2[field] += dif * dif * dv;
            a.linf[field] = std::max(dif, a.linf[field]);
            U[field][i] = A[field];
        }
        if (opts().problem == SOLID_SPHERE) {
            const auto a = solid_sphere_analytic_phi(X[0][i], X[1][i], X[2][i], 0.25);
            for (integer f = 0; f < 4; f++) {
                G[ig][f] = a[f];
            }
            U[pot_i][i] = a[0] * U[rho_i][i];
        }
    }
    return a;
}

void grid::allocate() {
    U_out0 = std::vector<real>(opts().n_fields, ZERO);
    U_out = std::vector<real>(opts().n_fields, ZERO);
    dphi_dt = std::vector<real>(intBox.volume());
    G.resize(intBox.volume());
    for (integer dim = 0; dim != NDIM; ++dim) {
        X[dim].resize(extBox.volume());
    }

    for (integer field = 0; field != opts().n_fields; ++field) {
        U0[field].resize(intBox.volume());
        U[field].resize(extBox.volume(), ZERO);
        Ushad[field].resize(extBoxShadow.volume(), ONE);
        dUdt[field].resize(intBox.volume());
        for (integer dim = 0; dim != NDIM; ++dim) {
            F[dim][field].resize(extBoxFlux.volume());
        }
    }
    L.resize(intBox.volume());
    L_c.resize(intBox.volume());
    integer nlevel = 0;
    com_ptr.resize(2);

    set_coordinates();

#ifdef OCTOTIGER_HAVE_GRAV_PAR
    L_mtx.reset(new hpx::spinlock);
#endif
}

grid::grid()
  : is_coarse(extBox.volume())
  , has_coarse(extBox.volume())
  , Ushad(opts().n_fields)
  , U(opts().n_fields)
  , U0(opts().n_fields)
  , dUdt(opts().n_fields)
  , F(NDIM)
  , X(NDIM)
  , G(NGF)
  , dphi_dt(extBox.volume())
  , is_root(false)
  , is_leaf(true)
  , U_out(opts().n_fields, ZERO)
  , U_out0(opts().n_fields, ZERO) {
    //	allocate();
}

grid::grid(const init_func_type& init_func, real _dx, std::array<real, NDIM> _xmin)
  : is_coarse(extBox.volume())
  , has_coarse(extBox.volume())
  , Ushad(opts().n_fields)
  , U(opts().n_fields)
  , U0(opts().n_fields)
  , dUdt(opts().n_fields)
  , F(NDIM)
  , X(NDIM)
  , G(NGF)
  , is_root(false)
  , is_leaf(true)
  , U_out(opts().n_fields, ZERO)
  , U_out0(opts().n_fields, ZERO)
  , dphi_dt(extBox.volume()) {
    dx = _dx;
    xmin = _xmin;
    allocate();
    for (integer iii = 0; iii != extBox.volume(); ++iii) {
        if (init_func != nullptr) {
            std::vector<real> this_u = init_func(X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii], dx);
            for (integer field = 0; field != opts().n_fields; ++field) {
                U[field][iii] = this_u[field];
            }
        } else {
            std::cerr << "Error: No problem specified\n";
            std::terminate();
        }
    }
    if (opts().gravity) {
        for (integer i = 0; i != intBox.volume(); ++i) {
            for (integer field = 0; field != NGF; ++field) {
                G[i][field] = ZERO;
            }
        }
    }
}

timestep_t grid::compute_fluxes() {
    PROFILE();
    static hpx::once_flag flag;
    hpx::call_once(flag, [this]() {
        physics<NDIM>::set_fgamma(fgamma);
        if (opts().eos == WD) {
            //			printf("%e %e\n", physcon().A, physcon().B);
            physics<NDIM>::set_degenerate_eos(physcon().A, physcon().B);
        } else if (opts().eos == IPR) {
            physics<NDIM>::set_ideal_plus_rad_eos(physcon().kb / physcon().mh,
                4 * physcon().sigma / physcon().c, opts().ipr_nr_tol, opts().ipr_nr_maxiter,
                opts().ipr_test, opts().ipr_eint_floor);
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
        for (integer i = spc_i; i < spc_i + opts().n_species; i++) {
            hydro.use_disc_detect(i);
        }
    }
    hydro.use_smooth_recon(pot_i);

    const interaction_host_kernel_type host_type = opts().hydro_host_kernel_type;
    const interaction_device_kernel_type device_type = opts().hydro_device_kernel_type;
    const size_t device_queue_length = opts().max_gpu_executor_queue_length;
    return launch_hydro_kernels(hydro, U, X, omega, F, host_type, device_type, device_queue_length);
}

real grid::compute_positivity_speed_limit() const {
    real max_lambda = ZERO;
    BoxIterator idxf(intBox, extBoxFlux);
    BoxIterator idxh(intBox, extBox);
    for (; !idxf.end(); ++idxf, ++idxh) {
        assert(!idxh.end());
        real drho_dt = ZERO;
        real dtau_dt = ZERO;
        for (integer d = 0; d < NDIM; d++) {
            drho_dt -= (F[d][rho_i][idxf + F_DN[d]] - F[d][rho_i][idxf]) / dx;
        }
        max_lambda = std::max(max_lambda, -drho_dt * dx / U[rho_i][idxh]);
        if (opts().eos != IPR) {    // For ipr eos, tau does not have the usual meanings
            for (integer d = 0; d < NDIM; d++) {
                dtau_dt -= (F[d][tau_i][idxf + F_DN[d]] - F[d][tau_i][idxf]) / dx;
            }
            max_lambda = std::max(max_lambda, -dtau_dt * dx / U[tau_i][idxh]);
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
        integer j = 0;
        for (BoxIterator idx(intBox, extBox); !idx.end(); ++idx, ++j) {
            U0[field][j] = U[field][idx];
        }
    }
    U_out0 = U_out;
}

void grid::restore() {
    for (integer field = 0; field != opts().n_fields; ++field) {
        integer j = 0;
        for (BoxIterator idx(intBox, extBox); !idx.end(); ++idx, ++j) {
            U[field][idx] = U0[field][j];
        }
    }
    U_out = U_out0;
}

void grid::set_physical_boundaries(const geo::face& f, real t) {
    PROFILE();
    integer const face = f;
    integer const dim = face >> 1;
    integer const side = face & 1;
    integer const lb = side ? H_NX - H_BW : 0;
    integer const ub = side ? H_NX : H_BW;
    integer const ib = side ? (H_NX - 1) : H_BW;
    auto const ibox = extBox.slice(dim, ib);
    BoxIterator ii(ibox, extBox);
    for (integer ob = lb; ob < ub; ++ob) {
        auto const obox = extBox.slice(dim, ob);
        BoxIterator oi(obox, extBox);
        ii.reset();
        for (; !oi.end(); ++oi, ++ii) {
            for (integer field = 0; field != opts().n_fields; ++field) {
                U[field][oi] = U[field][ii];
            }
        }
        oi.reset();
        ii.reset();
        for (BoxIterator oi(obox, extBox); !oi.end(); ++oi, ++ii) {
            auto& sn = U[sx_i + dim][oi];
            auto& egas = U[egas_i][oi];
            auto const& rho = U[rho_i][oi];
            auto const irho = ONE / rho;
            real const sn0 = ((NDIM >= 2) && (dim < ZDIM)) ?
                (real(2 * dim - 1) * X[1 - dim][oi] * omega * rho) :
                ZERO;
            egas -= HALF * irho * sqr(sn);
            sn -= sn0;
            if (opts().reflect_bc) {
                sn = -sn;
            } else {
                sn = side ? std::max(ZERO, sn) : std::min(ZERO, sn);
            }
            sn += sn0;
            egas += HALF * irho * sqr(sn);
        }
    }
}

void grid::compute_sources(real t, real rotational_time) {
    PROFILE();
    if constexpr (NDIM >= 2) {
        auto& src = dUdt;
        integer iii0 = 0;
        integer& iiig = iii0;
        for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii, ++iii0) {
            for (integer field = 0; field != opts().n_fields; ++field) {
                src[field][iii0] = ZERO;
            }
            real const rho = U[rho_i][iii];
            if (opts().gravity) {
                src[sx_i][iii0] += rho * G[iiig][gx_i];
                src[sy_i][iii0] += rho * G[iiig][gy_i];
                src[sz_i][iii0] += rho * G[iiig][gz_i];
            }
            if (opts().gravity) {
                src[egas_i][iii0] -= omega * X[YDIM][iii] * rho * G[iiig][gx_i];
                src[egas_i][iii0] += omega * X[XDIM][iii] * rho * G[iiig][gy_i];
            }
            if (opts().driving_rate != ZERO) {
                real const period_len = 2.0 * M_PI / grid::omega;
                if (opts().driving_time > rotational_time / (2.0 * M_PI)) {
                    real const ff = -opts().driving_rate / period_len;
                    ///	printf("%e %e %e\n", ff, opts().driving_rate, period_len);
                    real const rho = U[rho_i][iii];
                    real const sx = U[sx_i][iii];
                    real const sy = U[sy_i][iii];
                    real const x = X[XDIM][iii];
                    real const y = X[YDIM][iii];
                    real const R = std::sqrt(x * x + y * y);
                    real const lz = (x * sy - y * sx);
                    real const dsx = -y / R / R * lz * ff;
                    real const dsy = +x / R / R * lz * ff;
                    src[sx_i][iii0] += dsx;
                    src[sy_i][iii0] += dsy;
                    src[egas_i][iii0] += (sx * dsx + sy * dsy) / rho;
                }
            }
            if (opts().entropy_driving_rate != ZERO) {
                constexpr integer spc_ac_i = spc_i;
                constexpr integer spc_ae_i = spc_i + 1;

                real const period_len = 2.0 * M_PI / grid::omega;
                if (opts().entropy_driving_time > rotational_time / (2.0 * M_PI)) {
                    constexpr integer spc_ac_i = spc_i;
                    constexpr integer spc_ae_i = spc_i + 1;

                    real ff = +opts().entropy_driving_rate / period_len;
                    ff *= (U[spc_ac_i][iii] + U[spc_ae_i][iii]) / U[rho_i][iii];
                    real ek = ZERO;
                    real const irho = ONE / U[rho_i][iii];
                    for (integer d = 0; d < NDIM; d++) {
                        ek += irho * sqr(U[sx_i + d][iii]);
                    }
                    ek *= HALF;
                    real ei;
                    if (opts().eos == WD) {
                        ei = U[egas_i][iii] - ek - ztwd_energy(U[rho_i][iii]);
                    } else {
                        ei = U[egas_i][iii] - ek;
                    }
                    real et = U[egas_i][iii];
                    real dei;
                    if (opts().eos == IPR) {
                        dei = ff * std::max(opts().ipr_eint_floor, ei);
                    } else {
                        real tau;
                        if (ei < de_switch2 * et) {
                            tau = U[tau_i][iii];
                        } else {
                            tau = std::pow(ei, ONE / fgamma);
                        }
                        ei = std::pow(tau, fgamma);
                        real const dtau = ff * tau;
                        dei = dtau * ei / tau * fgamma;
                        src[tau_i][iii0] += dtau;
                    }
                    src[egas_i][iii0] += dei;
                }
            }
            src[sx_i][iii0] += omega * U[sy_i][iii];
            src[sy_i][iii0] -= omega * U[sx_i][iii];
        }
    }
}

void grid::compute_dudt() {
    PROFILE();
    real const idx = ONE / dx;
    integer iii0 = 0;
    for (BoxIterator iiif(intBoxFlux, extBoxFlux); iiif != iiif.end(); ++iiif, ++iii0) {
        for (integer field = 0; field != opts().n_fields; ++field) {
            for (integer d = 0; d < NDIM; d++) {
                dUdt[field][iii0] -= (F[d][field][iiif + F_DN[d]] - F[XDIM][field][iiif]) * idx;
            }
        }
        if (opts().gravity) {
            dUdt[egas_i][iii0] += dUdt[pot_i][iii0] - HALF * dUdt[rho_i][iii0] * G[iii0][phi_i];
            dUdt[pot_i][iii0] = ZERO;
        }
    }
}

void grid::egas_to_etot() {
    PROFILE();
    if (opts().gravity) {
        for (BoxIterator iii(intBox, extBox); iii != iii.end(); ++iii) {
            U[egas_i][iii] += HALF * U[pot_i][iii];
        }
    }
}

void grid::etot_to_egas() {
    PROFILE();
    if (opts().gravity) {
        for (BoxIterator iii(intBox, extBox); iii != iii.end(); ++iii) {
            U[egas_i][iii] -= HALF * U[pot_i][iii];
        }
    }
}

void grid::next_u(integer rk, real t, real dt) {
    PROFILE();
    if (!opts().hydro) return;
    integer iii0 = 0;
    for (BoxIterator iii(intBox, extBox); iii != iii.end(); ++iii, ++iii0) {
        dUdt[egas_i][iii0] += HALF * dphi_dt[iii0] * U[rho_i][iii];
    }

    std::vector<real> du_out(opts().n_fields, ZERO);

    std::vector<real> ds(NDIM, ZERO);
    real const w1 = rk_beta[rk];
    real const w0 = w1 - ONE;
    integer j = 0;
    for (BoxIterator i(intBox, extBox); i != i.end(); ++i, ++j) {
        for (integer f = 0; f != opts().n_fields; ++f) {
            real const du1 = dUdt[f][j] * dt;
            real const du0 = U[f][i] - U0[f][j];
            U[f][i] += w0 * du0 + w1 * du1;
        }
    }
    if constexpr (NDIM >= 2) {
        du_out[sx_i] += omega * U_out[sy_i] * dt;
        du_out[sy_i] -= omega * U_out[sx_i] * dt;
    }
    real const dS = pow(dx, NDIM - 1);
    for (integer d = 0; d < NDIM; d++) {
        auto const leftFluxBox = intBoxFlux.slice(d, 0);
        auto const rightFluxBox = intBoxFlux.slice(d, INX);
        auto const leftPosBox = intBox.slice(d, H_BW);
        auto const rightPosBox = intBox.slice(d, H_NX - H_BW);
        BoxIterator ifl(leftFluxBox, extBoxFlux);
        BoxIterator ifr(rightFluxBox, extBoxFlux);
        BoxIterator ixl(leftPosBox, extBoxFlux);
        BoxIterator ixr(rightPosBox, extBoxFlux);
        for (; ifl != ifl.end(); ++ifr, ++ifl, ++ixr, ++ixl) {
            std::vector<real> du(opts().n_fields);
            for (integer f = 0; f != opts().n_fields; ++f) {
                if (X[d][ixr] > scaling_factor) {
                    du_out[f] += (F[d][f][ifr]) * dS;
                }
                if (X[d][ixl] > scaling_factor) {
                    du_out[f] += (F[d][f][ifl]) * dS;
                }
            }
        }
    }
    for (integer f = 0; f != opts().n_fields; ++f) {
        real const du_out1 = du_out[f] * dt;
        real const du_out0 = U_out[f] - U_out0[f];
        U_out[f] += w0 * du_out0 + w1 * du_out1;
    }
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
        if ((opts().tau_floor > ZERO) && (opts().eos != IPR)) {
            U[tau_i][iii] = std::max(U[tau_i][iii], opts().tau_floor);
        } else if (U[tau_i][iii] < ZERO) {
            std::cout << "Tau is negative @ " << std::to_string(U[tau_i][iii]);
            for (integer d = 0; d < NDIM; d++) {
                std::cout << " " << std::to_string(iii[d]);
            }
            for (integer d = 0; d < NDIM; d++) {
                std::cout << " " << std::to_string(X[d][iii]);
            }
            std::cout << "Use tau_floor option" << std::endl;
            abort();
        }
        if (opts().rho_floor > ZERO) {
            real x = ZERO;
            for (integer s = 0; s < opts().n_species; s++) {
                U[spc_i + s][iii] = std::max(U[spc_i + s][iii], ZERO);
                x += U[spc_i + s][iii];
            }
            if (x != ZERO) {
                for (integer s = 0; s < opts().n_species; s++) {
                    U[spc_i + s][iii] /= x;
                }
            } else {
                U[spc_i + opts().n_species - 1][iii] = ONE;
            }
            if (U[rho_i][iii] < opts().rho_floor) {
                x = ONE - std::max(U[rho_i][iii], ZERO) / opts().rho_floor;
                U[rho_i][iii] = opts().rho_floor;
                if (opts().eos == IPR) {
                    U[egas_i][iii] += x * ((opts().ipr_eint_floor) - U[egas_i][iii]);
                    U[tau_i][iii] -= x * U[tau_i][iii];
                } else {
                    U[tau_i][iii] += x * (opts().tau_floor - U[tau_i][iii]);
                    U[egas_i][iii] += x * (std::pow(opts().tau_floor, fgamma) - U[egas_i][iii]);
                }
                for (integer d = 0; d < NDIM; ++d) {
                    U[sx_i + d][iii] *= (ONE - x);
                }
            }
            for (integer s = 0; s < opts().n_species; s++) {
                U[spc_i + s][iii] *= U[rho_i][iii];
            }

        } else if (U[rho_i][iii] <= ZERO) {
            std::cout << "Rho is non-positive @ " << std::to_string(U[rho_i][iii]);
            for (integer d = 0; d < NDIM; d++) {
                std::cout << " " << std::to_string(iii[d]);
            }
            for (integer d = 0; d < NDIM; d++) {
                std::cout << " " << std::to_string(X[d][iii]);
            }
            std::cout << "Use tau_floor option" << std::endl;
            abort();
        }
    }
}

void grid::dual_energy_update() {
    PROFILE();
    physics<NDIM>::post_process<INX>(U, X, dx);
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
        real rho_tot = ZERO;
        for (integer s = 0; s < opts().n_species; s++) {
            rho_tot += U[spc_i + s][iii];
        }
        U[rho_i][iii] = rho_tot;
    }
}

std::pair<real, real> grid::virial() const {
    std::pair<real, real> v;
    v.first = v.second = ZERO;
    real const dV = pow(dx, NDIM);
    for (BoxIterator iii(intBox, extBox); !iii.end(); ++iii) {
        real ei, p;
        real et = U[egas_i][iii];
        real ek = ZERO;
        real const irho = ONE / U[rho_i][iii];
        for (integer d = 0; d < NDIM; d++) {
            ek += irho * sqr(U[sx_i + d][iii]);
        }
        ek *= HALF;
        if (opts().eos == WD) {
            ei = U[egas_i][iii] - ek - ztwd_energy(U[rho_i][iii]);
        } else {
            ei = U[egas_i][iii] - ek;
        }
        if (opts().eos == IPR) {
            ei = std::max(opts().ipr_eint_floor, ei);
            specie_state_t<real> spc;
            real mmw_loc, X_loc, Z_loc;
            for (integer si = 0; si != opts().n_species; ++si) {
                spc[si] = U[spc_i + si][iii];
            }
            mean_ion_weight(spc, mmw_loc, X_loc, Z_loc);
            p = ipr_pressure(U[tau_i][iii], U[rho_i][iii], mmw_loc);
        } else {
            if (ei < de_switch2 * et) {
                ei = std::pow(U[tau_i][iii], fgamma);
            }
            real p = (fgamma - ONE) * ei;
            if (opts().eos == WD) {
                p += ztwd_pressure(U[rho_i][iii]);
            }
        }
        v.first += (2.0 * ek + 0.5 * U[pot_i][iii] + 3.0 * p) * dV;
        v.second += (2.0 * ek - 0.5 * U[pot_i][iii] + 3.0 * p) * dV;
    }
    return v;
}

std::vector<real> grid::conserved_outflows() const {
    if (!opts().gravity) return U_out;
    auto Uret = U_out;
    Uret[egas_i] += Uret[pot_i];
    return Uret;
}
#endif
