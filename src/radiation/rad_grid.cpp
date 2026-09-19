//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/math/Debug.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/space_vector.hpp"
#include "octotiger/test_problems/radiation.hpp"

#include <hpx/include/future.hpp>

#include "octotiger/radiation/m1.hpp"
#include "octotiger/radiation/subcycle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

using M1 = RadiationM1<Real, NDIM>;

std::unordered_map<std::string, int> rad_grid::str_to_index;
std::unordered_map<int, std::string> rad_grid::index_to_str;

radiationConservation::Totals rad_grid::takeConservation() {
    FpeGuard fpeGuard;
    radiationConservation::Totals result;
    long double const cellVolume = static_cast<long double>(dx) * dx * dx;
    // Only leaf interiors enter the distributed reduction. Ghost copies and
    // restricted parent states must not count the same radiation twice.
    for (integer i = RAD_BW; i < RAD_NX - RAD_BW; ++i) {
        for (integer j = RAD_BW; j < RAD_NX - RAD_BW; ++j) {
            for (integer k = RAD_BW; k < RAD_NX - RAD_BW; ++k) {
                integer const r = rindex(i, j, k);
                result.volume += cellVolume;
                for (int f = 0; f < NRF; ++f)
                    result.value[f] += static_cast<long double>(U[f][r]) * cellVolume;
            }
        }
    }
    result.boundary = conservationBoundary;
    result.source = conservationSource;
    conservationBoundary.fill(0);
    conservationSource.fill(0);
    return result;
}

void rad_grid::accountBoundaryFlux(Real dt, geo::face face) {
    FpeGuard fpeGuard;
    // Integrate the numerical physical-unit face flux used by advance(), after
    // refluxing. Call only for exterior nonperiodic faces of leaf blocks.
    integer const normal = face.get_dimension();
    bool const plus = face.get_side() == geo::PLUS;
    integer const edge = plus ? RAD_NX - RAD_BW : RAD_BW;
    long double const weight = (plus ? 1.L : -1.L) * dt * dx * dx;
    for (integer a = RAD_BW; a < RAD_NX - RAD_BW; ++a) {
        for (integer b = RAD_BW; b < RAD_NX - RAD_BW; ++b) {
            integer const r = normal == XDIM ? rindex(edge, a, b)
                : (normal == YDIM ? rindex(a, edge, b) : rindex(a, b, edge));
            for (int f = 0; f < NRF; ++f)
                conservationBoundary[f] += weight * flux[normal][f][r];
        }
    }
}

void rad_grid::static_init() {
    str_to_index["er"] = er_i;
    str_to_index["fx"] = fx_i;
    str_to_index["fy"] = fy_i;
    str_to_index["fz"] = fz_i;
    for (const auto& s : str_to_index) {
        index_to_str[s.second] = s.first;
    }
}

std::vector<std::string> rad_grid::get_field_names() {
    std::vector<std::string> rc;
    for (auto i : str_to_index) {
        rc.push_back(i.first);
    }
    return rc;
}

void rad_grid::set(const std::string name, Real* data) {
    FpeGuard fpeGuard;
    auto iter = str_to_index.find(name);
    Real eunit = opts().problem == MARSHAK ?
        1 :
        opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
    Real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
    if (iter != str_to_index.end()) {
        int f = iter->second;
        int jjj = 0;
        for (int i = 0; i < INX; i++) {
            for (int j = 0; j < INX; j++) {
                for (int k = 0; k < INX; k++) {
                    const int iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
                    data[jjj] /= f == er_i ? eunit : funit;
                    U[f][iii] = data[jjj];
                    jjj++;
                }
            }
        }
    }
}

std::vector<silo_var_t> rad_grid::var_data() const {
    FpeGuard fpeGuard;
    std::vector<silo_var_t> s;
    Real eunit = opts().problem == MARSHAK ?
        1 :
        opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
    Real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
    for (auto l : str_to_index) {
        const int f = l.second;
        std::string this_name = l.first;
        int jjj = 0;
        silo_var_t this_s(this_name);
        for (int i = 0; i < INX; i++) {
            for (int j = 0; j < INX; j++) {
                for (int k = 0; k < INX; k++) {
                    const int iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
                    this_s(jjj) = U[f][iii];
                    this_s(jjj) *= f == er_i ? eunit : funit;
                    this_s.set_range(this_s(jjj));
                    jjj++;
                }
            }
        }
        s.push_back(std::move(this_s));
    }
    return std::move(s);
}

constexpr auto _0 = Real(0);
constexpr auto _1 = Real(1);
constexpr auto _2 = Real(2);
constexpr auto _3 = Real(3);
constexpr auto _4 = Real(4);
constexpr auto _5 = Real(5);

using set_rad_grid_action_type = node_server::set_rad_grid_action;
HPX_REGISTER_ACTION(set_rad_grid_action_type);

hpx::future<void> node_client::set_rad_grid(
    std::vector<Real>&& g /*, std::vector<Real>&& o*/) const {
    return hpx::async<typename node_server::set_rad_grid_action>(get_unmanaged_gid(), g /*, o*/);
}

void node_server::set_rad_grid(const std::vector<Real>& data /*, std::vector<Real>&& outflows*/) {
    rad_grid_ptr->set_prolong(data /*, std::move(outflows)*/);
}

using send_rad_boundary_action_type = node_server::send_rad_boundary_action;
HPX_REGISTER_ACTION(send_rad_boundary_action_type);

using send_rad_flux_correct_action_type = node_server::send_rad_flux_correct_action;
HPX_REGISTER_ACTION(send_rad_flux_correct_action_type);

void node_client::send_rad_flux_correct(
    std::vector<Real>&& data, const geo::face& face, const geo::octant& ci, std::size_t cycle) const {
    hpx::apply<typename node_server::send_rad_flux_correct_action>(
        get_unmanaged_gid(), std::move(data), face, ci, cycle);
}

void node_server::recv_rad_flux_correct(
    std::vector<Real>&& data, const geo::face& face, const geo::octant& ci, std::size_t cycle) {
    const geo::quadrant index(ci, face.get_dimension());
    niece_rad_channels[face][index].set_value(std::move(data),cycle);
}

void node_client::send_rad_boundary(
    std::vector<Real>&& data, const geo::direction& dir, std::size_t cycle) const {
    hpx::apply<typename node_server::send_rad_boundary_action>(
        get_gid(), std::move(data), dir, cycle);
}

void node_server::recv_rad_boundary(
    std::vector<Real>&& bdata, const geo::direction& dir, std::size_t cycle) {
    sibling_rad_type tmp;
    tmp.data = std::move(bdata);
    tmp.direction = dir;
    sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

using send_rad_children_action_type = node_server::send_rad_children_action;
HPX_REGISTER_ACTION(send_rad_children_action_type);

void node_server::recv_rad_children(
    std::vector<Real>&& data, const geo::octant& ci, std::size_t cycle) {
    child_rad_channels[ci].set_value(std::move(data), cycle);
}

void node_client::send_rad_children(
    std::vector<Real>&& data, const geo::octant& ci, std::size_t cycle) const {
    hpx::apply<typename node_server::send_rad_children_action>(
        get_unmanaged_gid(), std::move(data), ci, cycle);
}

namespace {
// Thermal energy excludes bulk motion and, for WD gas, the cold EOS energy.
// Use the existing dual-energy fallback when total-energy subtraction is ill-conditioned.
Real radiationGasInternal(Real egas, Real tau, Real sx, Real sy, Real sz, Real rho) {
    FpeGuard fpeGuard;
    (void) expectPositive(rho);
    Real e = egas - 0.5 * ((sx / rho) * sx + (sy / rho) * sy + (sz / rho) * sz);
    if (opts().eos == WD)
        e -= ztwd_energy(rho);
    if (e < opts().dual_energy_sw1 * egas)
        e = std::pow(expectNonNegative(tau), grid::get_fgamma());
    return expectNonNegative(e);
}
}    // namespace

void rad_grid::rad_imp(std::vector<Real>& egas, std::vector<Real>& tau,
    std::vector<Real>& sx, std::vector<Real>& sy, std::vector<Real>& sz,
    const std::vector<Real>& rho, Real dt) {
    PROFILE();
    FpeGuard guard;
    // S&O (44)--(49): thermal exchange once on the GAS timestep. Momentum
    // damping belongs to the radiation subcycles, not this nonlinear solve.
    if (opts().rad_energy_mode != "thermal") return;
    Real const c=physcon().c, ratio=opts().rad_c_ratio, gamma=grid::get_fgamma();
    bool const marshak=opts().problem==MARSHAK;
    Real const logAr=marshak ? 0 : std::log(4*physcon().sigma/c);
    Real const tempFactor=(gamma-1)*physcon().mh/physcon().kb;
    long double const dv=static_cast<long double>(dx)*dx*dx;
    constexpr integer offset=H_BW-RAD_BW;
    for (integer i=RAD_BW; i<RAD_NX-RAD_BW; ++i)
      for (integer j=RAD_BW; j<RAD_NX-RAD_BW; ++j)
        for (integer k=RAD_BW; k<RAD_NX-RAD_BW; ++k) {
            auto const r=rindex(i,j,k), h=hindex(i+offset,j+offset,k+offset);
            Real const e=radiationGasInternal(egas[h],tau[h],sx[h],sy[h],sz[h],rho[h]);
            Real const chi=radiationAbsorption(rho[h],e,mmw[r],X_spc[r],Z_spc[r],gamma);
            Real const logAlpha=marshak ? 0 : logAr+4*std::log(tempFactor*mmw[r]/rho[h]);
            auto const next=M1::thermalExchange(e,U[0][r],dt*c*chi,logAlpha,
                marshak,ratio,opts().rad_theta);
            // A thermal-only split can exhaust E beneath an existing beam.
            // Never clip F or E to conceal this: choose absorption mode when
            // emission is absent, or shorten the gas step for thermal problems.
            if (!M1::admissible({next.radiation,U[1][r]/c,U[2][r]/c,U[3][r]/c}))
                throw std::runtime_error("Thermal split produced |F| > c E; reduce hard_dt "
                    "(use rad_energy_mode=absorption for an absorption-only medium)");
            conservationSource[0]+=(static_cast<long double>(next.radiation)-U[0][r])*dv;
            U[0][r]=next.radiation;
            egas[h]+=next.gas-e;
            tau[h]=std::pow(next.gas,1/gamma);
        }
}

void rad_grid::prepareSources(const std::vector<std::vector<Real>>& gas) {
    FpeGuard guard;
    sourceEnabled=opts().rad_implicit && !radiationFixedMediumProblem();
    for (auto& delta:gasDelta) std::fill(delta.begin(),delta.end(),Real(0));
    if (!sourceEnabled) return;
    compute_mmw(gas);
    constexpr integer offset=H_BW-RAD_BW;
    for (integer i=0; i<RAD_NX; ++i)
      for (integer j=0; j<RAD_NX; ++j)
        for (integer k=0; k<RAD_NX; ++k) {
            auto const r=rindex(i,j,k), h=hindex(i+offset,j+offset,k+offset);
            Real const rho=gas[rho_i][h];
            Real const e=radiationGasInternal(gas[egas_i][h],gas[tau_i][h],
                gas[sx_i][h],gas[sy_i][h],gas[sz_i][h],rho);
            chiAbsorption[r]=radiationAbsorption(rho,e,mmw[r],X_spc[r],Z_spc[r],grid::get_fgamma());
            chiTotal[r]=radiationTransport(rho,e,mmw[r],X_spc[r],Z_spc[r],grid::get_fgamma());
            if (!(std::isfinite(chiAbsorption[r]) && chiAbsorption[r]>=0 &&
                  std::isfinite(chiTotal[r]) && chiTotal[r]>=0))
                throw std::runtime_error("Invalid radiation opacity");
            for (int d=0; d<NDIM; ++d) gasVelocity[d][r]=gas[sx_i+d][h]/rho;
        }
}

void rad_grid::finishSources(std::vector<std::vector<Real>>& gas) {
    FpeGuard guard;
    if (!sourceEnabled) return;
    // S&O freeze the gas during subcycling and correct its source afterwards.
    // Here the correction uses the actually integrated radiation sources:
    // gas+E/(c_hat/c), and momentum+F/(c*c_hat), balance to roundoff.
    // It is outside hydro's RK stages, so the hydro/radiation split is order 1.
    constexpr integer offset=H_BW-RAD_BW;
    for (integer i=RAD_BW; i<RAD_NX-RAD_BW; ++i)
      for (integer j=RAD_BW; j<RAD_NX-RAD_BW; ++j)
        for (integer k=RAD_BW; k<RAD_NX-RAD_BW; ++k) {
            auto const r=rindex(i,j,k), h=hindex(i+offset,j+offset,k+offset);
            Real const rho=gas[rho_i][h];
            Real const e=radiationGasInternal(gas[egas_i][h],gas[tau_i][h],
                gas[sx_i][h],gas[sy_i][h],gas[sz_i][h],rho);
            Real dk=0;
            for (int d=0; d<NDIM; ++d)
                dk+=gasDelta[d+1][r]*(gas[sx_i+d][h]/rho+Real(.5)*gasDelta[d+1][r]/rho);
            Real const nextInternal=e+gasDelta[0][r]-dk;
            if (!(std::isfinite(nextInternal) && nextInternal>=0))
                throw std::runtime_error("Radiation feedback exhausted gas internal energy; reduce hard_dt");
            gas[egas_i][h]+=gasDelta[0][r];
            gas[tau_i][h]=std::pow(nextInternal,1/grid::get_fgamma());
            for (int d=0; d<NDIM; ++d) gas[sx_i+d][h]+=gasDelta[d+1][r];
        }
    sourceEnabled=false;
}

void rad_grid::set_dx(Real _dx) {
    dx = _dx;
}

void rad_grid::set_X(const std::vector<std::vector<Real>>& x) {
    X.resize(NDIM);
    for (integer d = 0; d != NDIM; ++d) {
        X[d].resize(RAD_N3);
        for (integer xi = 0; xi != RAD_NX; ++xi) {
            for (integer yi = 0; yi != RAD_NX; ++yi) {
                for (integer zi = 0; zi != RAD_NX; ++zi) {
                    const auto D = H_BW - RAD_BW;
                    integer const iiir = rindex(xi, yi, zi);
                    integer const iiih = hindex(xi + D, yi + D, zi + D);
                    //		printf( "%i %i %i %i %i %i \n", d, iiir, xi, yi, zi, iiih);
                    X[d][iiir] = x[d][iiih];
                }
            }
        }
    }
}

// Skinner & Ostriker (2013), (29), https://arxiv.org/abs/1306.0010:
// c_eff² = [γP + (4/9)ℰ(1 - exp(-ρ κ₀ Δx))]/ρ.
// Return only the radiation contribution's square root; hydro supplies γP/ρ.
// Here ℰ = U[er_i], ρ κ₀ = kappa_R(...). Use -expm1(-τ) for 1 - exp(-τ)
// at small τ = ρ κ₀ Δx. Transport separately uses the configured reduced light speed.
Real rad_grid::hydroSignalSpeed(const std::vector<Real>& egas, const std::vector<Real>& tau,
    const std::vector<Real>& sx, const std::vector<Real>& sy, const std::vector<Real>& sz,
    const std::vector<Real>& rho) {
    FpeGuard fpeGuard;
    Real speed2 = 0;
    constexpr integer offset = H_BW - RAD_BW;
    for (integer i = RAD_BW; i < RAD_NX - RAD_BW; ++i) {
        for (integer j = RAD_BW; j < RAD_NX - RAD_BW; ++j) {
            for (integer k = RAD_BW; k < RAD_NX - RAD_BW; ++k) {
                integer const r = rindex(i, j, k);
                integer const h = hindex(i + offset, j + offset, k + offset);
                Real const e = radiationGasInternal(egas[h], tau[h], sx[h], sy[h], sz[h], rho[h]);
                Real const opticalDepth = expectNonNegative(
                    radiationTransport(rho[h], e, mmw[r], X_spc[r], Z_spc[r], grid::get_fgamma()) * dx);
                speed2 = std::max(
                    speed2, (4.0 / 9.0) * U[er_i][r] / rho[h] * (-std::expm1(-opticalDepth)));
            }
        }
    }
    return std::sqrt(expectNonNegative(speed2));
}

void rad_grid::compute_mmw(const std::vector<std::vector<Real>>& U) {
    FpeGuard fpeGuard;
    mmw.resize(RAD_N3);
    X_spc.resize(RAD_N3);
    Z_spc.resize(RAD_N3);
    for (integer i = 0; i != RAD_NX; ++i) {
        for (integer j = 0; j != RAD_NX; ++j) {
            for (integer k = 0; k != RAD_NX; ++k) {
                integer const d = H_BW - RAD_BW;
                integer const iiir = rindex(i, j, k);
                integer const iiih = hindex(i + d, j + d, k + d);
                specie_state_t<Real> spc;
                for (integer si = 0; si != opts().n_species; ++si) {
                    spc[si] = U[spc_i + si][iiih];
                }
                mean_ion_weight(spc, mmw[iiir], X_spc[iiir], Z_spc[iiir]);
            }
        }
    }
}

void node_server::compute_radiation(Real dt, Real omega) {
    auto& rgrid=*rad_grid_ptr;
    rgrid.set_dx(grid_ptr->get_dx());
    rgrid.set_X(grid_ptr->get_X());
    // EVERY node uses these same globally reduced values. A local CFL or local
    // refinement level here would desynchronize sibling/parent radiation keys.
    radiation::SubcyclePlan const plan(dt,dt_.radiation_dt,
        opts().rad_subcycling && opts().hydro,opts().rad_max_subcycles);
    if (opts().rad_log_subcycles && my_location.level()==0)
        printf("RAD gas_dt=%.17e rad_limit=%.17e subcycles=%zu rad_dt=%.17e\n",
            dt,dt_.radiation_dt,plan.count,plan.dt(0));

    bool const coupled=opts().rad_implicit && !radiationFixedMediumProblem();
    if (coupled) {
        if (!is_refined) {
            if (!opts().gravity && opts().optimize_local_communication)
                GET(all_neighbors_got_hydro[(hcycle-1)%number_hydro_exchange_promises]);
            rgrid.compute_mmw(grid_ptr->U);
            rgrid.rad_imp(grid_ptr->get_field(egas_i),grid_ptr->get_field(tau_i),
                grid_ptr->get_field(sx_i),grid_ptr->get_field(sy_i),grid_ptr->get_field(sz_i),
                grid_ptr->get_field(rho_i),dt);
        }
        // One material halo refresh for the frozen source data, on ALL levels.
        // This uses hcycle; the following radiation substeps use only rcycle.
        all_hydro_bounds();
        rgrid.prepareSources(grid_ptr->U);
    }
    std::array<bool,2*NDIM> physical{};
    for (auto face:geo::face::full_set())
        physical[face]=!opts().periodic && my_location.is_physical_boundary(face);
    for (std::size_t sub=0; sub<plan.count; ++sub) {
        Real const subdt=plan.dt(sub);
        Real const time=current_time+plan.offset(sub);
        // One radiation state exchange per substep. VL builds the half-time
        // halo locally; only the FINAL time-centered flux is refluxed.
        all_rad_bounds(time);
        rgrid.compute_flux(subdt,omega,time,physical);
        GET(exchange_rad_flux_corrections());
        if (!is_refined) {
            for (auto face:geo::face::full_set())
                if (physical[face]) rgrid.accountBoundaryFlux(subdt,face);
            rgrid.advance(subdt,omega);
            if (radiationFixedMediumProblem()) rgrid.applyRegressionSource(subdt);
        }
    }
    if (coupled && !is_refined) {
        if (!opts().gravity && opts().optimize_local_communication)
            GET(all_neighbors_got_hydro[(hcycle-1)%number_hydro_exchange_promises]);
        rgrid.finishSources(grid_ptr->U);
    }
    // Final restriction/physical boundaries at the exact end of the GAS step.
    // Regridding/migration subsequently inherit this same rcycle on all nodes.
    all_rad_bounds(current_time+dt);
}

void rad_grid::applyRegressionSource(Real dt) {
    FpeGuard fpeGuard;
    // These regression media are held fixed externally: record their prescribed
    // scattering/emission increments without applying gas recoil or heating.
    auto const parameters = radiationTestParameters();
    bool const bulb = opts().problem == RADIATION_EQUILIBRIUM_SPHERE;
    long double const cellVolume = static_cast<long double>(dx) * dx * dx;
    for (integer i = RAD_BW; i < RAD_NX - RAD_BW; ++i) {
        for (integer j = RAD_BW; j < RAD_NX - RAD_BW; ++j) {
            for (integer k = RAD_BW; k < RAD_NX - RAD_BW; ++k) {
                integer const r = rindex(i, j, k);
                Real const emission = bulb ? radiationTests::bulbSourceAverage(
                    {X[0][r], X[1][r], X[2][r]}, dx, parameters) : 0;
                auto const updated = radiationTests::mediumSource(
                    {U[0][r], U[1][r], U[2][r], U[3][r]}, dt * opts().rad_c_ratio, parameters, emission);
                M1::ConservedState{updated[0], updated[1] / parameters.c,
                    updated[2] / parameters.c, updated[3] / parameters.c}.checkState();
                for (int f = 0; f < NRF; ++f) {
                    conservationSource[f] += (static_cast<long double>(updated[f]) - U[f][r]) * cellVolume;
                    U[f][r] = updated[f];
                }
            }
        }
    }
}

void rad_grid::allocate() {
    chiAbsorption.resize(RAD_N3);
    chiTotal.resize(RAD_N3);
    for (auto& velocity:gasVelocity) velocity.resize(RAD_N3);
    U.resize(NRF);
    Ushad.resize(NRF);
    flux.resize(NDIM);
    for (int d = 0; d < NDIM; d++) {
        flux[d].resize(NRF);
    }
    for (integer f = 0; f != NRF; ++f) {
        U[f].resize(RAD_N3);
        Ushad[f].resize(HS_N3);
        Uhalf[f].resize(RAD_N3);
        gasDelta[f].resize(RAD_N3);
        for (auto& face : faces)
            face[f].resize(RAD_N3);
        for (integer d = 0; d != NDIM; ++d) {
            flux[d][f].resize(RAD_N3);
        }
    }
}

void rad_grid::sanity_check() {
#ifndef NDEBUG
    FpeGuard fpeGuard;
    Real const c = expectPositive(physcon().c);
    for (integer i = RAD_BW; i < RAD_NX - RAD_BW; ++i) {
        for (integer j = RAD_BW; j < RAD_NX - RAD_BW; ++j) {
            for (integer k = RAD_BW; k < RAD_NX - RAD_BW; ++k) {
                integer const r = rindex(i, j, k);
                M1::ConservedState{U[0][r], U[1][r] / c, U[2][r] / c, U[3][r] / c}.checkState();
            }
        }
    }
#endif
}

Real rad_grid::maxTimestep(Real omega) const {
    FpeGuard fpeGuard;
    Real gridSpeed = 0;
    if (omega != 0) {
        // v_grid=(-omega*y, omega*x, 0); ghost centres also bound all face centres.
        for (int d = 0; d < 2; ++d) {
            for (Real x : X[d])
                gridSpeed = std::max(gridSpeed, std::abs(omega * x));
        }
    }
    return M1::transportTimestep(dx, physcon().c * opts().rad_c_ratio, opts().rad_cfl, gridSpeed);
}

void rad_grid::compute_flux(Real dt, Real omega, Real time,
    std::array<bool,2*NDIM> physicalFaces) {
    PROFILE();
    FpeGuard guard;
    static_assert(NDIM==3 && NRF==4 && RAD_BW>=3 && H_BW>=RAD_BW);
    Real const c=physcon().c, ratio=opts().rad_c_ratio, chat=c*ratio;
    auto load=[&](integer r) {
        return M1::ConservedState{U[0][r],U[1][r]/c,U[2][r]/c,U[3][r]/c};
    };
    auto half=[&](integer r) {
        return M1::ConservedState{Uhalf[0][r],Uhalf[1][r],Uhalf[2][r],Uhalf[3][r]};
    };
    auto speed=[&](int normal,integer left,integer right) {
        if (omega==0 || normal==ZDIM) return Real(0);
        int const transverse=normal==XDIM ? YDIM : XDIM;
        return (normal==XDIM ? -omega : omega)*Real(.5)*(X[transverse][left]+X[transverse][right]);
    };
    for (integer r=0; r<RAD_N3; ++r)
        for (int f=0; f<NRF; ++f) Uhalf[f][r]=U[f][r]/(f==0 ? 1 : c);

    // Athena VL predictor: first-order HLL fluxes from cell averages, an
    // unsplit half-step, and first-order sources. Predict TWO ghost layers
    // from the three-layer t^n halo; no halfway communication is necessary.
    for (integer i=1; i<RAD_NX-1; ++i)
      for (integer j=1; j<RAD_NX-1; ++j)
        for (integer k=1; k<RAD_NX-1; ++k) {
            integer const r=rindex(i,j,k);
            auto const old=load(r);
            M1::StateVector increment{};
            Real updateScale=old[0];
            for (int d=0; d<NDIM; ++d) {
                auto const stride=R_DN[d];
                updateScale+=Real(.5)*dt/dx*(chat+std::abs(speed(d,r-stride,r))+
                    std::abs(speed(d,r,r+stride)))*(U[0][r-stride]+2*old[0]+U[0][r+stride]);
                auto const minus=M1::hll(load(r-stride),old,d,chat,speed(d,r-stride,r));
                auto const plus=M1::hll(old,load(r+stride),d,chat,speed(d,r,r+stride));
                increment-=Real(.5)*dt/dx*(plus-minus);
                for (int f=0; f<NRF; ++f)
                    updateScale+=Real(.5)*dt/dx*(std::abs(plus[f])+std::abs(minus[f]));
            }
            if (sourceEnabled) {
                auto const src=M1::explicitSource(old,
                    {gasVelocity[0][r],gasVelocity[1][r],gasVelocity[2][r]},
                    chiAbsorption[r],chiTotal[r],c,ratio,opts().rad_velocity_terms);
                increment+=Real(.5)*dt*src;
                for (int f=0; f<NRF; ++f) updateScale+=Real(.5)*dt*std::abs(src[f]);
            }
            M1::ConservedState predicted=old+increment;
            if (sourceEnabled) {
                for (int f=1; f<NRF; ++f)
                    predicted[f]=M1::damp(old[f],increment[f],Real(.5)*dt*chat*chiTotal[r],1);
                if (opts().rad_energy_mode=="absorption")
                    predicted[0]=M1::damp(old[0],increment[0],Real(.5)*dt*chat*chiAbsorption[r],1);
            }
            if (omega!=0) {
                Real const cs=std::cos(Real(.5)*omega*dt), sn=std::sin(Real(.5)*omega*dt);
                Real const x=predicted[1], y=predicted[2];
                predicted[1]=cs*x+sn*y; predicted[2]=cs*y-sn*x;
            }
            predicted=M1::roundoffState(predicted,updateScale);
            for (int f=0; f<NRF; ++f) Uhalf[f][r]=predicted[f];
        }
    // Prescribed/time-dependent physical faces must be evaluated at half time,
    // not evolved as though they were interior cells (notably Marshak inflow).
    for (int face=0; face<2*NDIM; ++face)
        if (physicalFaces[face]) set_physical_boundaries(geo::face(face),time+Real(.5)*dt,true);

    // S&O 3.3: PLM of half-time E and F, then HLL. Dividing F by the constant
    // physical c merely rescales the identical component reconstruction.
    for (int normal=0; normal<NDIM; ++normal) {
        integer const stride=R_DN[normal];
        std::array<integer,NDIM> lo{RAD_BW,RAD_BW,RAD_BW};
        std::array<integer,NDIM> hi{RAD_NX-RAD_BW,RAD_NX-RAD_BW,RAD_NX-RAD_BW};
        --lo[normal]; ++hi[normal];
        for (integer i=lo[0]; i<hi[0]; ++i)
          for (integer j=lo[1]; j<hi[1]; ++j)
            for (integer k=lo[2]; k<hi[2]; ++k) {
                integer const r=rindex(i,j,k);
                auto const pair=M1::reconstruct(half(r-stride),half(r),half(r+stride));
                for (int f=0; f<NRF; ++f) {
                    faces[0][f][r]=pair.first[f]; faces[1][f][r]=pair.second[f];
                }
            }
        ++lo[normal];
        for (integer i=lo[0]; i<hi[0]; ++i)
          for (integer j=lo[1]; j<hi[1]; ++j)
            for (integer k=lo[2]; k<hi[2]; ++k) {
                integer const right=rindex(i,j,k), left=right-stride;
                M1::ConservedState l,r;
                for (int f=0; f<NRF; ++f) { l[f]=faces[1][f][left]; r[f]=faces[0][f][right]; }
                Real const velocity=speed(normal,left,right);
                auto const high=M1::hll(l,r,normal,chat,velocity);
                auto const F=M1::limitFlux(load(left),load(right),high,normal,chat,velocity,dt/dx);
                for (int f=0; f<NRF; ++f) flux[normal][f][right]=F[f]*(f==0 ? 1 : c);
            }
    }
}

void rad_grid::change_units(Real m, Real l, Real t, Real k) {
    FpeGuard fpeGuard;
    Real const l2 = l * l;
    Real const t2 = t * t;
    Real const t2inv = 1.0 * INVERSE(t2);
    Real const tinv = 1.0 * INVERSE(t);
    Real const l3 = l2 * l;
    Real const l3inv = 1.0 * INVERSE(l3);
    for (integer i = 0; i != RAD_N3; ++i) {
        U[er_i][i] *= (m * l2 * t2inv) * l3inv;
        U[fx_i][i] *= tinv * (m * t2inv);
        U[fy_i][i] *= tinv * (m * t2inv);
        U[fz_i][i] *= tinv * (m * t2inv);
    }
}

void rad_grid::advance(Real dt, Real omega) {
    FpeGuard guard;
    Real const c=physcon().c, ratio=opts().rad_c_ratio, chat=c*ratio;
    long double const dv=static_cast<long double>(dx)*dx*dx;
    for (integer i=RAD_BW; i<RAD_NX-RAD_BW; ++i)
      for (integer j=RAD_BW; j<RAD_NX-RAD_BW; ++j)
        for (integer k=RAD_BW; k<RAD_NX-RAD_BW; ++k) {
            integer const r=rindex(i,j,k);
            M1::ConservedState old, next;
            M1::StateVector transport{};
            Real updateScale=U[0][r];
            for (int d=0; d<NDIM; ++d)
                updateScale+=dt/dx*(chat+std::abs(omega)*(std::abs(X[0][r])+std::abs(X[1][r])))
                    *(Uhalf[0][r-R_DN[d]]+2*Uhalf[0][r]+Uhalf[0][r+R_DN[d]]);
            for (int f=0; f<NRF; ++f) {
                Real const unit=f==0 ? 1 : c;
                old[f]=U[f][r]/unit;
                for (int d=0; d<NDIM; ++d) {
                    transport[f]-=dt/dx*(flux[d][f][r+R_DN[d]]-flux[d][f][r])/unit;
                    updateScale+=dt/dx*(std::abs(flux[d][f][r+R_DN[d]])+std::abs(flux[d][f][r]))/unit;
                }
                next[f]=old[f]+transport[f];
            }
            if (sourceEnabled) {
                M1::ConservedState const mid{Uhalf[0][r],Uhalf[1][r],Uhalf[2][r],Uhalf[3][r]};
                auto const src=M1::explicitSource(mid,
                    {gasVelocity[0][r],gasVelocity[1][r],gasVelocity[2][r]},
                    chiAbsorption[r],chiTotal[r],c,ratio,opts().rad_velocity_terms);
                Real work=dt*src[0];
                next[0]+=work;
                if (opts().rad_energy_mode=="absorption") {
                    Real const eta=dt*chat*chiAbsorption[r];
                    Real theta=opts().rad_theta;
                    if ((1-theta)*eta>1) theta=1;
                    next[0]=M1::damp(old[0],transport[0]+work,eta,theta);
                    // Absorbed energy escapes this band, as in S&O (51)--(52).
                    // Only the velocity work has equal-and-opposite gas energy.
                    work/=1+theta*eta;
                }
                gasDelta[0][r]-=work/ratio;
                for (int f=1; f<NRF; ++f) {
                    next[f]=M1::damp(old[f],transport[f]+dt*src[f],
                        dt*chat*chiTotal[r],opts().rad_theta);
                    gasDelta[f][r]-=(next[f]-old[f]-transport[f])/chat;
                }
                for (int f=0; f<NRF; ++f)
                    conservationSource[f]+=(static_cast<long double>(next[f])-old[f]-transport[f])
                        *(f==0 ? 1 : c)*dv;
            }
            if (omega!=0) {
                // Integrating-factor correction: rotate U^n a full step and
                // the time-centered increment a half step. This complements
                // the predictor's half rotation without first-order splitting.
                Real const cs=std::cos(omega*dt), sn=std::sin(omega*dt);
                Real const ch=std::cos(Real(.5)*omega*dt), sh=std::sin(Real(.5)*omega*dt);
                Real const ix=next[1]-old[1], iy=next[2]-old[2];
                Real const x=cs*old[1]+sn*old[2]+ch*ix+sh*iy;
                Real const y=cs*old[2]-sn*old[1]+ch*iy-sh*ix;
                conservationSource[1]+=(static_cast<long double>(x)-next[1])*c*dv;
                conservationSource[2]+=(static_cast<long double>(y)-next[2])*c*dv;
                next[1]=x; next[2]=y;
            }
            auto const rounded=M1::roundoffState(next,updateScale+std::abs(next[0]));
            for (int f=0; f<NRF; ++f) {
                conservationSource[f]+=(static_cast<long double>(rounded[f])-next[f])*(f==0 ? 1 : c)*dv;
                U[f][r]=rounded[f]*(f==0 ? 1 : c);
            }
        }
}

void rad_grid::set_physical_boundaries(geo::face face, Real t, bool half) {
    FpeGuard fpeGuard;
    auto state=[&](int f,integer r)->Real& { return half ? Uhalf[f][r] : U[f][r]; };
    auto unit=[&](int f) { return half && f!=er_i ? Real(1)/physcon().c : Real(1); };
    for (integer i = 0; i != RAD_NX; ++i) {
        for (integer j = 0; j != RAD_NX; ++j) {
            for (integer k = 0; k != RAD_BW; ++k) {
                integer iii1, iii0;
                switch (face) {
                case 0:
                    iii1 = rindex(k, i, j);
                    iii0 = rindex(RAD_BW, i, j);
                    break;
                case 1:
                    iii1 = rindex(RAD_NX - 1 - k, i, j);
                    iii0 = rindex(RAD_NX - 1 - RAD_BW, i, j);
                    break;
                case 2:
                    iii1 = rindex(i, k, j);
                    iii0 = rindex(i, RAD_BW, j);
                    break;
                case 3:
                    iii1 = rindex(i, RAD_NX - 1 - k, j);
                    iii0 = rindex(i, RAD_NX - 1 - RAD_BW, j);
                    break;
                case 4:
                    iii1 = rindex(i, j, k);
                    iii0 = rindex(i, j, RAD_BW);
                    break;
                case 5:
                default:
                    iii1 = rindex(i, j, RAD_NX - 1 - k);
                    iii0 = rindex(i, j, RAD_NX - 1 - RAD_BW);
                }
                if (opts().problem == RADIATION_EQUILIBRIUM_SPHERE) {
                    // The reference includes incoming radiation at the box faces;
                    // ordinary outflow clipping would change this prescribed test.
                    auto const profile = radiationTests::sphereAverage(
                        {X[0][iii1], X[1][iii1], X[2][iii1]}, dx, radiationTestParameters());
                    for (int f = 0; f < NRF; ++f)
                        state(f,iii1) = profile[f]*unit(f);
                    continue;
                }
                for (integer f = 0; f != NRF; ++f) {
                    state(f,iii1) = state(f,iii0);
                }
                switch (face) {
                case 0:
                    if (opts().problem == MARSHAK) {
                        if (t > 0) {
                            auto u = marshak_wave_analytic(-opts().xscale, 0, 0, t);
                            state(fx_i,iii1) = u[opts().n_fields + fx_i]*unit(fx_i);
                            state(er_i,iii1) = std::max(u[opts().n_fields + er_i], 1.0e-10);
                        } else {
                            state(fx_i,iii1) = 0.0;
                            state(er_i,iii1) = 1.0e-10;
                        }
                    } else {
                        state(fx_i,iii1) = std::min(state(fx_i,iii1), 0.0);
                    }
                    break;
                case 1:
                    state(fx_i,iii1) = std::max(state(fx_i,iii1), 0.0);
                    break;
                case 2:
                    state(fy_i,iii1) = std::min(state(fy_i,iii1), 0.0);
                    break;
                case 3:
                    state(fy_i,iii1) = std::max(state(fy_i,iii1), 0.0);
                    break;
                case 4:
                    state(fz_i,iii1) = std::min(state(fz_i,iii1), 0.0);
                    break;
                case 5:
                    state(fz_i,iii1) = std::max(state(fz_i,iii1), 0.0);
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
            lb[XDIM] = lb[YDIM] = lb[ZDIM] = RAD_BW;
            ub[XDIM] = ub[YDIM] = ub[ZDIM] = INX + RAD_BW;
            if (f.get_side() == geo::MINUS) {
                lb[face_dim] = RAD_BW;
            } else {
                lb[face_dim] = INX + RAD_BW;
            }
            ub[face_dim] = lb[face_dim] + 1;
            auto data = rad_grid_ptr->get_flux_restrict(lb, ub, face_dim);
            this_aunt.send_rad_flux_correct(std::move(data), f.flip(), ci, rcycle);
        }
    }

    constexpr integer size = geo::face::count() * geo::quadrant::count();
    std::array<future<void>, size> futs;
    for (auto& f : futs) {
        f = hpx::make_ready_future();
    }
    integer index = 0;
    for (auto const& f : geo::face::full_set()) {
        if (this->nieces[f] == +1) {
            for (auto const& quadrant : geo::quadrant::full_set()) {
                futs[index++] = niece_rad_channels[f][quadrant].get_future(rcycle).then(
                    [this, f, quadrant](hpx::future<std::vector<Real>>&& fdata) -> void {
                        const auto face_dim = f.get_dimension();
                        std::array<integer, NDIM> lb, ub;
                        switch (face_dim) {
                        case XDIM:
                            lb[XDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
                            lb[YDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
                            lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
                            ub[XDIM] = lb[XDIM] + 1;
                            ub[YDIM] = lb[YDIM] + (INX / 2);
                            ub[ZDIM] = lb[ZDIM] + (INX / 2);
                            break;
                        case YDIM:
                            lb[XDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
                            lb[YDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
                            lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
                            ub[XDIM] = lb[XDIM] + (INX / 2);
                            ub[YDIM] = lb[YDIM] + 1;
                            ub[ZDIM] = lb[ZDIM] + (INX / 2);
                            break;
                        case ZDIM:
                        default:
                            lb[XDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
                            lb[YDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
                            lb[ZDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
                            ub[XDIM] = lb[XDIM] + (INX / 2);
                            ub[YDIM] = lb[YDIM] + (INX / 2);
                            ub[ZDIM] = lb[ZDIM] + 1;
                            break;
                        }
                        rad_grid_ptr->set_flux_restrict(GET(fdata), lb, ub, face_dim);
                    });
            }
        }
    }
    return hpx::when_all(std::move(futs)).then([](future<decltype(futs)> fout) {
        auto fin = GET(fout);
        for (auto& f : fin) {
            GET(f);
        }
    });
}

void rad_grid::set_flux_restrict(const std::vector<Real>& data, const std::array<integer, NDIM>& lb,
    const std::array<integer, NDIM>& ub, const geo::dimension& dim) {
    integer index = 0;
    for (integer field = 0; field != NRF; ++field) {
        for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
            for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
                for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
                    integer const iii = rindex(i, j, k);
                    flux[dim][field][iii] = data[index];
                    ++index;
                }
            }
        }
    }
}

std::vector<Real> rad_grid::get_flux_restrict(const std::array<integer, NDIM>& lb,
    const std::array<integer, NDIM>& ub, const geo::dimension& dim) const {
    FpeGuard fpeGuard;
    std::vector<Real> data;
    integer size = 1;
    for (auto& dim : geo::dimension::full_set()) {
        size *= (ub[dim] - lb[dim]);
    }
    size /= (NCHILD / 2);
    size *= NRF;
    data.reserve(size);
    integer const stride1 = (dim == XDIM) ? (RAD_NX) : (RAD_NX) * (RAD_NX);
    integer const stride2 = (dim == ZDIM) ? (RAD_NX) : 1;
    for (integer field = 0; field != NRF; ++field) {
        for (integer i = lb[XDIM]; i < ub[XDIM]; i += 2) {
            for (integer j = lb[YDIM]; j < ub[YDIM]; j += 2) {
                for (integer k = lb[ZDIM]; k < ub[ZDIM]; k += 2) {
                    integer const i00 = rindex(i, j, k);
                    integer const i10 = i00 + stride1;
                    integer const i01 = i00 + stride2;
                    integer const i11 = i00 + stride1 + stride2;
                    Real value = ZERO;
                    value += flux[dim][field][i00];
                    value += flux[dim][field][i10];
                    value += flux[dim][field][i01];
                    value += flux[dim][field][i11];
                    value /= Real(4);
                    data.push_back(value);
                }
            }
        }
    }
    return data;
}

void node_server::all_rad_bounds(Real boundary_time) {
    //	if( my_location.level() == 0 ) printf( "\nbounds 1\n");
    GET(exchange_interlevel_rad_data());
    //	if( my_location.level() == 0 ) printf( "\nbounds 2\n");
    collect_radiation_bounds(boundary_time);
    //	if( my_location.level() == 0 ) printf( "\nbounds 3\n");
    send_rad_amr_bounds();
    //	if( my_location.level() == 0 ) printf( "\nbounds 4\n");
    rcycle++;
}

hpx::future<void> node_server::exchange_interlevel_rad_data() {
    hpx::future<void> f = hpx::make_ready_future();
    integer ci = my_location.get_child_index();

    if (is_refined) {
        for (auto const& ci : geo::octant::full_set()) {
            auto data = GET(child_rad_channels[ci].get_future(rcycle));
            rad_grid_ptr->set_restrict(data, ci);
        }
    }
    if (my_location.level() > 0) {
        auto data = rad_grid_ptr->get_restrict();
        parent.send_rad_children(std::move(data), ci, rcycle);
    }
    return hpx::make_ready_future();
}

void node_server::collect_radiation_bounds(Real boundary_time) {
    rad_grid_ptr->clear_amr();
    for (auto const& dir : geo::direction::full_set()) {
        if (!neighbors[dir].empty()) {
            auto bdata = rad_grid_ptr->get_boundary(dir);
            neighbors[dir].send_rad_boundary(std::move(bdata), dir.flip(), rcycle);
        }
    }

    std::array<future<sibling_rad_type>, geo::direction::count()> results;
    integer index = 0;
    for (auto const& dir : geo::direction::full_set()) {
        if (!(neighbors[dir].empty() && my_location.level() == 0)) {
            results[index++] = sibling_rad_channels[dir].get_future(rcycle);
        }
    }
    // Receive concurrently, then unpack on this task: adjacent AMR messages
    // include overlapping coarse stencil halos. Concurrent writes to Ushad
    // would race even when the incoming floating-point values are identical.
    for (integer n = 0; n < index; ++n) {
        auto tmp = GET(results[n]);
        if (!neighbors[tmp.direction].empty()) {
            rad_grid_ptr->set_boundary(tmp.data, tmp.direction);
        } else {
            rad_grid_ptr->set_rad_amr_boundary(tmp.data, tmp.direction);
        }
    }
    rad_grid_ptr->complete_rad_amr_boundary();
    for (auto& face : geo::face::full_set()) {
        if (!opts().periodic && my_location.is_physical_boundary(face)) {
            rad_grid_ptr->set_physical_boundaries(face, boundary_time);
        }
    }
}

void rad_grid::initialize_erad(const std::vector<Real>&, const std::vector<Real>&) {
    // Problem initialization installs the E,F profiles directly. Preserve them,
    // including true vacuum; emission begins in the first source step. This hook
    // intentionally performs no equilibrium reset (as in the supplied source).
}

rad_grid::rad_grid(Real _dx)
  : dx(_dx)
  , is_coarse(HS_N3)
  , has_coarse(HS_N3) {
    allocate();
}

rad_grid::rad_grid()
  : is_coarse(HS_N3)
  , has_coarse(HS_N3) {
    allocate();
}

void rad_grid::set_boundary(const std::vector<Real>& data, const geo::direction& dir) {
    std::array<integer, NDIM> lb, ub;
    get_boundary_size(lb, ub, dir, OUTER, INX, RAD_BW);
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
}

std::vector<Real> rad_grid::get_boundary(const geo::direction& dir) {
    std::array<integer, NDIM> lb, ub;
    std::vector<Real> data;
    integer size = NRF * get_boundary_size(lb, ub, dir, INNER, INX, RAD_BW);
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

    return data;
}

void rad_grid::set_field(Real v, integer f, integer i, integer j, integer k) {
    U[f][rindex(i, j, k)] = v;
}

Real rad_grid::get_field(integer f, integer i, integer j, integer k) const {
    return U[f][rindex(i, j, k)];
}

void rad_grid::set_prolong(const std::vector<Real>& data) {
    integer index = 0;
    for (integer f = 0; f != NRF; ++f) {
        for (integer i = RAD_BW; i != RAD_NX - RAD_BW; ++i) {
            for (integer j = RAD_BW; j != RAD_NX - RAD_BW; ++j) {
                for (integer k = RAD_BW; k != RAD_NX - RAD_BW; ++k) {
                    integer const iii = rindex(i, j, k);
                    U[f][iii] = data[index];
                    ++index;
                }
            }
        }
    }
}

std::vector<Real> rad_grid::get_prolong(
    const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub) {
    std::vector<Real> data;
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
                    integer const iii = rindex(i / 2, j / 2, k / 2);
                    Real value = U[f][iii];
                    data.push_back(value);
                }
            }
        }
    }
    return data;
}

std::vector<Real> rad_grid::get_restrict() const {
    FpeGuard fpeGuard;
    std::vector<Real> data;
    for (integer f = 0; f != NRF; ++f) {
        for (integer i = RAD_BW; i < RAD_NX - RAD_BW; i += 2) {
            for (integer j = RAD_BW; j < RAD_NX - RAD_BW; j += 2) {
                for (integer k = RAD_BW; k < RAD_NX - RAD_BW; k += 2) {
                    integer const iii = rindex(i, j, k);
                    Real v = ZERO;
                    for (integer x = 0; x != 2; ++x) {
                        for (integer y = 0; y != 2; ++y) {
                            for (integer z = 0; z != 2; ++z) {
                                integer const jjj = iii + x * RAD_NX * RAD_NX + y * RAD_NX + z;
                                v += U[f][jjj];
                            }
                        }
                    }
                    v /= Real(NCHILD);
                    data.push_back(v);
                }
            }
        }
    }
    return data;
}

void rad_grid::set_restrict(const std::vector<Real>& data, const geo::octant& octant) {
    integer index = 0;
    integer const i0 = octant.get_side(XDIM) * (INX / 2);
    integer const j0 = octant.get_side(YDIM) * (INX / 2);
    integer const k0 = octant.get_side(ZDIM) * (INX / 2);
    for (integer f = 0; f != NRF; ++f) {
        for (integer i = RAD_BW; i != RAD_NX / 2; ++i) {
            for (integer j = RAD_BW; j != RAD_NX / 2; ++j) {
                for (integer k = RAD_BW; k != RAD_NX / 2; ++k) {
                    integer const iii = rindex(i + i0, j + j0, k + k0);
                    U[f][iii] = data[index];
                    ++index;
                    if (index > int(data.size())) {
                        printf(
                            "rad_grid::set_restrict error %i %i\n", int(index), int(data.size()));
                    }
                }
            }
        }
    }
};

void node_server::send_rad_amr_bounds() {
    if (is_refined) {
        constexpr auto full_set = geo::octant::full_set();
        for (auto& ci : full_set) {
            const auto& flags = amr_flags[ci];
            for (auto& dir : geo::direction::full_set()) {
                if (flags[dir]) {
                    std::array<integer, NDIM> lb, ub;
                    std::vector<Real> data;
                    get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
                    for (integer dim = 0; dim != NDIM; ++dim) {
                        lb[dim] = std::max(lb[dim] - 1, integer(0));
                        ub[dim] = std::min(ub[dim] + 1, integer(HS_NX));
                        lb[dim] = lb[dim] + ci.get_side(dim) * (INX / 2);
                        ub[dim] = ub[dim] + ci.get_side(dim) * (INX / 2);
                    }
                    data = rad_grid_ptr->get_subset(lb, ub);
                    children[ci].send_rad_amr_boundary(std::move(data), dir, rcycle);
                }
            }
        }
    }
}

using erad_init_action_type = node_server::erad_init_action;
HPX_REGISTER_ACTION(erad_init_action_type);

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

void rad_grid::clear_amr() {
    std::fill(is_coarse.begin(), is_coarse.end(), 0);
    std::fill(has_coarse.begin(), has_coarse.end(), 0);
}

void rad_grid::set_rad_amr_boundary(const std::vector<Real>& data, const geo::direction& dir) {
    PROFILE();

    std::array<integer, NDIM> lb, ub;
    int l = 0;
    get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
    for (int i = lb[0]; i < ub[0]; i++) {
        for (int j = lb[1]; j < ub[1]; j++) {
            for (int k = lb[2]; k < ub[2]; k++) {
                is_coarse[hSindex(i, j, k)]++;
                assert(i < H_BW || i >= HS_NX - H_BW || j < H_BW || j >= HS_NX - H_BW || k < H_BW ||
                    k >= HS_NX - H_BW);
            }
        }
    }

    for (int dim = 0; dim < NDIM; dim++) {
        lb[dim] = std::max(lb[dim] - 1, integer(0));
        ub[dim] = std::min(ub[dim] + 1, integer(HS_NX));
    }

    for (int f = 0; f < NRF; f++) {
        for (int i = lb[0]; i < ub[0]; i++) {
            for (int j = lb[1]; j < ub[1]; j++) {
                for (int k = lb[2]; k < ub[2]; k++) {
                    has_coarse[hSindex(i, j, k)]++;
                    Ushad[f][hSindex(i, j, k)] = data[l++];
                }
            }
        }
    }
    assert(l == data.size());
}

void rad_grid::complete_rad_amr_boundary() {
    PROFILE();
    FpeGuard fpeGuard;
    using PhysicalState = M1::StateVector;
    Real const c = expectPositive(physcon().c);
    constexpr std::array<integer, NDIM> stride{HS_DNX, HS_DNY, HS_DNZ};
    constexpr integer offset = H_BW - RAD_BW;
    // AMR volume interpolation is distinct from face reconstruction. Use the
    // same six-neighbour stencil, but limit physical conserved slopes together
    // so the eight child volumes average exactly to the coarse state.
    for (int i0 = 1; i0 < HS_NX - 1; ++i0) {
        for (int j0 = 1; j0 < HS_NX - 1; ++j0) {
            for (int k0 = 1; k0 < HS_NX - 1; ++k0) {
                integer const coarse = hSindex(i0, j0, k0);
                if (!is_coarse[coarse])
                    continue;
                PhysicalState center;
                std::array<PhysicalState, NDIM> slope{};
                for (int f = 0; f < NRF; ++f) {
                    center[f] = Ushad[f][coarse];
                    for (int d = 0; d < NDIM; ++d) {
                        if (has_coarse[coarse - stride[d]] && has_coarse[coarse + stride[d]]) {
                            slope[d][f] = 0.25 *
                                M1::plmSlope(center[f] - Ushad[f][coarse - stride[d]],
                                    Ushad[f][coarse + stride[d]] - center[f]);
                        }
                    }
                }
                M1::ConservedState{center[0], center[1] / c, center[2] / c, center[3] / c}.checkState();
                std::array<PhysicalState, NCHILD> children;
                Real scale = 1;
                // All children use one factor; limiting each child separately
                // would change the parent average. The admissible M1 cone is convex.
                for (;;) {
                    bool admissible = true;
                    for (int child = 0; child < NCHILD; ++child) {
                        auto& u = children[child];
                        for (int f = 0; f < NRF; ++f) {
                            u[f] = center[f];
                            for (int d = 0; d < NDIM; ++d) {
                                u[f] += scale * ((child & (1 << (2 - d))) ? 1 : -1) * slope[d][f];
                            }
                        }
                        Real const norm = std::hypot(u[1] / c, u[2] / c, u[3] / c);
                        admissible =
                            admissible && u[0] >= 0 && norm <= u[0] * (1 + M1::roundoff);
                    }
                    if (admissible)
                        break;
                    scale *= 0.5;
                    if (scale < std::numeric_limits<Real>::epsilon()) {
                        children.fill(center);
                        break;
                    }
                }
                for (int child = 0; child < NCHILD; ++child) {
                    // The coarse shadow grid uses hydro ghost coordinates.
                    // Convert to radiation coordinates explicitly, even if widths coincide.
                    integer const i = 2 * i0 - H_BW + ((child >> 2) & 1) - offset;
                    integer const j = 2 * j0 - H_BW + ((child >> 1) & 1) - offset;
                    integer const k = 2 * k0 - H_BW + (child & 1) - offset;
                    if (i < 0 || i >= RAD_NX || j < 0 || j >= RAD_NX || k < 0 || k >= RAD_NX)
                        continue;
                    integer const r = rindex(i, j, k);
                    for (int f = 0; f < NRF; f++) {
                        U[f][r] = children[child][f];
                    }
                }
            }
        }
    }
}

std::vector<Real> rad_grid::get_subset(
    const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub) {
    PROFILE();
    std::vector<Real> data;
    data.reserve(NRF * (ub[0] - lb[0]) * (ub[1] - lb[1]) * (ub[2] - lb[2]));
    constexpr integer offset = H_BW - RAD_BW;
    // The AMR message envelope uses hydro coordinates; the values live in U
    // with radiation strides. If hydro has a wider halo, its extra padding is
    // outside every stencil needed for a radiation ghost cell (RAD_BW >= 2).
    // Extend the nearest available value into that unused message padding.
    const auto radiation_index = [](integer x) {
        return std::clamp(x - offset, integer(0), integer(RAD_NX - 1));
    };
    for (int f = 0; f < NRF; f++) {
        for (int i = lb[0]; i < ub[0]; i++) {
            for (int j = lb[1]; j < ub[1]; j++) {
                for (int k = lb[2]; k < ub[2]; k++) {
                    data.push_back(
                        U[f][rindex(radiation_index(i), radiation_index(j), radiation_index(k))]);
                }
            }
        }
    }
    return std::move(data);
}

using send_rad_amr_boundary_action_type = node_server::send_rad_amr_boundary_action;
HPX_REGISTER_ACTION(send_rad_amr_boundary_action_type);

void node_server::recv_rad_amr_boundary(
    std::vector<Real>&& bdata, const geo::direction& dir, std::size_t cycle) {
    sibling_rad_type tmp;
    tmp.data = std::move(bdata);
    tmp.direction = dir;
    sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

void node_client::send_rad_amr_boundary(
    std::vector<Real>&& data, const geo::direction& dir, std::size_t cycle) const {
    hpx::apply<typename node_server::send_rad_amr_boundary_action>(
        get_unmanaged_gid(), std::move(data), dir, cycle);
}

#endif
