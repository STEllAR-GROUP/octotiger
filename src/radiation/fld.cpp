// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/fld.hpp"
#include "octotiger/radiation/fld_hydro.hpp"
#include "octotiger/radiation/fld_benchmarks.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/radiation/physical_opacity.hpp"
#include "octotiger/util.hpp"
#include "octotiger/test_problems/radiation.hpp"

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iterator>
#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <utility>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

using receive_fld_action_type = node_server::receive_fld_action;
HPX_REGISTER_ACTION(receive_fld_action_type);
using deliver_fld_action_type = node_server::deliver_fld_action;
HPX_REGISTER_ACTION(deliver_fld_action_type);
using refresh_fld_action_type = node_server::refresh_fld_action;
HPX_REGISTER_ACTION(refresh_fld_action_type);

void node_server::receive_fld(integer child, radiation::fld_packet packet, size_t cycle) {
    fld_children.at(child).set_value(std::move(packet), cycle);
}

void node_server::deliver_fld(radiation::fld_packet packet, size_t cycle) {
    fld_parent.set_value(std::move(packet), cycle);
}

namespace {
namespace FLD = radiation::fld;
namespace hydro = radiation::fld::hydro;
namespace elliptic = octotiger::elliptic;

template<class Function> void fld_interior(Function&& function) {
    for (integer i = 0; i < INX; ++i) {
        for (integer j = 0; j < INX; ++j) {
            for (integer k = 0; k < INX; ++k) {
                function(hindex(i + H_BW, j + H_BW, k + H_BW),
                    rindex(i + RAD_BW, j + RAD_BW, k + RAD_BW));
            }
        }
    }
}

std::array<double, 2> extinction(const radiation::fld_runtime_cell& cell, double internal) {
    // Benchmark branches already express inverse code lengths. The generic
    // fitted branch alone requires evaluation with physical CGS density.
    switch (opts().problem) {
    case MARSHAK:
    case RADIATION_TEST:
    case RADIATION_DIFFUSION:
    case RADIATION_COUPLING:
    case RADIATION_COUPLED_WAVE:
    case RADIATION_SUBCRITICAL_SHOCK:
    case RADIATION_SUPERCRITICAL_SHOCK:
    case RADIATION_BLAST_WAVE:
        return {{kappa_R(cell.density, internal, cell.mmw, cell.hydrogen, cell.metals, grid::get_fgamma()),
            kappa_p(cell.density, internal, cell.mmw, cell.hydrogen, cell.metals, grid::get_fgamma())}};
    case RADIATION_GAUSSIAN_PULSE:
    case RADIATION_EQUILIBRIUM_SPHERE:
        return {{opts().radTestChi, 0}}; // Fixed scattering medium.
    case RADIATION_STREAMING_WAVE:
    case RADIATION_STREAMING_FRONT:
        return {{opts().radTestChi, 0}};
    default:
        return radiation::fitted_extinction(cell.density,
            temperature(cell.density, internal, cell.mmw, grid::get_fgamma()),
            cell.hydrogen, cell.metals, opts().code_to_g, opts().code_to_cm);
    }
}

void solve_packet(radiation::fld_packet& packet, double dt, double time) {
    std::vector<elliptic::cell> geometry;
    std::vector<FLD::cell> state;
    std::vector<hydro::state> gas_state;
    auto material_cells = packet.cells;
    geometry.reserve(packet.cells.size());
    state.reserve(packet.cells.size());
    const double c = physcon().c;
    const double log_ar = std::log(4 * physcon().sigma / c);
    const double temperature_factor = (grid::get_fgamma() - 1) * physcon().mh / physcon().kb;
    for (const auto& cell : packet.cells) {
        geometry.push_back({cell.center, cell.width});
        const auto chi = extinction(cell, cell.internal_energy);
        const double log_alpha = opts().problem == MARSHAK ? 0
            : log_ar + 4 * std::log(temperature_factor * cell.mmw / cell.density);
        state.push_back({cell.energy, cell.internal_energy, chi[0], chi[1], log_alpha,
            opts().problem == MARSHAK});
        hydro::state gas{cell.density, cell.momentum, cell.internal_energy, cell.energy};
        gas.gas_energy += hydro::kinetic_energy(gas);
        gas_state.push_back(gas);
        state.back().velocity = hydro::velocity(gas);
    }
    const double scale = grid::get_scaling_factor();
    const elliptic::composite_mesh mesh(std::move(geometry), {{-scale, -scale, -scale}},
        {{scale, scale, scale}}, {{opts().periodic, opts().periodic, opts().periodic}});
    std::vector<FLD::boundary_condition> boundary(6);
    if (opts().fld_boundary == "vacuum") {
        std::fill(boundary.begin(), boundary.end(), FLD::boundary_condition{c / 2, 1, 0});
    }
    if (opts().problem == MARSHAK) {
        // Incoming half-space flux=1: c E_b/2 + D dE/dn = 2.
        boundary.assign(6, FLD::boundary_condition{});
        boundary[0] = {c / 2, 1, 2};
        boundary[1] = {c / 2, 1, 0};
    }
    hydro::boundary_callback gas_boundary;
    if (radiationShockProblem()) {
        // Same physical wall/reservoir as the M1 benchmark. Transverse faces
        // have zero normal diffusion gradient and extrapolated gas states.
        const auto values = radiationHydroInit(opts().xscale, 0, 0, 0);
        const hydro::state upstream{values[rho_i], {{values[sx_i], values[sy_i], values[sz_i]}},
            values[egas_i], values[opts().n_fields]};
        auto reservoir = material_cells.front();
        reservoir.density = upstream.density;
        const double chi = extinction(reservoir, hydro::internal_energy(upstream))[0];
        boundary.assign(6, FLD::boundary_condition{});
        boundary[1] = {1, 0, upstream.radiation_energy};
        gas_boundary = [upstream, chi](const elliptic::face& face, const hydro::state& gas,
            const hydro::closure& pressure) {
            if (face.axis == 0 && face.boundary_side < 0) {
                return hydro::reflecting_boundary(face, gas, pressure);
            }
            if (face.axis == 0 && face.boundary_side > 0) {
                return hydro::boundary_state{upstream, chi, {}};
            }
            return hydro::outflow_boundary(gas, pressure);
        };
    } else if (opts().reflect_bc) {
        gas_boundary = hydro::reflecting_boundary;
    }
    std::vector<double> prescribed_source;
    if (opts().problem == RADIATION_EQUILIBRIUM_SPHERE) {
        const auto parameters = radiationTestParameters();
        boundary = FLD::sphere_boundary(mesh, parameters);
        prescribed_source.reserve(state.size());
        for (const auto& cell : mesh.cells()) {
            prescribed_source.push_back(radiationTests::bulbSourceAverage(
                cell.center, cell.width, parameters));
        }
    }
    const FLD::settings settings{opts().fld_nonlinear_tolerance, opts().fld_linear_tolerance,
        static_cast<int>(opts().fld_max_iterations), static_cast<int>(opts().fld_max_linear_iterations)};
    const FLD::opacity_callback opacity = [&](size_t i, double u, double& rosseland, double& planck) {
        // Incoming cells were checked above. An inadmissible trial thermal
        // state rejects this nonlinear trial, so damping/retry can recover.
        try {
            const auto chi = extinction(material_cells[i], u);
            rosseland = chi[0];
            planck = chi[1];
            return true;
        } catch (const std::invalid_argument&) {
            return false;
        } catch (const std::runtime_error&) {
            return false;
        }
    };
    FLD::result next;
    double elapsed = 0;
    double trial_dt = dt;
    int substeps = 0, retries = 0, nonlinear_iterations = 0, linear_iterations = 0, vcycles = 0;
    double boundary_energy = 0, conservation_error = 0, nonlinear_residual = 0, linear_residual = 0;
    std::vector<double> source(state.size(), 0), incoming(state.size(), 0);
    std::vector<std::array<double, 5>> gas_incoming(state.size());
    if (dt == 0) {
        next = FLD::refresh(mesh, state, c, boundary, opacity);
        if (!next.success) {
            throw std::runtime_error("FLD diagnostic reconstruction: " + next.message);
        }
    } else {
        // Failed attempts modify only scratch. All accepted substeps must finish
        // before any leaf installs values or commits an interval budget.
        while (elapsed < dt) {
            trial_dt = std::min(trial_dt, dt - elapsed);
            if (!(trial_dt > 0) || time + elapsed + trial_dt == time + elapsed) {
                throw std::runtime_error("FLD timestep underflow during collective retry");
            }
            auto implicit_input = state;
            hydro::result advected;
            if (opts().hydro) {
                std::vector<double> chi(state.size());
                for (size_t i = 0; i < state.size(); ++i) {
                    chi[i] = state[i].chi_rosseland;
                }
                advected = hydro::hyperbolic_step(mesh, gas_state, chi, trial_dt,
                    c, grid::get_fgamma(), gas_boundary, opts().cfl);
                if (!advected.success) {
                    if (++retries > 32) {
                        throw std::runtime_error("FLD hydro retry limit: " + advected.message);
                    }
                    trial_dt = std::min(.5 * trial_dt, advected.stable_dt);
                    continue;
                }
                for (size_t i = 0; i < state.size(); ++i) {
                    const auto& gas = advected.state_values[i];
                    material_cells[i].density = gas.density;
                    auto& thermal = implicit_input[i];
                    thermal.energy = gas.radiation_energy;
                    thermal.internal_energy = hydro::internal_energy(gas);
                    thermal.velocity = hydro::velocity(gas);
                    thermal.log_alpha = log_ar + 4 * std::log(
                        temperature_factor * material_cells[i].mmw / gas.density);
                    const auto chi = extinction(material_cells[i], thermal.internal_energy);
                    thermal.chi_rosseland = chi[0];
                    thermal.chi_planck = chi[1];
                }
            }
            next = prescribed_source.empty()
                ? FLD::solve(mesh, implicit_input, trial_dt, c, boundary, settings, opacity)
                : FLD::solve_with_source(mesh, implicit_input, prescribed_source, trial_dt, c,
                    boundary, settings, opacity);
            if (!next.success) {
                if (++retries > 16) {
                    throw std::runtime_error("FLD failed after 16 timestep halvings: " + next.message);
                }
                trial_dt *= 0.5;
                continue;
            }
            state = next.state;
            if (opts().hydro) {
                gas_state = std::move(advected.state_values);
                for (size_t i = 0; i < state.size(); ++i) {
                    gas_state[i].radiation_energy = state[i].energy;
                    gas_state[i].gas_energy = state[i].internal_energy + hydro::kinetic_energy(gas_state[i]);
                    gas_incoming[i][0] += advected.mass_boundary_by_cell[i];
                    for (int d = 0; d < 3; ++d) {
                        gas_incoming[i][d + 1] += advected.momentum_boundary_by_cell[i][d];
                    }
                    gas_incoming[i][4] += advected.gas_boundary_by_cell[i];
                    incoming[i] += advected.radiation_boundary_by_cell[i];
                    source[i] += advected.radiation_work_by_cell[i];
                    boundary_energy += advected.radiation_boundary_by_cell[i];
                }
                conservation_error += advected.conservation_error;
            }
            elapsed += trial_dt;
            if (++substeps > 65536) {
                throw std::runtime_error("Excessive FLD implicit substep count");
            }
            nonlinear_iterations += next.iterations;
            linear_iterations += next.linear_iterations;
            vcycles += next.vcycles;
            nonlinear_residual = std::max(nonlinear_residual, next.nonlinear_residual);
            linear_residual = std::max(linear_residual, next.linear_residual);
            boundary_energy += next.boundary_energy;
            conservation_error += next.conservation_error;
            for (size_t i = 0; i < state.size(); ++i) {
                source[i] += next.source_energy_by_cell[i];
                incoming[i] += next.boundary_energy_by_cell[i];
            }
        }
    }
    std::vector<double> final_energy(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        final_energy[i] = next.state[i].energy;
    }
    for (size_t i = 0; i < packet.cells.size(); ++i) {
        packet.cells[i].energy = next.state[i].energy;
        packet.cells[i].internal_energy = next.state[i].internal_energy;
        packet.cells[i].flux = next.flux[i];
        if (opts().hydro) {
            hydro::closure pressure;
            if (!hydro::make_closure(final_energy[i], next.state[i].chi_rosseland,
                    mesh.cell_gradient(i, final_energy), c, pressure)) {
                throw std::runtime_error("FLD lab flux reconstruction failed");
            }
            const auto H = pressure.enthalpy_flux(hydro::velocity(gas_state[i]));
            for (int d = 0; d < 3; ++d) {
                packet.cells[i].flux[d] += H[d];
            }
            packet.cells[i].density = gas_state[i].density;
            packet.cells[i].momentum = gas_state[i].momentum;
            packet.cells[i].gas_boundary = gas_incoming[i];
        }
        packet.cells[i].source_energy = source[i];
        packet.cells[i].boundary_energy = incoming[i];
    }
    if (dt > 0) {
        std::printf("FLD accepted t=%.17g dt=%.17g substeps=%d retries=%d nonlinear_iterations=%d "
            "linear_iterations=%d vcycles=%d nonlinear_residual=%.17g linear_residual=%.17g "
            "boundary_energy=%.17g conservation_error=%.17g\n", time, dt, substeps, retries,
            nonlinear_iterations, linear_iterations, vcycles, nonlinear_residual, linear_residual,
            boundary_energy, conservation_error);
    }
}
} // namespace

void node_server::compute_fld(real dt) {
    // Synchronized tree topology stays fixed throughout this collective. Tags
    // separate completed-step solves from post-regrid/restart reconstruction;
    // receive buffers erase consumed messages, so repeated refresh is safe.
    const size_t cycle = 2 * static_cast<size_t>(step_num) + (dt == 0 ? 1 : 0);
    radiation::fld_packet packet;
    std::array<size_t, NCHILD> counts{};
    auto& gas = *grid_ptr;
    auto& rad = *rad_grid_ptr;
    rad.set_dx(gas.get_dx());
    rad.set_X(gas.get_X());
    if (is_refined) {
        for (integer child = 0; child < NCHILD; ++child) {
            auto part = GET(fld_children[child].get_future(cycle));
            if (!part.error.empty()) {
                packet.error = part.error;
            }
            counts[child] = part.cells.size();
            packet.cells.insert(packet.cells.end(), std::make_move_iterator(part.cells.begin()),
                std::make_move_iterator(part.cells.end()));
        }
    } else {
        try {
            packet.cells.reserve(INX * INX * INX);
            fld_interior([&](integer h, integer r) {
                radiation::fld_runtime_cell cell;
                for (int d = 0; d < NDIM; ++d) {
                    if (!opts().hydro && gas.U[sx_i + d][h] != 0) {
                        throw std::runtime_error("FLD stationary material requires zero cell momentum");
                    }
                    cell.center[d] = gas.get_X()[d][h];
                    cell.momentum[d] = gas.U[sx_i + d][h];
                }
                cell.width = gas.get_dx();
                cell.energy = rad.U[rad_grid::er_i][r];
                cell.internal_energy = gas.U[egas_i][h];
                cell.density = gas.U[rho_i][h];
                if (!(cell.density > 0) || !std::isfinite(cell.density)) {
                    throw std::runtime_error("FLD requires positive finite density");
                }
                for (int d = 0; d < NDIM; ++d) {
                    cell.internal_energy -= .5 * cell.momentum[d] * (cell.momentum[d] / cell.density);
                }
                // Only physical leaves enter the solve. Restart ghost cells
                // have not necessarily received material halos yet.
                specie_state_t<> species;
                double species_density = 0;
                for (integer sp = 0; sp < opts().n_species; ++sp) {
                    species[sp] = gas.U[spc_i + sp][h];
                    if (!(species[sp] >= 0) || !std::isfinite(species[sp])) {
                        throw std::runtime_error("FLD requires nonnegative finite species densities");
                    }
                    species_density += species[sp];
                }
                if (!(species_density > 0) || !std::isfinite(species_density)) {
                    throw std::runtime_error("FLD requires positive finite species density");
                }
                mean_ion_weight(species, cell.mmw, cell.hydrogen, cell.metals);
                if (!(cell.density > 0) || !std::isfinite(cell.density) || !(cell.mmw > 0)
                    || !std::isfinite(cell.mmw)) {
                    throw std::runtime_error("FLD requires positive finite density and molecular weight");
                }
                packet.cells.push_back(cell);
            });
        } catch (const std::exception& error) {
            packet.error = error.what();
        }
    }
    if (my_location.level() == 0) {
        if (packet.error.empty()) {
            try {
                solve_packet(packet, dt, current_time);
            } catch (const std::exception& error) {
                packet.error = error.what();
            }
        }
    } else {
        hpx::apply<receive_fld_action>(parent.get_unmanaged_gid(), my_location.get_child_index(),
            std::move(packet), cycle);
        packet = GET(fld_parent.get_future(cycle));
    }
    if (is_refined) {
        std::vector<hpx::future<void>> deliveries;
        size_t offset = 0;
        for (integer child = 0; child < NCHILD; ++child) {
            radiation::fld_packet part;
            part.error = packet.error;
            if (part.error.empty()) {
                part.cells.assign(packet.cells.begin() + offset, packet.cells.begin() + offset + counts[child]);
            }
            offset += counts[child];
            deliveries.push_back(hpx::async<deliver_fld_action>(
                children[child].get_unmanaged_gid(), std::move(part), cycle));
        }
        for (auto& delivery : deliveries) {
            GET(delivery);
        }
    }
    if (!packet.error.empty()) {
        // Propagate the same failure to every participant before throwing.
        throw std::runtime_error("Composite FLD rejected: " + packet.error);
    }
    if (!is_refined) {
        size_t i = 0;
        auto outflows = gas.get_outflows_raw();
        fld_interior([&](integer h, integer r) {
            const auto& cell = packet.cells.at(i++);
            rad.U[rad_grid::er_i][r] = cell.energy;
            for (int d = 0; d < NDIM; ++d) {
                rad.U[rad_grid::fx_i + d][r] = cell.flux[d];
            }
            // Regridding averages conserved thermal energy; reconstruct its
            // entropy tracer as well as the diagnostic radiation flux.
            gas.U[tau_i][h] = std::pow(cell.internal_energy, 1 / grid::get_fgamma());
            if (dt > 0) {
                gas.U[egas_i][h] = cell.internal_energy;
                if (opts().hydro) {
                    gas.U[rho_i][h] = gas.U[spc_i][h] = cell.density;
                    for (int d = 0; d < NDIM; ++d) {
                        gas.U[sx_i + d][h] = cell.momentum[d];
                        gas.U[egas_i][h] += .5 * cell.momentum[d] * (cell.momentum[d] / cell.density);
                        outflows[sx_i + d] -= cell.gas_boundary[d + 1];
                    }
                    gas.U[lx_i][h] = cell.center[1] * cell.momentum[2] - cell.center[2] * cell.momentum[1];
                    gas.U[ly_i][h] = cell.center[2] * cell.momentum[0] - cell.center[0] * cell.momentum[2];
                    gas.U[lz_i][h] = cell.center[0] * cell.momentum[1] - cell.center[1] * cell.momentum[0];
                    outflows[rho_i] -= cell.gas_boundary[0];
                    outflows[spc_i] -= cell.gas_boundary[0];
                    outflows[egas_i] -= cell.gas_boundary[4];
                }
                rad.conservationBoundary[0] -= cell.boundary_energy;
                rad.conservationSource[0] += cell.source_energy;
            }
        });
        gas.set_outflows(std::move(outflows));
    }
    // Existing volume restriction is valid for completed physical states only.
    // Solver correction transfers belong solely to the scalar multigrid core.
    all_rad_bounds(current_time + dt);
}

void node_server::refresh_fld() {
    std::vector<hpx::future<void>> work;
    if (is_refined) {
        for (const auto& child : children) {
            work.push_back(hpx::async<refresh_fld_action>(child.get_unmanaged_gid()));
        }
    }
    std::exception_ptr failure;
    try {
        compute_fld(0);
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& child : work) {
        try {
            GET(child);
        } catch (...) {
            failure = std::current_exception();
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

void node_server::radiation_fld_step(integer remaining_steps) {
    timestep_t local;
    local.dt = std::numeric_limits<real>::max();
    if (!is_refined) {
        // hard_dt controls the requested interval. Moving FLD further limits
        // every internal coupled substep by its radiation-modified gas CFL.
        const auto h = hindex(H_BW, H_BW, H_BW);
        local.ur.resize(opts().n_fields);
        local.ul.resize(opts().n_fields);
        for (int f = 0; f < opts().n_fields; ++f) {
            local.ur[f] = local.ul[f] = grid_ptr->U[f][h];
        }
        local.x = grid_ptr->get_X()[0][h];
        local.y = grid_ptr->get_X()[1][h];
        local.z = grid_ptr->get_X()[2][h];
        if (opts().hard_dt > 0) {
            local.dt = opts().hard_dt;
        }
        if (opts().stop_time > 0) {
            local.dt = std::min(local.dt, (opts().stop_time - current_time)
                / remaining_steps);
        }
    }
    auto future_dt = global_timestep_channel.get_future();
    local_timestep_channels[NCHILD].set_value(local);
    dt_ = GET(future_dt);
    if (!(dt_.dt > 0) || !std::isfinite(dt_.dt)) {
        throw std::runtime_error("FLD requires hard_dt or a finite positive stop_time");
    }
    compute_fld(dt_.dt);
    all_hydro_bounds(current_time + dt_.dt);
    // Stationary FLD has already installed consistent gas energy and tau.
    // Hydro post_process is inappropriate here: its kernel gamma is initialized
    // by hydro flux evaluation, which this transport-only path never invokes.
    current_time += dt_.dt;
    rotational_time = current_time;
    if (remaining_steps == 1 && opts().stop_time > 0 &&
        std::abs(current_time - opts().stop_time) <= 8 * std::numeric_limits<real>::epsilon()
            * std::max(std::abs(current_time), std::abs(opts().stop_time))) {
        current_time = opts().stop_time;
        rotational_time = current_time;
    }
}
#endif
