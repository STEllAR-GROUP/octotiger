// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/fld.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/radiation/physical_opacity.hpp"
#include "octotiger/util.hpp"

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
        return {{opts().radTestChi, 0}}; // Fixed scattering medium.
    case RADIATION_STREAMING_WAVE:
    case RADIATION_STREAMING_FRONT:
        return {{0, 0}};
    case RADIATION_EQUILIBRIUM_SPHERE:
        throw std::invalid_argument("The prescribed M1 sphere emission test is not an FLD reference");
    default:
        return radiation::fitted_extinction(cell.density,
            temperature(cell.density, internal, cell.mmw, grid::get_fgamma()),
            cell.hydrogen, cell.metals, opts().code_to_g, opts().code_to_cm);
    }
}

void solve_packet(radiation::fld_packet& packet, double dt, double time) {
    std::vector<elliptic::cell> geometry;
    std::vector<FLD::cell> state;
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
    const FLD::settings settings{opts().fld_nonlinear_tolerance, opts().fld_linear_tolerance,
        static_cast<int>(opts().fld_max_iterations), static_cast<int>(opts().fld_max_linear_iterations)};
    const FLD::opacity_callback opacity = [&](size_t i, double u, double& rosseland, double& planck) {
        // Incoming cells were checked above. An inadmissible trial thermal
        // state rejects this nonlinear trial, so damping/retry can recover.
        try {
            const auto chi = extinction(packet.cells[i], u);
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
            next = FLD::solve(mesh, state, trial_dt, c, boundary, settings, opacity);
            if (!next.success) {
                if (++retries > 16) {
                    throw std::runtime_error("FLD failed after 16 timestep halvings: " + next.message);
                }
                trial_dt *= 0.5;
                continue;
            }
            state = next.state;
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
    for (size_t i = 0; i < packet.cells.size(); ++i) {
        packet.cells[i].energy = next.state[i].energy;
        packet.cells[i].internal_energy = next.state[i].internal_energy;
        packet.cells[i].flux = next.flux[i];
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
                    if (gas.U[sx_i + d][h] != 0) {
                        throw std::runtime_error("FLD stationary material requires zero cell momentum");
                    }
                    cell.center[d] = gas.get_X()[d][h];
                }
                cell.width = gas.get_dx();
                cell.energy = rad.U[rad_grid::er_i][r];
                cell.internal_energy = gas.U[egas_i][h];
                cell.density = gas.U[rho_i][h];
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
                rad.conservationBoundary[0] -= cell.boundary_energy;
                rad.conservationSource[0] += cell.source_energy;
            }
        });
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
        // With matter held stationary there are no explicit propagation terms.
        // hard_dt (or the remaining stop-time batch) controls temporal accuracy.
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
