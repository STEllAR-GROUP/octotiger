// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>

#include "octotiger/fld_configuration.hpp"
#include "octotiger/fld_grid.hpp"
#include "octotiger/fld_hydro.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"

#if !defined(HPX_COMPUTE_DEVICE_CODE)
namespace {
using octotiger::multigrid::Field;
using octotiger::multigrid::Block;
using octotiger::fld::StageResult;

constexpr int neighborCount = NDIR - 1;
constexpr int parentDestination = neighborCount;
constexpr int faceChildCount = NCHILD / 2;
constexpr int dwdVacuumSpecies = 4;
constexpr double atmosphereTracerThreshold = 0.25;
constexpr double relaxationWeight = 0.8;
constexpr int leafPolishingSweeps = 8;
constexpr int rootSmoothingSweeps = 32;
constexpr int descendingSmoothingSweeps = 3;
constexpr int ascendingSmoothingSweeps = 4;
constexpr int maximumLineSearchTrials = 24;
constexpr double armijoSlope = 1.0e-4;

enum DataOperation {
    restrict_values, child_values, neighbor_values, child_correction,
    restrict_diffusion, restrict_wall_conductance
};

enum Stage {
    initialize_fld, restrict_field, halo, cell_coefficients, face_coefficients,
    linearize, restrict_coefficients, fluxes, reconcile_fluxes, residual,
    smooth, restrict_fas, coarse_rhs, prolong_correction, nonlinear_residual,
    positivity, trial_update, validate_commit, commit, fld_diagnostics, leaf_smooth
};
enum Which {
    energy, temperature, density, solution, reaction, defect, rhs,
    opacity, extinction, saved, energy_star, thermal_star, capacity, electron_fraction, ion_charge_squared
};

Field& field(fld_grid& grid, int which) {
    switch (which) {
    case energy: return grid.energy;
    case temperature: return grid.temperature;
    case density: return grid.density;
    case solution: return grid.linear.solution;
    case reaction: return grid.linear.reaction;
    case defect: return grid.linear.residual;
    case rhs: return grid.linear.rhs;
    case opacity: return grid.opacity;
    case extinction: return grid.extinction;
    case saved: return grid.linear.saved;
    case energy_star: return grid.energyStar;
    case thermal_star: return grid.thermalStar;
    case capacity: return grid.capacity;
    case electron_fraction: return grid.electronDensity;
    case ion_charge_squared: return grid.ionChargeSquaredDensity;
    default: throw std::logic_error("Invalid FLD field identifier");
    }
}

std::array<int, NDIM> lower(geo::direction const& direction, bool ghost) {
    std::array<int, NDIM> result{};
    for (int d = 0; d < NDIM; ++d) {
        result[d] = direction[d] < 0 ? (ghost ? 0 : 1) :
            direction[d] > 0 ? (ghost ? INX + 1 : INX) : 1;
    }
    return result;
}
std::array<int, NDIM> upper(geo::direction const& direction, bool ghost) {
    auto result = lower(direction, ghost);
    for (int d = 0; d < NDIM; ++d) {
        result[d] = direction[d] == 0 ? INX + 1 : result[d] + 1;
    }
    return result;
}
double nonlinear_norm(StageResult const& result) {
    auto const& norms = result.nonlinear;
    double const l2 = std::sqrt((norms.radiationSquared + norms.materialSquared) /
        std::max(norms.scaleSquared, std::numeric_limits<double>::min()));
    return std::max(l2, norms.maximum);
}
} // namespace

using fld_stage_action_type = node_server::fld_stage_action;
using fld_data_action_type = node_server::fld_data_action;
using fld_sample_action_type = node_server::fld_sample_action;
using fld_face_action_type = node_server::fld_face_action;
HPX_REGISTER_ACTION(fld_stage_action_type);
HPX_REGISTER_ACTION(fld_data_action_type);
HPX_REGISTER_ACTION(fld_sample_action_type);
HPX_REGISTER_ACTION(fld_face_action_type);

future<octotiger::fld::StageResult> node_client::fld_stage(
    int stage, int level, std::vector<double> parameters) const {
    return hpx::async<node_server::fld_stage_action>(
        get_unmanaged_gid(), stage, level, std::move(parameters));
}
future<std::vector<double>> node_client::fld_data(int which, int operation, int index) const {
    return hpx::async<node_server::fld_data_action>(get_unmanaged_gid(), which, operation, index);
}
future<std::vector<double>> node_client::fld_sample(int which, std::vector<double> positions) const {
    return hpx::async<node_server::fld_sample_action>(get_unmanaged_gid(), which, std::move(positions));
}
future<std::vector<double>> node_client::fld_face(int axis, int side, bool children) const {
    return hpx::async<node_server::fld_face_action>(get_unmanaged_gid(), axis, side, children);
}

void node_server::bind_fld_grid() {
    if (opts().fld) {
        fld_grid_ptr = std::make_shared<fld_grid>(grid_ptr->get_fld_energy(), grid_ptr->get_dx());
    } else {
        fld_grid_ptr.reset();
    }
}

std::vector<double> node_server::fld_data(int which, int operation, int index) {
    assert(fld_grid_ptr);
    auto const& block = fld_grid_ptr->linear;
    if (operation == child_correction) {
        return block.childCorrection(index);
    }
    if (operation == restrict_diffusion) {
        return block.restrictFaceCoefficients(which);
    }
    if (operation == restrict_wall_conductance) {
        return block.restrictWallConductance(which);
    }
    auto const& values = field(*fld_grid_ptr, which);
    if (operation == restrict_values) {
        if (which == temperature) {
            // Restrict thermal energy and capacity, not temperature itself.
            Field thermal = values;
            for (size_t cell = 0; cell < thermal.values.size(); ++cell) {
                thermal.values[cell] *= fld_grid_ptr->capacity.values[cell];
            }
            auto result = block.restrictField(thermal);
            auto const capacities = block.restrictField(fld_grid_ptr->capacity);
            for (size_t cell = 0; cell < result.size(); ++cell) {
                result[cell] /= capacities[cell];
            }
            return result;
        }
        return block.restrictField(values);
    }
    if (operation == child_values) {
        return block.childValues(values, index);
    }
    assert(operation == neighbor_values);
    auto const direction = geo::direction::full_set()[index];
    auto const lb = lower(direction, false);
    auto const ub = upper(direction, false);
    std::vector<double> result;
    for (int i = lb[0]; i < ub[0]; ++i) {
        for (int j = lb[1]; j < ub[1]; ++j) {
            for (int k = lb[2]; k < ub[2]; ++k) {
                result.push_back(values(i, j, k));
            }
        }
    }
    return result;
}

std::vector<double> node_server::fld_sample(int which, std::vector<double> positions) {
    assert(fld_grid_ptr && positions.size() % NDIM == 0);
    std::vector<double> result(positions.size() / NDIM);
    std::array<std::vector<double>, NDIR> pending;
    std::array<std::vector<size_t>, NDIR> indices;
    auto const& values = field(*fld_grid_ptr, which);
    double const extent = grid::get_scaling_factor();
    for (size_t p = 0; p < result.size(); ++p) {
        std::array<double, NDIM> x{}, logical{};
        std::array<int, NDIM> offset{};
        bool local = true;
        for (int d = 0; d < NDIM; ++d) {
            // Clamp thermodynamic/closure ghosts at the domain, including
            // corners. The physical radiation flux is imposed separately.
            x[d] = std::clamp(positions[NDIM * p + d], -extent + 0.25 * dx, extent - 0.25 * dx);
            logical[d] = (x[d] - xmin[d]) / dx + 0.5;
            offset[d] = logical[d] < 0.5 ? -1 : logical[d] > INX + 0.5 ? 1 : 0;
            local = local && offset[d] == 0;
        }
        if (local) {
            for (double& coordinate : logical) {
                coordinate = std::clamp(coordinate, 1.0, double(INX));
            }
            result[p] = Block::sampleNearestCell(values, logical);
        } else {
            geo::direction direction;
            direction.set(offset[0], offset[1], offset[2]);
            int const destination = neighbors[direction].empty() ? parentDestination : int(direction);
            assert(destination != parentDestination || !parent.empty());
            indices[destination].push_back(p);
            pending[destination].insert(pending[destination].end(), x.begin(), x.end());
        }
    }
    std::array<future<std::vector<double>>, NDIR> requests;
    for (int d = 0; d < NDIR; ++d) {
        if (!pending[d].empty()) {
            auto const& destination = d == parentDestination ? parent : neighbors[d];
            requests[d] = destination.fld_sample(which, std::move(pending[d]));
        }
    }
    for (int d = 0; d < NDIR; ++d) {
        if (!indices[d].empty()) {
            auto const incoming = requests[d].get();
            for (size_t p = 0; p < incoming.size(); ++p) {
                result[indices[d][p]] = incoming[p];
            }
        }
    }
    return result;
}

std::vector<double> node_server::fld_face(int axis, int side, bool from_children) {
    assert(fld_grid_ptr);
    if (!from_children) {
        return fld_grid_ptr->linear.coarseFaceFluxes(axis, side);
    }
    assert(is_refined);
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    std::array<future<std::vector<double>>, faceChildCount> requests;
    for (int q = 0; q < faceChildCount; ++q) {
        int const child = (side << axis) | ((q / 2) << a) | ((q % 2) << b);
        requests[q] = children[child].fld_face(axis, side, false);
    }
    std::vector<double> result;
    for (auto& request : requests) {
        auto const packet = request.get();
        result.insert(result.end(), packet.begin(), packet.end());
    }
    return result;
}

octotiger::fld::StageResult node_server::fld_stage(int stage, int target_level, std::vector<double> parameters) {
    StageResult result;
    int const level = int(my_location.level());
    std::vector<future<StageResult>> descendants;
    if (is_refined && (target_level < 0 || level < target_level)) {
        for (auto const& child : children) {
            descendants.push_back(child.fld_stage(stage, target_level, parameters));
        }
    }
    if (target_level < 0 || level == target_level) {
        assert(fld_grid_ptr);
        auto& radiation = *fld_grid_ptr;
        auto& block = radiation.linear;
        if (stage == initialize_fld) {
            octotiger::fld::Parameters p;
            static_cast<octotiger::fld::OpacityParameters &>(p) = octotiger::fld::fld_opacity_parameters();
            p.cv = opts().fld_cv;
            p.c = octotiger::fld::fld_effective_c();
            p.ar = octotiger::fld::fld_effective_ar();
            std::vector<real> rho(H_N3), thermal(H_N3), heatCapacity(H_N3), ye(H_N3), z2(H_N3);
            // Only interiors are physical here. Real halos are exchanged below;
            // seed the unused hydro halo without relying on stale RK ghost data.
            for (int i = 0; i < H_NX; ++i) {
                for (int j = 0; j < H_NX; ++j) {
                    for (int k = 0; k < H_NX; ++k) {
                        int const source = hindex(std::clamp(i, H_BW, H_BW + INX - 1),
                            std::clamp(j, H_BW, H_BW + INX - 1),
                            std::clamp(k, H_BW, H_BW + INX - 1));
                        int const dest = hindex(i, j, k);
                        double const density_value = grid_ptr->get_field(rho_i)[source];
                        auto const material = octotiger::fld::fld_material_state_from_fields(
                            [&](size_t const field, size_t const index) {
                                return double(grid_ptr->get_field(field)[index]);
                            }, source);
                        if (material.usedTau && !is_refined && dest == source) {
                            result.dualEnergyCells += 1.0;
                        }
                        rho[dest] = density_value;
                        thermal[dest] = material.thermalEnergy;
                        heatCapacity[dest] = material.capacity;
                        ye[dest] = material.electronFraction;
                        z2[dest] = material.ionChargeSquared;
                    }
                }
            }
            radiation.initialize(parameters.at(0), p, rho, thermal, heatCapacity, ye, z2);
            block.refined = is_refined;
            result.deepestLevel = level;
            result.leaves = is_refined ? 0.0 : 1.0;
        } else if (stage == restrict_field || stage == restrict_coefficients || stage == restrict_fas) {
            if (is_refined) {
                std::vector<int> which_fields;
                if (stage == restrict_field) {
                    which_fields = {int(parameters.at(0))};
                } else if (stage == restrict_coefficients) {
                    which_fields = {reaction};
                } else {
                    which_fields = {solution, defect};
                }
                for (int which : which_fields) {
                    std::array<future<std::vector<double>>, NCHILD> requests;
                    for (int c = 0; c < NCHILD; ++c) {
                        requests[c] = children[c].fld_data(which, restrict_values, 0);
                    }
                    for (int c = 0; c < NCHILD; ++c) {
                        block.setChild(field(radiation, which), c, requests[c].get());
                    }
                }
                if (stage == restrict_coefficients) {
                    for (int axis = 0; axis < NDIM; ++axis) {
                        std::array<future<std::vector<double>>, NCHILD> requests;
                        for (int c = 0; c < NCHILD; ++c) {
                            requests[c] = children[c].fld_data(axis, restrict_diffusion, 0);
                        }
                        for (int c = 0; c < NCHILD; ++c) {
                            block.setChildFaceCoefficients(axis, c, requests[c].get());
                        }
                        for (int c = 0; c < NCHILD; ++c) {
                            requests[c] = children[c].fld_data(axis, restrict_wall_conductance, 0);
                        }
                        for (int c = 0; c < NCHILD; ++c) {
                            block.setChildWallConductance(axis, c, requests[c].get());
                        }
                    }
                }
                if (stage == restrict_fas) {
                    block.saveSolution();
                }
            }
        } else if (stage == halo) {
            int const which = int(parameters.at(0));
            auto& values = field(radiation, which);
            std::array<future<std::vector<double>>, neighborCount> requests;
            for (auto const& direction : geo::direction::full_set()) {
                if (!neighbors[direction].empty()) {
                    requests[direction] = neighbors[direction].fld_data(which, neighbor_values, int(direction.flip()));
                }
            }
            std::vector<std::array<int, NDIM>> coarse_cells;
            std::vector<double> positions;
            for (auto const& direction : geo::direction::full_set()) {
                auto const lb = lower(direction, true);
                auto const ub = upper(direction, true);
                std::vector<double> incoming;
                if (!neighbors[direction].empty()) {
                    incoming = requests[direction].get();
                }
                size_t p = 0;
                for (int i = lb[0]; i < ub[0]; ++i) {
                    for (int j = lb[1]; j < ub[1]; ++j) {
                        for (int k = lb[2]; k < ub[2]; ++k) {
                            if (!incoming.empty()) {
                                values(i, j, k) = incoming.at(p++);
                                continue;
                            }
                            std::array<int, NDIM> const index{{i, j, k}};
                            std::array<double, NDIM> x{};
                            for (int d = 0; d < NDIM; ++d) {
                                x[d] = xmin[d] + (index[d] - 0.5) * dx;
                            }
                            // Route corners as well as faces by physical position.
                            // A ghost can cross a physical wall in one direction
                            // and a sibling boundary in another; clamping all
                            // coordinates to this block would give inconsistent
                            // transverse gradients on the shared sibling face.
                            coarse_cells.push_back(index);
                            positions.insert(positions.end(), x.begin(), x.end());
                        }
                    }
                }
            }
            if (!positions.empty()) {
                auto const sampled = fld_sample(which, std::move(positions));
                for (size_t p = 0; p < sampled.size(); ++p) {
                    auto const& i = coarse_cells[p];
                    values(i[0], i[1], i[2]) = sampled[p];
                }
            }
        } else if (stage == cell_coefficients) {
            radiation.refreshCellCoefficients();
        } else if (stage == face_coefficients) {
            radiation.refreshFaceCoefficients();
            block.solution = radiation.energy;
            std::array<bool, NFACE> physicalBoundaries{};
            for (auto const& face : geo::face::full_set()) {
                physicalBoundaries[2 * int(face.get_dimension()) + int(face.get_side())] =
                    my_location.is_physical_boundary(face);
            }
            for (auto const& face : geo::face::full_set()) {
                int const axis = int(face.get_dimension());
                int const side = int(face.get_side());
                if (my_location.is_physical_boundary(face)) {
                    radiation.refreshPhysicalBoundaryFaceCoefficients(
                        axis, side, false, physicalBoundaries);
                } else if (neighbors[face.to_direction()].empty()) {
                    radiation.refreshBoundaryFaceCoefficients(axis, side, 2.0);
                }
            }
        } else if (stage == linearize) {
            radiation.prepareLinearization();
        } else if (stage == fluxes) {
            block.computeFluxes();
        } else if (stage == reconcile_fluxes) {
            if (!is_refined) {
                for (auto const& face : geo::face::full_set()) {
                    if (nieces[face] != 1) {
                        continue;
                    }
                    int const axis = int(face.get_dimension());
                    int const side = int(face.get_side());
                    auto const packets = neighbors[face.to_direction()].fld_face(axis, side ^ 1, true).get();
                    size_t const count = INX * INX / faceChildCount;
                    for (int q = 0; q < faceChildCount; ++q) {
                        std::vector<double> patch(packets.begin() + q * count,
                            packets.begin() + (q + 1) * count);
                        block.setRestrictedFlux(axis, side ? INX + 1 : 1,
                            (q / 2) * (INX / 2), (q % 2) * (INX / 2), patch);
                    }
                }
            }
            block.updateFluxDefect();
        } else if (stage == residual) {
            block.computeResidual();
            if (!is_refined) {
                result.linearMaximum = radiation.computeLinearResidualMaximum();
                for (int i = 1; i <= INX; ++i) {
                    for (int j = 1; j <= INX; ++j) {
                        for (int k = 1; k <= INX; ++k) {
                            double const r = block.residual(i, j, k);
                            double const b = block.rhs(i, j, k);
                            if (!std::isfinite(r)) {
                                throw std::runtime_error("Nonfinite FLD linear residual");
                            }
                            result.linearResidualSquared += r * r * dx * dx * dx;
                            result.linearRhsSquared += b * b * dx * dx * dx;
                        }
                    }
                }
            }
        } else if (stage == smooth || stage == leaf_smooth) {
            if (stage == smooth || !is_refined) {
                block.relax(relaxationWeight);
            }
        } else if (stage == coarse_rhs) {
            if (is_refined) {
                block.formCoarseRhs(block.residual);
            }
        } else if (stage == prolong_correction) {
            if (!parent.empty()) {
                block.addCoarseCorrection(parent.fld_data(solution, child_correction,
                    int(my_location.get_child_index())).get());
            }
        } else if (stage == nonlinear_residual) {
            if (!is_refined) {
                result.nonlinear = radiation.computeNonlinearResidual();
            }
        } else if (stage == positivity) {
            if (!is_refined) {
                result.updateLimit = radiation.prepareUpdate();
            }
        } else if (stage == trial_update) {
            if (!is_refined) {
                radiation.applyUpdate(parameters.at(0));
            }
        } else if (stage == validate_commit || stage == commit) {
            if (!is_refined) {
                auto const delta = radiation.materialEnergyIncrement();
                for (int i = 1; i <= INX; ++i) {
                    for (int j = 1; j <= INX; ++j) {
                        for (int k = 1; k <= INX; ++k) {
                            int const h = hindex(i + H_BW - 1, j + H_BW - 1, k + H_BW - 1);
                            double const internal = radiation.thermalStar(i, j, k) + delta[h];
                            if (!(internal >= 0.0) || !std::isfinite(internal) ||
                                !std::isfinite(double(grid_ptr->get_field(egas_i)[h]) + delta[h])) {
                                throw std::runtime_error("FLD update would create negative gas thermal energy");
                            }
                            if (stage == commit) {
                                // Preserve the evolved inertial total energy even
                                // when tau supplied the material temperature.
                                grid_ptr->get_field(egas_i)[h] += delta[h];
                                grid_ptr->get_field(tau_i)[h] = octotiger::fld::thermal_tau(internal, grid::get_fgamma());
                            }
                        }
                    }
                }
                if (stage == commit) {
                    radiation.commit();
                }
            }
            if (stage == commit && !opts().hydro) {
                current_time += radiation.dt;
                rotational_time += grid::get_omega() == 0.0 ? radiation.dt : grid::get_omega() * radiation.dt;
                ++step_num;
                dt_.dt = radiation.dt;
            }
        } else if (stage == fld_diagnostics && !is_refined) {
            double const volume = dx * dx * dx;
            auto const delta = radiation.materialEnergyIncrement();
            for (int i = 1; i <= INX; ++i) {
                for (int j = 1; j <= INX; ++j) {
                    for (int k = 1; k <= INX; ++k) {
                        int const h = hindex(i + H_BW - 1, j + H_BW - 1, k + H_BW - 1);
                        double const gas_total = grid_ptr->get_field(egas_i)[h];
                        double const internal = radiation.thermalStar(i, j, k) + delta[h];
                        result.diagnostics.tauThermalEnergy += std::pow(double(grid_ptr->get_field(tau_i)[h]), grid::get_fgamma()) * volume;
                        result.diagnostics.gasTotalEnergy += gas_total * volume;
                        result.diagnostics.initialGasTotalEnergy += (gas_total - delta[h]) * volume;
                        double const e = (*radiation.sharedEnergy)[h];
                        if (opts().eos == WD) {
                            result.diagnostics.coldEnergy += double(ztwd_energy(radiation.density(i, j, k))) * volume;
                            // The vacuum origin tracer identifies the floor atmosphere.
                            if (opts().problem == DWD) {
                                double tracerTotal = 0.0;
                                for (int species = 0; species < opts().n_species; ++species) {
                                    tracerTotal += grid_ptr->get_field(spc_i + species)[h];
                                }
                                if (grid_ptr->get_field(spc_i + dwdVacuumSpecies)[h] >= atmosphereTracerThreshold * tracerTotal) {
                                    result.diagnostics.atmosphereThermalEnergy += internal * volume;
                                    result.diagnostics.atmosphereRadiationEnergy += e * volume;
                                }
                            }
                        }
                        result.diagnostics.radiationEnergy += e * volume;
                        result.diagnostics.thermalEnergy += internal * volume;
                        result.diagnostics.initialRadiationEnergy += radiation.energyStar(i, j, k) * volume;
                        result.diagnostics.initialThermalEnergy += radiation.thermalStar(i, j, k) * volume;
                        result.diagnostics.minimumEnergy = std::min(result.diagnostics.minimumEnergy, e);
                        result.diagnostics.maximumEnergy = std::max(result.diagnostics.maximumEnergy, e);
                        result.diagnostics.minimumTemperature = std::min(result.diagnostics.minimumTemperature, internal / radiation.capacity(i, j, k));
                    }
                }
            }
            result.diagnostics.opacityRegularizedCells = double(radiation.regularizedCells);
            result.diagnostics.unsharedBlocks = radiation.sharedEnergy == grid_ptr->get_fld_energy() ? 0.0 : 1.0;
        }
    }
    for (auto& child : descendants) {
        result.accumulate(child.get());
    }
    return result;
}

void node_server::compute_fld(double dt) {
    assert(my_location.level() == 0 && opts().fld);
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        throw std::invalid_argument("FLD requires a finite positive timestep");
    }
    bool const print_progress = opts().print_times_per_timestep;
    auto const started = print_progress ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    auto const log_progress = [&](char const* stage, auto const&... fields) {
        if (print_progress) {
            std::ostringstream line;
            line << std::setprecision(17) << "FLD_PROGRESS stage=" << stage;
            (line << ... << fields);
            line << " elapsed=" << std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            std::cerr << line.str() << '\n' << std::flush;
        }
    };
    log_progress("begin", " dt=", dt);
    auto run = [this](int stage, int level = -1, std::vector<double> p = {}) {
        return fld_stage(stage, level, std::move(p));
    };
    auto const topology = run(initialize_fld, -1, {dt});
    int const deepest = topology.deepestLevel;
    auto restrict = [&](int which) {
        for (int level = deepest - 1; level >= 0; --level) {
            run(restrict_field, level, {double(which)});
        }
    };
    auto exchange = [&](int which) {
        for (int level = 0; level <= deepest; ++level) {
            run(halo, level, {double(which)});
        }
    };
    auto update_flux = [&](int level) {
        run(fluxes, level);
        run(reconcile_fluxes, level);
    };
    auto update_linear_residual = [&]() {
        restrict(solution);
        exchange(solution);
        update_flux(-1);
        return run(residual);
    };
    auto refresh = [&]() {
        for (int which : {density, capacity, electron_fraction, ion_charge_squared, energy, temperature}) {
            restrict(which);
            exchange(which);
        }
        run(cell_coefficients);
        run(face_coefficients);
        update_flux(-1);
        return run(nonlinear_residual);
    };
    auto smooth_level = [&](int level, int sweeps, bool leaves_only = false) {
        if (leaves_only && level < deepest) {
            run(restrict_field, level, {double(solution)});
        }
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            run(halo, level, {double(solution)});
            update_flux(level);
            run(leaves_only ? leaf_smooth : smooth, level);
        }
        run(halo, level, {double(solution)});
        update_flux(level);
        run(residual, level);
    };
    auto norms = refresh();
    double norm = nonlinear_norm(norms);
    log_progress("first_refresh", " norm=", norm);
    if (!std::isfinite(norm)) {
        throw std::runtime_error("Nonfinite initial FLD nonlinear residual");
    }
    int iterations = 0;
    int total_cycles = 0;
    while (norm > opts().fld_tolerance && iterations < opts().fld_max_iterations) {
        if (!std::isfinite(norm)) {
            throw std::runtime_error("Nonfinite FLD nonlinear residual");
        }
        log_progress("outer_begin", " iteration=", iterations, " norm=", norm);
        run(linearize);
        for (int level = deepest - 1; level >= 0; --level) {
            run(restrict_coefficients, level);
        }
        auto linear_norms = update_linear_residual();
        double const initial = std::sqrt(linear_norms.linearResidualSquared);
        // Once the global residual is below the scaled absolute tolerance,
        // forcing another relative reduction can demand accuracy below
        // roundoff. The independent local guard still requires progress in
        // cold cells that the volume-weighted L2 norm can hide, and the full
        // nonlinear residual remains the final acceptance criterion.
        double const target = std::max(0.01 * initial,
            0.05 * opts().fld_tolerance * std::sqrt(std::max(norms.nonlinear.scaleSquared, std::numeric_limits<double>::min())));
        double const local_target = std::min(0.1 * linear_norms.linearMaximum,
            std::max(0.01 * linear_norms.linearMaximum, 0.05 * opts().fld_tolerance));
        if (!std::isfinite(initial) || !std::isfinite(target) || !std::isfinite(local_target)) {
            throw std::runtime_error("Nonfinite FLD linear solve tolerance");
        }
        int cycles = 0;
        auto const log_linear_progress = [&](char const* path, bool force = false) {
            if (print_progress && (force || cycles == 0 || (cycles & (cycles - 1)) == 0)) {
                log_progress("linear", " iteration=", iterations, " cycles=", cycles,
                    " path=", path, " l2=", std::sqrt(linear_norms.linearResidualSquared),
                    " target=", target, " local=", linear_norms.linearMaximum,
                    " local_target=", local_target);
            }
        };
        log_linear_progress("initial");
        while ((std::sqrt(linear_norms.linearResidualSquared) > target || linear_norms.linearMaximum > local_target) &&
            cycles < opts().fld_max_cycles) {
            if (std::sqrt(linear_norms.linearResidualSquared) <= target) {
                // Coarse corrections can repeatedly spread roundoff from a
                // heated cell into an almost empty thermal reservoir. Once
                // the global solve converges, finish the remaining local
                // error with relaxation and the same reconciled face fluxes.
                for (int level = deepest; level >= 0; --level) {
                    smooth_level(level, leafPolishingSweeps, true);
                }
                linear_norms = update_linear_residual();
                ++cycles;
                log_linear_progress("leaf_polish");
                continue;
            }
            for (int level = deepest; level >= 0; --level) {
                if (level < deepest) {
                    run(restrict_fas, level);
                    run(halo, level, {double(solution)});
                    update_flux(level);
                    run(coarse_rhs, level);
                }
                // The root block is the bottom grid. Extra smoothing solves its
                // reaction-diffusion equation without any root cell gathering.
                smooth_level(level, level == 0 ? rootSmoothingSweeps : descendingSmoothingSweeps);
            }
            for (int level = 1; level <= deepest; ++level) {
                run(prolong_correction, level);
                smooth_level(level, ascendingSmoothingSweeps);
            }
            linear_norms = update_linear_residual();
            ++cycles;
            log_linear_progress("multigrid");
        }
        total_cycles += cycles;
        if (std::sqrt(linear_norms.linearResidualSquared) > target || linear_norms.linearMaximum > local_target) {
            std::cerr << std::setprecision(17) << "FLD linear solve: iteration=" << iterations
                << " l2=" << std::sqrt(linear_norms.linearResidualSquared) << " target=" << target
                << " local=" << linear_norms.linearMaximum << " local_target=" << local_target
                << " cycles=" << cycles << '\n';
            throw std::runtime_error("FLD multigrid failed to converge; implicit state was not committed");
        }
        log_linear_progress("converged", true);
        double damping = run(positivity).updateLimit;
        bool accepted = false;
        double const previous = norm;
        double last_candidate = std::numeric_limits<double>::infinity();
        for (int trial = 0; trial < maximumLineSearchTrials && damping > 0.0; ++trial) {
            run(trial_update, -1, {damping});
            auto candidate = refresh();
            double const candidate_norm = nonlinear_norm(candidate);
            last_candidate = candidate_norm;
            if (std::isfinite(candidate_norm) &&
                (candidate_norm <= (1.0 - armijoSlope * damping) * previous || candidate_norm <= opts().fld_tolerance)) {
                norms = std::move(candidate);
                norm = candidate_norm;
                accepted = true;
                log_progress("outer_accepted", " iteration=", iterations,
                    " damping=", damping, " norm=", norm,
                    " trials=", trial + 1, " cycles=", cycles);
                break;
            }
            damping *= 0.5;
        }
        if (!accepted) {
            std::cerr << std::setprecision(17) << "FLD line search: iteration=" << iterations
                << " norm=" << previous << " last_candidate=" << last_candidate
                << " damping=" << damping << " linear_initial=" << initial
                << " linear_final=" << std::sqrt(linear_norms.linearResidualSquared)
                << " cycles=" << cycles << '\n';
            throw std::runtime_error("FLD nonlinear line search failed; implicit state was not committed");
        }
        ++iterations;
    }
    if (!std::isfinite(norm) || norm > opts().fld_tolerance) {
        std::cerr << "FLD nonlinear solve: iterations=" << iterations << " residual=" << norm << '\n';
        throw std::runtime_error("FLD nonlinear iteration limit reached; implicit state was not committed");
    }
    run(validate_commit);
    run(commit);
    auto const totals = run(fld_diagnostics).diagnostics;
    std::cout << std::setprecision(17) << "FLD_STEP dt=" << dt
        << " iterations=" << iterations << " cycles=" << total_cycles << " residual=" << norm
        << " radiation_energy=" << totals.radiationEnergy << " thermal_energy=" << totals.thermalEnergy
        << " initial_radiation_energy=" << totals.initialRadiationEnergy << " initial_thermal_energy=" << totals.initialThermalEnergy
        << " min_energy=" << totals.minimumEnergy << " max_energy=" << totals.maximumEnergy
        << " min_temperature=" << totals.minimumTemperature
        << " gas_total_energy=" << totals.gasTotalEnergy << " initial_gas_total_energy=" << totals.initialGasTotalEnergy
        << " tau_thermal_energy=" << totals.tauThermalEnergy << " dual_energy_cells=" << topology.dualEnergyCells
        << " opacity_regularized_cells=" << totals.opacityRegularizedCells
        << " cold_energy=" << totals.coldEnergy
        << " atmosphere_thermal_energy=" << totals.atmosphereThermalEnergy
        << " atmosphere_radiation_energy=" << totals.atmosphereRadiationEnergy
        << " shared=" << (totals.unsharedBlocks == 0.0 ? 1 : 0)
        << " leaves=" << topology.leaves << " cells=" << topology.leaves * INX * INX * INX << '\n';
}
#endif
