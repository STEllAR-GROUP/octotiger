// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>

#include "octotiger/multigrid/block.hpp"
#include "octotiger/multigrid/isolated_boundary.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"

#if !defined(HPX_COMPUTE_DEVICE_CODE)

namespace octotiger::multigrid {
struct GravityBoundary {
    bool multipole = false;
    std::array<double, 3> center{};
    IsolatedBoundaryMoments moments;
};
} // namespace octotiger::multigrid

namespace {
using octotiger::multigrid::Block;
using octotiger::multigrid::Field;

static_assert(INX >= 2 && INX % 2 == 0, "Multigrid requires a positive even grid size");

enum Stage {
    initialize_block, restrict_source, halo_phi, begin_down, restrict_correction,
    adjust_defect, save_rhs, halo_correction, relax, replace_mean,
    inject_correction, retire, restrict_phi, forces, restrict_forces,
    residual, errors, write_fields, release
};

Field& field(Block& block, int which) {
    assert(which >= 0 && which <= 2);
    if (which == 0) {
        return block.phi;
    }
    if (which == 1) {
        return block.correction;
    }
    return block.source;
}

std::array<double, 4> sphere(std::array<double, 3> position) {
    position[0] -= opts().solid_sphere_xcenter;
    position[1] -= opts().solid_sphere_ycenter;
    position[2] -= opts().solid_sphere_zcenter;
    double const r2 = position[0] * position[0] + position[1] * position[1] +
        position[2] * position[2];
    double const radius = opts().solid_sphere_radius;
    double const gm = physcon().G * opts().solid_sphere_mass;
    double const r = std::sqrt(r2);
    double const denominator = r < radius ? radius * radius * radius : r2 * r;
    std::array<double, 4> result{};
    result[0] = r < radius ? -gm * (3.0 * radius * radius - r2) /
        (2.0 * radius * radius * radius) : -gm / r;
    for (int d = 0; d < 3; ++d) {
        result[d + 1] = -gm * position[d] / denominator;
    }
    return result;
}

double physical_boundary(octotiger::multigrid::GravityBoundary const& boundary,
    std::array<double, 3> const& position) {
    return boundary.multipole ? boundary.moments.potential(position, boundary.center) :
        sphere(position)[0];
}

void accumulate(std::vector<double>& total, std::vector<double> const& part, int stage) {
    assert(part.size() == total.size());
    for (size_t i = 0; i < total.size(); ++i) {
        bool const maximum = (stage == initialize_block && i == 0) ||
            (stage == residual && i == 2) || (stage == errors && i == 2);
        total[i] = maximum ? std::max(total[i], part[i]) : total[i] + part[i];
    }
}

std::array<int, 3> lower_bound(geo::direction const& direction, int n, bool ghost) {
    std::array<int, 3> result{};
    for (int d = 0; d < 3; ++d) {
        result[d] = direction[d] < 0 ? (ghost ? 0 : 1) :
            direction[d] > 0 ? (ghost ? n + 1 : n) : 1;
    }
    return result;
}

std::array<int, 3> upper_bound(geo::direction const& direction, int n, bool ghost) {
    auto result = lower_bound(direction, n, ghost);
    for (int d = 0; d < 3; ++d) {
        result[d] = direction[d] == 0 ? n + 1 : result[d] + 1;
    }
    return result;
}
} // namespace

using multigrid_stage_action_type = node_server::multigrid_stage_action;
using multigrid_data_action_type = node_server::multigrid_data_action;
using multigrid_face_action_type = node_server::multigrid_face_action;
using multigrid_sample_action_type = node_server::multigrid_sample_action;
using multigrid_boundary_action_type = node_server::multigrid_boundary_action;
HPX_REGISTER_ACTION(multigrid_stage_action_type);
HPX_REGISTER_ACTION(multigrid_data_action_type);
HPX_REGISTER_ACTION(multigrid_face_action_type);
HPX_REGISTER_ACTION(multigrid_sample_action_type);
HPX_REGISTER_ACTION(multigrid_boundary_action_type);

future<std::vector<double>> node_client::multigrid_stage(int stage, int level) const {
    return hpx::async<node_server::multigrid_stage_action>(get_unmanaged_gid(), stage, level);
}

future<std::vector<double>> node_client::multigrid_data(int which, int operation, int index) const {
    return hpx::async<node_server::multigrid_data_action>(
        get_unmanaged_gid(), which, operation, index);
}

future<std::vector<double>> node_client::multigrid_face(int axis, int side, int kind) const {
    return hpx::async<node_server::multigrid_face_action>(get_unmanaged_gid(), axis, side, kind);
}

future<std::vector<double>> node_client::multigrid_sample(
    int which, std::vector<double> positions) const {
    return hpx::async<node_server::multigrid_sample_action>(
        get_unmanaged_gid(), which, std::move(positions));
}

future<std::vector<double>> node_client::multigrid_boundary(
    int stage, std::vector<double> parameters) const {
    return hpx::async<node_server::multigrid_boundary_action>(
        get_unmanaged_gid(), stage, std::move(parameters));
}

std::vector<double> node_server::multigrid_boundary(int stage, std::vector<double> parameters) {
    assert(multigrid_block_);
    assert(stage >= 0 && stage <= 2);
    assert(stage != 1 || parameters.size() == 3);
    assert(stage != 2 || parameters.size() == 54);
    std::vector<double> result(stage == 1 ? 51 : 4, 0.0);
    std::vector<future<std::vector<double>>> descendants;
    if (is_refined) {
        for (auto const& child : children) {
            descendants.push_back(child.multigrid_boundary(stage, parameters));
        }
    }
    if (stage == 2) {
        multigrid_boundary_ = std::make_shared<octotiger::multigrid::GravityBoundary>();
        multigrid_boundary_->multipole = parameters[0] != 0.0;
        for (int d = 0; d < 3; ++d) {
            multigrid_boundary_->center[d] = parameters[d + 1];
        }
        for (size_t m = 0; m < 25; ++m) {
            multigrid_boundary_->moments.real[m] = parameters[m + 4];
            multigrid_boundary_->moments.imaginary[m] = parameters[m + 29];
        }
    } else if (!is_refined) {
        octotiger::multigrid::SourceMoments source_moments;
        octotiger::multigrid::IsolatedBoundaryMoments boundary_moments;
        std::array<double, 3> center{};
        if (stage == 1) {
            std::copy_n(parameters.begin(), 3, center.begin());
        }
        double const volume = dx * dx * dx;
        for (int i = 1; i <= INX; ++i) {
            for (int j = 1; j <= INX; ++j) {
                for (int k = 1; k <= INX; ++k) {
                    std::array<double, 3> const x{{xmin[0] + (i - 0.5) * dx,
                        xmin[1] + (j - 0.5) * dx, xmin[2] + (k - 0.5) * dx}};
                    double const s = multigrid_block_->source(i, j, k);
                    if (stage == 0) {
                        source_moments.addSource(x, s, volume);
                    } else {
                        boundary_moments.addSource(x, s, volume, center);
                        if (s > 0.0) {
                            double radius2 = 0.0;
                            for (int d = 0; d < 3; ++d) {
                                radius2 += (x[d] - center[d]) * (x[d] - center[d]);
                            }
                            result[50] = std::max(result[50], radius2);
                        }
                    }
                }
            }
        }
        if (stage == 0) {
            result[0] = source_moments.mass;
            for (int d = 0; d < 3; ++d) {
                result[d + 1] = source_moments.firstMoment[d];
            }
        } else {
            std::copy(boundary_moments.real.begin(), boundary_moments.real.end(), result.begin());
            std::copy(boundary_moments.imaginary.begin(), boundary_moments.imaginary.end(), result.begin() + 25);
        }
    }
    for (auto& descendant : descendants) {
        auto const part = descendant.get();
        for (size_t p = 0; p < result.size(); ++p) {
            result[p] = stage == 1 && p == 50 ? std::max(result[p], part[p]) : result[p] + part[p];
        }
    }
    return result;
}

std::vector<double> node_server::multigrid_data(int which, int operation, int index) {
    assert(multigrid_block_);
    auto const& value = field(*multigrid_block_, which);
    if (operation == 1) {
        return multigrid_block_->restrictField(value);
    }
    if (operation == 2) {
        return multigrid_block_->childValues(value, index);
    }
    assert(operation == 3);
    assert(index >= 0 && index < geo::direction::count());
    // Read only interiors. Concurrent halo fills on this block never write them.
    auto const direction = geo::direction::full_set()[index];
    auto const lb = lower_bound(direction, INX, false);
    auto const ub = upper_bound(direction, INX, false);
    std::vector<double> result;
    result.reserve(size_t(ub[0] - lb[0]) * (ub[1] - lb[1]) * (ub[2] - lb[2]));
    for (int i = lb[0]; i < ub[0]; ++i) {
        for (int j = lb[1]; j < ub[1]; ++j) {
            for (int k = lb[2]; k < ub[2]; ++k) {
                result.push_back(value(i, j, k));
            }
        }
    }
    return result;
}

std::vector<double> node_server::multigrid_sample(int which, std::vector<double> positions) {
    assert(multigrid_block_);
    assert(multigrid_boundary_);
    assert(positions.size() % 3 == 0);
    std::vector<double> result(positions.size() / 3);
    std::array<std::vector<double>, 27> pending;
    std::array<std::vector<size_t>, 27> indices;
    auto const& value = field(*multigrid_block_, which);
    double const extent = grid::get_scaling_factor();
    for (size_t p = 0; p < result.size(); ++p) {
        std::array<double, 3> x{}, logical{};
        std::array<int, 3> offset{};
        bool exterior = false;
        bool local = true;
        for (int d = 0; d < 3; ++d) {
            x[d] = positions[3 * p + d];
            exterior = exterior || x[d] < -extent || x[d] > extent;
            logical[d] = (x[d] - xmin[d]) / dx + 0.5;
            offset[d] = logical[d] < 0.5 ? -1 : logical[d] > INX + 0.5 ? 1 : 0;
            local = local && offset[d] == 0;
        }
        if (exterior) {
            result[p] = which == 0 ? physical_boundary(*multigrid_boundary_, x) : 0.0;
        } else if (local) {
            // Callers sample a strictly coarser level after its halo barrier.
            result[p] = which == 0 ? Block::quadraticInterpolate(value, logical) :
                Block::sampleNearestCell(value, logical);
        } else {
            geo::direction direction;
            direction.set(offset[0], offset[1], offset[2]);
            int const destination = neighbors[direction].empty() ? 26 : int(direction);
            assert(destination != 26 || !parent.empty());
            indices[destination].push_back(p);
            pending[destination].insert(pending[destination].end(), x.begin(), x.end());
        }
    }
    std::array<future<std::vector<double>>, 27> requests;
    for (int d = 0; d < 27; ++d) {
        if (!pending[d].empty()) {
            auto const& destination = d == 26 ? parent : neighbors[d];
            requests[d] = destination.multigrid_sample(which, std::move(pending[d]));
        }
    }
    for (int d = 0; d < 27; ++d) {
        if (!indices[d].empty()) {
            auto values = requests[d].get();
            for (size_t p = 0; p < values.size(); ++p) {
                result[indices[d][p]] = values[p];
            }
        }
    }
    return result;
}

std::vector<double> node_server::multigrid_face(int axis, int side, int kind) {
    assert(multigrid_block_);
    assert(axis >= 0 && axis < 3);
    assert(side >= 0 && side <= 1);
    assert(kind >= 0 && kind <= 3);
    if (kind < 2) {
        return kind == 0 ? multigrid_block_->coarseFaceJumps(axis, side) :
            multigrid_block_->coarseFaceForces(axis, side);
    }
    assert(is_refined);
    int const a = (axis + 1) % 3;
    int const b = (axis + 2) % 3;
    std::array<future<std::vector<double>>, 4> requests;
    for (int q = 0; q < 4; ++q) {
        int const child = (side << axis) | ((q / 2) << a) | ((q % 2) << b);
        requests[q] = children[child].multigrid_face(axis, side, kind - 2);
    }
    std::vector<double> result;
    result.reserve(INX * INX);
    for (auto& request : requests) {
        auto values = request.get();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}

std::vector<double> node_server::multigrid_stage(int stage, int target_level) {
    std::vector<double> result(16, 0.0);
    int const level = int(my_location.level());
    std::vector<future<std::vector<double>>> descendants;
    if (is_refined && (target_level < 0 || level < target_level)) {
        for (auto const& child : children) {
            descendants.push_back(child.multigrid_stage(stage, target_level));
        }
    }
    if (target_level < 0 || level == target_level) {
        if (stage == initialize_block) {
            if (!(dx > 0.0) || !std::isfinite(dx)) {
                throw std::invalid_argument("Static gravity requires finite positive cell widths");
            }
            multigrid_block_ = std::make_shared<Block>(INX, dx);
            multigrid_block_->refined = is_refined;
            auto& source = multigrid_block_->source;
            auto const& density = grid_ptr->get_field(rho_i);
            double const scale = 4.0 * std::acos(-1.0) * physcon().G;
            for (int i = 1; i <= INX; ++i) {
                for (int j = 1; j <= INX; ++j) {
                    for (int k = 1; k <= INX; ++k) {
                        double const rho = density[hindex(i + H_BW - 1, j + H_BW - 1, k + H_BW - 1)];
                        if (!std::isfinite(rho) || rho < 0.0) {
                            throw std::invalid_argument("Static gravity requires finite nonnegative density");
                        }
                        source(i, j, k) = scale * rho;
                    }
                }
            }
            result[0] = level;
            result[1] = is_refined ? 0.0 : 1.0;
        } else if (stage == release) {
            multigrid_block_.reset();
            multigrid_boundary_.reset();
        } else {
            assert(multigrid_block_);
            auto& block = *multigrid_block_;
            if (stage == restrict_source || stage == restrict_phi || stage == restrict_correction) {
                int const which = stage == restrict_source ? 2 : stage == restrict_phi ? 0 : 1;
                if (is_refined) {
                    std::array<future<std::vector<double>>, 8> requests;
                    for (int c = 0; c < 8; ++c) {
                        requests[c] = children[c].multigrid_data(which, 1, 0);
                    }
                    for (int c = 0; c < 8; ++c) {
                        block.setChild(field(block, which), c, requests[c].get());
                    }
                }
            } else if (stage == halo_phi || stage == halo_correction || stage == inject_correction) {
                int const which = stage == halo_phi ? 0 : 1;
                auto& values = field(block, which);
                std::array<future<std::vector<double>>, 26> requests;
                for (auto const& direction : geo::direction::full_set()) {
                    if (which == 1 && !direction.is_face()) {
                        continue;
                    }
                    if (!neighbors[direction].empty()) {
                        requests[direction] = neighbors[direction].multigrid_data(
                            which, 3, int(direction.flip()));
                    }
                }
                std::vector<std::array<int, 3>> coarse_cells;
                std::vector<double> positions;
                double const extent = grid::get_scaling_factor();
                for (auto const& direction : geo::direction::full_set()) {
                    if (which == 1 && !direction.is_face()) {
                        continue;
                    }
                    auto const lb = lower_bound(direction, INX, true);
                    auto const ub = upper_bound(direction, INX, true);
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
                                std::array<int, 3> const index{{i, j, k}};
                                std::array<double, 3> x{};
                                bool exterior = false;
                                for (int d = 0; d < 3; ++d) {
                                    x[d] = xmin[d] + (index[d] - 0.5) * dx;
                                    exterior = exterior || x[d] < -extent || x[d] > extent;
                                }
                                if (exterior) {
                                    values(i, j, k) = which == 0 ?
                                        physical_boundary(*multigrid_boundary_, x) : 0.0;
                                } else if (stage != halo_correction) {
                                    assert(!parent.empty());
                                    coarse_cells.push_back(index);
                                    positions.insert(positions.end(), x.begin(), x.end());
                                }
                            }
                        }
                    }
                }
                if (!positions.empty()) {
                    auto sampled = parent.multigrid_sample(which, std::move(positions)).get();
                    for (size_t p = 0; p < sampled.size(); ++p) {
                        auto const& i = coarse_cells[p];
                        values(i[0], i[1], i[2]) = sampled[p];
                    }
                }
            } else if (stage == begin_down) {
                block.beginDown();
            } else if (stage == save_rhs) {
                block.saveCorrectionRhs();
            } else if (stage == relax) {
                block.relax();
            } else if (stage == replace_mean) {
                if (!parent.empty()) {
                    block.replaceCorrectionMean(parent.multigrid_data(
                        1, 2, int(my_location.get_child_index())).get());
                }
            } else if (stage == retire) {
                block.retireCorrection();
            } else if (stage == forces) {
                block.computeForces();
            } else if (stage == adjust_defect || stage == restrict_forces) {
                if (!is_refined) {
                    for (auto const& face : geo::face::full_set()) {
                        if (nieces[face] != 1) {
                            continue;
                        }
                        int const axis = int(face.get_dimension());
                        int const side = int(face.get_side());
                        auto packets = neighbors[face.to_direction()].multigrid_face(
                            axis, side ^ 1, stage == adjust_defect ? 2 : 3).get();
                        size_t const count = INX * INX / 4;
                        for (int q = 0; q < 4; ++q) {
                            std::vector<double> patch(packets.begin() + q * count,
                                packets.begin() + (q + 1) * count);
                            if (stage == adjust_defect) {
                                block.adjustFluxDefect(axis, side ? INX + 1 : 1,
                                    (q / 2) * (INX / 2), (q % 2) * (INX / 2), patch);
                            } else {
                                block.setRestrictedForce(axis, side ? INX + 1 : 1,
                                    (q / 2) * (INX / 2), (q % 2) * (INX / 2), patch);
                            }
                        }
                    }
                }
            } else if ((stage == residual || stage == errors || stage == write_fields) && !is_refined) {
                double const volume = dx * dx * dx;
                std::ofstream output;
                if (stage == write_fields) {
                    std::string const name = opts().data_dir + "/gravity_compare_" +
                        std::to_string(my_location.to_id()) + ".csv";
                    output.open(name);
                    if (!output) {
                        throw std::runtime_error("Cannot open " + name);
                    }
                    output << std::setprecision(17)
                        << "x,y,z,dx,rho,mg_phi,fmm_phi,analytic_phi,mg_gx,mg_gy,mg_gz,"
                           "fmm_gx,fmm_gy,fmm_gz,analytic_gx,analytic_gy,analytic_gz,residual\n";
                }
                for (int i = 1; i <= INX; ++i) {
                    for (int j = 1; j <= INX; ++j) {
                        for (int k = 1; k <= INX; ++k) {
                            double const r = block.physicalResidual(i, j, k);
                            double const s = block.source(i, j, k);
                            if (!std::isfinite(r) || !std::isfinite(block.phi(i, j, k))) {
                                throw std::runtime_error("Nonfinite multigrid solution or residual");
                            }
                            if (stage == residual) {
                                result[0] += r * r * volume;
                                result[1] += s * s * volume;
                                result[2] = std::max(result[2], std::abs(r));
                                result[3] += 1.0;
                                continue;
                            }
                            std::array<double, 3> const x{{xmin[0] + (i - 0.5) * dx,
                                xmin[1] + (j - 0.5) * dx, xmin[2] + (k - 0.5) * dx}};
                            auto const exact = sphere(x);
                            std::array<double, 4> mg{{block.phi(i, j, k),
                                0.5 * (block.force[0](i, j, k) + block.force[0](i + 1, j, k)),
                                0.5 * (block.force[1](i, j, k) + block.force[1](i, j + 1, k)),
                                0.5 * (block.force[2](i, j, k) + block.force[2](i, j, k + 1))}};
                            std::array<double, 4> fmm{};
                            for (int d = 0; d < 4; ++d) {
                                fmm[d] = grid_ptr->gravity_value(d, i - 1, j - 1, k - 1);
                                if (!std::isfinite(fmm[d]) || !std::isfinite(mg[d])) {
                                    throw std::runtime_error("Nonfinite gravity comparison field");
                                }
                            }
                            double const ep = mg[0] - exact[0];
                            result[0] += std::abs(ep) * volume;
                            result[1] += ep * ep * volume;
                            result[2] = std::max(result[2], std::abs(ep));
                            result[3] += (fmm[0] - exact[0]) * (fmm[0] - exact[0]) * volume;
                            result[4] += (mg[0] - fmm[0]) * (mg[0] - fmm[0]) * volume;
                            for (int d = 1; d < 4; ++d) {
                                result[5] += (mg[d] - exact[d]) * (mg[d] - exact[d]) * volume;
                                result[6] += (fmm[d] - exact[d]) * (fmm[d] - exact[d]) * volume;
                                result[7] += (mg[d] - fmm[d]) * (mg[d] - fmm[d]) * volume;
                            }
                            result[8] += volume;
                            if (output) {
                                output << x[0] << ',' << x[1] << ',' << x[2] << ',' << dx << ','
                                    << s / (4.0 * std::acos(-1.0) * physcon().G) << ',' << mg[0]
                                    << ',' << fmm[0] << ',' << exact[0];
                                for (int d = 1; d < 4; ++d) {
                                    output << ',' << mg[d];
                                }
                                for (int d = 1; d < 4; ++d) {
                                    output << ',' << fmm[d];
                                }
                                for (int d = 1; d < 4; ++d) {
                                    output << ',' << exact[d];
                                }
                                output << ',' << r << '\n';
                            }
                        }
                    }
                }
            }
        }
    }
    for (auto& descendant : descendants) {
        accumulate(result, descendant.get(), stage);
    }
    return result;
}

void node_server::compare_gravity_multigrid() {
    assert(my_location.level() == 0);
    if (opts().hydro || opts().radiation || !opts().gravity ||
        opts().problem != SOLID_SPHERE || !opts().restart_filename.empty()) {
        throw std::invalid_argument(
            "gravity_compare requires a fresh static solid_sphere run with gravity on and hydro/radiation off");
    }
    if (opts().gravity_mg_max_cycles <= 0 || !(opts().gravity_mg_tolerance > 0.0) ||
        !std::isfinite(opts().gravity_mg_tolerance) || !(opts().solid_sphere_radius > 0.0) ||
        !(opts().solid_sphere_mass > 0.0) || !(physcon().G > 0.0) ||
        !std::isfinite(opts().solid_sphere_radius) || !std::isfinite(opts().solid_sphere_mass) ||
        !std::isfinite(physcon().G)) {
        throw std::invalid_argument("Invalid static gravity comparison parameters");
    }
    if (opts().gravity_mg_boundary != "analytic" && opts().gravity_mg_boundary != "multipole") {
        throw std::invalid_argument("gravity_mg_boundary must be analytic or multipole");
    }
    if (opts().solid_sphere_rho_min != 0.0 || opts().rho_floor != 0.0) {
        throw std::invalid_argument(
            "Analytic sphere comparison requires solid_sphere_rho_min=0 and rho_floor=0");
    }
    if (opts().correct_am_grav) {
        throw std::invalid_argument(
            "Static vacuum sphere comparison requires correct_am_grav=off");
    }
    double const radius = opts().solid_sphere_radius;
    double const extent = grid::get_scaling_factor();
    if (!(extent > 0.0) || !std::isfinite(extent)) {
        throw std::invalid_argument("Static gravity requires a finite positive domain extent");
    }
    for (double center : {opts().solid_sphere_xcenter, opts().solid_sphere_ycenter,
             opts().solid_sphere_zcenter}) {
        if (!std::isfinite(center) || std::abs(center) + radius >= extent) {
            throw std::invalid_argument("Comparison sphere must lie strictly inside the domain");
        }
    }
    using clock = std::chrono::steady_clock;
    auto const fmm_start = clock::now();
    // Keep density and topology fixed, and do not alter the gas energy.
    solve_gravity(false, false);
    double const fmm_seconds = std::chrono::duration<double>(clock::now() - fmm_start).count();
    auto const mg_start = clock::now();
    auto const topology = multigrid_stage(initialize_block, -1);
    int const deepest = int(topology[0]);
    auto run = [this](int stage, int level) { return multigrid_stage(stage, level); };
    for (int level = deepest - 1; level >= 0; --level) {
        run(restrict_source, level);
    }
    // Reduce only a fixed number of moments. Physical cell arrays remain on
    // their owning HPX components, including during isolated-boundary setup.
    auto const source_moments = multigrid_boundary(0, {});
    if (!(source_moments[0] > 0.0) || !std::isfinite(source_moments[0])) {
        throw std::runtime_error("Static gravity comparison requires finite positive total mass");
    }
    std::vector<double> context(54, 0.0);
    context[0] = opts().gravity_mg_boundary == "multipole" ? 1.0 : 0.0;
    std::vector<double> center(3);
    for (int d = 0; d < 3; ++d) {
        center[d] = source_moments[d + 1] / source_moments[0];
        if (!std::isfinite(center[d])) {
            throw std::runtime_error("Nonfinite gravity source center of mass");
        }
        context[d + 1] = center[d];
    }
    if (context[0] != 0.0) {
        auto const moments = multigrid_boundary(1, center);
        double const minimum_boundary_radius = extent -
            std::max({std::abs(center[0]), std::abs(center[1]), std::abs(center[2])});
        if (!(std::sqrt(moments[50]) < minimum_boundary_radius)) {
            throw std::invalid_argument(
                "Gravity multipole boundary must lie outside the source-enclosing sphere");
        }
        std::copy_n(moments.begin(), 50, context.begin() + 4);
    }
    multigrid_boundary(2, std::move(context));
    auto update = [&]() {
        for (int level = deepest - 1; level >= 0; --level) {
            run(restrict_phi, level);
        }
        for (int level = 0; level <= deepest; ++level) {
            run(halo_phi, level);
        }
        run(forces, -1);
        run(restrict_forces, -1);
        return run(residual, -1);
    };
    auto norms = update();
    if (!std::isfinite(norms[0])) {
        throw std::runtime_error("Nonfinite initial multigrid residual norm");
    }
    double const source_norm = std::sqrt(norms[1]);
    if (!(source_norm > 0.0) || !std::isfinite(source_norm)) {
        throw std::runtime_error("Static gravity comparison requires a finite nonzero source norm");
    }
    int cycles = 0;
    while (std::sqrt(norms[0]) > opts().gravity_mg_tolerance * source_norm &&
        cycles < opts().gravity_mg_max_cycles) {
        for (int level = deepest; level >= 0; --level) {
            run(restrict_correction, level);
            run(begin_down, level);
            run(adjust_defect, level);
            run(save_rhs, level);
            for (int sweep = 0; sweep < 2; ++sweep) {
                run(halo_correction, level);
                run(relax, level);
            }
        }
        for (int level = 0; level <= deepest; ++level) {
            run(replace_mean, level);
            run(inject_correction, level);
            for (int sweep = 0; sweep < 4; ++sweep) {
                run(halo_correction, level);
                run(relax, level);
            }
        }
        run(retire, -1);
        norms = update();
        ++cycles;
        if (!std::isfinite(norms[0])) {
            throw std::runtime_error("Nonfinite multigrid residual norm");
        }
        std::cout << "MG_GRAVITY_CYCLE cycle=" << cycles
                  << " relative_residual=" << std::sqrt(norms[0]) / source_norm << '\n';
    }
    double const mg_seconds = std::chrono::duration<double>(clock::now() - mg_start).count();
    auto const error = run(errors, -1);
    if (opts().gravity_compare_output) {
        run(write_fields, -1);
    }
    double const volume = error[8];
    std::cout << std::setprecision(17) << "MG_GRAVITY_RESULT cycles=" << cycles
        << " residual=" << std::sqrt(norms[0])
        << " relative_residual=" << std::sqrt(norms[0]) / source_norm
        << " phi_l1=" << error[0] / volume
        << " phi_l2=" << std::sqrt(error[1] / volume)
        << " phi_linf=" << error[2]
        << " fmm_phi_l2=" << std::sqrt(error[3] / volume)
        << " delta_phi_l2=" << std::sqrt(error[4] / volume)
        << " force_l2=" << std::sqrt(error[5] / volume)
        << " fmm_force_l2=" << std::sqrt(error[6] / volume)
        << " delta_force_l2=" << std::sqrt(error[7] / volume)
        << " leaves=" << topology[1] << " cells=" << norms[3]
        << " mg_seconds=" << mg_seconds << " fmm_seconds=" << fmm_seconds
        << " boundary=" << (opts().gravity_mg_boundary == "multipole" ?
            "multipole_l4_ghost_centers" : "analytic_sphere_ghost_centers")
        << " represented_mass=" << source_moments[0] / physcon().G
        << " fmm_am_correction=" << opts().correct_am_grav
        << " rho_floor=" << opts().solid_sphere_rho_min << '\n';
    run(release, -1);
    if (std::sqrt(norms[0]) > opts().gravity_mg_tolerance * source_norm) {
        throw std::runtime_error("Static gravity multigrid reached its cycle limit without convergence");
    }
}

#endif
