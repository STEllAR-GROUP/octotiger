// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hpx/include/future.hpp>

#include "octotiger/node_server.hpp"
#include "octotiger/radiation/amr.hpp"
#include "octotiger/radiation/rad_grid.hpp"

#if !defined(HPX_COMPUTE_DEVICE_CODE)
using set_rad_grid_action_type = node_server::set_rad_grid_action;
HPX_REGISTER_ACTION(set_rad_grid_action_type);

using send_rad_boundary_action_type = node_server::send_rad_boundary_action;
HPX_REGISTER_ACTION(send_rad_boundary_action_type);

using send_rad_flux_correct_action_type = node_server::send_rad_flux_correct_action;
HPX_REGISTER_ACTION(send_rad_flux_correct_action_type);

using send_rad_children_action_type = node_server::send_rad_children_action;
HPX_REGISTER_ACTION(send_rad_children_action_type);

using send_rad_amr_boundary_action_type = node_server::send_rad_amr_boundary_action;
HPX_REGISTER_ACTION(send_rad_amr_boundary_action_type);

hpx::future<void> node_client::set_rad_grid(
    std::vector<real>&& g /*, std::vector<real>&& o*/) const {
    return hpx::async<typename node_server::set_rad_grid_action>(get_unmanaged_gid(), g /*, o*/);
}

void node_server::set_rad_grid(std::vector<real> const& data /*, std::vector<real>&& outflows*/) {
    rad_grid_ptr->set_prolong(data /*, std::move(outflows)*/);
}

void node_client::send_rad_flux_correct(
    std::vector<real>&& data, geo::face const& face, geo::octant const& ci) const {
    hpx::apply<typename node_server::send_rad_flux_correct_action>(
        get_unmanaged_gid(), std::move(data), face, ci);
}

void node_server::recv_rad_flux_correct(
    std::vector<real>&& data, geo::face const& face, geo::octant const& ci) {
    geo::quadrant const index(ci, face.get_dimension());
    niece_rad_channels[face][index].set_value(std::move(data));
}

void node_client::send_rad_boundary(
    std::vector<real>&& data, geo::direction const& dir, size_t cycle) const {
    hpx::apply<typename node_server::send_rad_boundary_action>(
        get_unmanaged_gid(), std::move(data), dir, cycle);
}

void node_server::recv_rad_boundary(
    std::vector<real>&& bdata, geo::direction const& dir, size_t cycle) {
    sibling_rad_type tmp;
    tmp.data = std::move(bdata);
    tmp.direction = dir;
    sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

void node_server::recv_rad_children(std::vector<real>&& data, geo::octant const& ci, size_t cycle) {
    child_rad_channels[ci].set_value(std::move(data), cycle);
}

void node_client::send_rad_children(
    std::vector<real>&& data, geo::octant const& ci, size_t cycle) const {
    hpx::apply<typename node_server::send_rad_children_action>(
        get_unmanaged_gid(), std::move(data), ci, cycle);
}

hpx::future<void> node_server::exchange_rad_flux_corrections() {
    geo::octant const ci = my_location.get_child_index();
    constexpr auto full_set = geo::face::full_set();
    for (auto& f : full_set) {
        auto const face_dim = f.get_dimension();
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
            this_aunt.send_rad_flux_correct(std::move(data), f.flip(), ci);
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
                futs[index++] = niece_rad_channels[f][quadrant].get_future().then(
                    [this, f, quadrant](hpx::future<std::vector<real>>&& fdata) -> void {
                        auto const face_dim = f.get_dimension();
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

void rad_grid::set_flux_restrict(std::vector<real> const& data, std::array<integer, NDIM> const& lb,
    std::array<integer, NDIM> const& ub, geo::dimension const& dim) {
    size_t expected = NRF;
    for (integer dimension = 0; dimension < NDIM; ++dimension) {
        expected *= ub[dimension] - lb[dimension];
    }
    if (data.size() != expected) {
        throw std::invalid_argument("Invalid radiation flux correction payload size");
    }
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

std::vector<real> rad_grid::get_flux_restrict(std::array<integer, NDIM> const& lb,
    std::array<integer, NDIM> const& ub, geo::dimension const& dim) const {
    std::vector<real> data;
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
                    real value = ZERO;
                    value += flux[dim][field][i00];
                    value += flux[dim][field][i10];
                    value += flux[dim][field][i01];
                    value += flux[dim][field][i11];
                    value /= real(4);
                    data.push_back(value);
                }
            }
        }
    }
    return data;
}

void node_server::all_rad_bounds(real boundary_time) {
    GET(exchange_interlevel_rad_data());
    collect_radiation_bounds(boundary_time);
    send_rad_amr_bounds();
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

void node_server::collect_radiation_bounds(real boundary_time) {
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
        if (my_location.is_physical_boundary(face)) {
            rad_grid_ptr->set_physical_boundaries(face, boundary_time);
        }
    }
}

void rad_grid::set_boundary(std::vector<real> const& data, geo::direction const& dir) {
    std::array<integer, NDIM> lb, ub;
    auto const cells = get_boundary_size(lb, ub, dir, OUTER, INX, RAD_BW);
    if (data.size() != size_t(NRF * cells)) {
        throw std::invalid_argument("Invalid radiation boundary payload size");
    }
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

std::vector<real> rad_grid::get_boundary(geo::direction const& dir) {
    std::array<integer, NDIM> lb, ub;
    std::vector<real> data;
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

void rad_grid::set_prolong(std::vector<real> const& data) {
    if (data.size() != size_t(NRF * INX * INX * INX)) {
        throw std::invalid_argument("Invalid radiation transfer payload size");
    }

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

std::vector<real> rad_grid::get_prolong(
    std::array<integer, NDIM> const& lb, std::array<integer, NDIM> const& ub) {
    std::vector<real> data;
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
                    real value = U[f][iii];
                    data.push_back(value);
                }
            }
        }
    }
    return data;
}

std::vector<real> rad_grid::get_restrict() const {
    std::vector<real> data;
    for (integer f = 0; f != NRF; ++f) {
        for (integer i = RAD_BW; i < RAD_NX - RAD_BW; i += 2) {
            for (integer j = RAD_BW; j < RAD_NX - RAD_BW; j += 2) {
                for (integer k = RAD_BW; k < RAD_NX - RAD_BW; k += 2) {
                    integer const iii = rindex(i, j, k);
                    real v = ZERO;
                    for (integer x = 0; x != 2; ++x) {
                        for (integer y = 0; y != 2; ++y) {
                            for (integer z = 0; z != 2; ++z) {
                                integer const jjj = iii + x * RAD_NX * RAD_NX + y * RAD_NX + z;
                                v += U[f][jjj];
                            }
                        }
                    }
                    v /= real(NCHILD);
                    data.push_back(v);
                }
            }
        }
    }
    return data;
}

void rad_grid::set_restrict(std::vector<real> const& data, geo::octant const& octant) {
    if (data.size() != size_t(NRF * INX * INX * INX / NCHILD)) {
        throw std::invalid_argument("Invalid radiation transfer payload size");
    }

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
}

void node_server::send_rad_amr_bounds() {
    if (is_refined) {
        constexpr auto full_set = geo::octant::full_set();
        for (auto& ci : full_set) {
            auto const& flags = amr_flags[ci];
            for (auto& dir : geo::direction::full_set()) {
                if (flags[dir]) {
                    std::array<integer, NDIM> lb, ub;
                    std::vector<real> data;
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

void rad_grid::clear_amr() {
    std::fill(is_coarse.begin(), is_coarse.end(), 0);
    std::fill(has_coarse.begin(), has_coarse.end(), 0);
}

void rad_grid::set_rad_amr_boundary(std::vector<real> const& data, geo::direction const& dir) {
    PROFILE();

    std::array<integer, NDIM> lb, ub;
    int l = 0;
    get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
    size_t expected = NRF;
    for (int dimension = 0; dimension < NDIM; ++dimension) {
        expected *=
            std::min(ub[dimension] + 1, integer(HS_NX)) - std::max(lb[dimension] - 1, integer(0));
    }
    if (data.size() != expected) {
        throw std::invalid_argument("Invalid radiation coarse boundary payload size");
    }

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
    using state_type = std::array<double, NRF>;
    constexpr std::array<integer, NDIM> stride{HS_DNX, HS_DNY, HS_DNZ};
    constexpr integer offset = H_BW - RAD_BW;
    for (integer i0 = 1; i0 < HS_NX - 1; ++i0) {
        for (integer j0 = 1; j0 < HS_NX - 1; ++j0) {
            for (integer k0 = 1; k0 < HS_NX - 1; ++k0) {
                integer const coarse = hSindex(i0, j0, k0);
                if (!is_coarse[coarse]) {
                    continue;
                }
                state_type center;
                std::array<state_type, NDIM> slope{};
                for (integer field = 0; field < NRF; ++field) {
                    center[field] = Ushad[field][coarse];
                    for (integer dimension = 0; dimension < NDIM; ++dimension) {
                        if (has_coarse[coarse - stride[dimension]] &&
                            has_coarse[coarse + stride[dimension]]) {
                            slope[dimension][field] = 0.25 *
                                radiation::limited_slope(
                                    center[field] - Ushad[field][coarse - stride[dimension]],
                                    Ushad[field][coarse + stride[dimension]] - center[field]);
                        }
                    }
                }
                auto const children = radiation::interpolate_children(
                    center, slope, radiation::finite_nonnegative_energy<NRF>);
                for (integer child = 0; child < NCHILD; ++child) {
                    // Shadow cells use hydro coordinates; fields use radiation strides.
                    integer const i = 2 * i0 - H_BW + ((child >> 2) & 1) - offset;
                    integer const j = 2 * j0 - H_BW + ((child >> 1) & 1) - offset;
                    integer const k = 2 * k0 - H_BW + (child & 1) - offset;
                    if (i < 0 || i >= RAD_NX || j < 0 || j >= RAD_NX || k < 0 || k >= RAD_NX) {
                        continue;
                    }
                    for (integer field = 0; field < NRF; ++field) {
                        U[field][rindex(i, j, k)] = children[child][field];
                    }
                }
            }
        }
    }
}

std::vector<real> rad_grid::get_subset(
    std::array<integer, NDIM> const& lb, std::array<integer, NDIM> const& ub) {
    PROFILE();
    std::vector<real> data;
    data.reserve(NRF * (ub[0] - lb[0]) * (ub[1] - lb[1]) * (ub[2] - lb[2]));
    constexpr integer offset = H_BW - RAD_BW;
    // The AMR message envelope uses hydro coordinates; the values live in U
    // with radiation strides. If hydro has a wider halo, its extra padding is
    // outside every stencil needed for a radiation ghost cell (RAD_BW >= 2).
    // Extend the nearest available value into that unused message padding.
    auto const radiation_index = [](integer x) {
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
    return data;
}

void node_server::recv_rad_amr_boundary(
    std::vector<real>&& bdata, geo::direction const& dir, size_t cycle) {
    sibling_rad_type tmp;
    tmp.data = std::move(bdata);
    tmp.direction = dir;
    sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

void node_client::send_rad_amr_boundary(
    std::vector<real>&& data, geo::direction const& dir, size_t cycle) const {
    hpx::apply<typename node_server::send_rad_amr_boundary_action>(
        get_unmanaged_gid(), std::move(data), dir, cycle);
}

#endif
