// Copyright (c) 2026 AUTHORS
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include <hpx/hpx_init.hpp>

#include "octotiger/compute_factor.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/grid_fmm.hpp"
#include "octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp"
#include "octotiger/options.hpp"

namespace {

constexpr real cell_width = 0.25;
int failures = 0;

struct result {
    std::vector<expansion> potential;
    std::vector<space_vector> correction;
};

void check(bool condition, char const *description) {
    if (!condition) {
        if (failures < 10) {
            std::cerr << "FAIL: " << description << '\n';
        }
        ++failures;
    }
}

void add_neighbor(std::shared_ptr<grid> const &target,
    std::vector<neighbor_gravity_type> &neighbors,
    std::array<bool, geo::direction::count()> &empty, geo::direction const &direction,
    bool monopole, gsolve_type type) {
    auto &neighbor = neighbors[direction];
    neighbor.is_monopole = monopole;
    neighbor.data.allocate();
    empty[direction] = false;
    for (auto const &entry : target->get_ilist_n_bnd(direction)) {
        real const mass = type == RHO ? 0.3 + 0.001 * entry.second :
            0.02 * (static_cast<int>(entry.second % 7) - 3);
        if (monopole) {
            neighbor.data.m->push_back(mass);
        } else {
            multipole moment;
            moment = 0.0;
            moment[0] = mass;
            if (type == DRHODT) {
                moment[1] = 0.07 * mass;
                moment[2] = -0.03 * mass;
                moment[3] = 0.05 * mass;
            }
            moment[4] = 0.01 * mass;
            moment[5] = -0.003 * mass;
            moment[9] = 0.02 * mass;
            moment[10] = 0.001 * mass;
            moment[14] = -0.002 * mass;
            neighbor.data.M->push_back(moment);
            space_vector center;
            for (integer d = 0; d < NDIM; ++d) {
                center[d] = (entry.x[d] + 0.5) * cell_width;
            }
            neighbor.data.x->push_back(center);
        }
    }
}

result evaluate(interaction_host_kernel_type host, gsolve_type type,
    bool monopole_neighbor, bool multipole_neighbor) {
    opts().monopole_host_kernel_type = host;
    auto target = std::make_shared<grid>(cell_width, std::array<real, NDIM>{0.0, 0.0, 0.0});
    std::fill(target->get_field(rho_i).begin(), target->get_field(rho_i).end(), 1.0);
    target->compute_multipoles(RHO);
    if (type == DRHODT) {
        auto &masses = target->get_mon();
        for (size_t i = 0; i < masses.size(); ++i) {
            masses[i] = 0.01 * (static_cast<int>(i % 11) - 5);
        }
    }
    std::fill(target->get_L().begin(), target->get_L().end(), 0.0);
    std::fill(target->get_L_c().begin(), target->get_L_c().end(), 0.0);

    std::vector<neighbor_gravity_type> neighbors(geo::direction::count());
    std::array<bool, geo::direction::count()> empty;
    empty.fill(true);
    for (auto const &direction : geo::direction::full_set()) {
        neighbors[direction].direction = direction;
        neighbors[direction].is_monopole = true;
        neighbors[direction].data.local_semaphore = nullptr;
    }
    if (monopole_neighbor) {
        geo::direction direction;
        direction.set(-1, 0, 0);
        add_neighbor(target, neighbors, empty, direction, true, type);
    }
    if (multipole_neighbor) {
        geo::direction direction;
        direction.set(0, 1, 0);
        add_neighbor(target, neighbors, empty, direction, false, type);
    }

    // Enter the same CPU fallback used by unsupported mixed-AMR device kernels.
    octotiger::fmm::monopole_interactions::monopole_interaction_interface interactor;
    interactor.compute_interactions(target->get_mon(), target->get_com_ptr(), neighbors,
        type, cell_width, empty, target, multipole_neighbor);
    return {target->get_L(), target->get_L_c()};
}

void compare(result const &actual, result const &expected) {
    for (size_t i = 0; i < actual.potential.size(); ++i) {
        for (integer component = 0; component < actual.potential[i].size(); ++component) {
            real const a = actual.potential[i][component];
            real const e = expected.potential[i][component];
            check(std::isfinite(a) && std::isfinite(e) &&
                    std::abs(a - e) <= 1.0e-12 * std::max(real(1.0), std::abs(e)),
                "DEVICE_ONLY fallback differs from LEGACY potential expansion");
        }
        for (integer d = 0; d < NDIM; ++d) {
            real const a = actual.correction[i][d];
            real const e = expected.correction[i][d];
            check(std::isfinite(a) && std::isfinite(e) &&
                    std::abs(a - e) <= 1.0e-12 * std::max(real(1.0), std::abs(e)),
                "DEVICE_ONLY fallback differs from LEGACY angular correction");
        }
    }
}

real difference(result const &left, result const &right) {
    real maximum = 0.0;
    for (size_t i = 0; i < left.potential.size(); ++i) {
        maximum = std::max(maximum, std::abs(left.potential[i][0] - right.potential[i][0]));
    }
    return maximum;
}

} // namespace

int hpx_main(hpx::program_options::variables_map &) {
    opts().n_species = 1;
    physics<NDIM>::set_n_species(1);
    physics<NDIM>::update_n_field();
    opts().n_fields = physics<NDIM>::field_count();
    opts().radiation = false;
    opts().gravity = true;
    opts().correct_am_grav = true;
    opts().theta = 0.5;
    compute_ilist();
    compute_factor();
    grid::static_init();

    for (gsolve_type const type : {RHO, DRHODT}) {
        auto const interior = evaluate(LEGACY, type, false, false);
        auto const monopoles = evaluate(LEGACY, type, true, false);
        auto const mixed = evaluate(LEGACY, type, true, true);
        check(difference(monopoles, interior) > 1.0e-8,
            "fixture must have a nonzero monopole boundary contribution");
        check(difference(mixed, monopoles) > 1.0e-8,
            "fixture must have a nonzero multipole boundary contribution");
        compare(evaluate(DEVICE_ONLY, type, true, false), monopoles);
        compare(evaluate(DEVICE_ONLY, type, true, true), mixed);
    }
    if (failures == 0) {
        std::cout << "DEVICE_ONLY CPU fallback matches LEGACY for RHO and DRHODT, "
                     "including monopole and multipole boundaries\n";
    } else {
        std::cerr << failures << " DEVICE_ONLY fallback checks failed\n";
    }
    hpx::finalize();
    return failures == 0 ? 0 : 1;
}

int main(int argc, char *argv[]) {
    hpx::init_params parameters;
    parameters.cfg = {"hpx.os_threads=1"};
    return hpx::init(argc, argv, parameters);
}
