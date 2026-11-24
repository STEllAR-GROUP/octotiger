// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <hpx/hpx_init.hpp>

#include "octotiger/compute_factor.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/grid_fmm.hpp"
#include "octotiger/monopole_interactions/monopole_kernel_interface.hpp"
#include "octotiger/options.hpp"

namespace {

int failures = 0;

std::array<integer, NDIM> coordinates(integer index) {
    return {index / (INX * INX), (index / INX) % INX, index % INX};
}

void check_results(grid &block, std::vector<real> const &sources, real dx,
    char const *path, gsolve_type type, bool sparse) {
    auto const &expansions = block.get_L();
    auto const &corrections = block.get_L_c();
    long double const tolerance = 128 * std::numeric_limits<real>::epsilon();
    for (integer target = 0; target != G_N3; ++target) {
        auto const target_coordinates = coordinates(target);
        std::array<long double, 4> expected{};
        std::array<long double, 4> magnitude{};
        for (integer source = 0; source != G_N3; ++source) {
            if (target == source) {
                continue;
            }
            auto const source_coordinates = coordinates(source);
            std::array<long double, NDIM> displacement;
            long double radius_squared = 0;
            for (integer dimension = 0; dimension != NDIM; ++dimension) {
                displacement[dimension] =
                    static_cast<long double>(target_coordinates[dimension] -
                        source_coordinates[dimension]) * dx;
                radius_squared += displacement[dimension] * displacement[dimension];
            }
            long double const radius = std::sqrt(radius_squared);
            std::array<long double, 4> contribution;
            contribution[0] = -sources[source] / radius;
            for (integer dimension = 0; dimension != NDIM; ++dimension) {
                contribution[dimension + 1] = sources[source] * displacement[dimension] /
                    (radius_squared * radius);
            }
            for (size_t component = 0; component != expected.size(); ++component) {
                expected[component] += contribution[component];
                magnitude[component] += std::abs(contribution[component]);
            }
        }
        for (size_t component = 0; component != expected.size(); ++component) {
            real const actual = expansions[target][component];
            long double const error = std::abs(actual - expected[component]);
            if (!std::isfinite(actual) ||
                error > tolerance * std::max(magnitude[component], 1.0e-30L)) {
                if (failures < 12) {
                    std::cerr << path << " type=" << static_cast<int>(type)
                              << " sparse=" << sparse << " dx=" << dx
                              << " target=" << target << " component=" << component
                              << ": expected " << expected[component] << ", got " << actual
                              << '\n';
                }
                ++failures;
            }
        }
        for (size_t component = 4; component != expansions[target].size(); ++component) {
            if (expansions[target][component] != 0.0) {
                ++failures;
            }
        }
        for (integer dimension = 0; dimension != NDIM; ++dimension) {
            if (corrections[target][dimension] != 0.0) {
                ++failures;
            }
        }
    }
}

void check_case(real dx, gsolve_type type, bool sparse) {
    auto block = std::make_shared<grid>(dx, std::array<real, NDIM>{-0.375, 0.125, -0.625});
    block->set_root();
    auto &density = block->get_field(rho_i);
    std::fill(density.begin(), density.end(), 1.0);
    block->compute_multipoles(RHO);

    // The interaction interface receives masses for RHO and signed mass rates for DRHODT.
    // Initialize geometry with positive density, then supply each source type directly.
    auto &sources = block->get_mon();
    for (integer index = 0; index != G_N3; ++index) {
        auto const position = coordinates(index);
        real value = 0.125 + ((3 * position[0] + 5 * position[1] + 7 * position[2]) % 23) / 29.0;
        if (sparse) {
            value = index == 0 ? 2.0 : (index == G_N3 - 1 ? 3.0 : 0.0);
        }
        if (type == DRHODT && (position[0] + position[1] + position[2]) % 2 != 0) {
            value = -value;
        }
        sources[index] = value * dx * dx * dx;
    }

    std::vector<neighbor_gravity_type> neighbors(geo::direction::count());
    for (auto const &direction : geo::direction::full_set()) {
        neighbors[direction].is_monopole = true;
        neighbors[direction].direction = direction;
        neighbors[direction].data.local_semaphore = nullptr;
    }
    std::array<bool, geo::direction::count()> empty_directions;
    empty_directions.fill(true);
    for (auto &entry : block->get_L()) {
        entry = 0.0;
    }
    std::fill(block->get_L_c().begin(), block->get_L_c().end(), space_vector(0.0));
    octotiger::fmm::monopole_interactions::monopole_kernel_interface(sources,
        block->get_com_ptr(), neighbors, type, dx, empty_directions, block, false);
    check_results(*block, sources, dx, "monopole interface", type, sparse);

    for (auto &entry : block->get_L()) {
        entry = 7.0;
    }
    std::fill(block->get_L_c().begin(), block->get_L_c().end(), space_vector(9.0));
    block->compute_interactions(type);
    check_results(*block, sources, dx, "grid interactions", type, sparse);
}

} // namespace

int hpx_main(hpx::program_options::variables_map &arguments) {
    bool const device_only = arguments.count("device-only") != 0;
    physics<NDIM>::set_n_species(1);
    physics<NDIM>::update_n_field();
    opts().n_species = 1;
    opts().n_fields = physics<NDIM>::field_count();
    opts().radiation = false;
    opts().gravity = true;
    opts().theta = 0.5;
    opts().monopole_host_kernel_type = device_only ? DEVICE_ONLY : LEGACY;
    opts().monopole_device_kernel_type = device_only ? KOKKOS_HIP : OFF;
    compute_factor();
    compute_ilist();
    grid::static_init();
    for (real const dx : {0.125, 0.7}) {
        for (gsolve_type const type : {RHO, DRHODT}) {
            for (bool const sparse : {false, true}) {
                check_case(dx, type, sparse);
            }
        }
    }
    if (failures != 0) {
        std::cerr << failures << " root-leaf gravity checks failed\n";
        hpx::finalize();
        return 1;
    }
    std::cout << "Root-leaf RHO and DRHODT potential and gradient match direct summation"
              << (device_only ? " with device-only options" : " with legacy host options") << '\n';
    return hpx::finalize();
}

int main(int argc, char **argv) {
    hpx::program_options::options_description description("Root-leaf gravity regression");
    description.add_options()("device-only", "Select device-only options without initializing a GPU");
    hpx::init_params parameters;
    parameters.desc_cmdline = description;
    parameters.cfg = {"hpx.os_threads=1"};
    return hpx::init(argc, argv, parameters);
}
