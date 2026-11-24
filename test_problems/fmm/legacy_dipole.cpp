// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>

#include <hpx/hpx_init.hpp>

#include "octotiger/compute_factor.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/grid_fmm.hpp"
#include "octotiger/options.hpp"

namespace {
    constexpr real cell_width = 0.25;
    size_t checks = 0;
    size_t failures = 0;

    void check(real actual, real expected, char const * description) {
        ++checks;
        if (!std::isfinite(actual)
            || std::abs(actual - expected) > 1.e-12 * std::max(real(1), std::abs(expected))) {
            if (failures < 12) {
                std::cerr << description << ": actual " << actual << ", expected "
                          << expected << '\n';
            }
            ++failures;
        }
    }

    std::shared_ptr<grid> make_grid(bool leaf) {
        auto target = std::make_shared<grid>(cell_width, std::array<real, NDIM>{0, 0, 0});
        std::fill(target->get_field(rho_i).begin(), target->get_field(rho_i).end(), 1.0);
        target->compute_multipoles(RHO);
        if (!leaf) {
            multipole_pass_type children;
            children.first.resize(G_N3);
            children.second = *target->get_com_ptr()[0];
            for (auto & moment : children.first) {
                moment = 0.0;
                moment[0] = 1.0;
            }
            target->set_leaf(false);
            target->set_root(true);
            target->compute_multipoles(RHO, &children);
        }
        return target;
    }

    std::array<real, 4> dipole_field(space_vector const & target,
        space_vector const & source, std::array<real, NDIM> const & dipole) {
        std::array<real, NDIM> displacement;
        real r2 = 0;
        real dot = 0;
        for (integer d = 0; d < NDIM; ++d) {
            displacement[d] = target[d] - source[d];
            r2 += displacement[d] * displacement[d];
            dot += displacement[d] * dipole[d];
        }
        real const r3 = r2 * std::sqrt(r2);
        std::array<real, 4> result;
        result[0] = -dot / r3;
        for (integer d = 0; d < NDIM; ++d) {
            result[d + 1] = -dipole[d] / r3 + 3 * displacement[d] * dot / (r2 * r3);
        }
        return result;
    }

    void interior() {
        auto target = make_grid(false);
        std::fill(target->get_M().begin(), target->get_M().end(), 0.0);
        std::array<integer, 2> const sources{0, G_N3 - 1};
        std::array<std::array<real, NDIM>, 2> const dipoles{{
            {0.17, -0.11, 0.23}, {-0.08, 0.19, -0.13}}};
        for (size_t s = 0; s < sources.size(); ++s) {
            for (integer d = 0; d < NDIM; ++d) {
                target->get_M()[sources[s]][d + 1] = dipoles[s][d];
            }
        }
        target->compute_interactions(DRHODT);
        auto const & centers = *target->get_com_ptr()[0];
        for (integer i = 0; i < INX; ++i) {
            for (integer j = 0; j < INX; ++j) {
                for (integer k = 0; k < INX; ++k) {
                    integer const index = gindex(i, j, k);
                    std::array<real, 4> expected{};
                    for (size_t s = 0; s < sources.size(); ++s) {
                        integer const coordinate = s == 0 ? 0 : INX - 1;
                        integer const distance2 = (i - coordinate) * (i - coordinate)
                            + (j - coordinate) * (j - coordinate)
                            + (k - coordinate) * (k - coordinate);
                        if (distance2 >= 4) {
                            auto const contribution = dipole_field(
                                centers[index], centers[sources[s]], dipoles[s]);
                            for (size_t d = 0; d < expected.size(); ++d) {
                                expected[d] += contribution[d];
                            }
                        }
                    }
                    for (size_t d = 0; d < expected.size(); ++d) {
                        check(target->get_L()[index][d], expected[d], "legacy interior dipole");
                    }
                }
            }
        }
    }

    void boundary(bool leaf) {
        auto target = make_grid(leaf);
        std::fill(target->get_L().begin(), target->get_L().end(), 0.0);
        std::fill(target->get_L_c().begin(), target->get_L_c().end(), 0.0);
        geo::direction direction;
        direction.set(1, 1, 1);
        auto const & list = target->get_ilist_n_bnd(direction);
        size_t selected = 0;
        while (selected < list.size() && list[selected].first.empty()) {
            ++selected;
        }
        if (selected == list.size()) {
            std::cerr << "No multipole boundary interaction available in the fixture\n";
            ++failures;
            return;
        }
        std::array<real, NDIM> const dipole{0.17, -0.11, 0.23};
        gravity_boundary_type data;
        data.allocate();
        for (size_t s = 0; s < list.size(); ++s) {
            multipole moment;
            moment = 0.0;
            space_vector center;
            for (integer d = 0; d < NDIM; ++d) {
                center[d] = (list[s].x[d] + 0.5) * cell_width;
                if (s == selected) {
                    moment[d + 1] = dipole[d];
                }
            }
            data.M->push_back(moment);
            data.x->push_back(center);
        }
        target->compute_boundary_interactions(DRHODT, direction, false, data);
        auto const & centers = *target->get_com_ptr()[0];
        size_t interacting = 0;
        for (integer i = 0; i < INX; ++i) {
            for (integer j = 0; j < INX; ++j) {
                for (integer k = 0; k < INX; ++k) {
                    std::array<integer, NDIM> const coordinates{i, j, k};
                    integer fine_distance2 = 0;
                    integer coarse_distance2 = 0;
                    for (integer d = 0; d < NDIM; ++d) {
                        integer const source = static_cast<integer>(list[selected].x[d]);
                        integer const fine = coordinates[d] - source;
                        integer const coarse = coordinates[d] / 2 - ((source + INX) / 2 - INX / 2);
                        fine_distance2 += fine * fine;
                        coarse_distance2 += coarse * coarse;
                    }
                    integer const index = gindex(i, j, k);
                    std::array<real, 4> expected{};
                    if (coarse_distance2 < 4 && (leaf || fine_distance2 >= 4)) {
                        expected = dipole_field(centers[index], (*data.x)[selected], dipole);
                        ++interacting;
                    }
                    for (size_t d = 0; d < expected.size(); ++d) {
                        check(target->get_L()[index][d], expected[d], leaf
                                ? "legacy multipole-monopole dipole"
                                : "legacy multipole-multipole dipole");
                    }
                }
            }
        }
        if (interacting == 0) {
            std::cerr << "Boundary fixture produced no dipole interactions\n";
            ++failures;
        }
    }
}

int hpx_main(hpx::program_options::variables_map &) {
    opts().n_species = 1;
    physics<NDIM>::set_n_species(1);
    physics<NDIM>::update_n_field();
    opts().n_fields = physics<NDIM>::field_count();
    opts().radiation = false;
    opts().gravity = true;
    opts().theta = 0.5;
    compute_ilist();
    compute_factor();
    grid::static_init();

    interior();
    boundary(false);
    boundary(true);
    std::cout << checks << " legacy dipole checks, " << failures << " failures\n";
    hpx::finalize();
    return failures == 0 ? 0 : 1;
}

int main(int argc, char * argv[]) {
    hpx::init_params parameters;
    parameters.cfg = {"hpx.os_threads=1"};
    return hpx::init(argc, argv, parameters);
}
