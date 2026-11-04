// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <hpx/hpx_init.hpp>
#ifdef OCTOTIGER_HAVE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#include "octotiger/radiation/rad_grid.hpp"

namespace {
void equal(double actual, double expected) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1.0e-11) {
        throw std::runtime_error("Radiation transfer mismatch: " + std::to_string(actual) +
            " != " + std::to_string(expected));
    }
}
double affine(int field, double i, double j, double k) {
    return (field == 0 ? 100.0 : -100.0 * field) + i + 2 * j + 3 * k;
}
void fill(rad_grid& grid) {
    for (int f = 0; f < NRF; ++f) {
        for (int i = 0; i < RAD_NX; ++i) {
            for (int j = 0; j < RAD_NX; ++j) {
                for (int k = 0; k < RAD_NX; ++k) {
                    grid.set_field(affine(f, i, j, k), f, i, j, k);
                }
            }
        }
    }
}
void transfers() {
    rad_grid source(1), target(1);
    fill(source);
    // All 26 same-level face, edge and corner messages use the opposite halo.
    for (auto const& dir : geo::direction::full_set()) {
        target.set_boundary(source.get_boundary(dir), dir.flip());
        std::array<integer, NDIM> lb, ub, src_lb, src_ub;
        get_boundary_size(lb, ub, dir.flip(), OUTER, INX, RAD_BW);
        get_boundary_size(src_lb, src_ub, dir, INNER, INX, RAD_BW);
        for (int f = 0; f < NRF; ++f) {
            for (int i = lb[0]; i < ub[0]; ++i) {
                for (int j = lb[1]; j < ub[1]; ++j) {
                    for (int k = lb[2]; k < ub[2]; ++k) {
                        equal(target.get_field(f, i, j, k),
                            affine(f, i - lb[0] + src_lb[0], j - lb[1] + src_lb[1],
                                k - lb[2] + src_lb[2]));
                    }
                }
            }
        }
    }
    // Creation prolongation is piecewise constant. Restricting all eight new
    // children must recover every parent interior value, including signed fields.
    for (auto const& octant : geo::octant::full_set()) {
        std::array<integer, NDIM> lb, ub;
        for (int d = 0; d < NDIM; ++d) {
            lb[d] = 2 * RAD_BW + octant.get_side(d) * INX;
            ub[d] = lb[d] + INX;
        }
        rad_grid child(0.5);
        child.set_prolong(source.get_prolong(lb, ub));
        target.set_restrict(child.get_restrict(), octant);
    }
    for (int f = 0; f < NRF; ++f) {
        for (int i = RAD_BW; i < RAD_BW + INX; ++i) {
            for (int j = RAD_BW; j < RAD_BW + INX; ++j) {
                for (int k = RAD_BW; k < RAD_BW + INX; ++k) {
                    equal(target.get_field(f, i, j, k), affine(f, i, j, k));
                }
            }
        }
    }
    // Refluxing uses area averages of four fine faces, for every orientation.
    for (auto const& dimension : geo::dimension::full_set()) {
        std::array<integer, NDIM> const all_lb{0, 0, 0};
        std::array<integer, NDIM> const all_ub{RAD_NX, RAD_NX, RAD_NX};
        source.set_flux_restrict(source.get_subset(all_lb, all_ub), all_lb, all_ub, dimension);
        for (int side = 0; side < 2; ++side) {
            std::array<integer, NDIM> lb{RAD_BW, RAD_BW, RAD_BW};
            std::array<integer, NDIM> ub{RAD_BW + INX, RAD_BW + INX, RAD_BW + INX};
            lb[dimension] += side * INX;
            ub[dimension] = lb[dimension] + 1;
            auto const data = source.get_flux_restrict(lb, ub, dimension);
            size_t index = 0;
            for (int f = 0; f < NRF; ++f) {
                for (int i = lb[0]; i < ub[0]; i += 2) {
                    for (int j = lb[1]; j < ub[1]; j += 2) {
                        for (int k = lb[2]; k < ub[2]; k += 2) {
                            equal(data.at(index++),
                                affine(f, i + (dimension == XDIM ? 0 : 0.5),
                                    j + (dimension == YDIM ? 0 : 0.5),
                                    k + (dimension == ZDIM ? 0 : 0.5)));
                        }
                    }
                }
            }
            if (index != data.size()) {
                throw std::runtime_error("Wrong reflux message length");
            }
        }
    }
    // Overlapping coarse envelopes exercise the production shadow packing and
    // interpolation for every child orientation and all halo directions.
    for (auto const& octant : geo::octant::full_set()) {
        rad_grid child(0.5);
        fill(child);
        child.clear_amr();
        for (auto const& dir : geo::direction::full_set()) {
            std::array<integer, NDIM> lb, ub;
            get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
            for (int d = 0; d < NDIM; ++d) {
                lb[d] = std::max(lb[d] - 1, integer(0)) + octant.get_side(d) * (INX / 2);
                ub[d] = std::min(ub[d] + 1, integer(HS_NX)) + octant.get_side(d) * (INX / 2);
            }
            child.set_rad_amr_boundary(source.get_subset(lb, ub), dir);
        }
        child.complete_rad_amr_boundary();
        for (int f = 0; f < NRF; ++f) {
            for (int i = 0; i < RAD_NX; ++i) {
                for (int j = 0; j < RAD_NX; ++j) {
                    for (int k = 0; k < RAD_NX; ++k) {
                        if (i >= RAD_BW && i < RAD_BW + INX && j >= RAD_BW && j < RAD_BW + INX &&
                            k >= RAD_BW && k < RAD_BW + INX) {
                            equal(child.get_field(f, i, j, k), affine(f, i, j, k));
                            continue;
                        }
                        auto position = [&](int x, int d) {
                            return RAD_BW + (x - RAD_BW + 0.5) * 0.5 - 0.5 +
                                octant.get_side(d) * (INX / 2);
                        };
                        equal(child.get_field(f, i, j, k),
                            affine(f, position(i, 0), position(j, 1), position(k, 2)));
                    }
                }
            }
        }
    }
    bool rejected = false;
    try {
        target.set_prolong({});
    } catch (std::invalid_argument const&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("Truncated prolongation accepted");
    }
}
}    // namespace
int hpx_main() {
#ifdef OCTOTIGER_HAVE_KOKKOS
    Kokkos::initialize();
#endif
    int result = 0;
    try {
        transfers();
        std::cout << "Production radiation grid transfer checks passed\n";
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
#ifdef OCTOTIGER_HAVE_KOKKOS
    Kokkos::finalize();
#endif
    hpx::finalize();
    return result;
}
int main(int argc, char** argv) {
    return hpx::init(argc, argv);
}
