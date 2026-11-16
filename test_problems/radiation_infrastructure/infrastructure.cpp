// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "octotiger/radiation/amr.hpp"
#include "octotiger/radiation/subcycle.hpp"

namespace {
void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
template <class Function>
void rejects(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (std::invalid_argument const&) {
        rejected = true;
    }
    require(rejected, "Invalid input was accepted");
}
}    // namespace
int main() {
    using state = std::array<double, 4>;
    state const center{10, -3, 2, -5};
    std::array<state, 3> slopes{{{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}}};
    auto const children =
        radiation::interpolate_children(center, slopes, radiation::finite_nonnegative_energy<4>);
    for (size_t field = 0; field < center.size(); ++field) {
        double average = 0;
        for (auto const& child : children) {
            average += child[field] / 8;
        }
        require(average == center[field], "Affine interpolation lost conservation");
        require(children[0][field] ==
                center[field] - slopes[0][field] - slopes[1][field] - slopes[2][field],
            "Affine interpolation changed a slope");
    }
    state const small{0.125, -3, 2, -5};
    auto const limited =
        radiation::interpolate_children(small, slopes, radiation::finite_nonnegative_energy<4>);
    for (size_t field = 0; field < small.size(); ++field) {
        double average = 0;
        for (auto const& child : limited) {
            require(child[0] >= 0, "Negative interpolated energy");
            average += child[field] / 8;
        }
        require(average == small[field], "Slope limiting lost conservation");
    }
    state const vacuum{};
    auto const empty =
        radiation::interpolate_children(vacuum, slopes, radiation::finite_nonnegative_energy<4>);
    for (auto const& child : empty) {
        require(child == vacuum, "Vacuum not preserved");
    }
    require(radiation::limited_slope(-2, 3) == 0, "Extremum not limited");
    require(radiation::limited_slope(3, 3) == 3, "Linear slope not preserved");
    rejects([&] {
        auto bad = center;
        bad[0] = -1;
        radiation::interpolate_children(bad, slopes, radiation::finite_nonnegative_energy<4>);
    });
    rejects([&] {
        auto bad = slopes;
        bad[0][0] = std::numeric_limits<double>::infinity();
        radiation::interpolate_children(center, bad, radiation::finite_nonnegative_energy<4>);
    });
    for (double dt : {0.0, 0.1, 1.0e-200, 1.0e200}) {
        for (int count : {1, 3, 7, 1009}) {
            double previous = 0;
            for (int index = 0; index < count; ++index) {
                auto const interval = radiation::substep(dt, count, index);
                require(interval.begin == previous, "Substep gap or overlap");
                require(interval.duration() >= 0, "Negative interval");
                previous = interval.end;
            }
            require(previous == dt, "Substeps missed endpoint");
        }
    }
    require(radiation::subcycle_count(0) == 1, "Zero interval count");
    require(radiation::subcycle_count(2.01) == 3, "CFL count rounded down");
    rejects([] { radiation::subcycle_count(-1); });
    rejects([] { radiation::subcycle_count(std::numeric_limits<double>::infinity()); });
    rejects([] { radiation::subcycle_count(std::numeric_limits<double>::quiet_NaN()); });
    rejects([] { radiation::subcycle_count(1.0e20); });
    rejects([] { radiation::substep(1, 0, 0); });
    rejects([] { radiation::substep(1, 3, 3); });
    std::cout << "Conservative interpolation and subcycle checks passed\n";
}
