// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "octotiger/fld_closure.hpp"

namespace {

using position = std::array<double, 3>;

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance, std::string const& message) {
    require(std::isfinite(actual) &&
            std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)),
        message + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

template <class Function>
void rejects(Function function, std::string const& message) {
    bool rejected = false;
    try {
        function();
    } catch (std::exception const&) {
        rejected = true;
    }
    require(rejected, message);
}

void testOpacityAndClosure() {
    using octotiger::fld::absorption;
    using octotiger::fld::closure;
    near(absorption(4.0, 2.0, 3.0, 1.5, -2.0), 6.0, 2.0e-15,
        "opacity uses the stated density and temperature exponents without another rho");
    near(absorption(2.0, 0.25, 0.0, -2.0, -3.5), 0.0, 0.0,
        "zero absorption remains zero for a scattering-only material");
    rejects([&] { absorption(1.0, 0.0, 1.0, 0.0, -3.5); },
        "zero temperature cannot enter a singular power-law opacity");
    rejects(
        [&] { absorption(-1.0, 1.0, 1.0, 1.0, 0.0); }, "negative density cannot enter the opacity");
    rejects(
        [&] { absorption(1.0e200, 1.0, 1.0, 2.0, 0.0); }, "overflowing opacity must be reported");

    double const energy = 3.0;
    auto const isotropic = closure(energy, {{0.0, 0.0, 0.0}}, 2.0);
    near(isotropic.lambda, 1.0 / 3.0, 0.0, "zero-gradient diffusion limit");
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            near(isotropic.pressure[row][column], row == column ? 1.0 : 0.0, 2.0e-15,
                "isotropic pressure tensor");
        }
    }
    position const direction{{0.6, -0.8, 0.0}};
    for (double const r : {1.0e-100, 1.0e-3, 0.1, 1.0, 10.0, 1.0e3, 1.0e100}) {
        position gradient{};
        for (size_t axis = 0; axis < 3; ++axis) {
            gradient[axis] = r * energy * direction[axis];
        }
        auto const result = closure(energy, gradient, 1.0);
        require(result.lambda >= 0.0 && result.lambda <= 1.0 / 3.0 + 1.0e-15,
            "flux limiter leaves its physical interval");
        require(result.lambda * r <= 1.0 + 1.0e-15, "FLD diffusion exceeds the streaming flux cE");
        require(result.eddington >= 1.0 / 3.0 - 1.0e-15 && result.eddington <= 1.0 + 1.0e-15,
            "Eddington factor leaves its physical interval");
        double trace = 0.0;
        double normalPressure = 0.0;
        for (size_t row = 0; row < 3; ++row) {
            trace += result.pressure[row][row];
            for (size_t column = 0; column < 3; ++column) {
                near(result.pressure[row][column], result.pressure[column][row], 2.0e-15,
                    "radiation pressure is symmetric");
                normalPressure += direction[row] * result.pressure[row][column] * direction[column];
            }
        }
        near(trace, energy, 3.0e-15, "radiation pressure trace equals radiation energy");
        near(normalPressure, result.eddington * energy, 3.0e-15,
            "gradient-direction pressure has the Eddington eigenvalue");
    }
    auto const transparent = closure(energy, direction, 0.0);
    near(transparent.lambda, 0.0, 0.0, "zero extinction has a finite streaming limiter");
    near(transparent.eddington, 1.0, 0.0, "zero extinction approaches a directed beam");
    rejects([&] { closure(-1.0, direction, 1.0); }, "negative radiation energy is rejected");
    rejects([&] { closure(1.0, {{0.0, std::numeric_limits<double>::infinity(), 0.0}}, 1.0); },
        "nonfinite radiation gradient is rejected");
}

} // namespace

int main() {
    try {
        testOpacityAndClosure();
        std::cout << "FLD closure tests passed\n";
    } catch (std::exception const& error) {
        std::cerr << "FLD numerical test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
