// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
// Standalone: c++ -std=c++20 -O2 -I. tests/radiation/test_marshak_boundary.cpp -o /tmp/test_marshak_boundary
#include "octotiger/radiation/marshak_boundary.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
size_t checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, long double expected, long double scale, const char* message,
          long double tolerance = 256 * std::numeric_limits<double>::epsilon()) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * scale, message);
}

void limiting_states() {
    const auto cold = radiation::marshakHalfSpaceGhost(0., 0., 4.);
    near(cold[0], 2, 2, "cold bath injects incoming hemispheric energy");
    near(cold[1], 1, 1, "cold bath has incident normalized flux one");
    const auto warm = radiation::marshakHalfSpaceGhost(4., 0., 4.);
    near(warm[0], 4, 4, "isotropic equilibrium energy is preserved");
    near(warm[1], 0, 4, "isotropic equilibrium has zero net flux");
    for (double energy : {1.e-20, .1, 4., 100., 1.e12}) {
        const auto incoming = radiation::marshakHalfSpaceGhost(energy, energy, 4.);
        near(incoming[0], 2, 2, "interior incoming beam adds no outgoing energy");
        near(incoming[1], 1, 1, "interior incoming beam adds no outgoing current");
        const auto outgoing = radiation::marshakHalfSpaceGhost(energy, -energy, 4.);
        near(outgoing[0], 2 + static_cast<long double>(energy), 2 + energy,
             "outgoing beam energy is retained");
        near(outgoing[1], 1 - static_cast<long double>(energy), 2 + energy,
             "strong outgoing beam may produce negative net boundary flux");
        for (double sign : {-1., 1.}) {
            const auto close = radiation::marshakHalfSpaceGhost(energy, sign * energy * (1 - 1.e-12), 4.);
            require(close[0] > 0 && std::abs(close[1]) <= close[0], "near-streaming ghost is realizable");
        }
    }
}

void independent_angular_quadrature() {
    // Integrate the nonnegative angular intensity numerically, including its
    // normalization and first moment. This checks the closed half-integrals
    // without copying their polynomial expressions into the oracle.
    constexpr size_t intervals = 32768;
    for (long double beta : {-.9L, -.7L, -.2L, 0.L, .2L, .7L, .9L}) {
        std::array<long double, 2> all{}, outgoing{};
        for (size_t i = 0; i <= intervals; ++i) {
            const long double mu = -1 + 2.L * i / intervals;
            const long double denominator = 1 - beta * mu;
            const long double value = 1 / (denominator * denominator * denominator * denominator);
            const long double weight = i == 0 || i == intervals ? 1 : (i % 2 ? 4 : 2);
            all[0] += weight * value;
            all[1] += weight * mu * value;
            if (i <= intervals / 2) {
                const long double half_weight = i == 0 || i == intervals / 2 ? 1 : (i % 2 ? 4 : 2);
                outgoing[0] += half_weight * value;
                outgoing[1] += half_weight * mu * value;
            }
        }
        constexpr double energy = 3.7, bath = 4;
        const double q = static_cast<double>(energy * all[1] / all[0]);
        const auto ghost = radiation::marshakHalfSpaceGhost(energy, q, bath);
        near(ghost[0], bath / 2 + energy * outgoing[0] / all[0], energy + bath,
             "ghost energy matches direct half-space intensity quadrature", 1.e-11);
        near(ghost[1], bath / 4 + energy * outgoing[1] / all[0], energy + bath,
             "ghost flux matches direct half-space intensity quadrature", 1.e-11);
    }
}

void realizability_and_diffusive_limit() {
    for (double energy : {1.e-20, 1.e-6, .1, 4., 100., 1.e12}) {
        for (int i = 0; i <= 1000; ++i) {
            const double f = -1 + 2. * i / 1000;
            const auto ghost = radiation::marshakHalfSpaceGhost(energy, f * energy, 4.);
            require(std::isfinite(ghost[0]) && std::isfinite(ghost[1])
                    && ghost[0] > 0 && std::abs(ghost[1]) <= ghost[0],
                    "nonnegative half-space assembly preserves the realizable cone");
        }
    }
    // P1 half-moment expansion of I(mu)=E/(4pi)+3Q*mu/(4pi):
    // Eout=E/2-3Q/4+O(f^2 E), Qout=-E/4+Q/2+O(f^2 E).
    // This checks the thick, nearly isotropic limit; it does not claim that
    // an M1/HLL face flux enforces the diffusion Robin condition exactly.
    for (double f : {-1.e-4, 1.e-4, -1.e-6, 1.e-6}) {
        constexpr double energy = 4, bath = 4;
        const double q = f * energy;
        const auto ghost = radiation::marshakHalfSpaceGhost(energy, q, bath);
        near(ghost[0], bath / 2 + energy / 2 - 3 * q / 4, energy,
             "outgoing energy tends to the P1 half moment", 2 * f * f);
        near(ghost[1], bath / 4 - energy / 4 + q / 2, energy,
             "outgoing flux tends to the P1 half moment", 2 * f * f);
    }
    for (const auto input : {std::array<double, 3>{1., 1.01, 4.}, {0., 1., 4.}, {-1., 0., 4.}}) {
        bool rejected = false;
        try {
            static_cast<void>(radiation::marshakHalfSpaceGhost(input[0], input[1], input[2]));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid interior moments are rejected rather than clipped");
    }
}
} // namespace

int main() {
    try {
        limiting_states();
        independent_angular_quadrature();
        realizability_and_diffusive_limit();
        std::cout << "PASS: " << checks << " Marshak half-space boundary checks\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
