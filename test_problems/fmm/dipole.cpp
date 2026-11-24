// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp"

namespace {
    using vector_type = std::array<double, 3>;
    using exponent_type = std::array<int, 3>;

    size_t checks = 0;
    size_t failures = 0;

    void check(char const * label, double actual, double expected, double scale = 0) {
        ++checks;
        double const tolerance = 3.e-12
            * std::max({std::abs(actual), std::abs(expected), scale, 1.e-8});
        if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
            if (failures < 12) {
                std::printf("%s: actual %.17g, expected %.17g\n", label, actual, expected);
            }
            ++failures;
        }
    }

    double factorial(int n) {
        return n < 2 ? 1 : n == 2 ? 2 : 6;
    }

    // Analytic derivatives of -1/r, independent of the production derivative generator.
    double derivative(vector_type const & r, exponent_type const & exponent) {
        double const r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        double const radius = std::sqrt(r2);
        int axes[3] = {};
        int degree = 0;
        for (int a = 0; a < 3; ++a) {
            for (int n = 0; n < exponent[a]; ++n) {
                axes[degree++] = a;
            }
        }
        if (degree == 0) {
            return -1 / radius;
        }
        int const i = axes[0];
        if (degree == 1) {
            return r[i] / (r2 * radius);
        }
        int const j = axes[1];
        if (degree == 2) {
            return (i == j ? 1.0 : 0.0) / (r2 * radius)
                - 3 * r[i] * r[j] / (r2 * r2 * radius);
        }
        int const k = axes[2];
        return 15 * r[i] * r[j] * r[k] / (r2 * r2 * r2 * radius)
            - 3 * ((i == j ? r[k] : 0) + (i == k ? r[j] : 0) + (j == k ? r[i] : 0))
                / (r2 * r2 * radius);
    }

    double evaluate(double const * local, vector_type const & offset,
        std::vector<exponent_type> const & exponents) {
        double value = 0;
        for (size_t i = 0; i < exponents.size(); ++i) {
            double term = local[i];
            for (int a = 0; a < 3; ++a) {
                term *= std::pow(offset[a], exponents[i][a]) / factorial(exponents[i][a]);
            }
            value += term;
        }
        return value;
    }
}

int main() {
    std::vector<exponent_type> exponents;
    for (int degree = 0; degree < 4; ++degree) {
        for (int x = degree; x >= 0; --x) {
            for (int y = degree - x; y >= 0; --y) {
                exponents.push_back({x, y, degree - x - y});
            }
        }
    }
    std::vector<vector_type> const separations = {
        {1, 2, 3}, {-2, 1, .7}, {.4, -1.1, 2.3}, {-.5, -2, -3},
        {.02, .03, -.04}, {11, -17, 23}};
    std::vector<vector_type> const dipoles = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {.7, -1.2, .3}, {-1.3, .2, -.8}};
    std::vector<vector_type> const points = {
        {-.3, .1, .2}, {.1, -.4, .3}, {.2, .3, -.1}, {.4, -.1, .1}};
    std::vector<double> const masses = {1, 2, 3, 4};
    vector_type center = {};
    for (size_t i = 0; i < points.size(); ++i) {
        for (int a = 0; a < 3; ++a) {
            center[a] += masses[i] * points[i][a] / 10;
        }
    }
    auto const maximum = [](double a, double b) { return std::max(a, b); };
    for (vector_type const & separation : separations) {
        for (vector_type const & dipole : dipoles) {
            for (double mass : {0.0, -.7}) {
                double target[3] = {separation[0], separation[1], separation[2]};
                double source[3] = {};
                double moments[20] = {mass, dipole[0], dipole[1], dipole[2]};
                double local[20] = {}, particle[4] = {}, reference[20] = {};
                octotiger::fmm::multipole_interactions::compute_kernel_non_rho(
                    target, source, moments, local, maximum);
                octotiger::fmm::monopole_interactions::compute_kernel_p2m_non_rho(
                    target, source, moments, particle, maximum);
                for (size_t i = 0; i < exponents.size(); ++i) {
                    reference[i] = mass * derivative(separation, exponents[i]);
                    double scale = std::abs(reference[i]);
                    int const degree = exponents[i][0] + exponents[i][1] + exponents[i][2];
                    if (degree < 3) {
                        for (int a = 0; a < 3; ++a) {
                            exponent_type shifted = exponents[i];
                            ++shifted[a];
                            double const term = dipole[a] * derivative(separation, shifted);
                            reference[i] -= term;
                            scale += std::abs(term);
                        }
                    }
                    check("M2L coefficient", local[i], reference[i], scale);
                    if (i < 4) {
                        check("P2M coefficient", particle[i], reference[i], scale);
                    }
                }
                double weighted_error = 0;
                double weighted_absolute_error = 0;
                for (size_t i = 0; i < points.size(); ++i) {
                    vector_type offset;
                    for (int a = 0; a < 3; ++a) {
                        offset[a] = points[i][a] - center[a];
                    }
                    double const actual = evaluate(local, offset, exponents);
                    double const expected = evaluate(reference, offset, exponents);
                    check("translated potential", actual, expected);
                    weighted_error += masses[i] * (actual - expected);
                    weighted_absolute_error += masses[i] * std::abs(actual - expected);
                }
                check("mass weighted error", weighted_error, 0,
                    weighted_absolute_error + std::abs(local[0]));
            }
        }
    }
    std::printf("%zu checks, %zu failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
