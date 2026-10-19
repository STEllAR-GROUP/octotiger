// Copyright (c) 2026 AUTHORS
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "octotiger/test_problems/wd_equilibrium.hpp"

namespace {

using octotiger::test_problems::one_solar_mass_wd;
using octotiger::test_problems::wd_equilibrium_model;

constexpr double pi = 3.141592653589793238462643383279502884;
int failures = 0;

void check(bool condition, char const *description) {
    if (!condition) {
        std::cerr << "FAIL: " << description << '\n';
        ++failures;
    }
}

void check_close(double actual, double expected, double relative_tolerance,
    char const *description) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > relative_tolerance * std::abs(expected)) {
        std::cerr << "FAIL: " << description << ": expected " << expected
                  << ", got " << actual << '\n';
        ++failures;
    }
}

double integrate_mass(wd_equilibrium_model const &model, double radius) {
    // Independent composite Simpson integration of the public density profile.
    constexpr size_t intervals = 8192;
    double const step = radius / intervals;
    double sum = radius * radius * model.density_at(radius);
    for (size_t i = 1; i < intervals; ++i) {
        double const r = i * step;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * r * r * model.density_at(r);
    }
    return 4.0 * pi * step * sum / 3.0;
}

long double pressure(wd_equilibrium_model const &model, double radius) {
    long double const density = model.density_at(radius);
    long double const x = std::cbrt(density / model.code_b());
    long double const x_squared = x * x;
    long double const a = model.code_a();
    // An independent pressure formula tests dP/dr = -rho M(r)/r^2.
    // The series avoids subtractive cancellation in dilute surface layers.
    if (x < 0.05L) {
        return a * x * x_squared * x_squared * (1.6L + x_squared *
            (-4.0L / 7.0L + x_squared * (1.0L / 3.0L + x_squared *
                (-5.0L / 22.0L + x_squared * 35.0L / 208.0L))));
    }
    return a * (x * (2.0L * x_squared - 3.0L) * std::sqrt(1.0L + x_squared) +
        3.0L * std::asinh(x));
}

void check_profile(wd_equilibrium_model const &model) {
    check_close(model.mass_cgs(), 1.989e33, 1.0e-15, "one solar mass in grams");
    check(model.radius_cgs() > 5.5e8 && model.radius_cgs() < 5.7e8,
        "physical radius sanity check");
    double const density_unit = model.mass_cgs() / std::pow(model.radius_cgs(), 3);
    check(model.central_density() * density_unit > 3.0e7 &&
            model.central_density() * density_unit < 3.3e7,
        "central physical density sanity check");
    check_close(model.code_b() * density_unit, 2.0 * 9.81011e5, 2.0e-14,
        "physical EOS density constant");
    double const pressure_unit = 6.67259e-8 * model.mass_cgs() * model.mass_cgs() /
        std::pow(model.radius_cgs(), 4);
    check_close(model.code_a() * pressure_unit, 6.00228e22, 2.0e-14,
        "physical EOS pressure constant");
    check_close(model.density_at(0.0), model.central_density(), 2.0e-14,
        "regular central density");
    check(model.enclosed_mass_at(0.0) == 0.0, "zero central enclosed mass");
    check(model.enthalpy_at(1.0) == 0.0 && model.density_at(1.0) == 0.0,
        "zero enthalpy and density at the surface");
    check(model.density_at(2.0) == 0.0 && model.enclosed_mass_at(2.0) == 1.0,
        "exterior vacuum and enclosed mass");

    double previous_density = model.density_at(0.0);
    double previous_mass = 0.0;
    for (size_t i = 1; i <= 10000; ++i) {
        double const radius = i / 10000.0;
        double const density = model.density_at(radius);
        double const mass = model.enclosed_mass_at(radius);
        check(std::isfinite(density) && density >= 0.0 && density <= previous_density,
            "monotone nonnegative density");
        check(std::isfinite(mass) && mass >= previous_mass && mass <= 1.0,
            "monotone bounded enclosed mass");
        previous_density = density;
        previous_mass = mass;
    }
    for (double const radius : {0.2, 0.5, 0.8, 0.99, 1.0}) {
        check_close(integrate_mass(model, radius), model.enclosed_mass_at(radius),
            2.0e-8, "independent integrated mass");
    }
    double maximum_hse_residual = 0.0;
    for (double const radius : {0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99}) {
        constexpr double difference_step = 1.0e-5;
        double const pressure_derivative = static_cast<double>(
            (pressure(model, radius + difference_step) - pressure(model, radius - difference_step)) /
            (2.0L * difference_step));
        double const gravity_force = -model.density_at(radius) * model.enclosed_mass_at(radius) /
            (radius * radius);
        maximum_hse_residual = std::max(maximum_hse_residual,
            std::abs(pressure_derivative / gravity_force - 1.0));
        check_close(pressure_derivative, gravity_force, 5.0e-6,
            "independent hydrostatic pressure gradient");
    }
    std::cout << "mass=" << integrate_mass(model, 1.0)
              << " radius_cgs=" << model.radius_cgs()
              << " central_density_cgs=" << model.central_density() * density_unit
              << " max_hse_relative_residual=" << maximum_hse_residual << '\n';
}

void check_convergence(wd_equilibrium_model const &model) {
    wd_equilibrium_model const coarse(2.0 * model.radial_step());
    wd_equilibrium_model const fine(0.5 * model.radial_step());
    check(fine.sample_count() > model.sample_count() &&
            model.sample_count() > coarse.sample_count(), "radial resolution refinement");
    check_close(model.radius_cgs(), fine.radius_cgs(), 2.0e-8, "half-step radius convergence");
    check_close(model.central_density(), fine.central_density(), 2.0e-8,
        "half-step central-density convergence");
    double const coarse_error = std::abs(coarse.radius_cgs() / fine.radius_cgs() - 1.0);
    double const standard_error = std::abs(model.radius_cgs() / fine.radius_cgs() - 1.0);
    check(standard_error <= coarse_error + 1.0e-12,
        "radius error decreases or reaches rounding precision");
    double maximum_density_change = 0.0;
    for (double const radius : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
        double const density = model.density_at(radius);
        double const finer_density = fine.density_at(radius);
        check_close(density, finer_density, 2.0e-7, "half-step density-profile convergence");
        maximum_density_change = std::max(maximum_density_change,
            std::abs(density / finer_density - 1.0));
    }
    std::cout << "coarse_radius_relative_error=" << coarse_error
              << " half_step_radius_relative_error=" << standard_error
              << " max_half_step_density_change=" << maximum_density_change << '\n';
}

} // namespace

int main() {
    std::cout << std::setprecision(15);
    try {
        wd_equilibrium_model const &model = one_solar_mass_wd();
        check(&model == &one_solar_mass_wd(), "cached immutable equilibrium model");
        check_profile(model);
        check_convergence(model);
        bool rejected = false;
        try {
            wd_equilibrium_model const invalid(0.0);
        } catch (std::invalid_argument const &) {
            rejected = true;
        }
        check(rejected, "reject a zero radial integration step");
        rejected = false;
        try {
            model.density_at(std::numeric_limits<double>::quiet_NaN());
        } catch (std::invalid_argument const &) {
            rejected = true;
        }
        check(rejected, "reject a NaN sampling radius");
    } catch (std::exception const &error) {
        std::cerr << "WD model test threw: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " WD equilibrium model checks failed\n";
        return 1;
    }
    std::cout << "WD equilibrium model checks passed\n";
    return 0;
}
