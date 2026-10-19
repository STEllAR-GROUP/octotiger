// Copyright (c) 2026 AUTHORS
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "octotiger/test_problems/wd_equilibrium.hpp"

namespace octotiger::test_problems {
namespace {

// Keep these consistent with the WD normalization in src/physcon.cpp.
constexpr double eos_a_cgs = 6.00228e22;
constexpr double eos_b_cgs = 2.0 * 9.81011e5;
constexpr double gravity_cgs = 6.67259e-8;
constexpr double pi = 3.141592653589793238462643383279502884;

struct state {
    double enthalpy;
    double mass;
};

struct natural_sample {
    double radius;
    state value;
};

struct integration_result {
    double radius;
    double mass;
    std::vector<natural_sample> profile;
};

// Here A = B = G = 1, so h = 8 (sqrt(1 + x^2) - 1) and rho = x^3.
// Factoring the inverse avoids subtracting nearly equal numbers near h = 0.
double natural_density(double enthalpy) {
    if (enthalpy <= 0.0) {
        return 0.0;
    }
    double const q = enthalpy / 8.0;
    double const x_squared = q * (q + 2.0);
    return x_squared * std::sqrt(x_squared);
}

state derivative(double radius, state const &value) {
    return {-value.mass / (radius * radius),
        4.0 * pi * radius * radius * natural_density(value.enthalpy)};
}

state advance(double radius, state const &value, double step) {
    state const k1 = derivative(radius, value);
    state const k2 = derivative(radius + 0.5 * step,
        {value.enthalpy + 0.5 * step * k1.enthalpy, value.mass + 0.5 * step * k1.mass});
    state const k3 = derivative(radius + 0.5 * step,
        {value.enthalpy + 0.5 * step * k2.enthalpy, value.mass + 0.5 * step * k2.mass});
    state const k4 = derivative(radius + step,
        {value.enthalpy + step * k3.enthalpy, value.mass + step * k3.mass});
    return {value.enthalpy + step *
            (k1.enthalpy + 2.0 * k2.enthalpy + 2.0 * k3.enthalpy + k4.enthalpy) / 6.0,
        value.mass + step * (k1.mass + 2.0 * k2.mass + 2.0 * k3.mass + k4.mass) / 6.0};
}

integration_result integrate(double central_x, double radial_step, bool save_profile) {
    double const central_enthalpy =
        8.0 * central_x * central_x / (std::sqrt(1.0 + central_x * central_x) + 1.0);
    double const central_density = central_x * central_x * central_x;
    double const density_derivative =
        3.0 / 8.0 * central_x * std::sqrt(1.0 + central_x * central_x);

    // Regular central expansion: h = h_c + a r^2 + b r^4 + O(r^6).
    // Starting RK4 at the first positive radius avoids its singular m/r^2 RHS.
    double const a = -2.0 * pi * central_density / 3.0;
    double const b = 2.0 * pi * pi * central_density * density_derivative / 15.0;
    double radius = radial_step;
    double const radius_squared = radius * radius;
    state value{central_enthalpy + radius_squared * (a + b * radius_squared),
        radius * radius_squared * (-2.0 * a - 4.0 * b * radius_squared)};
    if (!(value.enthalpy > 0.0 && value.mass > 0.0)) {
        throw std::runtime_error("WD radial step does not resolve the central expansion");
    }

    integration_result result{};
    if (save_profile) {
        result.profile.push_back({0.0, {central_enthalpy, 0.0}});
        result.profile.push_back({radius, value});
    }
    for (size_t step = 0; step < 1000000; ++step) {
        state const next = advance(radius, value, radial_step);
        if (!std::isfinite(next.enthalpy) || !std::isfinite(next.mass) ||
            next.enthalpy >= value.enthalpy || next.mass < value.mass) {
            throw std::runtime_error("WD radial integration lost a finite monotone solution");
        }
        if (next.enthalpy <= 0.0) {
            // Locate the surface using the same RK4 evolution over a shortened
            // final step. Linear interpolation of the overshoot is less accurate.
            double lower = 0.0;
            double upper = radial_step;
            for (size_t iteration = 0; iteration < 48; ++iteration) {
                double const midpoint = 0.5 * (lower + upper);
                if (advance(radius, value, midpoint).enthalpy > 0.0) {
                    lower = midpoint;
                } else {
                    upper = midpoint;
                }
            }
            double const last_step = 0.5 * (lower + upper);
            state const surface = advance(radius, value, last_step);
            result.radius = radius + last_step;
            result.mass = surface.mass;
            if (save_profile) {
                result.profile.push_back({result.radius, {0.0, result.mass}});
            }
            return result;
        }
        radius += radial_step;
        value = next;
        if (save_profile) {
            result.profile.push_back({radius, value});
        }
    }
    throw std::runtime_error("WD radial integration did not reach its surface");
}

// Cubic Hermite interpolation with a sufficient monotonicity limiter. The
// supplied slopes are physical derivatives, except when limiting is necessary.
double interpolate(double position, double lower_position, double upper_position,
    double lower_value, double upper_value, double lower_slope, double upper_slope) {
    double const width = upper_position - lower_position;
    double const secant = (upper_value - lower_value) / width;
    if (secant == 0.0) {
        return lower_value;
    }
    double const alpha = lower_slope / secant;
    double const beta = upper_slope / secant;
    if (alpha < 0.0 || beta < 0.0) {
        lower_slope = 0.0;
        upper_slope = 0.0;
    } else {
        double const norm_squared = alpha * alpha + beta * beta;
        if (norm_squared > 9.0) {
            double const scale = 3.0 / std::sqrt(norm_squared);
            lower_slope *= scale;
            upper_slope *= scale;
        }
    }
    double const fraction = (position - lower_position) / width;
    double const fraction_squared = fraction * fraction;
    double const fraction_cubed = fraction_squared * fraction;
    return (2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0) * lower_value +
        (fraction_cubed - 2.0 * fraction_squared + fraction) * width * lower_slope +
        (-2.0 * fraction_cubed + 3.0 * fraction_squared) * upper_value +
        (fraction_cubed - fraction_squared) * width * upper_slope;
}

void check_radius(double radius) {
    if (!(radius >= 0.0)) {
        throw std::invalid_argument("WD radius must be nonnegative and not NaN");
    }
}

} // namespace

wd_equilibrium_model::wd_equilibrium_model(double radial_step) : radial_step_(radial_step) {
    if (!(radial_step > 0.0) || !std::isfinite(radial_step) || radial_step > 0.01) {
        throw std::invalid_argument("WD radial step must be finite and in (0, 0.01]");
    }
    double const natural_mass_cgs =
        std::pow(eos_a_cgs / gravity_cgs, 1.5) / (eos_b_cgs * eos_b_cgs);
    double const natural_length_cgs = std::sqrt(eos_a_cgs / gravity_cgs) / eos_b_cgs;
    double const target_mass = solar_mass_cgs / natural_mass_cgs;

    double lower_x = 0.1;
    double upper_x = 20.0;
    if (!(integrate(lower_x, radial_step, false).mass < target_mass &&
            integrate(upper_x, radial_step, false).mass > target_mass)) {
        throw std::runtime_error("WD central density does not bracket one solar mass");
    }
    double best_x = lower_x;
    double best_error = std::numeric_limits<double>::infinity();
    for (size_t iteration = 0; iteration < 64; ++iteration) {
        double const midpoint = 0.5 * (lower_x + upper_x);
        double const mass = integrate(midpoint, radial_step, false).mass;
        double const error = std::abs(mass - target_mass);
        if (error < best_error) {
            best_x = midpoint;
            best_error = error;
        }
        if (error <= 8.0 * std::numeric_limits<double>::epsilon() * target_mass ||
            midpoint == lower_x || midpoint == upper_x) {
            break;
        }
        if (mass < target_mass) {
            lower_x = midpoint;
        } else {
            upper_x = midpoint;
        }
    }
    integration_result const solution = integrate(best_x, radial_step, true);
    if (std::abs(solution.mass / target_mass - 1.0) > 1.0e-12) {
        throw std::runtime_error("WD mass calibration did not converge");
    }

    radius_cgs_ = solution.radius * natural_length_cgs;
    double const radius_squared = solution.radius * solution.radius;
    code_a_ = radius_squared * radius_squared / (target_mass * target_mass);
    code_b_ = radius_squared * solution.radius / target_mass;
    central_density_ = code_b_ * best_x * best_x * best_x;
    profile_.reserve(solution.profile.size());
    for (natural_sample const &point : solution.profile) {
        profile_.push_back({point.radius / solution.radius,
            point.value.enthalpy * solution.radius / target_mass,
            point.value.mass / target_mass});
    }
    // The shooting residual is at rounding precision. Make the documented
    // code-unit surface values exact without changing any interior densities.
    profile_.back() = {1.0, 0.0, 1.0};
}

double wd_equilibrium_model::mass_cgs() const {
    return solar_mass_cgs;
}

double wd_equilibrium_model::radius_cgs() const {
    return radius_cgs_;
}

double wd_equilibrium_model::code_a() const {
    return code_a_;
}

double wd_equilibrium_model::code_b() const {
    return code_b_;
}

double wd_equilibrium_model::central_density() const {
    return central_density_;
}

double wd_equilibrium_model::radial_step() const {
    return radial_step_;
}

size_t wd_equilibrium_model::sample_count() const {
    return profile_.size();
}

size_t wd_equilibrium_model::interval_at(double radius) const {
    auto const upper = std::upper_bound(profile_.begin(), profile_.end(), radius,
        [](double value, sample const &point) { return value < point.radius; });
    return static_cast<size_t>(upper - profile_.begin()) - 1;
}

double wd_equilibrium_model::density_from_enthalpy(double enthalpy) const {
    double const natural_enthalpy = enthalpy * code_b_ / code_a_;
    return code_b_ * natural_density(natural_enthalpy);
}

double wd_equilibrium_model::enthalpy_at(double radius) const {
    check_radius(radius);
    if (radius >= 1.0) {
        return 0.0;
    }
    size_t const index = interval_at(radius);
    sample const &lower = profile_[index];
    sample const &upper = profile_[index + 1];
    double const lower_slope = lower.radius == 0.0 ? 0.0 :
        -lower.enclosed_mass / (lower.radius * lower.radius);
    double const upper_slope = -upper.enclosed_mass / (upper.radius * upper.radius);
    return std::max(0.0, interpolate(radius, lower.radius, upper.radius,
        lower.enthalpy, upper.enthalpy, lower_slope, upper_slope));
}

double wd_equilibrium_model::density_at(double radius) const {
    return density_from_enthalpy(enthalpy_at(radius));
}

double wd_equilibrium_model::enclosed_mass_at(double radius) const {
    check_radius(radius);
    if (radius >= 1.0) {
        return 1.0;
    }
    size_t const index = interval_at(radius);
    sample const &lower = profile_[index];
    sample const &upper = profile_[index + 1];
    double const lower_slope =
        4.0 * pi * lower.radius * lower.radius * density_from_enthalpy(lower.enthalpy);
    double const upper_slope =
        4.0 * pi * upper.radius * upper.radius * density_from_enthalpy(upper.enthalpy);
    return interpolate(radius, lower.radius, upper.radius,
        lower.enclosed_mass, upper.enclosed_mass, lower_slope, upper_slope);
}

wd_equilibrium_model const &one_solar_mass_wd() {
    static wd_equilibrium_model const model;
    return model;
}

} // namespace octotiger::test_problems
