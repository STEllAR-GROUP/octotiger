// Copyright (c) 2026 AUTHORS
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "octotiger/grid.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp"

namespace {
constexpr double eos_a = 2.0;
constexpr double eos_b = 3.0;
constexpr double dual_energy_switch = 1.0e-3;
int failures = 0;

// Evaluate the pressure independently of the production acoustic formula.
// The small-x expansion avoids cancellation when taking its finite difference.
long double cold_pressure(long double rho, long double a) {
    long double const x = std::cbrt(rho / eos_b);
    long double const x2 = x * x;
    if (x < 0.05L) {
        return a * x * x2 * x2 *
               (1.6L + x2 * (-4.0L / 7.0L +
                             x2 * (1.0L / 3.0L + x2 * (-5.0L / 22.0L + x2 * 35.0L / 208.0L))));
    }
    return a * (x * (2 * x2 - 3) * std::sqrt(1 + x2) + 3 * std::asinh(x));
}

double reference_speed(double rho, double ein, double gamma, double a) {
    long double const drho = rho * 1.0e-4L;
    long double const k = (gamma - 1.0L) * ein / std::pow(static_cast<long double>(rho), gamma);
    auto const pressure = [=](long double density) {
        return cold_pressure(density, a) + k * std::pow(density, gamma);
    };
    return std::sqrt((pressure(rho + drho) - pressure(rho - drho)) / (2 * drho));
}

void check(char const *path, double actual, double expected) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > 5.0e-6 * std::abs(expected)) {
        std::cerr << path << ": expected " << expected << ", got " << actual << '\n';
        ++failures;
    }
}

std::vector<safe_real> state(double rho, double ein, double velocity, double gamma, double a,
                             bool fallback) {
    std::vector<safe_real> u(physics<3>::field_count(), 0.0);
    double const x = std::pow(rho / eos_b, 1.0 / 3.0);
    // Construct total energy consistently with the production cold EOS; the
    // expected sound speed above is obtained independently from pressure.
    double const edeg = a == 0.0
                            ? 0.0
                            : (x > 0.001 ? rho * 8.0 * a / eos_b * (std::sqrt(x * x + 1.0) - 1.0) -
                                               physics<3>::deg_pres(x)
                                         : 2.4 * a * std::pow(x, 5));
    u[rho_i] = rho;
    u[sx_i] = rho * velocity;
    u[egas_i] = edeg + 0.5 * rho * velocity * velocity + (fallback ? 0.0 : ein);
    u[tau_i] = std::pow(ein, 1.0 / gamma);
    u[spc_i] = rho;
    return u;
}

template <class Simd>
void check_interface(std::vector<safe_real> const &right, std::vector<safe_real> const &left,
                     double a, double gamma, double expected_ap, double expected_am,
                     double expected_pressure) {
    std::array<Simd, OCTOTIGER_MAX_NUMBER_FIELDS> qr{}, ql{}, flux{};
    std::array<Simd, NDIM> position{}, grid_velocity{};
    for (size_t f = 0; f < right.size(); ++f) {
        qr[f] = Simd(right[f]);
        ql[f] = Simd(left[f]);
    }
    for (size_t dim = 0; dim < NDIM; ++dim) {
        position[dim] = Simd(0.0);
        grid_velocity[dim] = Simd(0.0);
    }
    Simd ap(0.0), am(0.0);
    Simd const speed = cell_inner_flux_loop_simd<Simd>(0.0, right.size(), a, eos_b, qr, ql, flux,
                                                       position, grid_velocity, ap, am, 0, 0, 1.0,
                                                       gamma, dual_energy_switch, 0);
    std::array<double, Simd::size()> ap_lanes{}, am_lanes{}, speed_lanes{};
    ap.copy_to(ap_lanes.data(), SIMD_NAMESPACE::element_aligned_tag{});
    am.copy_to(am_lanes.data(), SIMD_NAMESPACE::element_aligned_tag{});
    speed.copy_to(speed_lanes.data(), SIMD_NAMESPACE::element_aligned_tag{});
    for (size_t lane = 0; lane < Simd::size(); ++lane) {
        check("interface right wave", ap_lanes[lane], expected_ap);
        check("interface left wave", am_lanes[lane], expected_am);
        check("interface maximum speed", speed_lanes[lane], std::max(expected_ap, -expected_am));
    }

    // Equal interface states must retain total pressure in momentum/energy fluxes.
    ap = Simd(0.0);
    am = Simd(0.0);
    cell_inner_flux_loop_simd<Simd>(0.0, right.size(), a, eos_b, qr, qr, flux, position,
                                    grid_velocity, ap, am, 0, 0, 1.0, gamma, dual_energy_switch, 0);
    std::array<double, Simd::size()> momentum_flux{}, energy_flux{};
    flux[sx_i].copy_to(momentum_flux.data(), SIMD_NAMESPACE::element_aligned_tag{});
    flux[egas_i].copy_to(energy_flux.data(), SIMD_NAMESPACE::element_aligned_tag{});
    auto const velocity = right[sx_i] / right[rho_i];
    for (size_t lane = 0; lane < Simd::size(); ++lane) {
        check("uniform momentum flux", momentum_flux[lane],
              right[rho_i] * velocity * velocity + expected_pressure);
        check("uniform energy flux", energy_flux[lane],
              velocity * (right[egas_i] + expected_pressure));
    }
}

void check_case(double x, double thermal_fraction, double gamma, bool wd, bool fallback) {
    double const a = wd ? eos_a : 0.0;
    physics<3>::set_degenerate_eos(a, eos_b);
    physics<3>::set_fgamma(gamma);
    physics<3>::set_dual_energy_switches(dual_energy_switch, 0.1);
    physcon().A = a;
    physcon().B = eos_b;
    grid::set_fgamma(gamma);

    double const rho_right = eos_b * x * x * x;
    double const rho_left = 1.7 * rho_right;
    double const ein_right =
        thermal_fraction * (wd ? cold_pressure(rho_right, a) : rho_right) / (gamma - 1.0);
    double const ein_left =
        thermal_fraction * (wd ? cold_pressure(rho_left, a) : rho_left) / (gamma - 1.0);
    double const c_right = reference_speed(rho_right, ein_right, gamma, a);
    double const c_left = reference_speed(rho_left, ein_left, gamma, a);
    // Opposite velocities force each interface state to set a different bound.
    double const velocity = 3.0 * std::max(c_right, c_left);
    auto const right = state(rho_right, ein_right, velocity, gamma, a, fallback);
    auto const left = state(rho_left, ein_left, -velocity, gamma, a, fallback);
    double pressure, v, cs;
    physics<3>::to_prim(right, pressure, v, cs, 0);
    check("to_prim right sound speed", cs, c_right);
    check("to_prim right pressure", pressure,
          cold_pressure(rho_right, a) + (gamma - 1.0) * ein_right);
    check("to_prim right velocity", v, velocity);
    physics<3>::to_prim(left, pressure, v, cs, 0);
    check("to_prim left sound speed", cs, c_left);
    check("to_prim left pressure", pressure, cold_pressure(rho_left, a) + (gamma - 1.0) * ein_left);
    check("to_prim left velocity", v, -velocity);
    check("ztwd_sound_speed right", ztwd_sound_speed(rho_right, ein_right), c_right);
    check("ztwd_sound_speed left", ztwd_sound_speed(rho_left, ein_left), c_left);
    auto const expected_pressure = cold_pressure(rho_right, a) + (gamma - 1.0) * ein_right;
    check_interface<host_simd_t>(right, left, a, gamma, velocity + c_right, -velocity - c_left,
                                 expected_pressure);
    check_interface<device_simd_t>(right, left, a, gamma, velocity + c_right, -velocity - c_left,
                                   expected_pressure);
}
} // namespace

int main() {
    physics<3>::set_n_species(1);
    physics<3>::update_n_field();
    for (double const gamma : {5.0 / 3.0, 1.4}) {
        for (double const x : {5.0e-4, 0.1, 1.0, 10.0, 100.0}) {
            check_case(x, 0.0, gamma, true, false);
            check_case(x, 1.0, gamma, true, false);
            check_case(x, 1.0, gamma, true, true);
        }
        check_case(1.0, 1.0, gamma, false, false);
        check_case(1.0, 1.0, gamma, false, true);
    }
    if (failures != 0) {
        std::cerr << failures << " WD EOS checks failed\n";
        return 1;
    }
    std::cout << "WD EOS scalar, host SIMD (" << host_simd_t::size()
              << " lanes), and scalar device SIMD sound-speed checks passed\n";
    return 0;
}
