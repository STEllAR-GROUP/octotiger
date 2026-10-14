// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
// Standalone: c++ -std=c++20 -O2 -I. tests/radiation/test_ap_flux.cpp -o /tmp/test_ap_flux
// These are spatial face-flux checks, not a nonlinear M1 time-integration test.
#include "octotiger/radiation/ap_flux.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
size_t checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, long double expected, long double scale, const std::string& message,
    long double tolerance = 128 * std::numeric_limits<double>::epsilon()) {
    ++checks;
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * scale) {
        std::ostringstream error;
        error.precision(18);
        error << message << ": actual=" << actual << " expected=" << expected << " scale=" << scale;
        throw std::runtime_error(error.str());
    }
}

void thin_limit() {
    // Smooth linear cell averages, with fixed physical opacity. The leading
    // correction to the supplied reconstructed HLL flux must be quadratic in
    // dx; an O(chi*dx) blend would fail this convergence check.
    constexpr double chi = .8, chat = 3, hll = .7, gradient = .35;
    const double relative_gradient = gradient / (chi * 2);
    const double validity = 1 / (1 + relative_gradient * relative_gradient);
    const double expected_coefficient = validity * (-chi * chi * hll - chat * chi * gradient / 3);
    double previous_error = 0;
    for (double dx : {.04, .02, .01, .005}) {
        const double flux = radiation::apEnergyFlux(hll, 2 - .5 * gradient * dx,
            2 + .5 * gradient * dx, 0., 0., chi, chi, dx, chat);
        const double error = std::abs(flux - hll);
        require(error > 0, "thin correction is measurable");
        near((flux - hll) / (dx * dx), expected_coefficient, std::abs(expected_coefficient),
            "thin leading correction", .0015);
        if (previous_error > 0) {
            const double order = std::log2(previous_error / error);
            require(order > 1.997 && order < 2.003, "thin correction converges quadratically");
        }
        previous_error = error;
    }
    for (double dx : {.01, 1., 10.}) {
        near(radiation::apEnergyFlux(hll, 1., 3., -.2, .4, 0., 0., dx, chat),
            hll, std::abs(hll), "zero opacity recovers HLL exactly", 0);
    }
}

void thick_limit() {
    // Deliberately retain the unreconstructed isotropic HLL jump. This is the
    // difficult fixed-grid limit: numerical viscosity must vanish relative
    // to Fick diffusion, not merely in an absolute norm as chi grows.
    constexpr double dx = .25, chat = 7, left = 1, right = 1.4;
    const double hll = -chat * (right - left) / (2 * std::sqrt(3.));
    double previous_error = 1;
    for (double tau : {1.e2, 1.e4, 1.e6, 1.e8}) {
        const double chi = tau / dx;
        const double expected = -chat / (3 * chi) * (right - left) / dx;
        const double actual = radiation::apEnergyFlux(hll, left, right, 0., 0., chi, chi, dx, chat);
        const double error = std::abs(actual / expected - 1);
        require(error < 2 / tau, "HLL viscosity vanishes relative to thick Fick flux");
        require(error < previous_error / 50, "thick relative error converges as optical depth grows");
        previous_error = error;
        const double no_hll = radiation::apEnergyFlux(0., left, right, 0., 0., chi, chi, dx, chat);
        near(no_hll, expected, std::abs(expected), "thick Fick coefficient",
            2 / (tau * tau) + 128 * std::numeric_limits<double>::epsilon());
    }
}

void compact_diffusion_modes() {
    constexpr size_t cells = 32;
    constexpr double dx = .25, chi = 1.e10, chat = 7, background = 2, amplitude = .2;
    const long double diffusion = static_cast<long double>(chat) / (3 * chi);
    const long double pi = std::acos(-1.L);
    for (size_t mode : {size_t(1), size_t(4), size_t(15), size_t(16)}) {
        std::array<double, cells> energy{}, face{};
        for (size_t i = 0; i < cells; ++i) {
            energy[i] = background + amplitude * std::cos(2 * pi * mode * i / cells);
        }
        for (size_t i = 0; i < cells; ++i) {
            face[i] = radiation::apEnergyFlux(0., energy[i], energy[(i + 1) % cells],
                0., 0., chi, chi, dx, chat);
        }
        const long double s = std::sin(pi * mode / cells);
        const long double eigenvalue = -4 * diffusion * s * s / (dx * dx);
        long double total = 0;
        double largest_rate = 0;
        for (size_t i = 0; i < cells; ++i) {
            const double rate = -(face[i] - face[(i + cells - 1) % cells]) / dx;
            const long double expected = eigenvalue * (energy[i] - background);
            near(rate, expected, std::abs(eigenvalue * amplitude), "compact diffusion Fourier eigenvalue",
                1024 * std::numeric_limits<double>::epsilon());
            total += rate;
            largest_rate = std::max(largest_rate, std::abs(rate));
        }
        near(static_cast<double>(total), 0, std::abs(eigenvalue * amplitude),
            "periodic face flux conserves total energy");
        require(largest_rate > .99 * std::abs(eigenvalue * amplitude),
            "all modes decay, including the checkerboard mode");
    }
}

void material_interfaces() {
    constexpr double dx = .7, chat = 4, left = 1.2, right = 3.4;
    for (const auto& opacity : std::array<std::array<double, 2>, 4>{{
             {{1.e8, 1.e8}}, {{1.e4, 1.e8}}, {{1.e8, 0.}}, {{0., 1.e8}}}}) {
        // Steady Fick flux follows from adding the two half-cell resistances.
        // This oracle does not reproduce the AP interpolation formula.
        const long double resistance = 1.5L * dx * (opacity[0] + opacity[1]) / chat;
        const long double expected = -(static_cast<long double>(right) - left) / resistance;
        const double actual = radiation::apEnergyFlux(0., left, right, 0., 0.,
            opacity[0], opacity[1], dx, chat);
        near(actual, expected, std::abs(expected), "harmonic face diffusivity from material resistance");
        const double reversed = radiation::apEnergyFlux(0., right, left, 0., 0.,
            opacity[1], opacity[0], dx, chat);
        near(reversed, -actual, std::abs(actual), "reversing interface orientation reverses flux");
    }
    const double actual = radiation::apEnergyFlux(0., left, right, 0., 0., 1.e4, 1.e8, dx, chat);
    const double wrong_arithmetic_diffusion = .5 * (chat / (3.e4) + chat / (3.e8));
    const double wrong_flux = -wrong_arithmetic_diffusion * (right - left) / dx;
    require(std::abs(actual / wrong_flux) < .001,
        "opacity jump does not use the arithmetic mean of diffusivities");
}

void advective_flux() {
    constexpr double c = 30, energy = 2.5, dx = .4;
    for (double epsilon : {1., .1, .001}) {
        const double chat = epsilon * c;
        for (double velocity : {-.7, .3}) {
            // Isotropic lab-frame equilibrium Q=(E+P)*v/c. The AP primitive
            // receives Q, not physical F, and propagation uses c-hat once.
            const double q = 4 * energy * velocity / (3 * c);
            const double expected = epsilon * 4 * energy * velocity / 3;
            for (double tau : {0., 1.e-8, .5, 1., 2., 1.e8}) {
                near(radiation::apEnergyFlux(expected, energy, energy, q, q,
                         tau / dx, tau / dx, dx, chat),
                    expected, std::abs(expected), "uniform moving equilibrium is preserved");
            }
        }
        constexpr double left = 1, right = 1.2, q_left = -.1, q_right = .3, chi = 1.e9;
        const double fick = -chat / (3 * chi) * (right - left) / dx;
        const double expected = chat * .5 * (q_left + q_right) + fick;
        near(radiation::apEnergyFlux(-.7, left, right, q_left, q_right, chi, chi, dx, chat),
            expected, std::abs(expected), "thick flux includes centered advective Q and Fick diffusion");
    }
}

void gradient_validity() {
    constexpr double hll = .73, chat = 3, chi = 10, dx = .1;
    for (double small : {0., 1.e-14, 1.e-10}) {
        const double flux = radiation::apEnergyFlux(hll, small, 1., -.1, .2, chi, chi, dx, chat);
        near(flux, hll, std::abs(hll), "unresolved near-vacuum jump recovers HLL");
        const double reversed = radiation::apEnergyFlux(-hll, 1., small, -.2, .1, chi, chi, dx, chat);
        near(reversed, -flux, std::abs(flux), "gradient validity is invariant under face reversal");
    }
    near(radiation::apEnergyFlux(0., 0., 0., 0., 0., chi, chi, dx, chat),
        0, 1, "equal vacuum states have finite zero flux", 0);
    // At tau=1 the base HLL and diffusion weights are equal. Infer the
    // validity factor from two calls differing only in the supplied HLL
    // flux: this checks the implemented blend without duplicating its flux.
    for (double gradient : {.001, .1, 1., 100.}) {
        const double zero_hll = radiation::apEnergyFlux(0., 1., 1 + gradient,
            0., 0., chi, chi, dx, chat);
        const double unit_hll = radiation::apEnergyFlux(1., 1., 1 + gradient,
            0., 0., chi, chi, dx, chat);
        const double alpha = unit_hll - zero_hll;
        const double measured_validity = 2 * (1 - alpha);
        const double mean_free_path_gradient = gradient; // tau=min(E)=1
        const double expected = 1 / (1 + mean_free_path_gradient * mean_free_path_gradient);
        near(measured_validity, expected, 1., "diffusion validity follows the mean-free-path energy gradient");
        require(alpha >= .5 && alpha <= 1, "unresolved gradient only increases the HLL weight");
    }
    // For this extreme finite combination tau and E_R/E_L would both
    // overflow, but g=1. The surviving HLL coefficient must be one half.
    near(radiation::apEnergyFlux(1., 1.e-300, 1.e300, 0., 0., 1.e300, 1.e300, 1.e300, 3.),
        .5, .5, "compensating extreme opacity and energy contrast remain finite", 1.e-12L);
}

void extreme_optical_depths() {
    const double maximum = std::numeric_limits<double>::max();
    for (double chi : {0., std::numeric_limits<double>::denorm_min(),
             std::numeric_limits<double>::min(), 1.e-200, 1.e-155, 1.e-20,
             1., 1.e20, 1.e155, 1.e200, maximum}) {
        const double flux = radiation::apEnergyFlux(.7, 1., 1.2, .1, .3, chi, chi, 1., 3.);
        require(std::isfinite(flux), "finite flux at extreme representable optical depth");
        if (chi < 1.e-20) {
            near(flux, .7, .7, "extremely thin flux has HLL limit");
        }
        if (chi > 1.e20) {
            near(flux, .6, .6, "extremely thick flux has advective limit");
        }
    }
    // The product chi*dx exceeds the largest double, while every argument
    // remains finite. Computing tau directly must not raise FE_OVERFLOW.
    near(radiation::apEnergyFlux(.7, 1., 1.2, .1, .3, maximum, maximum, 2., 3.),
        .6, .6, "overflowing optical depth uses finite limiting weights");
}

void invalid_parameters() {
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double invalid : {-1., -infinity, infinity, nan}) {
        require(std::isnan(radiation::apEnergyFlux(.1, 1., 2., -.1, .2, invalid, 1., 1., 3.)),
            "invalid left extinction returns quiet NaN");
        require(std::isnan(radiation::apEnergyFlux(.1, 1., 2., -.1, .2, 1., invalid, 1., 3.)),
            "invalid right extinction returns quiet NaN");
    }
    for (double invalid : {0., -1., -infinity, infinity, nan}) {
        require(std::isnan(radiation::apEnergyFlux(.1, 1., 2., -.1, .2, 1., 1., invalid, 3.)),
            "invalid cell width returns quiet NaN");
        require(std::isnan(radiation::apEnergyFlux(.1, 1., 2., -.1, .2, 1., 1., 1., invalid)),
            "invalid reduced light speed returns quiet NaN");
    }
    for (double invalid : {-1., -infinity, infinity, nan}) {
        require(std::isnan(radiation::apEnergyFlux(.1, invalid, 2., -.1, .2, 1., 1., 1., 3.)),
            "invalid left energy returns quiet NaN");
        require(std::isnan(radiation::apEnergyFlux(.1, 1., invalid, -.1, .2, 1., 1., 1., 3.)),
            "invalid right energy returns quiet NaN");
    }
}
} // namespace

int main() {
    try {
        // The production application can enable these exceptions. The pure
        // flux primitive must not silently disable them or trigger them.
#if defined(__GLIBC__)
        const int previous_exceptions = feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#endif
        thin_limit();
        thick_limit();
        compact_diffusion_modes();
        material_interfaces();
        advective_flux();
        gradient_validity();
        extreme_optical_depths();
        invalid_parameters();
#if defined(__GLIBC__)
        require((fegetexcept() & (FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW)) ==
                (FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW),
            "AP flux preserves caller floating-point exception settings");
        fedisableexcept(FE_ALL_EXCEPT);
        feenableexcept(previous_exceptions);
#endif
        std::cout << checks << " AP face-flux checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "AP face-flux test failed: " << error.what() << '\n';
        return 1;
    }
}
