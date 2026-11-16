// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "octotiger/fld_hydro.hpp"

namespace {

using vector3 = std::array<double, 3>;
using tensor3 = std::array<vector3, 3>;

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

vector3 affineVelocity(vector3 const& position, tensor3 const& gradient) {
    vector3 result{{0.7, -1.1, 0.3}};
    for (size_t component = 0; component < 3; ++component) {
        for (size_t axis = 0; axis < 3; ++axis) {
            result[component] += gradient[component][axis] * position[axis];
        }
    }
    return result;
}

void testCentralVelocityGradient() {
    tensor3 const expected{{{{0.2, 2.0, -0.3}}, {{-0.25, -0.4, 0.8}}, {{1.3, -0.9, 0.7}}}};
    vector3 const center{{0.3, -0.5, 0.8}};
    double const dx = 0.125;
    tensor3 minus{};
    tensor3 plus{};
    for (size_t axis = 0; axis < 3; ++axis) {
        vector3 position = center;
        position[axis] -= dx;
        minus[axis] = affineVelocity(position, expected);
        position[axis] += 2.0 * dx;
        plus[axis] = affineVelocity(position, expected);
    }
    tensor3 const actual = octotiger::fld::velocity_gradient(minus, plus, dx);
    for (size_t component = 0; component < 3; ++component) {
        for (size_t axis = 0; axis < 3; ++axis) {
            near(actual[component][axis], expected[component][axis], 3.0e-15,
                "six-neighbor gradient reproduces an affine vector field");
        }
    }
    tensor3 pressure{};
    double const energy = 6.0;
    for (size_t axis = 0; axis < 3; ++axis) {
        pressure[axis][axis] = energy / 3.0;
    }
    near(octotiger::fld::pressure_work(pressure, actual), energy * 0.5 / 3.0, 3.0e-15,
        "isotropic pressure work is E/3 times velocity divergence");
}

void testAnisotropicShearAndRotation() {
    // E=6, f=0.8 and n=(1,1,0)/sqrt(2) give this realizable FLD tensor.
    tensor3 const pressure{{{{2.7, 2.1, 0.0}}, {{2.1, 2.7, 0.0}}, {{0.0, 0.0, 0.6}}}};
    tensor3 shear{};
    shear[0][1] = 2.0;
    shear[1][0] = -0.25;
    near(octotiger::fld::pressure_work(pressure, shear), 3.675, 2.0e-15,
        "trace-free shear still does anisotropic radiation pressure work");

    tensor3 rotation{};
    rotation[0][1] = 0.7;
    rotation[1][0] = -0.7;
    rotation[0][2] = -0.2;
    rotation[2][0] = 0.2;
    rotation[1][2] = 1.3;
    rotation[2][1] = -1.3;
    near(octotiger::fld::pressure_work(pressure, rotation), 0.0, 2.0e-15,
        "rigid rotation does no work against a symmetric pressure tensor");
}

void testRadiationSignalSpeed() {
    double const energy = 9.0;
    double const density = 2.0;
    near(octotiger::fld::radiation_sound_speed_squared(energy, density, 1.0 / 3.0, 1.0 / 3.0),
        4.0 * energy / (9.0 * density), 2.0e-15,
        "trapped isotropic radiation contributes 4E/(9rho)");
    near(octotiger::fld::radiation_sound_speed_squared(0.0, density, 1.0 / 3.0, 1.0 / 3.0), 0.0,
        0.0, "zero radiation leaves gas wave speed unchanged");
    near(octotiger::fld::radiation_sound_speed_squared(energy, density, 0.0, 1.0), 0.0, 0.0,
        "vanishing matter coupling has no radiation acoustic contribution");
    double const parallel =
        octotiger::fld::radiation_sound_speed_squared(energy, density, 0.1, 0.8);
    double const transverse =
        octotiger::fld::radiation_sound_speed_squared(energy, density, 0.1, 0.1);
    near(parallel / transverse, 1.8 / 1.1, 2.0e-15,
        "anisotropic normal pressure changes directional signal speed");
}

void testHllFluxLimits() {
    using octotiger::fld::hll_flux;
    near(hll_flux(2.0, 4.0, 6.0, 12.0, 0.0, 3.0), 6.0, 0.0,
        "right-moving fan selects the left physical flux");
    near(hll_flux(2.0, 4.0, -6.0, -12.0, -3.0, 0.0), -12.0, 0.0,
        "left-moving fan selects the right physical flux");
    near(hll_flux(2.0, 2.0, 0.6, 0.6, -1.0, 2.0), 0.6, 2.0e-15,
        "constant state has its physical radiation advection flux");
    near(hll_flux(1.0, 3.0, 0.4, 1.2, -2.0, 3.0), -1.68, 2.0e-15,
        "two-sided fan includes the conservative state-jump term");
    near(hll_flux(1.0, 2.0, 0.0, 0.0, 0.0, 0.0), 0.0, 0.0,
        "stationary zero-width fan produces a finite zero flux");
}

void testPeriodicAdvection() {
    int const cells = 64;
    double const dx = 1.0 / cells;
    double const velocity = 0.3;
    double const acousticSpeed = 1.2;
    double const dt = 0.4 * dx / (velocity + acousticSpeed);
    std::vector<double> energy(cells);
    std::vector<double> flux(cells);
    double initialSum = 0.0;
    for (int i = 0; i < cells; ++i) {
        energy[i] = 1.0 + 0.7 * std::sin(2.0 * std::acos(-1.0) * (i + 0.5) * dx);
        initialSum += energy[i];
    }
    for (int step = 0; step < 100; ++step) {
        for (int i = 0; i < cells; ++i) {
            int const next = (i + 1) % cells;
            flux[i] = octotiger::fld::hll_flux(energy[i], energy[next], velocity * energy[i],
                velocity * energy[next], velocity - acousticSpeed, velocity + acousticSpeed);
        }
        for (int i = 0; i < cells; ++i) {
            energy[i] -= dt / dx * (flux[i] - flux[(i + cells - 1) % cells]);
            require(energy[i] >= 0.3 && energy[i] <= 1.7,
                "CFL-limited first-order radiation advection preserves initial bounds");
        }
    }
    double finalSum = 0.0;
    for (double const value : energy) {
        finalSum += value;
    }
    near(finalSum, initialSum, 3.0e-15,
        "one shared periodic HLL face flux conserves radiation energy");
}

void testDualEnergyRecoveryAndHeating() {
    using octotiger::fld::recover_internal_energy;
    using octotiger::fld::thermal_tau;
    double const gamma = 5.0 / 3.0;
    near(thermal_tau(32.0, gamma), 8.0, 2.0e-15,
        "gamma=5/3 stores tau as the three-fifths power of internal energy");
    near(recover_internal_energy(5.0, 2.0, thermal_tau(42.0, gamma), gamma, 0.001), 3.0, 0.0,
        "resolved gas thermal energy takes precedence over stale tau");
    near(recover_internal_energy(1024.0, 1023.0, thermal_tau(3.0, gamma), gamma, 1.0 / 1024.0), 3.0,
        2.0e-15, "the dual-energy threshold includes equality");

    double const velocity = 1.0e9;
    double const kinetic = 0.5 * velocity * velocity;
    double total = kinetic + 1.0;
    double tau = thermal_tau(1.0, gamma);
    require(total - kinetic == 0.0,
        "high-Mach fixture must actually lose thermal energy in kinetic subtraction");
    double expected = 1.0;
    for (double const increment : {0.25, -0.375, 0.5, -0.125}) {
        double const internal = recover_internal_energy(total, kinetic, tau, gamma, 0.001);
        near(internal, expected, 3.0e-15,
            "high-Mach temperature recovery preserves prior implicit heating through tau");
        expected += increment;
        total += increment;
        tau = thermal_tau(internal + increment, gamma);
        near(std::pow(tau, gamma), expected, 3.0e-15,
            "implicit heating/cooling survives when total-energy increments are below an ulp");
    }
    require(total == kinetic,
        "high-Mach fixture must retain unresolved total-energy increments throughout");
    near(recover_internal_energy(total, kinetic, tau, gamma, 0.001), 1.25, 3.0e-15,
        "final high-Mach internal energy is recovered from the updated tau");
}

}    // namespace

int main() {
    try {
        testCentralVelocityGradient();
        testAnisotropicShearAndRotation();
        testRadiationSignalSpeed();
        testHllFluxLimits();
        testPeriodicAdvection();
        testDualEnergyRecoveryAndHeating();
        std::cout << "FLD explicit numerical tests passed\n";
    } catch (std::exception const& error) {
        std::cerr << "FLD explicit numerical test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
