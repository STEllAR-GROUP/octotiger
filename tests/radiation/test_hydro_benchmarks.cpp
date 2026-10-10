// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#include "test_problems/radiation/hydro_benchmarks.hpp"

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using radiationHydroTests::Case;
using radiationHydroTests::Point;
using radiationHydroTests::State;
using radiationHydroTests::Units;
using radiationHydroTests::initialState;
using radiationHydroTests::pointState;
using radiationHydroTests::setup;
constexpr double lightSpeed = 2.99792458e10;
constexpr double boltzmann = 1.380658e-16, hydrogenMass = 1.6733e-24;
constexpr double radiationConstant = 4 * 5.67051e-5 / lightSpeed;
constexpr double pi = std::numbers::pi;

void require(bool condition, const std::string& label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}
void near(double actual, double expected, const std::string& label, double tolerance = 4e-13) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * std::max(std::abs(expected), 1e-280)) {
        throw std::runtime_error(label + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
    }
}
template<class Function> void invalid(Function function, const std::string& label) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Invalid input accepted: " + label);
}
void physical(const State& u) {
    require(u.rho > 0 && u.internalEnergy > 0 && u.radiation[0] > 0, "positive matter/radiation state");
    require(std::isfinite(u.gasEnergy), "finite gas energy");
    const double flux = std::hypot(u.radiation[1], u.radiation[2], u.radiation[3]);
    require(flux <= lightSpeed * u.radiation[0] * (1 + 1e-12),
        "physical radiation realizability: |F|/(cE)=" + std::to_string(flux / (lightSpeed * u.radiation[0])));
}

void shocks() {
    for (auto problem : {Case::subcriticalShock, Case::supercriticalShock}) {
        const auto parameters = setup(problem);
        const double velocity = problem == Case::subcriticalShock ? -6e5 : -2e6;
        near(parameters.halfLength, 3.5e10, "shock domain half length");
        near(parameters.density, 7.78e-10, "Ensman original density");
        near(parameters.velocity, velocity, "shock piston-frame velocity");
        near(parameters.stopTime, problem == Case::subcriticalShock ? 38300 : 7500, "shock comparison time");
        for (const double x : {-7e10, -3.5e10, 0., 3.5e10, 7e10}) {
            const double temperature = x <= -3.5e10 ? 85 : x >= 3.5e10 ? 10 : 47.5;
            const auto u = initialState(problem, {x, 0, 0}, 0);
            physical(u);
            near(u.rho, 7.78e-10, "shock mass density");
            near(u.momentum[0] / u.rho, velocity, "shock velocity");
            near(u.internalEnergy, u.rho * boltzmann * temperature / (.5 * hydrogenMass * (2. / 3.)), "ionized gas EOS");
            near(u.radiation[0], radiationConstant * std::pow(temperature, 4), "shock LTE radiation");
            near(u.radiation[1], 4. / 3. * velocity * u.radiation[0], "advected LTE radiation flux");
            near(u.gasEnergy, u.internalEnergy + .5 * u.rho * velocity * velocity, "shock total gas energy");
            require(u.momentum[1] == 0 && u.momentum[2] == 0 && u.radiation[2] == 0 && u.radiation[3] == 0,
                "planar shock transverse fields");
        }
        // The exact mean of a fourth-power linear temperature catches pointwise
        // initialization of finite-volume radiation cells.
        const double dx = 2e10, midpointTemperature = 47.5, delta = 75 * dx / (2 * 7e10);
        const auto average = initialState(problem, {0, 0, 0}, dx);
        const double meanFourth = std::pow(midpointTemperature, 4) +
            2 * midpointTemperature * midpointTemperature * delta * delta + std::pow(delta, 4) / 5;
        near(average.radiation[0], radiationConstant * meanFourth, "exact shock cell-average LTE energy");
        require(average.radiation[0] > radiationConstant * std::pow(midpointTemperature, 4), "nonlinear LTE average");
    }
    near(radiationHydroTests::kappa * 7.78e-10, 3.115e-10, "absorption inverse mean free path");
}

void unitConversions() {
    const Units units{2.3e10, 7e8, 11.3};
    near(units.density(), 2.3e10 / std::pow(7e8, 3), "density unit");
    near(units.velocity(), 7e8 / 11.3, "velocity unit");
    near(units.energyDensity(), 2.3e10 / (7e8 * 11.3 * 11.3), "energy density unit");
    near(units.flux(), 2.3e10 / std::pow(11.3, 3), "physical flux unit");
    for (auto problem : {Case::subcriticalShock, Case::supercriticalShock, Case::blastWave}) {
        for (const Point position : {Point{0, 0, 0}, Point{2e10, -1e10, 3e10}, Point{2e12, 3e12, -4e12}}) {
            const double dx = 2e9;
            const auto physicalState = initialState(problem, position, dx);
            Point codePosition = position;
            for (double& coordinate : codePosition) {
                coordinate /= units.length;
            }
            const auto codeState = initialState(problem, codePosition, dx / units.length, units);
            near(codeState.rho * units.density(), physicalState.rho, "density conversion");
            near(codeState.internalEnergy * units.energyDensity(), physicalState.internalEnergy, "internal energy conversion");
            near(codeState.gasEnergy * units.energyDensity(), physicalState.gasEnergy, "total energy conversion");
            near(codeState.radiation[0] * units.energyDensity(), physicalState.radiation[0], "radiation energy conversion");
            for (size_t d = 0; d < 3; ++d) {
                near(codeState.momentum[d] * units.density() * units.velocity(), physicalState.momentum[d], "momentum conversion");
                near(codeState.radiation[d + 1] * units.flux(), physicalState.radiation[d + 1], "physical radiation flux conversion");
            }
        }
    }
}

void blast() {
    const auto parameters = setup(Case::blastWave);
    near(parameters.halfLength, 1e13, "blast domain");
    near(parameters.coreRadius, 1e12, "regularized blast core");
    near(parameters.depositedEnergy, 1e47, "blast deposited energy");
    near(parameters.luminosity, 1e34, "pre-explosion atmosphere luminosity");
    const auto center = pointState(Case::blastWave, {0, 0, 0});
    physical(center);
    require(center.radiation[1] == 0 && center.radiation[2] == 0 && center.radiation[3] == 0, "regular center flux");
    const auto heated = initialState(Case::blastWave, {0, 0, 0}, 0);
    const double expectedDeposit = 1e47 / (std::pow(pi, 1.5) * 1e36 * std::pow(std::erf(10.), 3));
    near(heated.internalEnergy - center.internalEnergy, expectedDeposit, "Gaussian deposit peak");
    near(heated.radiation[0], center.radiation[0], "energy initially deposited in gas");
    // Diffusion alone can exceed |F|=cE in the thin outer envelope even when
    // its endpoint is realizable. Scan the transition, not only the boundary.
    for (int sample = 0; sample <= 2048; ++sample) {
        physical(pointState(Case::blastWave, {1.8e13 * sample / 2048, 0, 0}));
    }
    for (double radius : {5e11, 1e12, 2e12, 8e12, 1e13, 1.5e13}) {
        const auto u = pointState(Case::blastWave, {radius, 0, 0});
        physical(u);
        near(u.rho, 1e-6 * std::pow(1e12 / std::max(radius, 1e12), 7), "r^-7 density envelope");
        const double luminosity = 4 * pi * radius * radius * u.radiation[1];
        near(luminosity, 1e34 * std::min(1., std::pow(radius / 1e12, 3)), "initial atmosphere luminosity");
        const auto opposite = pointState(Case::blastWave, {-radius, 0, 0});
        near(opposite.radiation[0], u.radiation[0], "spherical atmosphere energy symmetry");
        near(opposite.radiation[1], -u.radiation[1], "radial atmosphere flux symmetry");
        const auto rotated = pointState(Case::blastWave, {radius / 3, 2 * radius / 3, 2 * radius / 3});
        near(rotated.rho, u.rho, "rotational density symmetry");
        near(rotated.gasEnergy, u.gasEnergy, "rotational gas-energy symmetry");
        near(rotated.radiation[0], u.radiation[0], "rotational radiation-energy symmetry");
        near(rotated.radiation[1], u.radiation[1] / 3, "rotated radial x flux");
        near(rotated.radiation[2], 2 * u.radiation[1] / 3, "rotated radial y flux");
        near(rotated.radiation[3], 2 * u.radiation[1] / 3, "rotated radial z flux");
    }
    const auto edge = pointState(Case::blastWave, {1e13, 0, 0});
    // The initialization reserves 1e-12 realizability headroom at streaming.
    near(edge.radiation[0], 1e34 / (4 * pi * lightSpeed * 1e26), "outer streaming boundary energy", 2e-12);
    const auto left = pointState(Case::blastWave, {1e12 * (1 - 1e-8), 0, 0});
    const auto right = pointState(Case::blastWave, {1e12 * (1 + 1e-8), 0, 0});
    near(left.radiation[0], right.radiation[0], "continuous regularized atmosphere energy", 2e-7);
    near(left.radiation[1], right.radiation[1], "continuous regularized atmosphere flux", 4e-8);
    // Integrating all cells must recover 1e47 erg even before the central
    // deposition is spatially resolved. The positive atmosphere contributes
    // less than 0.1 percent of this budget; it is not subtracted numerically.
    for (int cells : {8, 16, 32}) {
        const double dx = 2e13 / cells;
        long double total = 0;
        for (int i = 0; i < cells; ++i) {
            for (int j = 0; j < cells; ++j) {
                for (int k = 0; k < cells; ++k) {
                    const auto u = initialState(Case::blastWave,
                        {(i + .5) * dx - 1e13, (j + .5) * dx - 1e13, (k + .5) * dx - 1e13}, dx);
                    physical(u);
                    total += static_cast<long double>(u.internalEnergy) * dx * dx * dx;
                }
            }
        }
        require(total >= 1e47 * (1 - 1e-13) && total < 1.001e47, "resolution-independent deposited energy budget");
    }
}

void invalidInputs() {
    const auto problem = Case::subcriticalShock;
    for (double bad : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        invalid([&]() { initialState(problem, {0, 0, 0}, 1, {bad, 1, 1}); }, "mass unit");
        invalid([&]() { initialState(problem, {0, 0, 0}, 1, {1, bad, 1}); }, "length unit");
        invalid([&]() { initialState(problem, {0, 0, 0}, 1, {1, 1, bad}); }, "time unit");
    }
    invalid([&]() { initialState(problem, {0, 0, 0}, -1); }, "negative cell width");
    invalid([&]() { initialState(problem, {std::numeric_limits<double>::infinity(), 0, 0}, 1); }, "nonfinite position");
}
}

int main() {
    try {
        shocks();
        unitConversions();
        blast();
        invalidInputs();
        std::cout << "PASS: Ensman Cartesian profiles, CGS/LTE/flux units, conservative blast deposition and regularized atmosphere\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
