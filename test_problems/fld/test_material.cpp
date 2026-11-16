// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "octotiger/fld_material.hpp"

namespace {

void require(bool const condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double const actual, double const expected, double const tolerance,
    std::string const& message) {
    double const scale = std::max(std::abs(actual), std::abs(expected));
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * scale,
        message + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

template <class Function>
void rejects(Function const& function, std::string const& message) {
    bool rejected = false;
    try {
        function();
    } catch (std::exception const&) {
        rejected = true;
    }
    require(rejected, message);
}

void recovery() {
    using octotiger::fld::recover_material_state;
    double const gamma = 5.0 / 3.0;
    auto const warm = recover_material_state(24.0, 5.0L, 7.0, 1.0,
        gamma, 1.0e-3, 3.0, 2.0, 0.5, 3.0);
    require(!warm.usedTau, "resolved thermal state must use conserved energy");
    near(warm.thermalEnergy, 12.0, 0.0, "subtract kinetic and cold energy");
    near(warm.temperature, 2.0, 0.0, "temperature from thermal reservoir only");
    near(warm.capacity, 6.0, 0.0, "capacity is per volume");
    near(warm.coldEnergy, 7.0, 0.0, "cold reservoir is retained separately");
    near(warm.electronFraction, 0.5, 0.0, "electron moment passed to opacity");
    near(warm.ionChargeSquared, 3.0, 0.0, "ion moment passed to opacity");

    double const tau = std::pow(2.0e-20, 1.0 / gamma);
    for (double const cold : {1.0e-10, 1.0, 1.0e12, 1.0e100}) {
        auto const state = recover_material_state(cold, 0.0L, cold, tau,
            gamma, 1.0e-3, 2.0, 3.0);
        require(state.usedTau, "unresolved degenerate thermal reservoir must use tau");
        near(state.thermalEnergy, 2.0e-20, 1.0e-14, "cold subtraction cannot erase tau energy");
        near(state.temperature, 2.0e-20 / 6.0, 1.0e-14, "cold magnitude cannot heat material");
    }
    auto const moving = recover_material_state(1.0e30, 1.0e30L, 5.0, tau,
        gamma, 1.0e-3, 2.0, 3.0);
    require(moving.usedTau, "high kinetic energy must use tau");
    near(moving.thermalEnergy, 2.0e-20, 1.0e-14, "high Mach recovery retains thermal energy");

    auto const boundary = recover_material_state(1000.0, 999.0L, 0.0, 0.0,
        gamma, 0.001, 1.0, 1.0);
    require(boundary.usedTau, "switch equality uses tau as hydro does");
    near(boundary.temperature, 0.0, 0.0, "zero tau permits exactly cold material");
    auto const above = recover_material_state(1000.0, 998.0L, 0.0, 0.0,
        gamma, 0.001, 1.0, 1.0);
    require(!above.usedTau, "resolved branch must ignore stale zero tau");
    near(above.temperature, 2.0, 0.0, "resolved branch temperature");

    // A cold-star atmosphere can temporarily undershoot conserved gas energy
    // during a hydro stage. Its existing pressure recovery selects valid tau.
    for (double const switch1 : {0.0, 1.0e-3}) {
        double const density = 3.2960793299917288e-9;
        double const cv = 3.8952680110186311e-10;
        auto const undershoot = recover_material_state(-1.7889351976846259e-10,
            3.8497058175403119e-11L, 7.3177239150951723e-15, 1.0e-30,
            gamma, switch1, density, cv, 0.5, 1.0);
        require(undershoot.usedTau, "finite hydro undershoot must follow its tau fallback");
        near(undershoot.thermalEnergy, 1.0e-50, 1.0e-14,
            "undershoot recovery retains only the existing thermal floor");
        near(undershoot.temperature, 1.0e-50 / (density * cv), 1.0e-14,
            "undershoot must not supply artificial heating");
    }
    rejects([&]() { recover_material_state(-1.0, 0.0L, 0.0, 1.0,
                gamma, 2.0, 1.0, 1.0); },
        "negative recovered energy is rejected when the hydro switch does not select tau");
    rejects([&]() { recover_material_state(-1.0, 0.0L, 0.0, -1.0,
                gamma, 0.0, 1.0, 1.0); }, "undershoot cannot hide invalid tau");
    rejects([&]() { recover_material_state(-std::numeric_limits<double>::infinity(),
                0.0L, 0.0, 1.0, gamma, 0.0, 1.0, 1.0); },
        "nonfinite conserved energy is always rejected");

    rejects([&]() { recover_material_state(1.0, 0.0L, 0.0, -1.0,
                gamma, 1.0e-3, 1.0, 1.0); }, "negative tau must be rejected");
    rejects([&]() { recover_material_state(1.0, 0.0L, 0.0, 1.0,
                gamma, 1.0e-3, 1.0, 0.0); }, "zero specific heat must be rejected");
    rejects([&]() { recover_material_state(1.0, 0.0L, 0.0, 1.0,
                gamma, 1.0e-3, std::numeric_limits<double>::infinity(), 1.0); },
        "nonfinite density must be rejected");
}

void composition() {
    using octotiger::fld::composition_moments;
    using octotiger::fld::fully_ionized_cv;
    std::array<double, 3> const mass{{4.0, 12.0, 16.0}};
    std::array<double, 3> const charge{{2.0, 6.0, 8.0}};
    auto const evaluate = [&](std::array<double, 3> const& fraction) {
        return composition_moments(3,
            [&](size_t const i) { return fraction[i]; },
            [&](size_t const i) { return mass[i]; },
            [&](size_t const i) { return charge[i]; });
    };
    auto const helium = evaluate({{1.0, 0.0, 0.0}});
    auto const carbon = evaluate({{0.0, 1.0, 0.0}});
    auto const mixture = evaluate({{0.0, 0.5, 0.5}});
    near(helium.particleFraction, 0.75, 1.0e-15, "helium mean particle mass");
    near(carbon.particleFraction, 7.0 / 12.0, 1.0e-15, "carbon mean particle mass");
    near(mixture.particleFraction, 55.0 / 96.0, 1.0e-15, "CO mixture heat capacity moment");
    near(helium.electronFraction, 0.5, 1.0e-15, "helium electron fraction");
    near(mixture.electronFraction, 0.5, 1.0e-15, "CO electron fraction");
    near(helium.ionChargeSquared, 1.0, 1.0e-15, "helium free-free moment");
    near(mixture.ionChargeSquared, 3.5, 1.0e-15, "CO free-free moment");

    // Accretor, donor, and atmosphere are origin tracers. The accretor tracer
    // itself contains equal carbon/oxygen mass, with number-weighted ion means.
    std::array<double, 3> const meanMass{{96.0 / 7.0, 4.0, 4.0}};
    std::array<double, 3> const meanCharge{{48.0 / 7.0, 2.0, 2.0}};
    std::array<double, 3> const meanChargeSquared{{48.0, 4.0, 4.0}};
    for (auto const& origins : {std::array<double, 3>{{1.0, 0.0, 0.0}},
             std::array<double, 3>{{0.8, 0.15, 0.05}},
             std::array<double, 3>{{0.0, 0.5, 0.5}}}) {
        auto const compressed = composition_moments(3,
            [&](size_t const i) { return origins[i]; },
            [&](size_t const i) { return meanMass[i]; },
            [&](size_t const i) { return meanCharge[i]; },
            [&](size_t const i) { return meanChargeSquared[i]; });
        auto const explicitMixture = evaluate(
            {{origins[1] + origins[2], 0.5 * origins[0], 0.5 * origins[0]}});
        near(compressed.particleFraction, explicitMixture.particleFraction, 1.0e-15,
            "compound origin tracer retains exact mixture heat capacity");
        near(compressed.electronFraction, explicitMixture.electronFraction, 1.0e-15,
            "compound origin tracer retains exact mixture electron density");
        near(compressed.ionChargeSquared, explicitMixture.ionChargeSquared, 1.0e-15,
            "compound origin tracer retains exact mixture free-free moment");
    }
    auto const collapsedCharge = composition_moments(1,
        [](size_t) { return 1.0; },
        [](size_t) { return 96.0 / 7.0; },
        [](size_t) { return 48.0 / 7.0; });
    near(collapsedCharge.ionChargeSquared, 24.0 / 7.0, 1.0e-15,
        "legacy effective ion retains its historical squared mean charge");
    require(std::abs(collapsedCharge.ionChargeSquared - mixture.ionChargeSquared) > 0.07,
        "a compound ion needs the independent second charge moment");
    for (double const invalid : {-1.0, std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()}) {
        rejects([&]() {
            composition_moments(1, [](size_t) { return 1.0; },
                [](size_t) { return 4.0; }, [](size_t) { return 2.0; },
                [&](size_t) { return invalid; });
        }, "invalid squared charge moments rejected");
    }
    near(fully_ionized_cv(2.0, 5.0 / 3.0, helium.particleFraction), 2.25,
        1.0e-15, "fully ionized heat capacity includes ions and electrons");
    rejects([&]() { evaluate({{0.0, 0.5, 0.25}}); }, "unnormalized fractions rejected");
    rejects([&]() { evaluate({{0.0, -0.5, 1.5}}); }, "negative fractions rejected");
    rejects([&]() { fully_ionized_cv(2.0, 1.0, 0.5); }, "gamma must exceed one");
}

void units() {
    using namespace octotiger::fld;
    auto const cgs = physical_units(1.0, 1.0, 1.0);
    near(cgs.c, 2.99792458e10, 1.0e-15, "CGS light speed");
    near(cgs.ar, 4.0 * 5.67051e-5 / 2.99792458e10, 1.0e-15, "CGS radiation constant");
    auto const scaled = physical_units(8.0, 2.0, 4.0);
    near(scaled.densityUnit, 1.0, 0.0, "density unit M/L^3");
    near(scaled.lengthUnit, 2.0, 0.0, "length unit");
    near(scaled.energyDensityUnit, 0.25, 0.0, "energy-density unit M/(L t^2)");
    near(scaled.c, 2.0 * cgs.c, 1.0e-15, "light speed scales as t/L");
    near(scaled.ar, 4.0 * cgs.ar, 1.0e-15, "radiation constant scales as L t^2/M");
    near(scaled.kbOverMh, 4.0 * cgs.kbOverMh, 1.0e-15, "specific heat scales as t^2/L^2");

    // The same physical CO cell has identical Kelvin temperature in both units.
    double const rhoCgs = 1.0e6;
    double const targetTemperature = 1.0e7;
    double const cvCgs = fully_ionized_cv(cgs.kbOverMh, 5.0 / 3.0, 55.0 / 96.0);
    double const uCgs = rhoCgs * cvCgs * targetTemperature;
    auto const stellar = physical_units(1.0e33, 1.0e9, 10.0);
    double const cvCode = fully_ionized_cv(stellar.kbOverMh, 5.0 / 3.0, 55.0 / 96.0);
    auto const state = recover_material_state(uCgs / stellar.energyDensityUnit,
        0.0L, 0.0, 0.0, 5.0 / 3.0, 1.0e-3, rhoCgs / stellar.densityUnit, cvCode);
    near(state.temperature, targetTemperature, 3.0e-15, "physical temperature remains Kelvin");
    near(stellar.ar * std::pow(targetTemperature, 4) * stellar.energyDensityUnit,
        cgs.ar * std::pow(targetTemperature, 4), 3.0e-15,
        "equilibrium radiation energy uses the same energy conversion");
    rejects([&]() { physical_units(0.0, 1.0, 1.0); }, "zero mass unit rejected");
}

void exchange() {
    using octotiger::fld::recover_material_state;
    double const gamma = 5.0 / 3.0;
    double const radiationStar = 0.25;
    double const thermalStar = 12.0;
    double const tauStar = std::pow(thermalStar, 1.0 / gamma);
    double const ar = 0.3;
    for (double const cold : {0.0, 1.0e8, 1.0e24}) {
        double const kinetic = 2.0 * cold;
        double const gasStar = kinetic + cold + thermalStar;
        auto const before = recover_material_state(gasStar, kinetic, cold,
            tauStar, gamma, 1.0e-3, 2.0, 3.0);
        for (double const coupling : {1.0e-8, 0.1, 1.0e12}) {
            // Independent scalar backward-Euler reference: total thermal plus
            // radiation energy is fixed and T is the sole remaining unknown.
            long double const available =
                static_cast<long double>(before.thermalEnergy) + radiationStar;
            long double lower = 0.0L;
            long double upper = available / before.capacity;
            for (int iteration = 0; iteration < 160; ++iteration) {
                long double const t = (lower + upper) / 2.0L;
                long double const thermal = before.capacity * t;
                long double const residual = thermal - before.thermalEnergy -
                    coupling * (available - thermal - ar * t * t * t * t);
                if (residual > 0.0L) {
                    upper = t;
                } else {
                    lower = t;
                }
            }
            double const temperature = static_cast<double>((lower + upper) / 2.0L);
            double const radiation = static_cast<double>(available - before.capacity *
                ((lower + upper) / 2.0L));
            double const delta = radiationStar - radiation;
            double const thermal = before.thermalEnergy + delta;
            double const gas = gasStar + delta;
            auto const after = recover_material_state(gas, kinetic, cold,
                std::pow(thermal, 1.0 / gamma), gamma, 1.0e-3, 2.0, 3.0);
            near(after.temperature, temperature, 4.0e-15,
                "BE exchange recovers accepted temperature with a fixed cold reservoir");
            near(after.thermalEnergy + radiation, before.thermalEnergy + radiationStar,
                4.0e-15, "thermal plus radiation ledger resolves the exchanged energy");
            near(after.coldEnergy, before.coldEnergy, 0.0,
                "thermal exchange cannot alter cold degeneracy energy");
            require(after.thermalEnergy >= 0.0 && radiation >= 0.0,
                "reference exchange preserves nonnegative thermal and radiation states");

            // When the thermal increment is below one total-energy ulp, tau
            // still carries it. Check the two ledgers at their own scales.
            long double const gasBalance = static_cast<long double>(gas) - gasStar +
                static_cast<long double>(radiation) - radiationStar;
            double const roundoff = 4.0 * std::numeric_limits<double>::epsilon() *
                std::max(gasStar, before.thermalEnergy + radiationStar);
            require(std::abs(gasBalance) <= roundoff,
                "gas-total exchange must balance to total-energy rounding precision");
        }
    }
}

} // namespace

int main() {
    try {
        recovery();
        composition();
        units();
        exchange();
        std::cout << "FLD material recovery, composition, units, and exchange tests passed\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
