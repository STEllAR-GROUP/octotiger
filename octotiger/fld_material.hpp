// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace octotiger::fld {

struct ThermalState {
    double coldEnergy = 0.0;
    double thermalEnergy = 0.0;
    double temperature = 0.0;
    double capacity = 0.0;
    double electronFraction = 0.0;
    double ionChargeSquared = 0.0;
    bool usedTau = false;
};

struct CompositionMoments {
    double particleFraction = 0.0; // Sum X_s (Z_s + 1) / A_s.
    double electronFraction = 0.0; // Sum X_s Z_s / A_s.
    double ionChargeSquared = 0.0; // Sum X_s <Z_s^2> / <A_s>.
};

// These match the constants used by src/physcon.cpp. Temperature is Kelvin.
inline constexpr double material_boltzmann_cgs = 1.380658e-16;
inline constexpr double material_hydrogen_mass_cgs = 1.6733e-24;
inline constexpr double material_light_speed_cgs = 2.99792458e10;
inline constexpr double material_stefan_boltzmann_cgs = 5.67051e-5;

struct MaterialUnits {
    double densityUnit = 1.0;       // g cm^-3 per code density.
    double lengthUnit = 1.0;        // cm per code length.
    double energyDensityUnit = 1.0; // erg cm^-3 per code energy density.
    double c = material_light_speed_cgs;
    double ar = 4.0 * material_stefan_boltzmann_cgs / material_light_speed_cgs;
    double kbOverMh = material_boltzmann_cgs / material_hydrogen_mass_cgs;
};

inline MaterialUnits physical_units(
    double const massUnit, double const lengthUnit, double const timeUnit) {
    if (!(massUnit > 0.0) || !(lengthUnit > 0.0) || !(timeUnit > 0.0) ||
        !std::isfinite(massUnit) || !std::isfinite(lengthUnit) || !std::isfinite(timeUnit)) {
        throw std::invalid_argument("FLD physical unit conversions must be finite and positive");
    }
    long double const velocity = static_cast<long double>(lengthUnit) / timeUnit;
    long double const density = static_cast<long double>(massUnit) /
        lengthUnit / lengthUnit / lengthUnit;
    long double const energyDensity = density * velocity * velocity;
    MaterialUnits result;
    result.lengthUnit = lengthUnit;
    result.densityUnit = static_cast<double>(density);
    result.energyDensityUnit = static_cast<double>(energyDensity);
    result.c = static_cast<double>(material_light_speed_cgs / velocity);
    result.ar = static_cast<double>(4.0L * material_stefan_boltzmann_cgs /
        material_light_speed_cgs / energyDensity);
    result.kbOverMh = static_cast<double>(static_cast<long double>(material_boltzmann_cgs) /
        material_hydrogen_mass_cgs / velocity / velocity);
    for (double const value : {result.densityUnit, result.energyDensityUnit,
             result.c, result.ar, result.kbOverMh}) {
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::invalid_argument("FLD physical constants exceed the numerical range");
        }
    }
    return result;
}

// A tracer may represent a mixture: mass, charge, and chargeSquared return
// its number-weighted ion means <A>, <Z>, and <Z^2>, respectively.
template <class FractionGetter, class MassGetter, class ChargeGetter, class ChargeSquaredGetter>
CompositionMoments composition_moments(size_t const count, FractionGetter const& fraction,
    MassGetter const& mass, ChargeGetter const& charge, ChargeSquaredGetter const& chargeSquaredMean) {
    if (count == 0) {
        throw std::invalid_argument("FLD composition must contain at least one species");
    }
    long double total = 0.0L;
    long double particles = 0.0L;
    long double electrons = 0.0L;
    long double chargeSquared = 0.0L;
    for (size_t species = 0; species < count; ++species) {
        double const x = fraction(species);
        double const a = mass(species);
        double const z = charge(species);
        long double const z2 = chargeSquaredMean(species);
        if (!(x >= 0.0) || !(a > 0.0) || !(z >= 0.0) || !(z2 >= 0.0L) ||
            !std::isfinite(x) || !std::isfinite(a) || !std::isfinite(z) || !std::isfinite(z2)) {
            throw std::invalid_argument("FLD requires nonnegative composition and positive atomic masses");
        }
        long double const ions = static_cast<long double>(x) / a;
        total += x;
        particles += ions * (z + 1.0L);
        electrons += ions * z;
        chargeSquared += ions * z2;
    }
    long double const tolerance = 1.0e-12L +
        64.0L * std::numeric_limits<double>::epsilon() * count;
    if (std::abs(total - 1.0L) > tolerance) {
        throw std::invalid_argument("FLD species mass fractions must sum to one");
    }
    CompositionMoments const result{static_cast<double>(particles),
        static_cast<double>(electrons), static_cast<double>(chargeSquared)};
    if (!(result.particleFraction > 0.0) || !std::isfinite(result.particleFraction) ||
        !std::isfinite(result.electronFraction) || !std::isfinite(result.ionChargeSquared)) {
        throw std::invalid_argument("FLD composition moments exceed the numerical range");
    }
    return result;
}

template <class FractionGetter, class MassGetter, class ChargeGetter>
CompositionMoments composition_moments(size_t const count, FractionGetter const& fraction,
    MassGetter const& mass, ChargeGetter const& charge) {
    return composition_moments(count, fraction, mass, charge, [&](size_t const species) {
        long double const z = charge(species);
        return z * z;
    });
}

inline double fully_ionized_cv(
    double const kbOverMh, double const gamma, double const particleFraction) {
    if (!(kbOverMh > 0.0) || !(gamma > 1.0) || !(particleFraction > 0.0) ||
        !std::isfinite(kbOverMh) || !std::isfinite(gamma) || !std::isfinite(particleFraction)) {
        throw std::invalid_argument("Invalid FLD fully ionized heat-capacity parameters");
    }
    double const cv = static_cast<double>(static_cast<long double>(kbOverMh) *
        particleFraction / (gamma - 1.0));
    if (!(cv > 0.0) || !std::isfinite(cv)) {
        throw std::invalid_argument("FLD specific heat exceeds the numerical range");
    }
    return cv;
}

// Density and composition are frozen during the implicit exchange. This only
// recovers the thermal reservoir; it never reconstructs evolved total energy.
inline ThermalState recover_material_state(double const total, long double const kinetic,
    double const cold, double const tau, double const gamma, double const switch1,
    double const density, double const cv, double const electronFraction = 0.0,
    double const ionChargeSquared = 0.0) {
    if (!(kinetic >= 0.0L) || !(cold >= 0.0) || !(tau >= 0.0) ||
        !(gamma > 1.0) || !(switch1 >= 0.0) || !(density > 0.0) || !(cv > 0.0) ||
        !(electronFraction >= 0.0) || !(ionChargeSquared >= 0.0) ||
        !std::isfinite(total) || !std::isfinite(kinetic) || !std::isfinite(cold) ||
        !std::isfinite(tau) || !std::isfinite(gamma) || !std::isfinite(switch1) ||
        !std::isfinite(density) || !std::isfinite(cv) || !std::isfinite(electronFraction) ||
        !std::isfinite(ionChargeSquared)) {
        std::ostringstream message;
        message.precision(17);
        message << "Invalid FLD thermal recovery state: total=" << total
                << " kinetic=" << kinetic << " cold=" << cold << " tau=" << tau
                << " gamma=" << gamma << " sw1=" << switch1 << " rho=" << density
                << " cv=" << cv << " Ye=" << electronFraction << " Z2=" << ionChargeSquared;
        throw std::invalid_argument(message.str());
    }
    // Match physics::to_prim even when a provisional hydro update undershoots
    // total energy. A finite negative total is usable only if the same sw1
    // criterion selects admissible thermal tau. No conserved energy is reset.
    long double const thermal = static_cast<long double>(total) - kinetic - cold;
    bool const usedTau = thermal <= static_cast<long double>(switch1) * total;
    double const recovered = usedTau ? std::pow(tau, gamma) : static_cast<double>(thermal);
    double const capacity = static_cast<double>(static_cast<long double>(density) * cv);
    double const temperature = recovered / capacity;
    if (!(recovered >= 0.0) || !(capacity > 0.0) || !(temperature >= 0.0) ||
        !std::isfinite(recovered) || !std::isfinite(capacity) || !std::isfinite(temperature)) {
        throw std::runtime_error("FLD thermal state exceeds the numerical range");
    }
    return {cold, recovered, temperature, capacity, electronFraction, ionChargeSquared, usedTau};
}

} // namespace octotiger::fld
