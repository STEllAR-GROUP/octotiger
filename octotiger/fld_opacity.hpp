// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "octotiger/fld_closure.hpp"
#include "octotiger/fld_rosseland_cache.hpp"

namespace octotiger::fld {

enum class OpacityModel { legacy, independent, free_free_thomson };

struct OpacityParameters
{
    OpacityModel model = OpacityModel::legacy;
    double absorption = 1.0;
    double scattering = 0.0; // Legacy/independent input: code area per code mass.
    double densityExponent = 0.0;
    double temperatureExponent = 0.0;
    double rosseland = 1.0; // Independent mode: TOTAL extinction, not absorption.
    double rosselandDensityExponent = 0.0;
    double rosselandTemperatureExponent = 0.0;
    double temperatureFloor = 0.0; // Opacity evaluation only; never an emission temperature.
    double gaunt = 1.0; // Constant free-free Gaunt factor; zero selects pure Thomson.
    double lengthUnit = 1.0; // cm per code length.
    double densityUnit = 1.0; // g cm^-3 per code density.
};

struct OpacityCoefficients
{
    double planckAbsorption = 0.0;
    double rosselandExtinction = 0.0;
    double scatteringExtinction = 0.0;
    bool regularized = false;
};

namespace opacity_detail {

constexpr long double pi = 3.141592653589793238462643383279502884L;
constexpr long double planckNormalization = pi * pi * pi * pi / 15.0L;
constexpr long double rosselandNormalization = 4.0L * planckNormalization;

// Precompute the spectral factors, not a table of already averaged opacities.
// x=h*nu/(k*T), q=(1-exp(-x))/x^3, w=x^4*exp(-x)/(1-exp(-x))^2.
// The omitted x>80 tail has relative weight below 1e-24 even for pure free-free,
// whose inverse-opacity integrand has the slowest tail, proportional to x^7 exp(-x).
template <size_t Order>
struct RosselandQuadrature
{
    static constexpr size_t intervals = 6;
    std::array<long double, intervals * Order> absorptionShape{};
    std::array<long double, intervals * Order> weight{};

    RosselandQuadrature() {
        std::array<long double, intervals + 1> const edges{{0.0L, 0.125L, 0.5L,
            2.0L, 8.0L, 24.0L, 80.0L}};
        for (size_t root = 0; root < (Order + 1) / 2; ++root) {
            long double z = std::cos(pi * (root + 0.75L) / (Order + 0.5L));
            long double derivative = 0.0L;
            bool converged = false;
            for (int iteration = 0; iteration < 32; ++iteration) {
                long double previous = 1.0L;
                long double value = z;
                for (size_t degree = 2; degree <= Order; ++degree) {
                    long double const next = ((2 * degree - 1) * z * value -
                        (degree - 1) * previous) / degree;
                    previous = value;
                    value = next;
                }
                derivative = Order * (z * value - previous) / (z * z - 1.0L);
                long double const correction = value / derivative;
                z -= correction;
                if (std::abs(correction) <= 8.0L * std::numeric_limits<long double>::epsilon()) {
                    converged = true;
                    break;
                }
            }
            if (!converged) {
                throw std::runtime_error("FLD opacity quadrature construction failed");
            }
            long double const rootWeight = 2.0L / ((1.0L - z * z) * derivative * derivative);
            for (size_t interval = 0; interval < intervals; ++interval) {
                long double const halfWidth = 0.5L * (edges[interval + 1] - edges[interval]);
                long double const center = 0.5L * (edges[interval + 1] + edges[interval]);
                for (size_t side = 0; side < 2; ++side) {
                    size_t const index = interval * Order + (side == 0 ? root : Order - 1 - root);
                    long double const x = center + (side == 0 ? -z : z) * halfWidth;
                    long double const stimulated = -std::expm1(-x);
                    absorptionShape[index] = stimulated / (x * x * x);
                    weight[index] = halfWidth * rootWeight * x * x * x * x * std::exp(-x) /
                        (stimulated * stimulated);
                }
            }
        }
    }

    long double integral(long double const freeFree, long double const thomson) const {
        long double sum = 0.0L;
        for (size_t index = 0; index < weight.size(); ++index) {
            sum += weight[index] / (freeFree * absorptionShape[index] + thomson);
        }
        return sum;
    }
};

template <size_t Order>
inline long double rosseland_integral(long double const freeFree, long double const thomson) {
    static RosselandQuadrature<Order> const quadrature;
    return quadrature.integral(freeFree, thomson);
}

// freeFree is the amplitude multiplying q(x), not its Planck or Rosseland mean.
// Scattering enters inside the harmonic average. Scale first to avoid overflow
// or loss of the weaker contribution when physical coefficients are extreme.
inline long double rosseland_mean_quadrature(long double const freeFree, long double const thomson) {
    if (!(freeFree >= 0.0L) || !(thomson >= 0.0L) ||
        !std::isfinite(freeFree) || !std::isfinite(thomson)) {
        throw std::invalid_argument("Invalid monochromatic FLD opacity amplitudes");
    }
    if (freeFree == 0.0L) {
        return thomson;
    }
    long double const scale = std::max(freeFree, thomson);
    long double const a = freeFree / scale;
    long double const s = thomson / scale;
    long double previous = rosseland_integral<16>(a, s);
    long double current = rosseland_integral<32>(a, s);
    constexpr long double tolerance = 2.0e-11L;
    if (std::abs(current - previous) > tolerance * current) {
        previous = current;
        current = rosseland_integral<64>(a, s);
        if (std::abs(current - previous) > tolerance * current) {
            previous = current;
            current = rosseland_integral<128>(a, s);
            if (std::abs(current - previous) > tolerance * current) {
                throw std::runtime_error("FLD Rosseland opacity quadrature did not converge");
            }
        }
    }
    return scale * (rosselandNormalization / current);
}

inline long double log_rosseland_ratio(long double const logRatio) {
    // Normalize before exponentiation, including when A_ff/sigma itself would
    // overflow. The cached ordinate is universal for this monochromatic model.
    long double const value = logRatio >= 0.0L ?
        logRatio + std::log(rosseland_mean_quadrature(1.0L, std::exp(-logRatio))) :
        std::log(rosseland_mean_quadrature(std::exp(logRatio), 1.0L));
    return std::max(0.0L, value);
}

inline LogRosselandCache const& rosseland_cache() {
    static LogRosselandCache const cache(log_rosseland_ratio);
    return cache;
}

inline long double rosseland_mean(long double const freeFree, long double const thomson) {
    if (!(freeFree >= 0.0L) || !(thomson >= 0.0L) ||
        !std::isfinite(freeFree) || !std::isfinite(thomson)) {
        throw std::invalid_argument("Invalid monochromatic FLD opacity amplitudes");
    }
    if (freeFree == 0.0L) {
        return thomson;
    }
    if (thomson == 0.0L) {
        // integral w/q dx = 2520*(zeta(6)+zeta(7)); no interpolation or
        // quadrature is needed for the exact constant-Gaunt free-free limit.
        constexpr long double zeta6 = pi * pi * pi * pi * pi * pi / 945.0L;
        constexpr long double zeta7 = 1.00834927738192282683979754984979676L;
        return freeFree * (rosselandNormalization / (2520.0L * (zeta6 + zeta7)));
    }
    long double const logRatio = std::log(freeFree) - std::log(thomson);
    long double const logMean = rosseland_cache().evaluate(logRatio);
    // Reconstruct from the larger amplitude so exp(logMean) need not fit even
    // when the final mean does. The interpolator never extrapolates its table.
    return logRatio >= 0.0L ? freeFree * std::exp(logMean - logRatio) :
        thomson * std::exp(logMean);
}

inline double checked_coefficient(long double const value) {
    if (!(value >= 0.0L) || !std::isfinite(value) ||
        value > std::numeric_limits<double>::max()) {
        throw std::runtime_error("Nonfinite FLD mean opacity in code units");
    }
    return static_cast<double>(value);
}

inline double power_law(double const rho, double const temperature, double const coefficient,
    double const densityExponent, double const temperatureExponent) {
    if (temperature > 0.0) {
        // Preserve the original positive-temperature evaluation exactly.
        return absorption(rho, temperature, coefficient, densityExponent, temperatureExponent);
    }
    if (!(rho > 0.0) || temperature != 0.0 || !(coefficient >= 0.0) ||
        !std::isfinite(rho) || !std::isfinite(coefficient) ||
        !std::isfinite(densityExponent) || !std::isfinite(temperatureExponent)) {
        throw std::invalid_argument("Invalid zero-temperature FLD opacity power law");
    }
    if (coefficient == 0.0 || temperatureExponent > 0.0) {
        return 0.0;
    }
    if (temperatureExponent == 0.0) {
        return absorption(rho, 1.0, coefficient, densityExponent, temperatureExponent);
    }
    throw std::invalid_argument("Negative temperature exponent requires an explicit FLD opacity cutoff");
}

}    // namespace opacity_detail

// Temperature remains a caller-owned material quantity. This function only
// regularizes coefficient evaluation; callers must use the ORIGINAL material
// temperature in a_r*T^4. Ye=sum(X_i Z_i/A_i), Z2=sum(X_i Z_i^2/A_i).
// The composition defaults describe fully ionized helium, not a general WD.
inline OpacityCoefficients evaluate_opacity(OpacityParameters const& p, double const rho,
    double const temperature, double const electronFraction = 0.5,
    double const ionChargeSquared = 1.0) {
    if (!(rho > 0.0) || !(temperature >= 0.0) || !(p.temperatureFloor >= 0.0) ||
        !std::isfinite(rho) || !std::isfinite(temperature) || !std::isfinite(p.temperatureFloor)) {
        throw std::invalid_argument("FLD opacity requires positive density and nonnegative temperature/floor");
    }
    double const evaluatedTemperature = std::max(temperature, p.temperatureFloor);
    OpacityCoefficients result;
    result.regularized = temperature < p.temperatureFloor;
    if (p.model == OpacityModel::legacy || p.model == OpacityModel::independent) {
        result.planckAbsorption = opacity_detail::power_law(rho, evaluatedTemperature, p.absorption,
            p.densityExponent, p.temperatureExponent);
        result.scatteringExtinction = scattering(rho, p.scattering);
        result.rosselandExtinction = p.model == OpacityModel::legacy ?
            opacity_detail::checked_coefficient(static_cast<long double>(result.planckAbsorption) +
                result.scatteringExtinction) :
            opacity_detail::power_law(rho, evaluatedTemperature, p.rosseland,
                p.rosselandDensityExponent, p.rosselandTemperatureExponent);
        return result;
    }
    if (p.model != OpacityModel::free_free_thomson) {
        throw std::invalid_argument("Unknown FLD opacity model");
    }
    if (!(p.temperatureFloor > 0.0) || !(p.gaunt >= 0.0) || !(p.lengthUnit > 0.0) ||
        !(p.densityUnit > 0.0) || !(electronFraction >= 0.0) || !(ionChargeSquared >= 0.0) ||
        !std::isfinite(p.gaunt) || !std::isfinite(p.lengthUnit) || !std::isfinite(p.densityUnit) ||
        !std::isfinite(electronFraction) || !std::isfinite(ionChargeSquared)) {
        throw std::invalid_argument("Invalid physical FLD opacity units, composition, Gaunt factor or cutoff");
    }

    // Classical Maxwellian electron-ion free-free absorption, including
    // stimulated emission: alpha_nu=3.692e8*g*ne*sum(Z_i^2 ni)*T^-1/2*
    // nu^-3*(1-exp(-h nu/kT)) [cm^-1]. See Rybicki & Lightman, eq. 5.14b;
    // the explicit normalization is also given in the A&A paper, eq. B.2:
    // https://doi.org/10.1051/0004-6361/202451835 . A constant g is a deliberate
    // approximation. This model omits degeneracy, ionization and Compton effects.
    // h, k and mH match src/physcon.cpp; sigmaT is in cm^2 per free electron.
    constexpr long double hydrogenMass = 1.6733e-24L;
    constexpr long double planckConstant = 6.6260755e-27L;
    constexpr long double boltzmannConstant = 1.380658e-16L;
    constexpr long double thomsonCrossSection = 6.6524587321e-25L;
    long double const density = static_cast<long double>(rho) * p.densityUnit;
    long double const thomson = density / hydrogenMass * electronFraction * thomsonCrossSection;
    long double freeFree = 0.0L;
    if (p.gaunt > 0.0 && electronFraction > 0.0 && ionChargeSquared > 0.0) {
        long double const logAmplitude = std::log(3.692e8L) + std::log(p.gaunt) +
            2.0L * (std::log(density) - std::log(hydrogenMass)) + std::log(electronFraction) +
            std::log(ionChargeSquared) + 3.0L * std::log(planckConstant / boltzmannConstant) -
            3.5L * std::log(evaluatedTemperature);
        freeFree = std::exp(logAmplitude);
    }
    // Integrating alpha_nu B_nu reduces to integral exp(-x) dx = 1.
    result.planckAbsorption = opacity_detail::checked_coefficient(
        freeFree / opacity_detail::planckNormalization * p.lengthUnit);
    result.scatteringExtinction = opacity_detail::checked_coefficient(thomson * p.lengthUnit);
    result.rosselandExtinction = opacity_detail::checked_coefficient(
        opacity_detail::rosseland_mean(freeFree, thomson) * p.lengthUnit);
    return result;
}

}    // namespace octotiger::fld
