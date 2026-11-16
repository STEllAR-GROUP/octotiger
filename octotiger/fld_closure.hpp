// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "octotiger/defs.hpp"

namespace octotiger::fld {

struct Closure
{
    double lambda = 1.0 / 3.0;
    double eddington = 1.0 / 3.0;
    std::array<std::array<double, NDIM>, NDIM> pressure{};
};

inline double absorption(double rho, double temperature, double coefficient,
    double densityExponent, double temperatureExponent) {
    if (!(rho > 0.0) || !(temperature > 0.0) || !(coefficient >= 0.0) ||
        !std::isfinite(rho) || !std::isfinite(temperature) || !std::isfinite(coefficient) ||
        !std::isfinite(densityExponent) || !std::isfinite(temperatureExponent)) {
        throw std::invalid_argument("FLD opacity requires finite positive density and temperature");
    }
    if (coefficient == 0.0) {
        return 0.0;
    }
    double const result = std::exp(std::log(coefficient) + densityExponent * std::log(rho) +
        temperatureExponent * std::log(temperature));
    if (!(result >= 0.0) || !std::isfinite(result)) {
        throw std::runtime_error("Nonfinite FLD absorption coefficient");
    }
    return result;
}

// Convert a scattering opacity per unit mass to extinction per unit length.
inline double scattering(double rho, double opacity) {
    if (!(rho > 0.0) || !(opacity >= 0.0) ||
        !std::isfinite(rho) || !std::isfinite(opacity)) {
        throw std::invalid_argument(
            "FLD scattering requires finite positive density and nonnegative mass opacity");
    }
    double const result = rho * opacity;
    if (!std::isfinite(result)) {
        throw std::runtime_error("Nonfinite FLD scattering extinction");
    }
    return result;
}

// Levermore--Pomraning closure. Extinction is per unit length.
inline Closure closure(double energy, std::array<double, NDIM> const& gradient, double extinction) {
    if (!(energy >= 0.0) || !(extinction >= 0.0) || !std::isfinite(energy) ||
        !std::isfinite(extinction)) {
        throw std::invalid_argument("FLD closure requires finite nonnegative energy and extinction");
    }
    for (double const value : gradient) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Nonfinite FLD energy gradient");
        }
    }
    double const norm = std::hypot(gradient[0], gradient[1], gradient[2]);
    if (!std::isfinite(norm)) {
        throw std::invalid_argument("Nonfinite FLD gradient magnitude");
    }
    Closure result;
    if (norm > 0.0) {
        double const denominator = extinction * energy;
        double const inverseR = denominator / norm;
        double lambdaR;
        // This branch avoids R^2 overflow in the free-streaming limit.
        if (inverseR < 1.0) {
            lambdaR = (1.0 + 2.0 * inverseR) /
                (1.0 + 3.0 * inverseR + 6.0 * inverseR * inverseR);
            result.lambda = inverseR * lambdaR;
        } else {
            double const r = 1.0 / inverseR;
            result.lambda = (2.0 + r) / (6.0 + 3.0 * r + r * r);
            lambdaR = result.lambda * r;
        }
        result.eddington = result.lambda + lambdaR * lambdaR;
    }
    for (int d = 0; d < NDIM; ++d) {
        for (int e = 0; e < NDIM; ++e) {
            double const nd = norm > 0.0 ? gradient[d] / norm : 0.0;
            double const ne = norm > 0.0 ? gradient[e] / norm : 0.0;
            result.pressure[d][e] = energy *
                ((d == e ? 0.5 * (1.0 - result.eddington) : 0.0) +
                    0.5 * (3.0 * result.eddington - 1.0) * nd * ne);
        }
    }
    return result;
}

}    // namespace octotiger::fld
