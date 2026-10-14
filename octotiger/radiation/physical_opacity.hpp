// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace radiation {

// The fitted constants below have CGS dimensions. Convert density before
// evaluating them, then convert rho*kappa [cm^-1] to inverse code length.
// Temperature is in kelvin in both the physical and code-unit ideal-gas EOS.
inline std::array<double, 2> fitted_extinction(double density, double temperature,
    double hydrogen, double metals, double mass_unit, double length_unit) {
    if (!(density > 0) || !(temperature > 0) || !(mass_unit > 0) || !(length_unit > 0)
        || !std::isfinite(density) || !std::isfinite(temperature)
        || !std::isfinite(mass_unit) || !std::isfinite(length_unit)
        || !std::isfinite(hydrogen) || !std::isfinite(metals)
        || hydrogen < 0 || metals < 0) {
        throw std::invalid_argument("Fitted FLD opacity requires positive finite density, temperature and units");
    }
    const long double rho = static_cast<long double>(density) * mass_unit
        / length_unit / length_unit / length_unit;
    const long double T = temperature;
    const long double absorption = 4.e25L * (1 + hydrogen) * rho * std::pow(T, -3.5L);
    const long double scattering = (1 + hydrogen) * 0.2L
        / (1 + 2.7e11L * rho / T / T) / (1 + std::pow(T / 4.5e8L, 0.86L));
    std::array<double, 2> result{{
        static_cast<double>(rho * length_unit * (absorption * (metals + 0.001L) + scattering)),
        static_cast<double>(rho * length_unit * 30.262L * absorption * (metals + 0.0001L))}};
    for (double chi : result) {
        if (!(chi >= 0) || !std::isfinite(chi)) {
            throw std::runtime_error("Nonfinite fitted FLD extinction");
        }
    }
    return result;
}

} // namespace radiation
