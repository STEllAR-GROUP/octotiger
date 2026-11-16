#pragma once

#include <array>
#include <cmath>

#include "octotiger/defs.hpp"

namespace octotiger::fld {

inline double recover_internal_energy(double const total, double const kinetic,
    double const tau, double const gamma, double const switch1) {
    double const internal = total - kinetic;
    return internal <= switch1 * total ? std::pow(tau, gamma) : internal;
}

inline double thermal_tau(double const internal, double const gamma) {
    return std::pow(internal, 1.0 / gamma);
}

inline std::array<std::array<double, NDIM>, NDIM> velocity_gradient(
    std::array<std::array<double, NDIM>, NDIM> const& minus,
    std::array<std::array<double, NDIM>, NDIM> const& plus, double const dx) {
    std::array<std::array<double, NDIM>, NDIM> result{};
    for (int row = 0; row < NDIM; ++row) {
        for (int column = 0; column < NDIM; ++column) {
            result[row][column] = (plus[column][row] - minus[column][row]) / (2.0 * dx);
        }
    }
    return result;
}

inline double pressure_work(std::array<std::array<double, NDIM>, NDIM> const& pressure,
    std::array<std::array<double, NDIM>, NDIM> const& gradient) {
    double result = 0.0;
    for (int row = 0; row < NDIM; ++row) {
        for (int column = 0; column < NDIM; ++column) {
            result += pressure[row][column] * gradient[row][column];
        }
    }
    return result;
}

inline double radiation_sound_speed_squared(
    double const energy, double const density, double const lambda, double const eddington_normal) {
    return lambda * (1.0 + eddington_normal) * energy / density;
}

inline double hll_flux(double const ul, double const ur, double const fl, double const fr,
    double const am, double const ap) {
    if (am > 0.0) {
        return fl;
    }
    if (ap < 0.0) {
        return fr;
    }
    if (ap == am) {
        return 0.5 * (fl + fr);
    }
    return (ap * fl - am * fr + ap * am * (ur - ul)) / (ap - am);
}

}    // namespace octotiger::fld
