// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

// Cartesian adaptations of Ensman (1994), ApJ 424, 275, sections 5.1--5.3,
// https://doi.org/10.1086/173889. See radiation_results/ENSMAN.md for departures
// from the original spherical shells, moving piston, and relaxed atmosphere.
namespace radiationHydroTests {
enum class Case { subcriticalShock, supercriticalShock, blastWave };
using Point = std::array<double, 3>;
inline constexpr double c = 2.99792458e10, kb = 1.380658e-16, mh = 1.6733e-24;
inline constexpr double sigma = 5.67051e-5, ar = 4 * sigma / c;
inline constexpr double gamma = 5. / 3., mu = .5;
inline constexpr double kappa = 3.115e-10 / 7.78e-10; // cm^2/g; pure absorption
inline constexpr double pi = std::numbers::pi;

struct Units {
    double mass = 1, length = 1, time = 1;
    void validate() const {
        for (double x : {mass, length, time}) {
            if (!std::isfinite(x) || x <= 0) {
                throw std::invalid_argument("Radiation-hydro unit factors must be finite and positive");
            }
        }
    }
    double density() const { return mass / (length * length * length); }
    double velocity() const { return length / time; }
    double energyDensity() const { return mass / (length * time * time); }
    double flux() const { return energyDensity() * velocity(); }
};
struct Setup {
    double halfLength, density, velocity, stopTime;
    double coreRadius = 0, luminosity = 0, depositedEnergy = 0;
};
inline Setup setup(Case problem) {
    if (problem == Case::blastWave) {
        return {1e13, 1e-6, 0, 2.2e4, 1e12, 1e34, 1e47};
    }
    return {3.5e10, 7.78e-10, problem == Case::subcriticalShock ? -6e5 : -2e6,
        problem == Case::subcriticalShock ? 3.83e4 : 7.5e3};
}
struct State {
    double rho = 0;
    Point momentum{};
    double gasEnergy = 0, internalEnergy = 0;
    std::array<double, 4> radiation{}; // E, physical Fx,Fy,Fz
};
inline State pointState(Case problem, Point position) {
    const auto p = setup(problem);
    State u;
    double temperature;
    if (problem != Case::blastWave) {
        // Planar wall-piston frame, with a fixed upstream reservoir at x=+L/2.
        const double s = std::clamp((position[0] + p.halfLength) / (2 * p.halfLength), 0., 1.);
        temperature = 85 - 75 * s;
        u.rho = p.density;
        u.momentum[0] = u.rho * p.velocity;
        u.radiation[0] = ar * std::pow(temperature, 4);
        // Isotropic comoving LTE transformed to the lab frame to first order v/c.
        u.radiation[1] = (4. / 3.) * p.velocity * u.radiation[0];
    } else {
        const double r = std::hypot(position[0], position[1], position[2]);
        const double radius = std::max(r, p.coreRadius);
        u.rho = p.density * std::pow(p.coreRadius / radius, 7);
        const double chi0 = kappa * p.density;
        const double edgeEnergy = p.luminosity / (4 * pi * c * p.halfLength * p.halfLength);
        const double potential = 3 * chi0 * p.luminosity / (32 * pi * c * p.coreRadius);
        // Integrate dE/dr=-3 chi L/(4 pi c r^2) through the r^-7 envelope.
        // Fill the central hole with a constant-density, uniformly luminous core.
        const double clippedRadius = std::min(radius, p.halfLength);
        u.radiation[0] = edgeEnergy + potential *
            (std::pow(p.coreRadius / clippedRadius, 8) - std::pow(p.coreRadius / p.halfLength, 8));
        if (r < p.coreRadius) {
            u.radiation[0] += 3 * chi0 * p.luminosity / (8 * pi * c * p.coreRadius) *
                (1 - r * r / (p.coreRadius * p.coreRadius));
        }
        const double fluxCoefficient = p.luminosity / (4 * pi * radius * radius * radius);
        for (int d = 0; d < 3; ++d) {
            u.radiation[1 + d] = fluxCoefficient * position[d];
        }
        // Diffusion ceases to apply in the optically thin outer envelope.
        // Join it to the streaming energy floor while preserving luminosity;
        // a small margin keeps positive cell quadrature inside the M1 cone.
        u.radiation[0] = std::max(u.radiation[0],
            (1 + 1e-12) * fluxCoefficient * r / c);
        temperature = std::pow(u.radiation[0] / ar, .25);
    }
    u.internalEnergy = u.rho * kb * temperature / (mu * mh * (gamma - 1));
    u.gasEnergy = u.internalEnergy;
    for (double momentum : u.momentum) {
        u.gasEnergy += .5 * momentum * momentum / u.rho;
    }
    return u;
}

inline State initialState(Case problem, Point position, double dx, Units units = {}) {
    units.validate();
    if (!std::isfinite(dx) || dx < 0) {
        throw std::invalid_argument("Invalid radiation-hydro cell width");
    }
    for (double& x : position) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument("Invalid radiation-hydro cell position");
        }
        x *= units.length;
    }
    dx *= units.length;
    // Positive quadrature averages conserved quantities, including T^4, rather
    // than assigning a point temperature to a finite-volume radiation cell.
    constexpr double nodes[]{-.8611363115940526, -.3399810435848563, .3399810435848563, .8611363115940526};
    constexpr double weights[]{.3478548451374538, .6521451548625461, .6521451548625461, .3478548451374538};
    State u;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                const double weight = weights[i] * weights[j] * weights[k] / 8;
                const auto sample = pointState(problem,
                    {position[0] + dx * nodes[i] / 2, position[1] + dx * nodes[j] / 2, position[2] + dx * nodes[k] / 2});
                u.rho += weight * sample.rho;
                u.gasEnergy += weight * sample.gasEnergy;
                u.internalEnergy += weight * sample.internalEnergy;
                for (int d = 0; d < 3; ++d) {
                    u.momentum[d] += weight * sample.momentum[d];
                }
                for (int f = 0; f < 4; ++f) {
                    u.radiation[f] += weight * sample.radiation[f];
                }
            }
        }
    }
    if (problem == Case::blastWave) {
        const auto p = setup(problem);
        // A resolved Gaussian replaces the unspecified innermost Lagrangian
        // zone. Integrate exactly per cell so the total deposited energy is
        // 1e47 erg at every uniform-grid resolution, even for coarse smoke runs.
        double density = p.depositedEnergy;
        for (double x : position) {
            const double factor = dx == 0 ? std::exp(-x * x / (p.coreRadius * p.coreRadius)) /
                (std::sqrt(pi) * p.coreRadius) :
                .5 * (std::erf((x + dx / 2) / p.coreRadius) - std::erf((x - dx / 2) / p.coreRadius)) / dx;
            density *= factor / std::erf(p.halfLength / p.coreRadius);
        }
        u.internalEnergy += density;
        u.gasEnergy += density;
    }
    u.rho /= units.density();
    u.internalEnergy /= units.energyDensity();
    u.gasEnergy /= units.energyDensity();
    for (double& momentum : u.momentum) {
        momentum /= units.density() * units.velocity();
    }
    u.radiation[0] /= units.energyDensity();
    for (int f = 1; f < 4; ++f) {
        u.radiation[f] /= units.flux();
    }
    return u;
}
}
