// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace radiation {

// Slab-symmetric kinetic ghost state at the left Marshak boundary. Fluxes Q
// are Fphysical/c; bath_energy = 4 Fin/c specifies the incoming hemispheric
// intensity. The outgoing hemisphere is supplied by the interior M1 state.
//
// Hanawa & Audit (2014), eqs. (15)-(16), doi:10.1016/j.jqsrt.2014.04.014:
// f = 4 beta/(3+beta^2). The corresponding nonnegative Levermore intensity is
// proportional to (1-beta*mu)^(-4). Integrating it over -1 <= mu <= 0 gives
// Eout/E = (1-beta)^3 (beta^2+3 beta+3)/(2(3+beta^2)),
// Qout/E = -(1-beta)^3 (beta+3)/(4(3+beta^2)). These polynomial forms include
// both streaming limits without subtracting nearly equal antiderivatives.
// Adding the incoming isotropic half moments (bath_energy/2,bath_energy/4)
// preserves realizability by construction and gives (2,1) for a cold medium
// with c=1 and Fin=1. No diffusion-profile flux is imposed at startup.
//
// This is a kinetic ghost prescription followed by the existing M1/HLL face
// solver, not an exact enforcement of a diffusion Robin condition at a face.
// Su-Olson remains a diffusion-model reference: reclosure of the mixed angular
// distribution, finite-grid boundary error and the early hyperbolic transient
// can differ. See Menon et al. (2022), section 3.4 and eq. (57),
// https://www.mso.anu.edu.au/~krumholz/publications/2022/menon22a.pdf .
template <class R>
std::array<R, 2> marshakHalfSpaceGhost(R energy, R normal_q, R bath_energy) {
    if (!std::isfinite(energy) || !std::isfinite(normal_q) || !std::isfinite(bath_energy)
        || energy < R(0) || bath_energy < R(0)) {
        throw std::invalid_argument("Invalid Marshak boundary energy or flux");
    }
    R f = 0;
    if (energy == R(0)) {
        if (normal_q != R(0)) {
            throw std::invalid_argument("Marshak boundary vacuum has nonzero flux");
        }
    } else {
        f = normal_q / energy;
        constexpr R tolerance = R(4096) * std::numeric_limits<R>::epsilon();
        if (!std::isfinite(f) || std::abs(f) > R(1) + tolerance) {
            throw std::invalid_argument("Marshak boundary interior flux is not realizable");
        }
        // Only canonicalize the same accumulated-roundoff allowance as M1.
        f = std::clamp(f, R(-1), R(1));
    }
    const R beta = R(3) * f / (R(2) + std::sqrt(R(4) - R(3) * f * f));
    const R beta2 = beta * beta;
    const R one_minus = R(1) - beta;
    const R denominator = R(4) * (R(3) + beta2);
    const R weight = one_minus * one_minus * one_minus / denominator;
    const R outgoing_energy = energy * (R(2) * weight * (beta2 + R(3) * beta + R(3)));
    const R outgoing_q = -energy * (weight * (beta + R(3)));
    return {bath_energy / R(2) + outgoing_energy, bath_energy / R(4) + outgoing_q};
}

} // namespace radiation
