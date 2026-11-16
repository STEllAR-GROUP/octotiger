// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "octotiger/defs.hpp"

namespace octotiger::fld {

namespace boundary_detail {

inline long double vacuumFluxFactor(long double const r) {
    if (!(r >= 0.0L)) {
        throw std::invalid_argument("FLD boundary gradient parameter must be nonnegative");
    }
    if (std::isinf(r)) {
        return 1.0L;
    }
    long double beta;
    if (r < 1.0e-3L) {
        long double const r2 = r * r;
        beta = 0.5L + r2 * (1.0L / 12.0L + r2 * (-1.0L / 60.0L + r2 / 280.0L));
    } else if (r < 20.0L) {
        // cosh(r)-1 = 2 sinh(r/2)^2 avoids cancellation near the origin.
        long double const halfSinh = std::sinh(0.5L * r);
        beta = std::log1p(2.0L * halfSinh * halfSinh) / (r * std::tanh(r));
    } else {
        // Divide before adding r, avoiding overflow and loss of the correction.
        beta = (1.0L + (std::log1p(std::exp(-r) * std::exp(-r)) - std::log(2.0L)) / r) /
            std::tanh(r);
    }
    return std::clamp(beta, 0.5L, 1.0L);
}

} // namespace boundary_detail

// Modified Marshak (Sanchez--Pomraning) vacuum factor: F_n = beta*c*E_boundary
// for zero incident partial flux. beta=log(cosh R)/(R*tanh R) ranges from 1/2
// for an outgoing hemisphere to 1 for normally directed streaming radiation.
// Castro calls half this factor FLDalpha, including for its rational LP limiter:
// https://github.com/AMReX-Astro/Castro/blob/main/Source/radiation/fluxlimiter.H
// Positive infinity denotes the analytic streaming limit; NaN is invalid.
inline double vacuumFluxFactor(double const r) {
    return static_cast<double>(boundary_detail::vacuumFluxFactor(r));
}

// Return the outgoing face conductance H [length/time], so F_n=H*E_cell.
// E and gradient have the same conventions as fld::closure; extinction is an
// inverse length, c is the transport speed, and halfCellWidth is center-to-face
// distance. The caller supplies lagged INTERIOR closure data. Combining
//   F_n = D*(E_cell-E_boundary)/halfCellWidth = beta*c*E_boundary
// gives H=c/(1/beta+halfCellWidth*extinction/lambda), with D=c*lambda/extinction.
// This is a positive half-cell Robin elimination of the continuum condition;
// it is not Castro's cell-value/extrapolation boundary stencil. Refresh H in
// each nonlinear iteration and use the same flux in the operator and energy
// budget. No negative ghost energy is required.
//
// The interior rational Levermore--Pomraning lambda is evaluated algebraically
// as in fld_closure.hpp, using extended/scaled ratios to preserve extreme
// states without forming D or R^2 when R is large. In exact vacuum, a uniform
// state uses the isotropic H=c/2 limit; a nonzero gradient uses D=c*E/|grad E|
// and beta=1. Zero E with nonzero gradient gives H=0; any uniform E=0 state has
// a finite conductance but still zero flux. Zero incoming flux is a moment
// boundary approximation, not an angularly resolved transport solution.
inline double vacuumConductance(double const energy, std::array<double, NDIM> const& gradient,
    double const extinction, double const c, double const halfCellWidth) {
    if (!(energy >= 0.0) || !std::isfinite(energy) ||
        !(extinction >= 0.0) || !std::isfinite(extinction) ||
        !(c > 0.0) || !std::isfinite(c) ||
        !(halfCellWidth > 0.0) || !std::isfinite(halfCellWidth)) {
        throw std::invalid_argument("Invalid FLD vacuum boundary state, speed or face distance");
    }
    for (double const value : gradient) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Nonfinite FLD vacuum boundary energy gradient");
        }
    }
    long double const norm = std::hypot(static_cast<long double>(gradient[0]),
        static_cast<long double>(gradient[1]), static_cast<long double>(gradient[2]));
    long double beta = 0.5L;
    // inverseDiffusionLength = c/D = chi/lambda, in inverse length units.
    long double inverseDiffusionLength = 3.0L * extinction;
    if (norm > 0.0L) {
        if (energy == 0.0) {
            return 0.0;
        }
        long double const reducedGradient = norm / energy;
        if (extinction == 0.0) {
            beta = 1.0L;
            inverseDiffusionLength = reducedGradient;
        } else {
            long double const inverseR = static_cast<long double>(extinction) / reducedGradient;
            if (inverseR < 1.0L) {
                long double const lambdaR = (1.0L + 2.0L * inverseR) /
                    (1.0L + 3.0L * inverseR + 6.0L * inverseR * inverseR);
                beta = boundary_detail::vacuumFluxFactor(1.0L / inverseR);
                inverseDiffusionLength = reducedGradient / lambdaR;
            } else {
                long double const r = 1.0L / inverseR;
                long double const lambda = (2.0L + r) / (6.0L + 3.0L * r + r * r);
                beta = boundary_detail::vacuumFluxFactor(r);
                inverseDiffusionLength = static_cast<long double>(extinction) / lambda;
            }
        }
    }
    long double const resistance = 1.0L / beta + halfCellWidth * inverseDiffusionLength;
    return static_cast<double>(static_cast<long double>(c) / resistance);
}

} // namespace octotiger::fld
