// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <cmath>
#include <limits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OCTOTIGER_AP_INLINE __host__ __device__ inline
#else
#define OCTOTIGER_AP_INLINE inline
#endif

namespace radiation {
namespace ap_detail {
template<class T> inline constexpr T invalid_flux = std::numeric_limits<T>::quiet_NaN();
}

// A conservative face-energy flux with a compact diffusion limit. chi is an
// inverse length (rho*kappa). Average chi, equivalently use the harmonic mean
// of the two diffusion coefficients, for a face between equal-width cells.
//
// For fixed physical chi and dx -> 0, 1-alpha=O(dx^2), so the correction does
// not lower the spatial order of the reconstructed HLL flux. For chi*dx ->
// infinity at fixed mesh, alpha=O((chi*dx)^-2): even an unreconstructed HLL
// jump then vanishes relative to the physical diffusion flux. The remaining
// face flux is c_hat*Q_adv-c_hat*(E_R-E_L)/(3*chi*dx). Using cell averages in
// that term gives a compact Laplacian instead of a checkerboard-null central
// gradient followed by another central derivative. It is used in an
// explicit radiation stage under the transport CFL; it is not a claim of a
// positivity-preserving flux (stage rejection remains necessary).
//
// Diffusion is only consistent when the energy varies little over a mean free
// path. Multiply its blend by phi=1/(1+g^2), where
// g=|E_R-E_L|/(chi*dx*min(E_L,E_R)). An unresolved jump into vacuum therefore
// uses HLL. For fixed positive cell energies and chi -> infinity, 1-phi is
// O(chi^-2), retaining the compact diffusion limit. This consistency blend
// does not replace the stage's physical-state check or timestep rejection.
template<class T>
OCTOTIGER_AP_INLINE T apEnergyFlux(T hll, T leftEnergy, T rightEnergy,
    T leftAdvectiveQ, T rightAdvectiveQ, T leftChi, T rightChi, T dx, T chat,
    T* compactWeight = nullptr)
{
    if (compactWeight) {
        *compactWeight = T(0);
    }
    if (!std::isfinite(leftEnergy) || !std::isfinite(rightEnergy)
        || leftEnergy < T(0) || rightEnergy < T(0)
        || !std::isfinite(leftChi) || !std::isfinite(rightChi)
        || !std::isfinite(dx) || !std::isfinite(chat)
        || leftChi < T(0) || rightChi < T(0) || !(dx > T(0)) || !(chat > T(0))) {
        return ap_detail::invalid_flux<T>;
    }
    const T chi = T(.5) * leftChi + T(.5) * rightChi;
    if (chi == T(0)) {
        return hll;
    }
    // Compare products without forming an overflowing optical depth. In the
    // thick branch compute its reciprocal using the larger factor first.
    const bool thin = (chi <= T(1) && dx <= T(1))
        || (dx > T(1) ? chi <= T(1) / dx : dx <= T(1) / chi);
    T alpha, weight, diffusionWeight, tauOrInverse;
    if (thin) {
        const T tau = chi * dx;
        tauOrInverse = tau;
        alpha = T(1) / (T(1) + tau * tau);
        weight = tau * tau * alpha;
        diffusionWeight = tau * alpha;
    } else {
        const T inverse = dx > T(1) ? (T(1) / dx) / chi : (T(1) / chi) / dx;
        tauOrInverse = inverse;
        weight = T(1) / (T(1) + inverse * inverse);
        alpha = inverse * inverse * weight;
        diffusionWeight = inverse * weight;
    }
    const T jump = ::fabs(rightEnergy - leftEnergy);
    const T minimumEnergy = leftEnergy < rightEnergy ? leftEnergy : rightEnergy;
    if (jump != T(0)) {
        if (minimumEnergy == T(0)) {
            return hll;
        }
        // Scale before squaring. The normal path never multiplies by an
        // optical depth greater than one and uses no logarithms.
        const T numerator = thin ? tauOrInverse * minimumEnergy : minimumEnergy;
        const T denominator = thin ? jump : tauOrInverse * jump;
        T phi, complement;
        if (numerator == T(0) || denominator == T(0)) {
            // A product (or inverse optical depth) underflowed. Its factors
            // can still give an order-one g when the energies also span a
            // very large range, so do not silently assume phi is zero/one.
            const T logGradient = ::log(jump) - ::log(minimumEnergy) - ::log(chi) - ::log(dx);
            const T ratio = ::exp(-::fabs(logGradient));
            const T smallWeight = ratio * ratio / (T(1) + ratio * ratio);
            const T largeWeight = T(1) / (T(1) + ratio * ratio);
            phi = logGradient >= T(0) ? smallWeight : largeWeight;
            complement = logGradient >= T(0) ? largeWeight : smallWeight;
        } else if (numerator >= denominator) {
            const T ratio = denominator / numerator;
            phi = T(1) / (T(1) + ratio * ratio);
            complement = ratio * ratio * phi;
        } else {
            const T ratio = numerator / denominator;
            complement = T(1) / (T(1) + ratio * ratio);
            phi = ratio * ratio * complement;
        }
        // Avoid 1-weight*phi: cancellation would lose the O(chi^-2) HLL
        // coefficient in the diffusion limit.
        alpha += weight * complement;
        weight *= phi;
        diffusionWeight *= phi;
    }
    if (compactWeight) {
        *compactWeight = weight;
    }
    return alpha * hll + weight * chat * T(.5) * (leftAdvectiveQ + rightAdvectiveQ)
        - chat / T(3) * diffusionWeight * (rightEnergy - leftEnergy);
}

} // namespace radiation
#undef OCTOTIGER_AP_INLINE
