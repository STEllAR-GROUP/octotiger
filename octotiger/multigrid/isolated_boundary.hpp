// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace octotiger::multigrid {

namespace isolated_boundary_detail {

inline constexpr double fourPi = 12.566370614359172953850573533118;
inline constexpr int maximumDegree = 4;
inline constexpr int coefficientCount = 25;

inline constexpr int index(int l, int m) {
    return l * l + l + m;
}

inline void validatePosition(std::array<double, 3> const& position) {
    for (double const value : position) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Isolated gravity requires finite positions");
        }
    }
}

inline double sourceMass(std::array<double, 3> const& position,
    double source, double volume) {
    validatePosition(position);
    if (!std::isfinite(source) || source < 0.0 || !std::isfinite(volume)
        || volume <= 0.0) {
        throw std::invalid_argument(
            "Isolated gravity requires a nonnegative source and positive cell volume");
    }
    double const mass = (source / fourPi) * volume;
    if (!std::isfinite(mass)) {
        throw std::overflow_error("Isolated gravity source mass overflow");
    }
    return mass;
}

inline std::array<double, 3> displacement(std::array<double, 3> const& position,
    std::array<double, 3> const& center) {
    validatePosition(position);
    validatePosition(center);
    std::array<double, 3> result{};
    for (int axis = 0; axis != 3; ++axis) {
        result[axis] = position[axis] - center[axis];
    }
    validatePosition(result);
    return result;
}

// q(l,m) = sqrt((l-m)!/(l+m)!) P_l^m, including the Condon-Shortley
// phase. Thus q(l,-m) = (-1)^m q(l,m). Summing both signs of m gives
// the ordinary Legendre addition theorem without any extra 4*pi factor.
inline std::array<double, coefficientCount> angularBasis(double cosine) {
    double const mu = std::clamp(cosine, -1.0, 1.0);
    double const sine = std::sqrt((1.0 - mu) * (1.0 + mu));
    std::array<double, coefficientCount> result{};
    result[index(0, 0)] = 1.0;
    for (int m = 1; m <= maximumDegree; ++m) {
        result[index(m, m)] = -(2 * m - 1) * sine * result[index(m - 1, m - 1)];
    }
    for (int m = 0; m < maximumDegree; ++m) {
        result[index(m + 1, m)] = (2 * m + 1) * mu * result[index(m, m)];
        for (int l = m + 2; l <= maximumDegree; ++l) {
            result[index(l, m)] = ((2 * l - 1) * mu * result[index(l - 1, m)]
                - (l + m - 1) * result[index(l - 2, m)]) / (l - m);
        }
    }
    constexpr std::array<double, 9> factorial{
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0};
    for (int l = 1; l <= maximumDegree; ++l) {
        for (int m = 1; m <= l; ++m) {
            result[index(l, m)] *= std::sqrt(factorial[l - m] / factorial[l + m]);
            result[index(l, -m)] = (m % 2 == 0 ? 1.0 : -1.0) * result[index(l, m)];
        }
    }
    return result;
}

} // namespace isolated_boundary_detail

// The two-pass isolated boundary follows dmarce1/scorpio commit
// b39beb91df1c15631a102f8502867ddabc6eca3e: src/multigrid/multigrid.cpp,
// accumulate_com(), poles_compute(), and compute_phi(). Only active physical
// cells contribute, once each. Source is S = 4*pi*G*rho for Laplacian(phi) = S;
// stored masses include G, so the returned potential needs no further G factor.
struct SourceMoments {
    double mass = 0.0;
    std::array<double, 3> firstMoment{};

    void addSource(std::array<double, 3> const& position, double source, double volume) {
        double const dm = isolated_boundary_detail::sourceMass(position, source, volume);
        mass += dm;
        for (int axis = 0; axis != 3; ++axis) {
            firstMoment[axis] += dm * position[axis];
        }
    }

    SourceMoments& operator+=(SourceMoments const& other) {
        mass += other.mass;
        for (int axis = 0; axis != 3; ++axis) {
            firstMoment[axis] += other.firstMoment[axis];
        }
        return *this;
    }

    // Empty density has no preferred center; the origin is deterministic.
    std::array<double, 3> centerOfMass() const {
        if (!std::isfinite(mass)) {
            throw std::overflow_error("Isolated gravity total mass overflow");
        }
        std::array<double, 3> center{};
        if (mass != 0.0) {
            for (int axis = 0; axis != 3; ++axis) {
                center[axis] = firstMoment[axis] / mass;
            }
            isolated_boundary_detail::validatePosition(center);
        }
        return center;
    }

    template<class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & mass & firstMoment;
    }
};

// COM-centered, degree-four spherical multipoles in Scorpio's real/imaginary
// representation, indexed by l*l+l+m. Each reduction must use the SAME center.
// This intentionally fixes src/legendre.cpp's negative-even-m factor of two:
// using that factor in both source and field basis violates rotation invariance.
// The associated-Legendre recurrence also avoids the old division by sin(theta).
struct IsolatedBoundaryMoments {
    std::array<double, isolated_boundary_detail::coefficientCount> real{};
    std::array<double, isolated_boundary_detail::coefficientCount> imaginary{};

    void addSource(std::array<double, 3> const& position, double source, double volume,
        std::array<double, 3> const& center) {
        double const dm = isolated_boundary_detail::sourceMass(position, source, volume);
        auto const offset = isolated_boundary_detail::displacement(position, center);
        if (dm == 0.0) {
            return;
        }
        double const radius = std::hypot(offset[0], offset[1], offset[2]);
        if (!std::isfinite(radius)) {
            throw std::overflow_error("Isolated gravity source radius overflow");
        }
        // A source exactly at the expansion center has only a monopole.
        if (radius == 0.0) {
            real[0] += dm;
            return;
        }
        auto const basis = isolated_boundary_detail::angularBasis(offset[2] / radius);
        double const azimuth = std::atan2(offset[1], offset[0]);
        double radialMass = dm;
        for (int l = 0; l <= isolated_boundary_detail::maximumDegree; ++l) {
            for (int m = -l; m <= l; ++m) {
                int const p = isolated_boundary_detail::index(l, m);
                double const value = radialMass * basis[p];
                real[p] += value * std::cos(m * azimuth);
                imaginary[p] -= value * std::sin(m * azimuth);
            }
            if (l < isolated_boundary_detail::maximumDegree) {
                radialMass *= radius;
            }
        }
    }

    IsolatedBoundaryMoments& operator+=(IsolatedBoundaryMoments const& other) {
        for (int p = 0; p != isolated_boundary_detail::coefficientCount; ++p) {
            real[p] += other.real[p];
            imaginary[p] += other.imaginary[p];
        }
        return *this;
    }

    // Evaluate at a physical ghost-cell CENTER, as in Scorpio. This exterior
    // expansion converges outside the source-enclosing sphere about center;
    // truncation at l=4 is a boundary approximation, not a direct gravity sum.
    double potential(std::array<double, 3> const& position,
        std::array<double, 3> const& center) const {
        auto const offset = isolated_boundary_detail::displacement(position, center);
        if (real[0] == 0.0) {
            return 0.0;
        }
        double const radius = std::hypot(offset[0], offset[1], offset[2]);
        if (!std::isfinite(radius) || radius == 0.0) {
            throw std::invalid_argument("Isolated gravity field radius must be finite and positive");
        }
        auto const basis = isolated_boundary_detail::angularBasis(offset[2] / radius);
        double const azimuth = std::atan2(offset[1], offset[0]);
        double inversePower = 1.0 / radius;
        double result = 0.0;
        for (int l = 0; l <= isolated_boundary_detail::maximumDegree; ++l) {
            for (int m = -l; m <= l; ++m) {
                int const p = isolated_boundary_detail::index(l, m);
                result -= basis[p] * inversePower * (real[p] * std::cos(m * azimuth)
                    - imaginary[p] * std::sin(m * azimuth));
            }
            inversePower /= radius;
        }
        if (!std::isfinite(result)) {
            throw std::overflow_error("Isolated gravity potential overflow");
        }
        return result;
    }

    template<class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & real & imaginary;
    }
};

} // namespace octotiger::multigrid
