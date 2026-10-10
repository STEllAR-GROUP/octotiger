// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gradientRefinement {

enum class Mode { absolute, relative };

inline bool isFinite(double value) noexcept {
    // The application enables -ffast-math, which can erase std::isfinite.
    // Inspect the IEEE exponent so invalid configuration/data stays rejected.
    static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    constexpr std::uint64_t exponent = UINT64_C(0x7ff0000000000000);
    return (bits & exponent) != exponent;
}

struct Parameters {
    Mode mode = Mode::absolute;
    double threshold = 1;
    double floor = 1e-30;
    double coarsenFactor = .8;
    int buffer = 0;

    void validate() const {
        if (mode != Mode::absolute && mode != Mode::relative) {
            throw std::invalid_argument("Invalid gradient refinement mode");
        }
        if (!isFinite(threshold) || threshold <= 0 ||
            !isFinite(floor) || floor <= 0 ||
            !isFinite(coarsenFactor) || coarsenFactor <= 0 || coarsenFactor > 1 ||
            buffer < 0 || buffer > 2) {
            throw std::invalid_argument("Invalid gradient refinement threshold, floor, coarsen factor or buffer");
        }
    }

    long double effectiveThreshold(bool alreadyRefined) const {
        // A refined parent is retained down to a lower threshold. The tree's
        // existing refinement state supplies the hysteresis; no history is added.
        return static_cast<long double>(threshold) * (alreadyRefined ? coarsenFactor : 1);
    }
};

// Maximum magnitude over the six one-sided cell edges: an axiswise infinity
// norm, not a centered derivative or Euclidean gradient norm. A centered
// difference would miss an isolated peak with equal neighbors on both sides.
// Absolute units are field/code-length; relative units are inverse code-length.
// The floor has the selected field's units and applies only inside the logarithm.
inline long double edgeGradient(double center, const std::array<double, 6>& neighbors,
    double dx, Mode mode, double floor) {
    if (!isFinite(dx) || dx <= 0 || !isFinite(floor) || floor <= 0 ||
        (mode != Mode::absolute && mode != Mode::relative)) {
        throw std::invalid_argument("Invalid gradient refinement spacing, mode or floor");
    }
    if (!isFinite(center)) {
        throw std::runtime_error("Nonfinite gradient refinement field");
    }
    const auto transform = [mode, floor](double value) -> long double {
        if (!isFinite(value)) {
            throw std::runtime_error("Nonfinite gradient refinement neighbor");
        }
        return mode == Mode::relative ? std::log(static_cast<long double>(std::max(value, floor))) : value;
    };
    const long double transformedCenter = transform(center);
    long double difference = 0;
    for (double neighbor : neighbors) {
        difference = std::max(difference, std::abs(transform(neighbor) - transformedCenter));
    }
    const long double gradient = difference / static_cast<long double>(dx);
    if (!std::isfinite(gradient)) {
        throw std::runtime_error("Nonfinite gradient refinement indicator");
    }
    return gradient;
}

} // namespace gradientRefinement
