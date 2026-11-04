// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace radiation {

inline int subcycle_count(double maximum_ratio) {
    if (!(maximum_ratio >= 0.0) || !std::isfinite(maximum_ratio) ||
        maximum_ratio > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Invalid or excessive radiation substep count");
    }
    return std::max(1, static_cast<int>(std::ceil(maximum_ratio)));
}

struct substep_interval
{
    double begin;
    double end;
    double duration() const {
        return end - begin;
    }
};

// Offsets from the still-unadvanced node clock. Form each endpoint from its
// fraction of the common interval, with the final endpoint exactly dt.
inline substep_interval substep(double dt, int count, int index) {
    if (!(dt >= 0.0) || !std::isfinite(dt) || count < 1 || index < 0 || index >= count) {
        throw std::invalid_argument("Invalid radiation substep interval");
    }
    double const begin = static_cast<double>(static_cast<long double>(dt) * index / count);
    double const end = index + 1 == count ?
        dt :
        static_cast<double>(static_cast<long double>(dt) * (index + 1) / count);
    if (dt > 0 && !(end > begin)) {
        throw std::invalid_argument("Radiation substep underflow");
    }
    return {begin, end};
}

}    // namespace radiation
