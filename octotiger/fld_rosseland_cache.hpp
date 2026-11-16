// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <utility>

#include "octotiger/interpolated_cache.hpp"

namespace octotiger::fld::opacity_detail {

// The physical wrapper stores log(chi_R/sigma) against log(A_ff/sigma).
// Nonnegative log opacity enforces chi_R >= sigma at the scattering asymptote;
// the reusable interpolator itself imposes no sign or monotonicity restriction.
class LogRosselandCache
{
    InterpolatedCache interpolator_;

public:
    static constexpr long double spacing = 1.0L / 128.0L;
    using Statistics = InterpolatedCache::Statistics;

    explicit LogRosselandCache(InterpolatedCache::Sampler sampler)
      : interpolator_(std::move(sampler), spacing, 128) {
    }

    long double evaluate(long double const coordinate) const {
        return std::max(0.0L, interpolator_.evaluate(coordinate));
    }

    Statistics statistics() const {
        return interpolator_.statistics();
    }
};

} // namespace octotiger::fld::opacity_detail
