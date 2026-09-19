// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace radiation {
// Construct ONLY from globally reduced values. Local dx/opacity must never
// choose a block's cycle count: all levels must issue the same channel keys.
struct SubcyclePlan {
    double duration;
    std::size_t count;
    SubcyclePlan(double gasDt, double radiationDt, bool enabled, std::size_t maximum)
      : duration(gasDt), count(1) {
        if (!(std::isfinite(gasDt) && gasDt>0 && std::isfinite(radiationDt) && radiationDt>0) || maximum==0)
            throw std::runtime_error("Invalid global radiation subcycle interval");
        double const ratio=gasDt/radiationDt;
        double const tolerance=64*std::numeric_limits<double>::epsilon()*std::max(1.0,ratio);
        double const required=std::ceil(ratio-tolerance);
        if ((!enabled && ratio>1+64*std::numeric_limits<double>::epsilon()) || required>double(maximum))
            throw std::runtime_error("Global gas timestep exceeds the allowed radiation subcycle budget");
        if (enabled) count=static_cast<std::size_t>(std::max(1.0,required));
    }
    double offset(std::size_t index) const {
        return index==count ? duration : duration*(double(index)/double(count));
    }
    double dt(std::size_t index) const { return offset(index+1)-offset(index); }
};
}
