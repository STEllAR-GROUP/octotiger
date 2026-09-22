// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/gravity/gravityFields.hpp"

#include <stdexcept>

namespace octotiger::gravity {

void validateDimensionCount(int dimensionCount) {
    if (dimensionCount != mesh::maximumDimensionCount) {
        throw std::invalid_argument("Gravity requires mesh.ndim=3");
    }
}

} // namespace octotiger::gravity
