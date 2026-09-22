// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/math/Vector.hpp"
#include "octotiger/mesh.hpp"

namespace octotiger::gravity {

class State : public Vector<Real, 4> {
public:
    using Vector<Real, 4>::Vector;
    constexpr State() = default;
    constexpr State(Vector<Real, 4> const& state) :
        Vector<Real, 4>(state) {
    }

    constexpr Real& potential() {
        return (*this)[0];
    }
    constexpr Real potential() const {
        return (*this)[0];
    }
    constexpr Real& acceleration(int axis) {
        return (*this)[axis + 1];
    }
    constexpr Real acceleration(int axis) const {
        return (*this)[axis + 1];
    }
};

using Fields = mesh::PatchData<State>;

void validateDimensionCount(int dimensionCount);

} // namespace octotiger::gravity
