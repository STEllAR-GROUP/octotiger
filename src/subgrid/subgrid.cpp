// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/subgrid/subgrid.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace octotiger {

Subgrid::Subgrid(mesh::MeshLayout layout, Real cellWidth,
    mesh::PhysicalCoordinates lower) :
    layout_(std::move(layout)),
    cellWidth_(cellWidth),
    lower_(lower) {
    if (!(cellWidth_ > 0) || !std::isfinite(cellWidth_)) {
        throw std::invalid_argument("Subgrid cell width must be positive and finite");
    }
    for (int axis = 0; axis < mesh::maximumDimensionCount; ++axis) {
        if (!std::isfinite(lower_[axis]) ||
            (!layout_.isActive(axis) && lower_[axis] != 0)) {
            throw std::invalid_argument(
                "Subgrid bounds must be finite and inactive axes must start at zero");
        }
    }
}

mesh::MeshLayout const& Subgrid::layout() const {
    return layout_;
}

Real Subgrid::cellWidth() const {
    return cellWidth_;
}

mesh::PhysicalCoordinates const& Subgrid::lower() const {
    return lower_;
}

hydro::Fields& Subgrid::enableHydro() {
    if (!hydroData_) {
        hydroData_.emplace(layout_, cellWidth_, lower_);
    }
    return *hydroData_;
}

radiation::Fields& Subgrid::enableRadiation() {
    if (!radiationData_) {
        radiationData_.emplace(layout_, cellWidth_, lower_);
    }
    return *radiationData_;
}

gravity::Fields& Subgrid::enableGravity() {
    gravity::validateDimensionCount(layout_.dimensionCount());
    if (!gravityData_) {
        gravityData_.emplace(layout_, cellWidth_, lower_);
    }
    return *gravityData_;
}

void Subgrid::initializeIndependentShadow() {
    // Each enabled field set is copied only once, at the common synchronization
    // time. Subsequent calls may initialize a newly enabled field set, but can
    // never overwrite an evolved shadow with restricted or otherwise updated
    // fine data.
    if (hydroData_ && !shadowData_.hydro) {
        shadowData_.hydro = hydroData_;
    }
    if (radiationData_ && !shadowData_.radiation) {
        shadowData_.radiation = radiationData_;
    }
    if (gravityData_ && !shadowData_.gravity) {
        shadowData_.gravity = gravityData_;
    }
}

std::optional<hydro::Fields> const& Subgrid::hydroFields() const {
    return hydroData_;
}

std::optional<hydro::Fields>& Subgrid::hydroFields() {
    return hydroData_;
}

std::optional<radiation::Fields> const& Subgrid::radiationFields() const {
    return radiationData_;
}

std::optional<radiation::Fields>& Subgrid::radiationFields() {
    return radiationData_;
}

std::optional<gravity::Fields> const& Subgrid::gravityFields() const {
    return gravityData_;
}

std::optional<gravity::Fields>& Subgrid::gravityFields() {
    return gravityData_;
}

ShadowState const& Subgrid::shadowState() const {
    return shadowData_;
}

hydro::Fields& Subgrid::hydroShadowFields() {
    if (!shadowData_.hydro) {
        throw std::logic_error("Hydro shadow state has not been initialized");
    }
    return *shadowData_.hydro;
}

radiation::Fields& Subgrid::radiationShadowFields() {
    if (!shadowData_.radiation) {
        throw std::logic_error("Radiation shadow state has not been initialized");
    }
    return *shadowData_.radiation;
}

gravity::Fields& Subgrid::gravityShadowFields() {
    if (!shadowData_.gravity) {
        throw std::logic_error("Gravity shadow state has not been initialized");
    }
    return *shadowData_.gravity;
}

} // namespace octotiger
