// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/mesh.hpp"

#include <algorithm>
#include <functional>
#include <limits>

namespace octotiger::mesh {

MeshLayout::MeshLayout(int dimensionCount, int cellsPerActiveDimension, int ghostWidth) :
    dimensionCount_(dimensionCount),
    cellsPerActiveDimension_(cellsPerActiveDimension),
    ghostWidth_(ghostWidth) {
    rebuild();
}

int MeshLayout::dimensionCount() const {
    return dimensionCount_;
}

int MeshLayout::cellsPerActiveDimension() const {
    return cellsPerActiveDimension_;
}

int MeshLayout::ghostWidth() const {
    return ghostWidth_;
}

bool MeshLayout::isActive(int axis) const {
    validateAxis(axis);
    return axis < dimensionCount_;
}

int MeshLayout::interiorExtent(int axis) const {
    validateAxis(axis);
    return interiorExtents_[axis];
}

int MeshLayout::extent(int axis) const {
    validateAxis(axis);
    return extents_[axis];
}

Coordinates MeshLayout::interiorExtents() const {
    return interiorExtents_;
}

Coordinates MeshLayout::extents() const {
    return extents_;
}

std::size_t MeshLayout::interiorCellCount() const {
    return product(interiorExtents_);
}

std::size_t MeshLayout::cellCount() const {
    return product(extents_);
}

int MeshLayout::childCount() const {
    return 1 << dimensionCount_;
}

bool MeshLayout::isActiveChild(int childSlot) const {
    return childSlot >= 0 && childSlot < childCount();
}

std::vector<int> MeshLayout::activeChildSlots() const {
    std::vector<int> slots(childCount());
    for (int slot = 0; slot < childCount(); ++slot) {
        slots[slot] = slot;
    }
    return slots;
}

std::size_t MeshLayout::index(Coordinates const& storageCoordinates) const {
    std::size_t result = 0;
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        if (storageCoordinates[axis] < 0 || storageCoordinates[axis] >= extents_[axis]) {
            throw std::out_of_range("Mesh storage coordinate is outside the allocated extent");
        }
        result += static_cast<std::size_t>(storageCoordinates[axis]) * strides_[axis];
    }
    return result;
}

std::size_t MeshLayout::index(int x, int y, int z) const {
    return index(Coordinates{x, y, z});
}

Coordinates MeshLayout::storageCoordinates(Coordinates const& interiorCoordinates) const {
    Coordinates result{};
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        if (interiorCoordinates[axis] < 0 || interiorCoordinates[axis] >= interiorExtents_[axis]) {
            throw std::out_of_range("Mesh interior coordinate is outside the active extent");
        }
        result[axis] = isActive(axis) ? interiorCoordinates[axis] + ghostWidth_ : 0;
    }
    return result;
}

Coordinates MeshLayout::interiorCoordinates(Coordinates const& storageCoordinates) const {
    Coordinates result{};
    if (!isInterior(storageCoordinates)) {
        throw std::out_of_range("Mesh storage coordinate is not an interior cell");
    }
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        result[axis] = isActive(axis) ? storageCoordinates[axis] - ghostWidth_ : 0;
    }
    return result;
}

bool MeshLayout::isInterior(Coordinates const& storageCoordinates) const {
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        if (storageCoordinates[axis] < 0 || storageCoordinates[axis] >= extents_[axis]) {
            return false;
        }
        if (isActive(axis) && (storageCoordinates[axis] < ghostWidth_ ||
                                  storageCoordinates[axis] >= ghostWidth_ + cellsPerActiveDimension_)) {
            return false;
        }
        if (!isActive(axis) && storageCoordinates[axis] != 0) {
            return false;
        }
    }
    return true;
}

Coordinates MeshLayout::faceExtents(int normal) const {
    validateAxis(normal);
    if (!isActive(normal)) {
        throw std::invalid_argument("Inactive axes do not own face arrays");
    }
    Coordinates result = interiorExtents_;
    ++result[normal];
    return result;
}

std::size_t MeshLayout::faceCount(int normal) const {
    return product(faceExtents(normal));
}

std::size_t MeshLayout::faceIndex(int normal, Coordinates const& faceCoordinates) const {
    Coordinates const dimensions = faceExtents(normal);
    std::size_t stride = 1;
    std::size_t result = 0;
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        if (faceCoordinates[axis] < 0 || faceCoordinates[axis] >= dimensions[axis]) {
            throw std::out_of_range("Face coordinate is outside the face array");
        }
        result += static_cast<std::size_t>(faceCoordinates[axis]) * stride;
        stride *= static_cast<std::size_t>(dimensions[axis]);
    }
    return result;
}

Real MeshLayout::cellMeasure(Real cellWidth) const {
    if (!(cellWidth > 0) || !std::isfinite(cellWidth)) {
        throw std::invalid_argument("Cell width must be positive and finite");
    }
    Real result = 1;
    for (int axis = 0; axis < dimensionCount_; ++axis) {
        result *= cellWidth;
    }
    return result;
}

Real MeshLayout::faceMeasure(Real cellWidth) const {
    if (!(cellWidth > 0) || !std::isfinite(cellWidth)) {
        throw std::invalid_argument("Cell width must be positive and finite");
    }
    Real result = 1;
    for (int axis = 1; axis < dimensionCount_; ++axis) {
        result *= cellWidth;
    }
    return result;
}

PhysicalCoordinates MeshLayout::cellCenter(PhysicalCoordinates const& lower,
    Real cellWidth, Coordinates const& interiorCoordinates) const {
    PhysicalCoordinates result{};
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        if (interiorCoordinates[axis] < 0 || interiorCoordinates[axis] >= interiorExtents_[axis]) {
            throw std::out_of_range("Cell-center coordinate is outside the interior");
        }
        result[axis] = isActive(axis) ? lower[axis] + (interiorCoordinates[axis] + Real(0.5)) * cellWidth : Real(0);
    }
    return result;
}

void MeshLayout::rebuild() {
    if (dimensionCount_ < 1 || dimensionCount_ > maximumDimensionCount) {
        throw std::invalid_argument("Mesh dimension count must be 1, 2, or 3");
    }
    if (cellsPerActiveDimension_ < 2 || cellsPerActiveDimension_ % 2 != 0) {
        throw std::invalid_argument("Active mesh extent must be a positive even number");
    }
    if (ghostWidth_ < 0) {
        throw std::invalid_argument("Mesh ghost width cannot be negative");
    }
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        bool const active = axis < dimensionCount_;
        interiorExtents_[axis] = active ? cellsPerActiveDimension_ : 1;
        extents_[axis] = active ? cellsPerActiveDimension_ + 2 * ghostWidth_ : 1;
    }
    strides_[0] = 1;
    for (int axis = 1; axis < maximumDimensionCount; ++axis) {
        strides_[axis] = strides_[axis - 1] * extents_[axis - 1];
    }
}

void MeshLayout::validateAxis(int axis) const {
    if (axis < 0 || axis >= maximumDimensionCount) {
        throw std::out_of_range("Mesh axis must be 0, 1, or 2");
    }
}

std::size_t MeshLayout::product(Coordinates const& extents) {
    std::size_t result = 1;
    for (int extent : extents) {
        result *= static_cast<std::size_t>(extent);
    }
    return result;
}

bool BlockLocation::isRoot() const {
    return level == 0;
}

BlockLocation BlockLocation::parent() const {
    if (isRoot()) {
        throw std::logic_error("The root block has no parent");
    }
    if (level < 0 || dimensionCount < 1 ||
        dimensionCount > maximumDimensionCount) {
        throw std::invalid_argument("Invalid block location");
    }
    BlockLocation result = *this;
    --result.level;
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        result.coordinates[axis] = axis < dimensionCount ? coordinates[axis] / 2 : 0;
    }
    return result;
}

int BlockLocation::childSlot() const {
    if (isRoot()) {
        throw std::logic_error("The root block is not a child");
    }
    if (level < 0 || dimensionCount < 1 ||
        dimensionCount > maximumDimensionCount) {
        throw std::invalid_argument("Invalid block location");
    }
    int result = 0;
    for (int axis = 0; axis < dimensionCount; ++axis) {
        result |= (coordinates[axis] & 1) << axis;
    }
    return result;
}

BlockLocation BlockLocation::child(int slot) const {
    if (level < 0 ||
        level >= std::numeric_limits<unsigned>::digits - 2 ||
        dimensionCount < 1 || dimensionCount > maximumDimensionCount ||
        slot < 0 || slot >= (1 << dimensionCount)) {
        throw std::invalid_argument("Invalid child slot for block dimensionality");
    }
    BlockLocation result = *this;
    ++result.level;
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        result.coordinates[axis] = axis < dimensionCount ? 2 * coordinates[axis] + ((slot >> axis) & 1) : 0;
    }
    return result;
}

std::size_t BlockLocationHash::operator()(BlockLocation const& location) const noexcept {
    std::size_t result = std::hash<int>{}(location.level);
    result ^= std::hash<int>{}(location.dimensionCount) + 0x9e3779b9 + (result << 6) + (result >> 2);
    for (int coordinate : location.coordinates) {
        result ^= std::hash<int>{}(coordinate) + 0x9e3779b9 + (result << 6) + (result >> 2);
    }
    return result;
}

Real TimeState::nextTime() const {
    return time + stepSize;
}

bool TimeState::synchronizedWith(TimeState const& other, Real tolerance) const {
    Real const scale = std::max({Real(1), std::abs(time), std::abs(other.time)});
    return std::abs(time - other.time) <= tolerance * scale;
}

void TimeState::completeStep(Real completedStepSize) {
    if (!(completedStepSize > 0) || !std::isfinite(completedStepSize)) {
        throw std::invalid_argument("Completed timestep must be positive and finite");
    }
    stepSize = completedStepSize;
    time += completedStepSize;
    ++step;
    ++substep;
}

Real TimeInterval::duration() const {
    if (end < begin) {
        throw std::logic_error("Time interval ends before it begins");
    }
    return end - begin;
}

bool TimeInterval::contains(Real sampleTime, Real tolerance) const {
    Real const scale = std::max({Real(1), std::abs(begin), std::abs(end), std::abs(sampleTime)});
    return sampleTime >= begin - tolerance * scale && sampleTime <= end + tolerance * scale;
}

} // namespace octotiger::mesh
