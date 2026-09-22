// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/math/Real.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace octotiger::mesh {

inline constexpr int maximumDimensionCount = 3;
using Coordinates = std::array<int, maximumDimensionCount>;
using PhysicalCoordinates = std::array<Real, maximumDimensionCount>;

// MeshLayout owns all dimensional indexing. Inactive axes have exactly one
// stored cell and no ghost cells, giving 1x1xN, 1xNxN, and NxNxN storage.
class MeshLayout {
public:
    MeshLayout() = default;
    MeshLayout(int dimensionCount, int cellsPerActiveDimension, int ghostWidth = 0);

    int dimensionCount() const;
    int cellsPerActiveDimension() const;
    int ghostWidth() const;
    bool isActive(int axis) const;
    int interiorExtent(int axis) const;
    int extent(int axis) const;
    Coordinates interiorExtents() const;
    Coordinates extents() const;
    std::size_t interiorCellCount() const;
    std::size_t cellCount() const;
    int childCount() const;
    bool isActiveChild(int childSlot) const;
    std::vector<int> activeChildSlots() const;

    std::size_t index(Coordinates const& storageCoordinates) const;
    std::size_t index(int x, int y = 0, int z = 0) const;
    Coordinates storageCoordinates(Coordinates const& interiorCoordinates) const;
    Coordinates interiorCoordinates(Coordinates const& storageCoordinates) const;
    bool isInterior(Coordinates const& storageCoordinates) const;

    Coordinates faceExtents(int normal) const;
    std::size_t faceCount(int normal) const;
    std::size_t faceIndex(int normal, Coordinates const& faceCoordinates) const;

    Real cellMeasure(Real cellWidth) const;
    Real faceMeasure(Real cellWidth) const;
    PhysicalCoordinates cellCenter(PhysicalCoordinates const& lower,
        Real cellWidth, Coordinates const& interiorCoordinates) const;

    template <class Function>
    void forEachInterior(Function&& function) const {
        for (int z = 0; z < interiorExtent(2); ++z) {
            for (int y = 0; y < interiorExtent(1); ++y) {
                for (int x = 0; x < interiorExtent(0); ++x) {
                    Coordinates const interior{x, y, z};
                    function(interior, index(storageCoordinates(interior)));
                }
            }
        }
    }

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & dimensionCount_;
        archive & cellsPerActiveDimension_;
        archive & ghostWidth_;
        rebuild();
    }

private:
    int dimensionCount_ = maximumDimensionCount;
    int cellsPerActiveDimension_ = 2;
    int ghostWidth_ = 0;
    Coordinates interiorExtents_{2, 2, 2};
    Coordinates extents_{2, 2, 2};
    Coordinates strides_{1, 2, 4};

    void rebuild();
    void validateAxis(int axis) const;
    static std::size_t product(Coordinates const& extents);
};

struct BlockLocation {
    int level = 0;
    Coordinates coordinates{0, 0, 0};
    int dimensionCount = maximumDimensionCount;

    bool isRoot() const;
    BlockLocation parent() const;
    int childSlot() const;
    BlockLocation child(int slot) const;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & level;
        archive & coordinates;
        archive & dimensionCount;
    }

    friend bool operator==(BlockLocation const&, BlockLocation const&) = default;
};

struct BlockLocationHash {
    std::size_t operator()(BlockLocation const& location) const noexcept;
};

// Time is patch-local even while the first implementation advances every
// level synchronously. This deliberately leaves room for temporal refinement:
// boundary and reflux operations can be tagged by physical time intervals,
// rather than assuming one global cycle number.
struct TimeState {
    Real time = 0;
    Real stepSize = 0;
    std::uint64_t step = 0;
    int temporalLevel = 0;
    int substep = 0;

    Real nextTime() const;
    bool synchronizedWith(TimeState const& other,
        Real tolerance = 64 * epsilonR) const;
    void completeStep(Real completedStepSize);

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & time;
        archive & stepSize;
        archive & step;
        archive & temporalLevel;
        archive & substep;
    }
};

struct TimeInterval {
    Real begin = 0;
    Real end = 0;

    Real duration() const;
    bool contains(Real sampleTime, Real tolerance = 64 * epsilonR) const;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & begin;
        archive & end;
    }
};

template <class State>
class PatchData {
public:
    PatchData() = default;
    PatchData(MeshLayout layout, Real cellWidth, PhysicalCoordinates lower = {}) :
        layout_(std::move(layout)),
        cellWidth_(cellWidth),
        lower_(lower),
        values_(layout_.cellCount()) {
        if (!(cellWidth_ > 0) || !std::isfinite(cellWidth_)) {
            throw std::invalid_argument("Patch cell width must be positive and finite");
        }
        for (int axis = 0; axis < maximumDimensionCount; ++axis) {
            if (!std::isfinite(lower_[axis]) ||
                (!layout_.isActive(axis) && lower_[axis] != 0)) {
                throw std::invalid_argument(
                    "Patch bounds must be finite and inactive axes must start at zero");
            }
        }
    }

    MeshLayout const& layout() const {
        return layout_;
    }
    Real cellWidth() const {
        return cellWidth_;
    }
    PhysicalCoordinates const& lower() const {
        return lower_;
    }
    TimeState const& timeState() const {
        return timeState_;
    }
    TimeState& timeState() {
        return timeState_;
    }
    std::vector<State> const& values() const {
        return values_;
    }
    std::vector<State>& values() {
        return values_;
    }
    State const& atStorage(Coordinates const& coordinates) const {
        return values_.at(layout_.index(coordinates));
    }
    State& atStorage(Coordinates const& coordinates) {
        return values_.at(layout_.index(coordinates));
    }
    State const& atInterior(Coordinates const& coordinates) const {
        return atStorage(layout_.storageCoordinates(coordinates));
    }
    State& atInterior(Coordinates const& coordinates) {
        return atStorage(layout_.storageCoordinates(coordinates));
    }

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & layout_;
        archive & cellWidth_;
        archive & lower_;
        archive & timeState_;
        archive & values_;
    }

private:
    MeshLayout layout_;
    Real cellWidth_ = 1;
    PhysicalCoordinates lower_{0, 0, 0};
    TimeState timeState_{};
    std::vector<State> values_;
};

} // namespace octotiger::mesh
