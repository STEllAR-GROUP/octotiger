// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace octotiger::mesh {

template <class State>
struct ShadowError {
    static constexpr int fieldCount = State::size();

    Real maximumNormalized = 0;
    std::array<Real, fieldCount> maximumAbsolute{};
    std::array<Real, fieldCount> meanAbsolute{};
};

// A data-only representation of the independently evolved hierarchy used for
// refinement estimates. In the HPX runtime each node can own its corresponding
// entry; keeping locations explicit makes the local and distributed forms
// numerically identical.
template <class State>
class ShadowHierarchy {
public:
    using Patch = PatchData<State>;
    static constexpr int fieldCount = State::size();

    void addPatch(BlockLocation location, Patch patch) {
        validateLocation(location, patch.layout());
        auto const [iterator, inserted] = patches_.emplace(location, std::move(patch));
        if (!inserted) {
            throw std::logic_error("A shadow patch already exists at this location");
        }
        (void) iterator;
    }

    bool contains(BlockLocation const& location) const {
        return patches_.contains(location);
    }

    Patch const& patch(BlockLocation const& location) const {
        return patches_.at(location);
    }

    Patch& patch(BlockLocation const& location) {
        return patches_.at(location);
    }

    std::size_t size() const {
        return patches_.size();
    }

    // Each patch chooses its own stable step and carries its own time state.
    // Today callers may return the same global step for every level; a future
    // time-refined driver can return level-dependent steps without changing
    // storage, refinement errors, or messages.
    template <class StableStep, class Advance>
    void advanceTo(BlockLocation const& location, Real targetTime,
        StableStep&& stableStep, Advance&& advance) {
        Patch& shadowPatch = patch(location);
        auto& timeState = shadowPatch.timeState();
        Real const scale = std::max({Real(1), std::abs(timeState.time), std::abs(targetTime)});
        if (targetTime < timeState.time - 64 * epsilonR * scale) {
            throw std::invalid_argument("A shadow patch cannot advance backward in time");
        }
        while (timeState.time < targetTime - 64 * epsilonR * scale) {
            Real const proposed = stableStep(shadowPatch);
            if (!(proposed > 0) || !std::isfinite(proposed)) {
                throw std::runtime_error("Shadow hierarchy received an invalid stable timestep");
            }
            Real const stepSize = std::min(proposed, targetTime - timeState.time);
            Real const oldTime = timeState.time;
            advance(shadowPatch, stepSize);
            if (timeState.time == oldTime) {
                timeState.completeStep(stepSize);
            } else {
                Real const expected = oldTime + stepSize;
                Real const tolerance = 128 * epsilonR * std::max(Real(1), std::abs(expected));
                if (std::abs(timeState.time - expected) > tolerance) {
                    throw std::runtime_error("Shadow stepper advanced to an unexpected physical time");
                }
            }
        }
    }

    // Synchronous driver for a coupled shadow hierarchy. All boundaries are
    // prepared from the same time-level snapshot before any patch advances,
    // which gives the same update as a separately stored hierarchy one level
    // coarser. The per-patch TimeState and time-tagged interval remain valid
    // when level-dependent stepping is added later.
    template <class StableStep, class PrepareBoundaries, class Advance>
    void advanceSynchronizedTo(Real targetTime, StableStep&& stableStep,
        PrepareBoundaries&& prepareBoundaries, Advance&& advance) {
        if (patches_.empty()) {
            return;
        }
        while (true) {
            Real const currentTime = patches_.begin()->second.timeState().time;
            Real const scale = std::max({Real(1), std::abs(currentTime),
                std::abs(targetTime)});
            if (targetTime < currentTime - 64 * epsilonR * scale) {
                throw std::invalid_argument("A shadow hierarchy cannot advance backward in time");
            }
            for (auto const& [location, shadowPatch] : patches_) {
                (void) location;
                TimeState reference;
                reference.time = currentTime;
                if (!shadowPatch.timeState().synchronizedWith(reference)) {
                    throw std::runtime_error("Synchronous shadow patches have different physical times");
                }
            }
            if (currentTime >= targetTime - 64 * epsilonR * scale) {
                break;
            }

            Real stepSize = targetTime - currentTime;
            for (auto const& [location, shadowPatch] : patches_) {
                (void) location;
                Real const proposed = stableStep(shadowPatch);
                if (!(proposed > 0) || !std::isfinite(proposed)) {
                    throw std::runtime_error("Shadow hierarchy received an invalid stable timestep");
                }
                stepSize = std::min(stepSize, proposed);
            }
            TimeInterval const interval{currentTime, currentTime + stepSize};
            prepareBoundaries(*this, interval);
            for (auto& [location, shadowPatch] : patches_) {
                (void) location;
                Real const oldTime = shadowPatch.timeState().time;
                advance(shadowPatch, stepSize);
                if (shadowPatch.timeState().time == oldTime) {
                    shadowPatch.timeState().completeStep(stepSize);
                }
                Real const tolerance = 128 * epsilonR *
                    std::max(Real(1), std::abs(interval.end));
                if (std::abs(shadowPatch.timeState().time - interval.end) > tolerance) {
                    throw std::runtime_error("Shadow patch advanced to an unexpected physical time");
                }
            }
        }
    }

    // Construct the leaf-sized representation of its independently evolved
    // parent. No fine-to-coarse restriction appears anywhere in this path.
    template <class Validator>
    [[nodiscard]] Patch projectParent(BlockLocation const& leafLocation,
        TimeState const& leafTime, Validator&& validState) const {
        if (leafLocation.isRoot()) {
            throw std::logic_error("The root has no coarser shadow solution");
        }
        Patch const& parentPatch = patch(leafLocation.parent());
        if (!parentPatch.timeState().synchronizedWith(leafTime)) {
            throw std::runtime_error("Fine and shadow states must be compared at the same physical time");
        }
        MeshLayout const& parentLayout = parentPatch.layout();
        if (parentLayout.cellsPerActiveDimension() % 2 != 0 || parentLayout.ghostWidth() < 1) {
            throw std::logic_error("Shadow projection requires an even extent and at least one ghost cell");
        }

        PhysicalCoordinates childLower = parentPatch.lower();
        Real const parentWidth = parentLayout.cellsPerActiveDimension() * parentPatch.cellWidth();
        int const childSlot = leafLocation.childSlot();
        for (int axis = 0; axis < parentLayout.dimensionCount(); ++axis) {
            childLower[axis] += ((childSlot >> axis) & 1) * Real(0.5) * parentWidth;
        }
        MeshLayout childLayout(parentLayout.dimensionCount(),
            parentLayout.cellsPerActiveDimension(), parentLayout.ghostWidth());
        Patch result(childLayout, Real(0.5) * parentPatch.cellWidth(), childLower);
        result.timeState() = parentPatch.timeState();

        childLayout.forEachInterior([&](Coordinates const& fineCoordinates, std::size_t fineIndex) {
            Coordinates coarseCoordinates{};
            std::array<Real, maximumDimensionCount> offsets{};
            for (int axis = 0; axis < maximumDimensionCount; ++axis) {
                if (parentLayout.isActive(axis)) {
                    int const halfOffset = ((childSlot >> axis) & 1) *
                        (parentLayout.cellsPerActiveDimension() / 2);
                    coarseCoordinates[axis] = halfOffset + fineCoordinates[axis] / 2;
                    offsets[axis] = fineCoordinates[axis] % 2 == 0 ? Real(-0.25) : Real(0.25);
                }
            }

            State const center = parentPatch.atInterior(coarseCoordinates);
            if (!validState(center)) {
                throw std::runtime_error(
                    "Cannot project an inadmissible parent shadow state");
            }
            State correction{};
            for (int axis = 0; axis < parentLayout.dimensionCount(); ++axis) {
                Coordinates leftCoordinates = parentLayout.storageCoordinates(coarseCoordinates);
                Coordinates rightCoordinates = leftCoordinates;
                --leftCoordinates[axis];
                ++rightCoordinates[axis];
                State const& left = parentPatch.atStorage(leftCoordinates);
                State const& right = parentPatch.atStorage(rightCoordinates);
                for (int field = 0; field < fieldCount; ++field) {
                    correction[field] += offsets[axis] * minmod(
                        center[field] - left[field], right[field] - center[field]);
                }
            }

            State candidate = center + correction;
            if (!validState(candidate)) {
                Real low = 0;
                Real high = 1;
                for (int iteration = 0; iteration < 56; ++iteration) {
                    Real const fraction = Real(0.5) * (low + high);
                    if (validState(center + fraction * correction)) {
                        low = fraction;
                    } else {
                        high = fraction;
                    }
                }
                candidate = center + low * correction;
            }
            result.values()[fineIndex] = candidate;
        });
        return result;
    }

    template <class Validator>
    [[nodiscard]] ShadowError<State> estimateLeafError(BlockLocation const& leafLocation,
        Patch const& finePatch, std::array<Real, fieldCount> const& absoluteTolerance,
        std::array<Real, fieldCount> const& relativeTolerance,
        Validator&& validState) const {
        for (int field = 0; field < fieldCount; ++field) {
            if (!std::isfinite(absoluteTolerance[field]) ||
                !std::isfinite(relativeTolerance[field]) ||
                absoluteTolerance[field] < 0 || relativeTolerance[field] < 0 ||
                (absoluteTolerance[field] == 0 && relativeTolerance[field] == 0)) {
                throw std::invalid_argument(
                    "Shadow error tolerances must be finite, nonnegative, and nonzero");
            }
        }
        Patch const projection = projectParent(leafLocation,
            finePatch.timeState(), std::forward<Validator>(validState));
        if (finePatch.layout().dimensionCount() != projection.layout().dimensionCount() ||
            finePatch.layout().interiorExtents() != projection.layout().interiorExtents()) {
            throw std::invalid_argument("Fine and projected shadow layouts do not match");
        }
        Real const widthScale = std::max({Real(1), std::abs(finePatch.cellWidth()),
            std::abs(projection.cellWidth())});
        if (std::abs(finePatch.cellWidth() - projection.cellWidth()) >
            64 * epsilonR * widthScale) {
            throw std::invalid_argument("Fine and projected shadow cell widths do not match");
        }
        for (int axis = 0; axis < maximumDimensionCount; ++axis) {
            Real const lowerScale = std::max({Real(1),
                std::abs(finePatch.lower()[axis]),
                std::abs(projection.lower()[axis])});
            if (std::abs(finePatch.lower()[axis] - projection.lower()[axis]) >
                64 * epsilonR * lowerScale) {
                throw std::invalid_argument("Fine and projected shadow bounds do not match");
            }
        }

        ShadowError<State> result;
        finePatch.layout().forEachInterior([&](Coordinates const& coordinates, std::size_t) {
            State const& fine = finePatch.atInterior(coordinates);
            State const& coarse = projection.atInterior(coordinates);
            if (!validState(fine)) {
                throw std::invalid_argument("Cannot estimate error for an inadmissible fine state");
            }
            for (int field = 0; field < fieldCount; ++field) {
                if (!std::isfinite(fine[field]) || !std::isfinite(coarse[field])) {
                    throw std::invalid_argument("Shadow error states must be finite");
                }
                Real const difference = std::abs(fine[field] - coarse[field]);
                Real const scale = absoluteTolerance[field] + relativeTolerance[field] *
                    std::max(std::abs(fine[field]), std::abs(coarse[field]));
                if (!(scale > 0) || !std::isfinite(scale)) {
                    throw std::invalid_argument("Shadow error tolerances must define a positive finite scale");
                }
                result.maximumAbsolute[field] = std::max(result.maximumAbsolute[field], difference);
                result.meanAbsolute[field] += difference;
                result.maximumNormalized = std::max(result.maximumNormalized, difference / scale);
            }
        });
        for (Real& mean : result.meanAbsolute) {
            mean /= static_cast<Real>(finePatch.layout().interiorCellCount());
        }
        return result;
    }

private:
    std::unordered_map<BlockLocation, Patch, BlockLocationHash> patches_;

    static Real minmod(Real left, Real right) {
        if (left * right <= 0) {
            return 0;
        }
        return std::copysign(std::min(std::abs(left), std::abs(right)), left);
    }

    static void validateLocation(BlockLocation const& location, MeshLayout const& layout) {
        if (location.level < 0 ||
            location.level >= std::numeric_limits<unsigned>::digits - 1 ||
            location.dimensionCount != layout.dimensionCount()) {
            throw std::invalid_argument("Shadow location and patch dimensionality do not match");
        }
        unsigned const locationsPerAxis = 1u << location.level;
        for (int axis = 0; axis < maximumDimensionCount; ++axis) {
            if (axis < location.dimensionCount) {
                if (location.coordinates[axis] < 0 ||
                    static_cast<unsigned>(location.coordinates[axis]) >= locationsPerAxis) {
                    throw std::invalid_argument("Shadow block coordinate is outside its refinement level");
                }
            } else if (location.coordinates[axis] != 0) {
                throw std::invalid_argument("Inactive shadow coordinates must be zero");
            }
        }
    }
};

} // namespace octotiger::mesh
