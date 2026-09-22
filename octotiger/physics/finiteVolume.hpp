// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace octotiger::physics {

std::string_view finiteVolumeSchemeName();

enum class Limiter {
    Minmod,
    VanLeer,
    MinmodTheta
};

enum class BoundaryCondition {
    Outflow,
    Periodic,
    Reflecting
};

struct BoundaryConditions {
    std::array<BoundaryCondition, mesh::maximumDimensionCount> lower{
        BoundaryCondition::Outflow, BoundaryCondition::Outflow, BoundaryCondition::Outflow};
    std::array<BoundaryCondition, mesh::maximumDimensionCount> upper{
        BoundaryCondition::Outflow, BoundaryCondition::Outflow, BoundaryCondition::Outflow};

    [[nodiscard]] static BoundaryConditions periodic(int dimensionCount) {
        if (dimensionCount < 1 || dimensionCount > mesh::maximumDimensionCount) {
            throw std::invalid_argument("Boundary dimensionality must lie in [1,3]");
        }
        BoundaryConditions result;
        for (int axis = 0; axis < dimensionCount; ++axis) {
            result.lower[axis] = BoundaryCondition::Periodic;
            result.upper[axis] = BoundaryCondition::Periodic;
        }
        return result;
    }
};

inline Real limitedSlope(Real leftDifference, Real rightDifference,
    Limiter limiter, Real theta = Real(1.5)) {
    if (!((leftDifference > 0 && rightDifference > 0) ||
            (leftDifference < 0 && rightDifference < 0))) {
        return 0;
    }
    Real const sign = std::copysign(Real(1), leftDifference);
    Real const left = std::abs(leftDifference);
    Real const right = std::abs(rightDifference);
    switch (limiter) {
    case Limiter::Minmod:
        return sign * std::min(left, right);
    case Limiter::VanLeer:
        return sign * (Real(2) * left * right / (left + right));
    case Limiter::MinmodTheta: {
        Real const centered = Real(0.5) * (left + right);
        return sign * std::min({theta * left, centered, theta * right});
    }
    }
    throw std::logic_error("Unknown finite-volume limiter");
}

template <class System>
void fillGhostCells(mesh::PatchData<typename System::State>& patch,
    BoundaryConditions const& boundaries, System const& system) {
    using State = typename System::State;
    mesh::MeshLayout const& layout = patch.layout();
    int const ghostWidth = layout.ghostWidth();
    if (ghostWidth == 0) {
        return;
    }
    mesh::Coordinates const extents = layout.extents();
    for (int z = 0; z < extents[2]; ++z) {
        for (int y = 0; y < extents[1]; ++y) {
            for (int x = 0; x < extents[0]; ++x) {
                mesh::Coordinates const destination{x, y, z};
                if (layout.isInterior(destination)) {
                    continue;
                }
                mesh::Coordinates source = destination;
                std::array<bool, mesh::maximumDimensionCount> reflect{};
                for (int axis = 0; axis < layout.dimensionCount(); ++axis) {
                    int coordinate = destination[axis] - ghostWidth;
                    int const count = layout.cellsPerActiveDimension();
                    if (coordinate < 0) {
                        switch (boundaries.lower[axis]) {
                        case BoundaryCondition::Periodic:
                            coordinate = (coordinate % count + count) % count;
                            break;
                        case BoundaryCondition::Outflow:
                            coordinate = 0;
                            break;
                        case BoundaryCondition::Reflecting:
                            coordinate = -coordinate - 1;
                            reflect[axis] = true;
                            break;
                        }
                    } else if (coordinate >= count) {
                        switch (boundaries.upper[axis]) {
                        case BoundaryCondition::Periodic:
                            coordinate %= count;
                            break;
                        case BoundaryCondition::Outflow:
                            coordinate = count - 1;
                            break;
                        case BoundaryCondition::Reflecting:
                            coordinate = 2 * count - coordinate - 1;
                            reflect[axis] = true;
                            break;
                        }
                    }
                    source[axis] = coordinate + ghostWidth;
                }
                State state = patch.atStorage(source);
                for (int axis = 0; axis < layout.dimensionCount(); ++axis) {
                    if (reflect[axis]) {
                        state = system.reflected(state, axis);
                    }
                }
                patch.atStorage(destination) = state;
            }
        }
    }
}

// Dimensionally unsplit MUSCL-Hancock. Reconstruction is directional, the
// half-step predictor contains the divergence from every active direction,
// and the only intercell quantities are face-centered fluxes.
template <class System, int dimensionCount>
class MusclHancock {
public:
    static_assert(dimensionCount >= 1 && dimensionCount <= mesh::maximumDimensionCount);
    using State = typename System::State;
    using Reconstruction = typename System::Reconstruction;
    using FaceFluxes = std::array<std::vector<State>, dimensionCount>;

    struct StepResult {
        mesh::TimeInterval timeInterval;
        FaceFluxes faceFluxes;
    };

    MusclHancock(System system, Limiter limiter = Limiter::VanLeer,
        Real theta = Real(1.5)) :
        system_(std::move(system)),
        limiter_(limiter),
        theta_(theta) {
        if (!(theta_ >= 1 && theta_ <= 2)) {
            throw std::invalid_argument("minmod-theta parameter must lie in [1,2]");
        }
    }

    [[nodiscard]] Real stableTimestep(mesh::PatchData<State> const& patch,
        Real courantNumber) const {
        if (patch.layout().dimensionCount() != dimensionCount ||
            !(courantNumber > 0 && courantNumber <= Real(0.5))) {
            throw std::invalid_argument("Invalid MUSCL-Hancock layout or Courant number");
        }
        std::array<Real, dimensionCount> maximumSpeed{};
        patch.layout().forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
            State const& state = patch.values()[index];
            for (int axis = 0; axis < dimensionCount; ++axis) {
                maximumSpeed[axis] = std::max(maximumSpeed[axis],
                    system_.maximumSignalSpeed(state, axis));
            }
        });
        Real inverseStep = 0;
        for (Real speed : maximumSpeed) {
            inverseStep += speed / patch.cellWidth();
        }
        if (!(inverseStep > 0) || !std::isfinite(inverseStep)) {
            throw std::runtime_error("No finite positive signal speed in finite-volume patch");
        }
        return courantNumber / inverseStep;
    }

    StepResult advance(mesh::PatchData<State>& patch, Real stepSize,
        BoundaryConditions const& boundaries) const {
        return advanceWithBoundaryUpdater(patch, stepSize,
            [&](mesh::PatchData<State>& boundaryPatch, Real) {
                fillGhostCells(boundaryPatch, boundaries, system_);
            });
    }

    // AMR drivers can supply same-level, coarse/fine, or physical boundary
    // data at the requested physical time. The time-tagged interface avoids
    // baking a global-cycle assumption into the spatial integrator and can
    // later interpolate boundaries for level-dependent timesteps.
    template <class BoundaryUpdater>
    StepResult advanceWithBoundaryUpdater(mesh::PatchData<State>& patch,
        Real stepSize, BoundaryUpdater&& updateBoundaries) const {
        mesh::MeshLayout const& layout = patch.layout();
        if (layout.dimensionCount() != dimensionCount || layout.ghostWidth() < 2) {
            throw std::invalid_argument("MUSCL-Hancock requires matching dimensionality and two ghost cells");
        }
        if (!(stepSize > 0) || !std::isfinite(stepSize)) {
            throw std::invalid_argument("MUSCL-Hancock timestep must be positive and finite");
        }
        mesh::TimeInterval const timeInterval{
            patch.timeState().time, patch.timeState().time + stepSize};
        updateBoundaries(patch, timeInterval.begin);

        std::array<std::vector<State>, dimensionCount> minus;
        std::array<std::vector<State>, dimensionCount> plus;
        for (int axis = 0; axis < dimensionCount; ++axis) {
            minus[axis].resize(layout.cellCount());
            plus[axis].resize(layout.cellCount());
        }
        predictFaceStates(patch, stepSize, minus, plus);

        FaceFluxes fluxes;
        for (int axis = 0; axis < dimensionCount; ++axis) {
            fluxes[axis].resize(layout.faceCount(axis));
            mesh::Coordinates const faceExtents = layout.faceExtents(axis);
            for (int z = 0; z < faceExtents[2]; ++z) {
                for (int y = 0; y < faceExtents[1]; ++y) {
                    for (int x = 0; x < faceExtents[0]; ++x) {
                        mesh::Coordinates const face{x, y, z};
                        mesh::Coordinates left = layout.storageCoordinates({
                            std::min(x, layout.interiorExtent(0) - 1),
                            std::min(y, layout.interiorExtent(1) - 1),
                            std::min(z, layout.interiorExtent(2) - 1)});
                        mesh::Coordinates right = left;
                        left[axis] = layout.ghostWidth() + face[axis] - 1;
                        right[axis] = layout.ghostWidth() + face[axis];
                        State const& leftState = plus[axis][layout.index(left)];
                        State const& rightState = minus[axis][layout.index(right)];
                        State highOrderFlux = system_.riemann(leftState, rightState, axis);
                        highOrderFlux = system_.limitFlux(patch.atStorage(left),
                            patch.atStorage(right), highOrderFlux, axis,
                            stepSize / patch.cellWidth(), dimensionCount);
                        fluxes[axis][layout.faceIndex(axis, face)] = highOrderFlux;
                    }
                }
            }
        }

        std::vector<State> next = patch.values();
        layout.forEachInterior([&](mesh::Coordinates const& cell, std::size_t cellIndex) {
            State update{};
            for (int axis = 0; axis < dimensionCount; ++axis) {
                mesh::Coordinates lowerFace = cell;
                mesh::Coordinates upperFace = cell;
                ++upperFace[axis];
                update += fluxes[axis][layout.faceIndex(axis, lowerFace)] -
                    fluxes[axis][layout.faceIndex(axis, upperFace)];
            }
            State candidate = patch.values()[cellIndex] +
                (stepSize / patch.cellWidth()) * update;
            candidate = system_.correctRoundoff(candidate, abs(update) *
                (stepSize / patch.cellWidth()));
            if (!system_.admissible(candidate)) {
                throw std::runtime_error("MUSCL-Hancock update produced an inadmissible state");
            }
            next[cellIndex] = candidate;
        });
        patch.values().swap(next);
        patch.timeState().completeStep(stepSize);
        updateBoundaries(patch, timeInterval.end);
        return StepResult{timeInterval, std::move(fluxes)};
    }

private:
    System system_;
    Limiter limiter_;
    Real theta_;

    void predictFaceStates(mesh::PatchData<State> const& patch, Real stepSize,
        std::array<std::vector<State>, dimensionCount>& minus,
        std::array<std::vector<State>, dimensionCount>& plus) const {
        mesh::MeshLayout const& layout = patch.layout();
        int const ghostWidth = layout.ghostWidth();
        mesh::Coordinates begin{};
        mesh::Coordinates end{};
        for (int axis = 0; axis < mesh::maximumDimensionCount; ++axis) {
            if (axis < dimensionCount) {
                begin[axis] = ghostWidth - 1;
                end[axis] = ghostWidth + layout.cellsPerActiveDimension() + 1;
            } else {
                begin[axis] = 0;
                end[axis] = 1;
            }
        }

        for (int z = begin[2]; z < end[2]; ++z) {
            for (int y = begin[1]; y < end[1]; ++y) {
                for (int x = begin[0]; x < end[0]; ++x) {
                    mesh::Coordinates const cell{x, y, z};
                    std::size_t const cellIndex = layout.index(cell);
                    State const centerState = patch.values()[cellIndex];
                    Reconstruction const center = system_.reconstructionVariables(centerState);
                    std::array<Reconstruction, dimensionCount> slopes{};
                    for (int axis = 0; axis < dimensionCount; ++axis) {
                        mesh::Coordinates left = cell;
                        mesh::Coordinates right = cell;
                        --left[axis];
                        ++right[axis];
                        Reconstruction const leftState = system_.reconstructionVariables(patch.atStorage(left));
                        Reconstruction const rightState = system_.reconstructionVariables(patch.atStorage(right));
                        for (int field = 0; field < Reconstruction::size(); ++field) {
                            slopes[axis][field] = limitedSlope(center[field] - leftState[field],
                                rightState[field] - center[field], limiter_, theta_);
                        }
                    }

                    Real slopeFraction = 1;
                    auto validFaces = [&](Real fraction) {
                        for (int axis = 0; axis < dimensionCount; ++axis) {
                            State const lower = system_.conservedState(center - Real(0.5) * fraction * slopes[axis]);
                            State const upper = system_.conservedState(center + Real(0.5) * fraction * slopes[axis]);
                            if (!system_.admissible(lower) || !system_.admissible(upper)) {
                                return false;
                            }
                        }
                        return true;
                    };
                    if (!validFaces(slopeFraction)) {
                        Real low = 0;
                        Real high = 1;
                        for (int iteration = 0; iteration < 48; ++iteration) {
                            Real const fraction = Real(0.5) * (low + high);
                            if (validFaces(fraction)) {
                                low = fraction;
                            } else {
                                high = fraction;
                            }
                        }
                        slopeFraction = low;
                    }

                    State predictor{};
                    for (int axis = 0; axis < dimensionCount; ++axis) {
                        minus[axis][cellIndex] = system_.conservedState(
                            center - Real(0.5) * slopeFraction * slopes[axis]);
                        plus[axis][cellIndex] = system_.conservedState(
                            center + Real(0.5) * slopeFraction * slopes[axis]);
                        predictor += system_.physicalFlux(minus[axis][cellIndex], axis) -
                            system_.physicalFlux(plus[axis][cellIndex], axis);
                    }
                    predictor *= Real(0.5) * stepSize / patch.cellWidth();
                    bool validPrediction = true;
                    for (int axis = 0; axis < dimensionCount; ++axis) {
                        minus[axis][cellIndex] += predictor;
                        plus[axis][cellIndex] += predictor;
                        validPrediction = validPrediction &&
                            system_.admissible(minus[axis][cellIndex]) &&
                            system_.admissible(plus[axis][cellIndex]);
                    }
                    if (!validPrediction) {
                        for (int axis = 0; axis < dimensionCount; ++axis) {
                            minus[axis][cellIndex] = centerState;
                            plus[axis][cellIndex] = centerState;
                        }
                    }
                }
            }
        }
    }
};

} // namespace octotiger::physics
