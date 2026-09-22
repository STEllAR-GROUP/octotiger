// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/radiation/radiationTransport.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace octotiger::radiation {

RadiationSystem::RadiationSystem(Real reducedLightSpeed) :
    reducedLightSpeed_(reducedLightSpeed) {
    if (!(reducedLightSpeed_ > 0) || !std::isfinite(reducedLightSpeed_)) {
        throw std::invalid_argument("Reduced light speed must be positive and finite");
    }
}

Real RadiationSystem::reducedLightSpeed() const {
    return reducedLightSpeed_;
}

RadiationSystem::Reconstruction RadiationSystem::reconstructionVariables(State const& state) const {
    state.checkState("radiation reconstruction");
    return state;
}

RadiationSystem::State RadiationSystem::conservedState(Reconstruction const& state) const {
    return state;
}

RadiationSystem::State RadiationSystem::physicalFlux(State const& state, int normal) const {
    return State(state.physicalFlux(normal, reducedLightSpeed_).flux);
}

RadiationSystem::State RadiationSystem::riemann(State const& left,
    State const& right, int normal) const {
    return State(Method::hll(left, right, normal, reducedLightSpeed_));
}

RadiationSystem::State RadiationSystem::reflected(State state, int normal) const {
    state[normal + 1] = -state[normal + 1];
    return state;
}

Real RadiationSystem::maximumSignalSpeed(State const& state, int normal) const {
    auto const waves = state.physicalFlux(normal, reducedLightSpeed_);
    return std::max(std::abs(waves.minus), std::abs(waves.plus));
}

bool RadiationSystem::admissible(State const& state) const {
    return Method::admissible(state);
}

RadiationSystem::State RadiationSystem::correctRoundoff(State state, Real updateScale) const {
    return Method::roundoffState(state, updateScale);
}

RadiationSystem::State RadiationSystem::limitFlux(State const& left,
    State const& right, State const& highOrderFlux, int normal,
    Real stepOverCellWidth, int dimensionCount) const {
    if (stepOverCellWidth == 0) {
        return highOrderFlux;
    }
    State const leftPhysical = physicalFlux(left, normal);
    State const rightPhysical = physicalFlux(right, normal);
    Real const factor = Real(2 * dimensionCount) * stepOverCellWidth;
    Real const tolerance = Method::roundoff * (left[0] + right[0]);
    auto validState = [&](State const& state) {
        if (admissible(state)) {
            return true;
        }
        if (!std::isfinite(state[0]) || state[0] < -tolerance) {
            return false;
        }
        Real magnitude = 0;
        for (int field = 1; field < State::size(); ++field) {
            if (!std::isfinite(state[field])) {
                return false;
            }
            magnitude = std::hypot(magnitude, state[field]);
        }
        return magnitude - std::max(Real(0), state[0]) <= tolerance;
    };
    auto validFlux = [&](State const& flux) {
        return validState(State(left - factor * (flux - leftPhysical))) &&
            validState(State(right + factor * (flux - rightPhysical)));
    };
    if (validFlux(highOrderFlux)) {
        return highOrderFlux;
    }

    Real const speed = reducedLightSpeed_ * (Real(1) + Method::roundoff);
    State const lowOrderFlux = State(Real(0.5) *
        (leftPhysical + rightPhysical - speed * (right - left)));
    if (!validFlux(lowOrderFlux)) {
        throw std::runtime_error("First-order M1 flux violates realizability at this timestep");
    }
    Real low = 0;
    Real high = 1;
    for (int iteration = 0; iteration < 56; ++iteration) {
        Real const fraction = Real(0.5) * (low + high);
        State const candidate = State(lowOrderFlux + fraction *
            (highOrderFlux - lowOrderFlux));
        if (validFlux(candidate)) {
            low = fraction;
        } else {
            high = fraction;
        }
    }
    return State(lowOrderFlux + low * (highOrderFlux - lowOrderFlux));
}

RadiationSystem::State RadiationSystem::fromPhysical(Real energyDensity,
    std::array<Real, 3> const& physicalFlux, Real physicalLightSpeed) {
    if (!(physicalLightSpeed > 0) || !std::isfinite(physicalLightSpeed)) {
        throw std::invalid_argument("Physical light speed must be positive and finite");
    }
    State result;
    result[0] = energyDensity;
    for (int axis = 0; axis < 3; ++axis) {
        result[axis + 1] = physicalFlux[axis] / physicalLightSpeed;
    }
    result.checkState("physical radiation state");
    return result;
}

std::array<Real, 3> RadiationSystem::toPhysicalFlux(State const& state,
    Real physicalLightSpeed) {
    state.checkState("physical radiation output");
    if (!(physicalLightSpeed > 0) || !std::isfinite(physicalLightSpeed)) {
        throw std::invalid_argument("Physical light speed must be positive and finite");
    }
    std::array<Real, 3> result{};
    for (int axis = 0; axis < 3; ++axis) {
        result[axis] = physicalLightSpeed * state[axis + 1];
    }
    return result;
}

} // namespace octotiger::radiation
