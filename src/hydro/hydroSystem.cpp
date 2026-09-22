// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/hydro/hydroSystem.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace octotiger::hydro {

HydroSystem::HydroSystem(Real adiabaticIndex, Real densityFloor, Real pressureFloor) :
    adiabaticIndex_(adiabaticIndex),
    densityFloor_(densityFloor),
    pressureFloor_(pressureFloor) {
    if (!(adiabaticIndex_ > 1) || !(densityFloor_ > 0) ||
        !(pressureFloor_ > 0)) {
        throw std::invalid_argument("Hydro EOS and positivity floors must be positive");
    }
}

Real HydroSystem::adiabaticIndex() const {
    return adiabaticIndex_;
}

PrimitiveState HydroSystem::reconstructionVariables(ConservedState const& state) const {
    if (!admissible(state)) {
        throw std::runtime_error("Cannot convert an inadmissible hydro state to primitive variables");
    }
    PrimitiveState result;
    result.density() = state.density();
    for (int axis = 0; axis < 3; ++axis) {
        result.velocity(axis) = state.momentum(axis) / state.density();
    }
    result.pressure() = pressure(state);
    return result;
}

ConservedState HydroSystem::conservedState(PrimitiveState const& state) const {
    if (!(state.density() >= densityFloor_) ||
        !(state.pressure() >= pressureFloor_) ||
        !std::isfinite(state.density()) || !std::isfinite(state.pressure())) {
        throw std::invalid_argument("Cannot convert an inadmissible hydro primitive state");
    }
    ConservedState result;
    result.density() = state.density();
    Real speedSquared = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(state.velocity(axis))) {
            throw std::invalid_argument("Hydro primitive velocity must be finite");
        }
        result.momentum(axis) = state.density() * state.velocity(axis);
        speedSquared += state.velocity(axis) * state.velocity(axis);
    }
    result.totalEnergy() = state.pressure() / (adiabaticIndex_ - 1) +
        Real(0.5) * state.density() * speedSquared;
    return result;
}

ConservedState HydroSystem::physicalFlux(ConservedState const& state, int normal) const {
    PrimitiveState const primitive = reconstructionVariables(state);
    Real const normalVelocity = primitive.velocity(normal);
    ConservedState result;
    result.density() = state.density() * normalVelocity;
    for (int axis = 0; axis < 3; ++axis) {
        result.momentum(axis) = state.momentum(axis) * normalVelocity +
            (axis == normal ? primitive.pressure() : Real(0));
    }
    result.totalEnergy() = (state.totalEnergy() + primitive.pressure()) * normalVelocity;
    return result;
}

ConservedState HydroSystem::riemann(ConservedState const& left,
    ConservedState const& right, int normal) const {
    PrimitiveState const leftPrimitive = reconstructionVariables(left);
    PrimitiveState const rightPrimitive = reconstructionVariables(right);
    Real const leftSound = std::sqrt(adiabaticIndex_ * leftPrimitive.pressure() / leftPrimitive.density());
    Real const rightSound = std::sqrt(adiabaticIndex_ * rightPrimitive.pressure() / rightPrimitive.density());
    Real const leftSpeed = std::min(leftPrimitive.velocity(normal) - leftSound,
        rightPrimitive.velocity(normal) - rightSound);
    Real const rightSpeed = std::max(leftPrimitive.velocity(normal) + leftSound,
        rightPrimitive.velocity(normal) + rightSound);
    ConservedState const leftFlux = physicalFlux(left, normal);
    ConservedState const rightFlux = physicalFlux(right, normal);
    if (leftSpeed >= 0) {
        return leftFlux;
    }
    if (rightSpeed <= 0) {
        return rightFlux;
    }

    Real const leftDenominator = leftPrimitive.density() *
        (leftSpeed - leftPrimitive.velocity(normal));
    Real const rightDenominator = rightPrimitive.density() *
        (rightSpeed - rightPrimitive.velocity(normal));
    Real const denominator = leftDenominator - rightDenominator;
    Real const denominatorScale = std::max({Real(1),
        std::abs(leftDenominator), std::abs(rightDenominator)});
    Real const waveScale = std::max({Real(1),
        std::abs(leftSpeed), std::abs(rightSpeed)});
    if (std::abs(denominator) <= 64 * epsilonR * denominatorScale) {
        return hll(left, right, normal);
    }
    Real const contactSpeed = (rightPrimitive.pressure() - leftPrimitive.pressure() +
        leftPrimitive.density() * leftPrimitive.velocity(normal) *
            (leftSpeed - leftPrimitive.velocity(normal)) -
        rightPrimitive.density() * rightPrimitive.velocity(normal) *
            (rightSpeed - rightPrimitive.velocity(normal))) /
        denominator;
    if (!std::isfinite(contactSpeed) || contactSpeed < leftSpeed ||
        contactSpeed > rightSpeed) {
        return hll(left, right, normal);
    }

    auto starState = [&](ConservedState const& state, PrimitiveState const& primitive,
                         Real waveSpeed) {
        Real const waveDifference = waveSpeed - primitive.velocity(normal);
        Real const starDifference = waveSpeed - contactSpeed;
        if (std::abs(starDifference) <= 64 * epsilonR * waveScale ||
            std::abs(waveDifference) <= 64 * epsilonR * waveScale) {
            return ConservedState{};
        }
        ConservedState star;
        star.density() = primitive.density() * waveDifference / starDifference;
        for (int axis = 0; axis < 3; ++axis) {
            Real const velocity = axis == normal ? contactSpeed : primitive.velocity(axis);
            star.momentum(axis) = star.density() * velocity;
        }
        star.totalEnergy() = star.density() *
            (state.totalEnergy() / primitive.density() +
                (contactSpeed - primitive.velocity(normal)) *
                    (contactSpeed + primitive.pressure() /
                        (primitive.density() * waveDifference)));
        return star;
    };

    if (contactSpeed >= 0) {
        ConservedState const star = starState(left, leftPrimitive, leftSpeed);
        if (!admissible(star)) {
            return hll(left, right, normal);
        }
        return leftFlux + leftSpeed * (star - left);
    }
    ConservedState const star = starState(right, rightPrimitive, rightSpeed);
    if (!admissible(star)) {
        return hll(left, right, normal);
    }
    return rightFlux + rightSpeed * (star - right);
}

ConservedState HydroSystem::reflected(ConservedState state, int normal) const {
    state.momentum(normal) = -state.momentum(normal);
    return state;
}

Real HydroSystem::maximumSignalSpeed(ConservedState const& state, int normal) const {
    PrimitiveState const primitive = reconstructionVariables(state);
    return std::abs(primitive.velocity(normal)) +
        std::sqrt(adiabaticIndex_ * primitive.pressure() / primitive.density());
}

bool HydroSystem::admissible(ConservedState const& state) const {
    for (int field = 0; field < State::size(); ++field) {
        if (!std::isfinite(state[field])) {
            return false;
        }
    }
    return state.density() >= densityFloor_ &&
        pressure(state) >= pressureFloor_;
}

ConservedState HydroSystem::correctRoundoff(ConservedState state, Real updateScale) const {
    (void) updateScale;
    return state;
}

ConservedState HydroSystem::limitFlux(ConservedState const& left,
    ConservedState const& right, ConservedState const& highOrderFlux,
    int normal, Real stepOverCellWidth, int dimensionCount) const {
    if (stepOverCellWidth == 0) {
        return highOrderFlux;
    }
    ConservedState const leftPhysical = physicalFlux(left, normal);
    ConservedState const rightPhysical = physicalFlux(right, normal);
    Real const factor = Real(2 * dimensionCount) * stepOverCellWidth;
    auto validFlux = [&](ConservedState const& flux) {
        return admissible(left - factor * (flux - leftPhysical)) &&
            admissible(right + factor * (flux - rightPhysical));
    };
    if (validFlux(highOrderFlux)) {
        return highOrderFlux;
    }
    Real const speed = std::max(maximumSignalSpeed(left, normal),
        maximumSignalSpeed(right, normal));
    ConservedState const lowOrderFlux = Real(0.5) *
        (leftPhysical + rightPhysical - speed * (right - left));
    if (!validFlux(lowOrderFlux)) {
        throw std::runtime_error("First-order hydro flux is not positivity preserving at this timestep");
    }
    Real low = 0;
    Real high = 1;
    for (int iteration = 0; iteration < 56; ++iteration) {
        Real const fraction = Real(0.5) * (low + high);
        if (validFlux(lowOrderFlux + fraction * (highOrderFlux - lowOrderFlux))) {
            low = fraction;
        } else {
            high = fraction;
        }
    }
    return lowOrderFlux + low * (highOrderFlux - lowOrderFlux);
}

Real HydroSystem::pressure(ConservedState const& state) const {
    if (!(state.density() > 0) || !std::isfinite(state.density())) {
        return -std::numeric_limits<Real>::infinity();
    }
    Real momentumSquared = 0;
    for (int axis = 0; axis < 3; ++axis) {
        momentumSquared += state.momentum(axis) * state.momentum(axis);
    }
    return (adiabaticIndex_ - 1) *
        (state.totalEnergy() - Real(0.5) * momentumSquared / state.density());
}

ConservedState HydroSystem::hll(ConservedState const& left,
    ConservedState const& right, int normal) const {
    PrimitiveState const leftPrimitive = reconstructionVariables(left);
    PrimitiveState const rightPrimitive = reconstructionVariables(right);
    Real const leftSound = std::sqrt(adiabaticIndex_ * leftPrimitive.pressure() / leftPrimitive.density());
    Real const rightSound = std::sqrt(adiabaticIndex_ * rightPrimitive.pressure() / rightPrimitive.density());
    Real const leftSpeed = std::min(leftPrimitive.velocity(normal) - leftSound,
        rightPrimitive.velocity(normal) - rightSound);
    Real const rightSpeed = std::max(leftPrimitive.velocity(normal) + leftSound,
        rightPrimitive.velocity(normal) + rightSound);
    ConservedState const leftFlux = physicalFlux(left, normal);
    ConservedState const rightFlux = physicalFlux(right, normal);
    if (leftSpeed >= 0) {
        return leftFlux;
    }
    if (rightSpeed <= 0) {
        return rightFlux;
    }
    return (rightSpeed * leftFlux - leftSpeed * rightFlux +
               leftSpeed * rightSpeed * (right - left)) /
        (rightSpeed - leftSpeed);
}

} // namespace octotiger::hydro
