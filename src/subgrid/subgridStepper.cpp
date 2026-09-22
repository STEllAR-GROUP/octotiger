// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/subgrid/subgridStepper.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace octotiger {
namespace {

template <class State, std::size_t dimensionCount>
DynamicStepResult<State> dynamicResult(
    mesh::TimeInterval timeInterval,
    std::array<std::vector<State>, dimensionCount>&& faceFluxes) {
    DynamicStepResult<State> result;
    result.timeInterval = timeInterval;
    result.faceFluxes.reserve(dimensionCount);
    for (auto& flux : faceFluxes) {
        result.faceFluxes.push_back(std::move(flux));
    }
    return result;
}

template <int dimensionCount, class System>
Real stableTimestep(mesh::PatchData<typename System::State> const& fields,
    System const& system, physics::Limiter limiter, Real limiterTheta,
    Real courantNumber) {
    physics::MusclHancock<System, dimensionCount> const solver(
        system, limiter, limiterTheta);
    return solver.stableTimestep(fields, courantNumber);
}

template <int dimensionCount, class System>
DynamicStepResult<typename System::State> advance(
    mesh::PatchData<typename System::State>& fields, System const& system,
    physics::Limiter limiter, Real limiterTheta, Real stepSize,
    physics::BoundaryConditions const& boundaries) {
    physics::MusclHancock<System, dimensionCount> const solver(
        system, limiter, limiterTheta);
    auto result = solver.advance(fields, stepSize, boundaries);
    return dynamicResult<typename System::State>(result.timeInterval,
        std::move(result.faceFluxes));
}

template <class Fields, class StableStep, class Advance>
void advanceTo(Fields& fields, Real targetTime, StableStep&& stableStep,
    Advance&& advanceStep) {
    Real const scale = std::max({Real(1), std::abs(fields.timeState().time),
        std::abs(targetTime)});
    if (targetTime < fields.timeState().time - 64 * epsilonR * scale) {
        throw std::invalid_argument("A subgrid shadow cannot advance backward in time");
    }
    while (fields.timeState().time < targetTime - 64 * epsilonR * scale) {
        Real const proposedStep = stableStep(fields);
        if (!(proposedStep > 0) || !std::isfinite(proposedStep)) {
            throw std::runtime_error("A subgrid shadow received an invalid stable timestep");
        }
        Real const stepSize = std::min(proposedStep,
            targetTime - fields.timeState().time);
        Real const oldTime = fields.timeState().time;
        advanceStep(fields, stepSize);
        if (fields.timeState().time == oldTime) {
            fields.timeState().completeStep(stepSize);
        } else {
            Real const expectedTime = oldTime + stepSize;
            Real const tolerance = 128 * epsilonR *
                std::max(Real(1), std::abs(expectedTime));
            if (std::abs(fields.timeState().time - expectedTime) > tolerance) {
                throw std::runtime_error(
                    "Subgrid shadow stepper advanced to an unexpected physical time");
            }
        }
    }
}

} // namespace

SubgridStepper::SubgridStepper(hydro::HydroSystem hydroSystem,
    radiation::RadiationSystem radiationSystem, physics::Limiter limiter,
    Real limiterTheta) :
    hydroSystem_(std::move(hydroSystem)),
    radiationSystem_(std::move(radiationSystem)),
    limiter_(limiter),
    limiterTheta_(limiterTheta) {
    if (!(limiterTheta_ >= 1 && limiterTheta_ <= 2)) {
        throw std::invalid_argument("Subgrid limiter theta must lie in [1,2]");
    }
}

Real SubgridStepper::stableHydroTimestep(Subgrid const& subgrid,
    Real courantNumber) const {
    if (!subgrid.hydroFields()) {
        throw std::logic_error("Hydro fields are not enabled on this subgrid");
    }
    auto const& fields = *subgrid.hydroFields();
    switch (subgrid.layout().dimensionCount()) {
    case 1:
        return stableTimestep<1>(fields, hydroSystem_, limiter_, limiterTheta_, courantNumber);
    case 2:
        return stableTimestep<2>(fields, hydroSystem_, limiter_, limiterTheta_, courantNumber);
    case 3:
        return stableTimestep<3>(fields, hydroSystem_, limiter_, limiterTheta_, courantNumber);
    }
    throw std::logic_error("Invalid subgrid dimensionality");
}

Real SubgridStepper::stableRadiationTimestep(Subgrid const& subgrid,
    Real courantNumber) const {
    if (!subgrid.radiationFields()) {
        throw std::logic_error("Radiation fields are not enabled on this subgrid");
    }
    auto const& fields = *subgrid.radiationFields();
    switch (subgrid.layout().dimensionCount()) {
    case 1:
        return stableTimestep<1>(fields, radiationSystem_, limiter_, limiterTheta_, courantNumber);
    case 2:
        return stableTimestep<2>(fields, radiationSystem_, limiter_, limiterTheta_, courantNumber);
    case 3:
        return stableTimestep<3>(fields, radiationSystem_, limiter_, limiterTheta_, courantNumber);
    }
    throw std::logic_error("Invalid subgrid dimensionality");
}

DynamicStepResult<hydro::ConservedState> SubgridStepper::advanceHydro(
    Subgrid& subgrid, Real stepSize,
    physics::BoundaryConditions const& boundaries) const {
    if (!subgrid.hydroFields()) {
        throw std::logic_error("Hydro fields are not enabled on this subgrid");
    }
    auto& fields = *subgrid.hydroFields();
    switch (subgrid.layout().dimensionCount()) {
    case 1:
        return advance<1>(fields, hydroSystem_, limiter_, limiterTheta_, stepSize, boundaries);
    case 2:
        return advance<2>(fields, hydroSystem_, limiter_, limiterTheta_, stepSize, boundaries);
    case 3:
        return advance<3>(fields, hydroSystem_, limiter_, limiterTheta_, stepSize, boundaries);
    }
    throw std::logic_error("Invalid subgrid dimensionality");
}

DynamicStepResult<radiation::RadiationSystem::State>
SubgridStepper::advanceRadiation(Subgrid& subgrid, Real stepSize,
    physics::BoundaryConditions const& boundaries) const {
    if (!subgrid.radiationFields()) {
        throw std::logic_error("Radiation fields are not enabled on this subgrid");
    }
    auto& fields = *subgrid.radiationFields();
    switch (subgrid.layout().dimensionCount()) {
    case 1:
        return advance<1>(fields, radiationSystem_, limiter_, limiterTheta_, stepSize, boundaries);
    case 2:
        return advance<2>(fields, radiationSystem_, limiter_, limiterTheta_, stepSize, boundaries);
    case 3:
        return advance<3>(fields, radiationSystem_, limiter_, limiterTheta_, stepSize, boundaries);
    }
    throw std::logic_error("Invalid subgrid dimensionality");
}

void SubgridStepper::advanceHydroShadowTo(Subgrid& subgrid, Real targetTime,
    Real courantNumber, physics::BoundaryConditions const& boundaries) const {
    if (!subgrid.shadowState().hydro) {
        throw std::logic_error("Hydro shadow state has not been initialized");
    }
    auto& fields = subgrid.hydroShadowFields();
    int const dimensionCount = subgrid.layout().dimensionCount();
    advanceTo(fields, targetTime,
        [&](hydro::Fields const& current) {
            switch (dimensionCount) {
            case 1:
                return stableTimestep<1>(current, hydroSystem_, limiter_, limiterTheta_, courantNumber);
            case 2:
                return stableTimestep<2>(current, hydroSystem_, limiter_, limiterTheta_, courantNumber);
            case 3:
                return stableTimestep<3>(current, hydroSystem_, limiter_, limiterTheta_, courantNumber);
            }
            throw std::logic_error("Invalid subgrid dimensionality");
        },
        [&](hydro::Fields& current, Real stepSize) {
            switch (dimensionCount) {
            case 1:
                (void) advance<1>(current, hydroSystem_, limiter_, limiterTheta_, stepSize, boundaries);
                return;
            case 2:
                (void) advance<2>(current, hydroSystem_, limiter_, limiterTheta_, stepSize, boundaries);
                return;
            case 3:
                (void) advance<3>(current, hydroSystem_, limiter_, limiterTheta_, stepSize, boundaries);
                return;
            }
            throw std::logic_error("Invalid subgrid dimensionality");
        });
}

void SubgridStepper::advanceRadiationShadowTo(Subgrid& subgrid,
    Real targetTime, Real courantNumber,
    physics::BoundaryConditions const& boundaries) const {
    if (!subgrid.shadowState().radiation) {
        throw std::logic_error("Radiation shadow state has not been initialized");
    }
    auto& fields = subgrid.radiationShadowFields();
    int const dimensionCount = subgrid.layout().dimensionCount();
    advanceTo(fields, targetTime,
        [&](radiation::Fields const& current) {
            switch (dimensionCount) {
            case 1:
                return stableTimestep<1>(current, radiationSystem_, limiter_, limiterTheta_, courantNumber);
            case 2:
                return stableTimestep<2>(current, radiationSystem_, limiter_, limiterTheta_, courantNumber);
            case 3:
                return stableTimestep<3>(current, radiationSystem_, limiter_, limiterTheta_, courantNumber);
            }
            throw std::logic_error("Invalid subgrid dimensionality");
        },
        [&](radiation::Fields& current, Real stepSize) {
            switch (dimensionCount) {
            case 1:
                (void) advance<1>(current, radiationSystem_, limiter_, limiterTheta_, stepSize, boundaries);
                return;
            case 2:
                (void) advance<2>(current, radiationSystem_, limiter_, limiterTheta_, stepSize, boundaries);
                return;
            case 3:
                (void) advance<3>(current, radiationSystem_, limiter_, limiterTheta_, stepSize, boundaries);
                return;
            }
            throw std::logic_error("Invalid subgrid dimensionality");
        });
}

hydro::HydroSystem const& SubgridStepper::hydroSystem() const {
    return hydroSystem_;
}

radiation::RadiationSystem const& SubgridStepper::radiationSystem() const {
    return radiationSystem_;
}

} // namespace octotiger
