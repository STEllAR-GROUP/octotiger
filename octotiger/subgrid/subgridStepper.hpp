// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/subgrid/subgrid.hpp"

#include <vector>

namespace octotiger {

template <class State>
struct DynamicStepResult {
    mesh::TimeInterval timeInterval;
    std::vector<std::vector<State>> faceFluxes;
};

// Runtime dimensional dispatch lives at the subgrid boundary. The numerical
// kernels remain dimension-specialized, while mesh.ndim can be selected once
// at startup without spreading switches throughout the hydro and radiation
// implementations.
class SubgridStepper {
public:
    SubgridStepper(hydro::HydroSystem hydroSystem = hydro::HydroSystem{},
        radiation::RadiationSystem radiationSystem = radiation::RadiationSystem{Real(1)},
        physics::Limiter limiter = physics::Limiter::VanLeer,
        Real limiterTheta = Real(1.5));

    [[nodiscard]] Real stableHydroTimestep(Subgrid const& subgrid,
        Real courantNumber) const;
    [[nodiscard]] Real stableRadiationTimestep(Subgrid const& subgrid,
        Real courantNumber) const;

    DynamicStepResult<hydro::ConservedState> advanceHydro(Subgrid& subgrid,
        Real stepSize, physics::BoundaryConditions const& boundaries) const;
    DynamicStepResult<radiation::RadiationSystem::State> advanceRadiation(
        Subgrid& subgrid, Real stepSize,
        physics::BoundaryConditions const& boundaries) const;

    void advanceHydroShadowTo(Subgrid& subgrid, Real targetTime,
        Real courantNumber, physics::BoundaryConditions const& boundaries) const;
    void advanceRadiationShadowTo(Subgrid& subgrid, Real targetTime,
        Real courantNumber, physics::BoundaryConditions const& boundaries) const;

    hydro::HydroSystem const& hydroSystem() const;
    radiation::RadiationSystem const& radiationSystem() const;

private:
    hydro::HydroSystem hydroSystem_;
    radiation::RadiationSystem radiationSystem_;
    physics::Limiter limiter_;
    Real limiterTheta_;
};

} // namespace octotiger
