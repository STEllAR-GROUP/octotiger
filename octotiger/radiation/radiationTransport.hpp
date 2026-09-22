// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/mesh.hpp"
#include "octotiger/physics/finiteVolume.hpp"
#include "octotiger/radiation/m1.hpp"

#include <array>

namespace octotiger::radiation {

class RadiationSystem {
public:
    using Method = RadiationM1<Real, 3>;
    using State = Method::ConservedState;
    using Reconstruction = Method::ConservedState;

    explicit RadiationSystem(Real reducedLightSpeed);

    Real reducedLightSpeed() const;
    [[nodiscard]] Reconstruction reconstructionVariables(State const& state) const;
    [[nodiscard]] State conservedState(Reconstruction const& state) const;
    State physicalFlux(State const& state, int normal) const;
    State riemann(State const& left, State const& right, int normal) const;
    [[nodiscard]] State reflected(State state, int normal) const;
    Real maximumSignalSpeed(State const& state, int normal) const;
    bool admissible(State const& state) const;
    [[nodiscard]] State correctRoundoff(State state, Real updateScale) const;
    [[nodiscard]] State limitFlux(State const& left, State const& right,
        State const& highOrderFlux, int normal, Real stepOverCellWidth,
        int dimensionCount) const;

    // The transport state is (E,Q=F/c). These adapters keep checkpoint and
    // output storage in physical cgs radiation flux units.
    [[nodiscard]] static State fromPhysical(Real energyDensity,
        std::array<Real, 3> const& physicalFlux, Real physicalLightSpeed);
    [[nodiscard]] static std::array<Real, 3> toPhysicalFlux(State const& state,
        Real physicalLightSpeed);

private:
    Real reducedLightSpeed_;
};

template <int dimensionCount>
using Solver = physics::MusclHancock<RadiationSystem, dimensionCount>;

using Fields = mesh::PatchData<RadiationSystem::State>;

} // namespace octotiger::radiation
