// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/mesh/shadowHierarchy.hpp"
#include "octotiger/subgrid/subgrid.hpp"

#include <algorithm>
#include <array>
#include <optional>

namespace octotiger {

template <class State>
struct ErrorTolerances {
    std::array<Real, State::size()> absolute{};
    std::array<Real, State::size()> relative{};

    [[nodiscard]] static ErrorTolerances uniform(Real absoluteTolerance,
        Real relativeTolerance) {
        ErrorTolerances result;
        result.absolute.fill(absoluteTolerance);
        result.relative.fill(relativeTolerance);
        return result;
    }
};

struct SubgridShadowError {
    std::optional<mesh::ShadowError<hydro::ConservedState>> hydro;
    std::optional<mesh::ShadowError<radiation::RadiationSystem::State>> radiation;
    std::optional<mesh::ShadowError<gravity::State>> gravity;

    Real maximumNormalized() const {
        Real result = 0;
        if (hydro) {
            result = std::max(result, hydro->maximumNormalized);
        }
        if (radiation) {
            result = std::max(result, radiation->maximumNormalized);
        }
        if (gravity) {
            result = std::max(result, gravity->maximumNormalized);
        }
        return result;
    }
};

// Compare a physical leaf against its immediate parent's independently evolved
// shadow state. The parent shadow is never reconstructed from or overwritten
// by the fine children.
[[nodiscard]] SubgridShadowError estimateLeafShadowError(
    mesh::BlockLocation const& leafLocation, Subgrid const& leaf,
    Subgrid const& parent,
    ErrorTolerances<hydro::ConservedState> const& hydroTolerances,
    ErrorTolerances<radiation::RadiationSystem::State> const& radiationTolerances,
    ErrorTolerances<gravity::State> const& gravityTolerances,
    hydro::HydroSystem const& hydroSystem,
    radiation::RadiationSystem const& radiationSystem);

} // namespace octotiger
