// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/subgrid/shadowError.hpp"

#include <cmath>
#include <stdexcept>

namespace octotiger {

SubgridShadowError estimateLeafShadowError(
    mesh::BlockLocation const& leafLocation, Subgrid const& leaf,
    Subgrid const& parent,
    ErrorTolerances<hydro::ConservedState> const& hydroTolerances,
    ErrorTolerances<radiation::RadiationSystem::State> const& radiationTolerances,
    ErrorTolerances<gravity::State> const& gravityTolerances,
    hydro::HydroSystem const& hydroSystem,
    radiation::RadiationSystem const& radiationSystem) {
    if (leafLocation.isRoot()) {
        throw std::invalid_argument("The AMR root has no parent shadow error");
    }
    if (leaf.layout().dimensionCount() != parent.layout().dimensionCount() ||
        leafLocation.dimensionCount != leaf.layout().dimensionCount()) {
        throw std::invalid_argument("Leaf, parent, and AMR location dimensions do not match");
    }

    bool const hasLeafHydro = leaf.hydroFields().has_value();
    bool const hasParentHydro = parent.shadowState().hydro.has_value();
    bool const hasLeafRadiation = leaf.radiationFields().has_value();
    bool const hasParentRadiation = parent.shadowState().radiation.has_value();
    bool const hasLeafGravity = leaf.gravityFields().has_value();
    bool const hasParentGravity = parent.shadowState().gravity.has_value();
    if (hasLeafHydro != hasParentHydro ||
        hasLeafRadiation != hasParentRadiation ||
        hasLeafGravity != hasParentGravity) {
        throw std::logic_error("Leaf fields and parent shadow fields do not match");
    }

    SubgridShadowError result;
    mesh::BlockLocation const parentLocation = leafLocation.parent();
    if (hasLeafHydro) {
        mesh::ShadowHierarchy<hydro::ConservedState> hierarchy;
        hierarchy.addPatch(parentLocation, *parent.shadowState().hydro);
        result.hydro = hierarchy.estimateLeafError(leafLocation,
            *leaf.hydroFields(), hydroTolerances.absolute,
            hydroTolerances.relative,
            [&](hydro::ConservedState const& state) {
                return hydroSystem.admissible(state);
            });
    }
    if (hasLeafRadiation) {
        mesh::ShadowHierarchy<radiation::RadiationSystem::State> hierarchy;
        hierarchy.addPatch(parentLocation, *parent.shadowState().radiation);
        result.radiation = hierarchy.estimateLeafError(leafLocation,
            *leaf.radiationFields(), radiationTolerances.absolute,
            radiationTolerances.relative,
            [&](radiation::RadiationSystem::State const& state) {
                return radiationSystem.admissible(state);
            });
    }
    if (hasLeafGravity) {
        mesh::ShadowHierarchy<gravity::State> hierarchy;
        hierarchy.addPatch(parentLocation, *parent.shadowState().gravity);
        result.gravity = hierarchy.estimateLeafError(leafLocation,
            *leaf.gravityFields(), gravityTolerances.absolute,
            gravityTolerances.relative,
            [](gravity::State const& state) {
                for (int field = 0; field < gravity::State::size(); ++field) {
                    if (!std::isfinite(state[field])) {
                        return false;
                    }
                }
                return true;
            });
    }
    if (!result.hydro && !result.radiation && !result.gravity) {
        throw std::logic_error("No evolved field set is available for a shadow error");
    }
    return result;
}

} // namespace octotiger
