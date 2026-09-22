// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/gravity/gravityFields.hpp"
#include "octotiger/hydro/hydroSystem.hpp"
#include "octotiger/mesh.hpp"
#include "octotiger/radiation/radiationTransport.hpp"

#include <optional>

namespace octotiger {

struct ShadowState {
    std::optional<hydro::Fields> hydro;
    std::optional<radiation::Fields> radiation;
    std::optional<gravity::Fields> gravity;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & hydro;
        archive & radiation;
        archive & gravity;
    }
};

// The one structural subgrid owned by nodeServer. Physics field sets are
// composable attributes; they are not separate grids or HPX components.
class Subgrid {
public:
    Subgrid() = default;
    Subgrid(mesh::MeshLayout layout, Real cellWidth,
        mesh::PhysicalCoordinates lower = {});

    mesh::MeshLayout const& layout() const;
    Real cellWidth() const;
    mesh::PhysicalCoordinates const& lower() const;

    hydro::Fields& enableHydro();
    radiation::Fields& enableRadiation();
    gravity::Fields& enableGravity();
    void initializeIndependentShadow();

    std::optional<hydro::Fields> const& hydroFields() const;
    std::optional<hydro::Fields>& hydroFields();
    std::optional<radiation::Fields> const& radiationFields() const;
    std::optional<radiation::Fields>& radiationFields();
    std::optional<gravity::Fields> const& gravityFields() const;
    std::optional<gravity::Fields>& gravityFields();
    ShadowState const& shadowState() const;
    hydro::Fields& hydroShadowFields();
    radiation::Fields& radiationShadowFields();
    gravity::Fields& gravityShadowFields();

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & layout_;
        archive & cellWidth_;
        archive & lower_;
        archive & hydroData_;
        archive & radiationData_;
        archive & gravityData_;
        archive & shadowData_;
    }

private:
    mesh::MeshLayout layout_;
    Real cellWidth_ = 1;
    mesh::PhysicalCoordinates lower_{};
    std::optional<hydro::Fields> hydroData_;
    std::optional<radiation::Fields> radiationData_;
    std::optional<gravity::Fields> gravityData_;
    ShadowState shadowData_;
};

} // namespace octotiger
