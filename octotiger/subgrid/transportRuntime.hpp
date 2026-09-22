// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once
#include "octotiger/subgrid/hydroExchange.hpp"
#include "octotiger/subgrid/radiationExchange.hpp"

namespace octotiger {
inline constexpr Real physicalLightSpeed = Real(2.99792458e10);

// Explicit flags keep the wire archive independent of optional serialization
// support in the installed HPX version. Disabled Fields remain empty.
struct TransportSnapshot {
    mesh::BlockLocation location;
    bool hydroEnabled = false;
    bool radiationEnabled = false;
    hydro::Fields hydro;
    radiation::Fields radiation;
    template<class Archive> void serialize(Archive& archive, unsigned) {
        archive & location & hydroEnabled & radiationEnabled & hydro & radiation;
    }
    mesh::MeshLayout const& layout() const {
        if (!hydroEnabled && !radiationEnabled) throw std::logic_error("Empty transport snapshot");
        return hydroEnabled ? hydro.layout() : radiation.layout();
    }
    mesh::PhysicalCoordinates const& lower() const {
        return hydroEnabled ? hydro.lower() : radiation.lower();
    }
    Real cellWidth() const { return hydroEnabled ? hydro.cellWidth() : radiation.cellWidth(); }
};
}
