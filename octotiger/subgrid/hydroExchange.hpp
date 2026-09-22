// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once
#include "octotiger/hydro/hydroSystem.hpp"
#include "octotiger/subgrid/transportExchange.hpp"
namespace octotiger {
using HydroSnapshot = FieldSnapshot<hydro::ConservedState>;
using HydroFluxPacket = FieldFluxPacket<hydro::ConservedState>;
using HydroDomain = ExchangeDomain;
using HydroLeafError = FieldLeafError<hydro::ConservedState>;
void fillHydroHalo(hydro::Fields& target,
    std::vector<HydroSnapshot> const& sources, HydroDomain const& domain,
    Real time, bool includeInterior = false);
Real hydroStableStep(hydro::Fields const& fields, Real gamma, Real cfl);
HydroFluxPacket advanceHydroPatch(mesh::BlockLocation const& location,
    hydro::Fields& fields, std::vector<HydroSnapshot> const& sources,
    HydroDomain const& domain, Real gamma, Real stepSize);
void refluxHydroPatch(hydro::Fields& fields, HydroFluxPacket const& own,
    std::vector<HydroFluxPacket> const& packets, HydroDomain const& domain,
    Real gamma);
std::vector<HydroLeafError> hydroLeafErrors(
    std::vector<HydroSnapshot> const& physical,
    std::vector<HydroSnapshot> const& shadows, HydroDomain const& domain,
    Real gamma);
Real hydroShadowError(std::vector<HydroSnapshot> const& physical,
    std::vector<HydroSnapshot> const& shadows, HydroDomain const& domain, Real gamma);
} // namespace octotiger
