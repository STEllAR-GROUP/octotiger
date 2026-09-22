// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once
#include "octotiger/radiation/radiationTransport.hpp"
#include "octotiger/subgrid/transportExchange.hpp"
namespace octotiger {
using RadiationSnapshot = FieldSnapshot<radiation::RadiationSystem::State>;
using RadiationFluxPacket = FieldFluxPacket<radiation::RadiationSystem::State>;
using RadiationDomain = ExchangeDomain;
using RadiationLeafError = FieldLeafError<radiation::RadiationSystem::State>;
void fillRadiationHalo(radiation::Fields& target,
    std::vector<RadiationSnapshot> const& sources, RadiationDomain const& domain,
    Real time, bool includeInterior = false);
Real radiationStableStep(radiation::Fields const& fields, Real reducedLightSpeed, Real cfl);
RadiationFluxPacket advanceRadiationPatch(mesh::BlockLocation const& location,
    radiation::Fields& fields, std::vector<RadiationSnapshot> const& sources,
    RadiationDomain const& domain, Real reducedLightSpeed, Real stepSize);
void refluxRadiationPatch(radiation::Fields& fields, RadiationFluxPacket const& own,
    std::vector<RadiationFluxPacket> const& packets, RadiationDomain const& domain,
    Real reducedLightSpeed);
std::vector<RadiationLeafError> radiationLeafErrors(
    std::vector<RadiationSnapshot> const& physical,
    std::vector<RadiationSnapshot> const& shadows, RadiationDomain const& domain,
    Real reducedLightSpeed);
// Radiation fields use (E,Q=F/c). Transport uses cHat; output uses physical c.
} // namespace octotiger
