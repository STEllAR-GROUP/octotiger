// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/subgrid/hydroExchange.hpp"
namespace octotiger {
void fillHydroHalo(hydro::Fields& target,
    std::vector<HydroSnapshot> const& sources, HydroDomain const& domain,
    Real time, bool includeInterior) {
    transportExchange::fillHalo(target, sources, domain, time, includeInterior);
}
Real hydroStableStep(hydro::Fields const& fields, Real gamma, Real cfl) {
    return transportExchange::stableStep(fields, hydro::HydroSystem(gamma), cfl);
}
HydroFluxPacket advanceHydroPatch(mesh::BlockLocation const& location,
    hydro::Fields& fields, std::vector<HydroSnapshot> const& sources,
    HydroDomain const& domain, Real gamma, Real stepSize) {
    return transportExchange::advancePatch(location, fields, sources, domain,
        hydro::HydroSystem(gamma), stepSize);
}
void refluxHydroPatch(hydro::Fields& fields, HydroFluxPacket const& own,
    std::vector<HydroFluxPacket> const& packets, HydroDomain const& domain,
    Real gamma) {
    transportExchange::refluxPatch(fields, own, packets, domain, hydro::HydroSystem(gamma));
}
std::vector<HydroLeafError> hydroLeafErrors(
    std::vector<HydroSnapshot> const& physical,
    std::vector<HydroSnapshot> const& shadows, HydroDomain const& domain,
    Real gamma) {
    return transportExchange::leafErrors(physical, shadows, domain, hydro::HydroSystem(gamma));
}
Real hydroShadowError(std::vector<HydroSnapshot> const& physical,
    std::vector<HydroSnapshot> const& shadows, HydroDomain const& domain, Real gamma) {
    Real maximum = 0;
    for (auto const& leaf : hydroLeafErrors(physical, shadows, domain, gamma))
        maximum = std::max(maximum, leaf.error.maximumNormalized);
    return maximum;
}
} // namespace octotiger
