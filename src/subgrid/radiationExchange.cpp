// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/subgrid/radiationExchange.hpp"
namespace octotiger {
void fillRadiationHalo(radiation::Fields& target,
    std::vector<RadiationSnapshot> const& sources, RadiationDomain const& domain,
    Real time, bool includeInterior) {
    transportExchange::fillHalo(target, sources, domain, time, includeInterior);
}
Real radiationStableStep(radiation::Fields const& fields, Real reducedLightSpeed, Real cfl) {
    return transportExchange::stableStep(fields, radiation::RadiationSystem(reducedLightSpeed), cfl);
}
RadiationFluxPacket advanceRadiationPatch(mesh::BlockLocation const& location,
    radiation::Fields& fields, std::vector<RadiationSnapshot> const& sources,
    RadiationDomain const& domain, Real reducedLightSpeed, Real stepSize) {
    return transportExchange::advancePatch(location, fields, sources, domain,
        radiation::RadiationSystem(reducedLightSpeed), stepSize);
}
void refluxRadiationPatch(radiation::Fields& fields, RadiationFluxPacket const& own,
    std::vector<RadiationFluxPacket> const& packets, RadiationDomain const& domain,
    Real reducedLightSpeed) {
    transportExchange::refluxPatch(fields, own, packets, domain, radiation::RadiationSystem(reducedLightSpeed));
}
std::vector<RadiationLeafError> radiationLeafErrors(
    std::vector<RadiationSnapshot> const& physical,
    std::vector<RadiationSnapshot> const& shadows, RadiationDomain const& domain,
    Real reducedLightSpeed) {
    return transportExchange::leafErrors(physical, shadows, domain, radiation::RadiationSystem(reducedLightSpeed));
}

} // namespace octotiger
