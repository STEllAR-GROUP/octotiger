// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/node_server.hpp"
#include "octotiger/subgrid/subgrid.hpp"
#include <hpx/include/lcos.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
using hydroSnapshotActionType = node_server::hydroSnapshotAction;
using modularStableStepActionType = node_server::modularStableStepAction;
using modularAdvanceActionType = node_server::modularAdvanceAction;
using modularRefluxActionType = node_server::modularRefluxAction;
using modularRefreshActionType = node_server::modularRefreshAction;
HPX_REGISTER_ACTION(hydroSnapshotActionType);
HPX_REGISTER_ACTION(modularStableStepActionType);
HPX_REGISTER_ACTION(modularAdvanceActionType);
HPX_REGISTER_ACTION(modularRefluxActionType);
HPX_REGISTER_ACTION(modularRefreshActionType);
using transportSnapshotActionType = node_server::transportSnapshotAction;
using radiationSnapshotActionType = node_server::radiationSnapshotAction;
using modularRadiationStableStepActionType = node_server::modularRadiationStableStepAction;
using modularRadiationAdvanceActionType = node_server::modularRadiationAdvanceAction;
using modularRadiationRefluxActionType = node_server::modularRadiationRefluxAction;
using modularRadiationRefreshActionType = node_server::modularRadiationRefreshAction;
HPX_REGISTER_ACTION(transportSnapshotActionType);
HPX_REGISTER_ACTION(radiationSnapshotActionType);
HPX_REGISTER_ACTION(modularRadiationStableStepActionType);
HPX_REGISTER_ACTION(modularRadiationAdvanceActionType);
HPX_REGISTER_ACTION(modularRadiationRefluxActionType);
HPX_REGISTER_ACTION(modularRadiationRefreshActionType);

node_server::node_server(octotiger::TransportSnapshot initial) :
    position(0), refinement_flag(0), step_num(0), rcycle(0), hcycle(0), gcycle(0),
    current_time(0), rotational_time(0),
    subgridPtr_(std::make_shared<octotiger::Subgrid>(initial.layout(), initial.cellWidth(), initial.lower())),
    modularNode_(true), modularLocation_(initial.location), is_refined(false),
    child_descendant_count{}, xmin(initial.lower()), dx(initial.cellWidth()),
    number_hydro_exchange_promises(0) {
    if (initial.location.dimensionCount != initial.layout().dimensionCount() ||
        (initial.hydroEnabled && initial.hydro.values().size() != initial.hydro.layout().cellCount()) ||
        (initial.radiationEnabled && initial.radiation.values().size() != initial.radiation.layout().cellCount()))
        throw std::invalid_argument("Invalid transport location or field storage");
    if (initial.hydroEnabled) {
        current_time = initial.hydro.timeState().time;
        subgridPtr_->enableHydro() = std::move(initial.hydro);
    }
    if (initial.radiationEnabled) {
        if (initial.radiation.layout().dimensionCount() != subgridPtr_->layout().dimensionCount() ||
            initial.radiation.layout().extents() != subgridPtr_->layout().extents() ||
            initial.radiation.layout().interiorExtents() != subgridPtr_->layout().interiorExtents() ||
            initial.radiation.layout().ghostWidth() != subgridPtr_->layout().ghostWidth() ||
            initial.radiation.lower() != subgridPtr_->lower() ||
            initial.radiation.cellWidth() != subgridPtr_->cellWidth() ||
            (initial.hydroEnabled && current_time != initial.radiation.timeState().time))
            throw std::invalid_argument("Transport field geometry/time mismatch");
        current_time = initial.radiation.timeState().time;
        subgridPtr_->enableRadiation() = std::move(initial.radiation);
    }
}

octotiger::TransportSnapshot node_server::transportSnapshot() const {
    if (!modularNode_ || modularFlux_ || modularRadiationFlux_)
        throw std::logic_error("Transport snapshot outside synchronized phase");
    octotiger::TransportSnapshot result;
    result.location = modularLocation_;
    result.hydroEnabled = subgridPtr_->hydroFields().has_value();
    result.radiationEnabled = subgridPtr_->radiationFields().has_value();
    if (result.hydroEnabled) result.hydro = *subgridPtr_->hydroFields();
    if (result.radiationEnabled) result.radiation = *subgridPtr_->radiationFields();
    if (result.hydroEnabled && result.radiationEnabled &&
        result.hydro.timeState().time != result.radiation.timeState().time)
        throw std::logic_error("Transport snapshot fields are not synchronized");
    return result;
}

octotiger::RadiationSnapshot node_server::radiationSnapshot() const {
    if (!modularNode_ || !subgridPtr_->radiationFields() || modularRadiationFlux_)
        throw std::logic_error("Radiation snapshot before reflux or fields not enabled");
    return {modularLocation_, *subgridPtr_->radiationFields()};
}
Real node_server::modularRadiationStableStep(Real reducedLightSpeed, Real cfl) const {
    if (!modularNode_ || !subgridPtr_->radiationFields() || modularRadiationFlux_)
        throw std::logic_error("Radiation CFL before reflux or fields not enabled");
    return octotiger::radiationStableStep(*subgridPtr_->radiationFields(), reducedLightSpeed, cfl);
}
octotiger::RadiationFluxPacket node_server::modularRadiationAdvance(
    std::vector<octotiger::RadiationSnapshot> const& snapshots,
    octotiger::RadiationDomain const& domain, Real reducedLightSpeed, Real stepSize) {
    if (!modularNode_ || !subgridPtr_->radiationFields() || modularRadiationFlux_)
        throw std::logic_error("Invalid radiation advance phase");
    modularRadiationFlux_ = octotiger::advanceRadiationPatch(modularLocation_,
        *subgridPtr_->radiationFields(), snapshots, domain, reducedLightSpeed, stepSize);
    current_time = subgridPtr_->radiationFields()->timeState().time;
    return *modularRadiationFlux_;
}
void node_server::modularRadiationReflux(std::vector<octotiger::RadiationFluxPacket> const& packets,
    octotiger::RadiationDomain const& domain, Real reducedLightSpeed) {
    if (!modularNode_ || !modularRadiationFlux_) throw std::logic_error("Invalid radiation reflux phase");
    octotiger::refluxRadiationPatch(*subgridPtr_->radiationFields(), *modularRadiationFlux_, packets, domain, reducedLightSpeed);
    modularRadiationFlux_.reset();
}
void node_server::modularRadiationRefresh(std::vector<octotiger::RadiationSnapshot> const& snapshots,
    octotiger::RadiationDomain const& domain, Real time) {
    if (!modularNode_ || !subgridPtr_->radiationFields() || modularRadiationFlux_)
        throw std::logic_error("Invalid radiation refresh phase");
    octotiger::fillRadiationHalo(*subgridPtr_->radiationFields(), snapshots, domain, time);
}

future<octotiger::TransportSnapshot> node_client::transportSnapshot() const {
    return hpx::async<node_server::transportSnapshotAction>(get_unmanaged_gid());
}
future<octotiger::RadiationSnapshot> node_client::radiationSnapshot() const {
    return hpx::async<node_server::radiationSnapshotAction>(get_unmanaged_gid());
}
future<Real> node_client::modularRadiationStableStep(Real reducedLightSpeed, Real cfl) const {
    return hpx::async<node_server::modularRadiationStableStepAction>(get_unmanaged_gid(), reducedLightSpeed, cfl);
}
future<octotiger::RadiationFluxPacket> node_client::modularRadiationAdvance(
    std::vector<octotiger::RadiationSnapshot> const& snapshots,
    octotiger::RadiationDomain const& domain, Real reducedLightSpeed, Real stepSize) const {
    return hpx::async<node_server::modularRadiationAdvanceAction>(get_unmanaged_gid(), snapshots, domain, reducedLightSpeed, stepSize);
}
future<void> node_client::modularRadiationReflux(std::vector<octotiger::RadiationFluxPacket> const& packets,
    octotiger::RadiationDomain const& domain, Real reducedLightSpeed) const {
    return hpx::async<node_server::modularRadiationRefluxAction>(get_unmanaged_gid(), packets, domain, reducedLightSpeed);
}
future<void> node_client::modularRadiationRefresh(std::vector<octotiger::RadiationSnapshot> const& snapshots,
    octotiger::RadiationDomain const& domain, Real time) const {
    return hpx::async<node_server::modularRadiationRefreshAction>(get_unmanaged_gid(), snapshots, domain, time);
}

node_server::node_server(octotiger::HydroSnapshot initial) :
    position(0), refinement_flag(0), step_num(0), rcycle(0), hcycle(0), gcycle(0),
    current_time(initial.fields.timeState().time), rotational_time(0),
    subgridPtr_(std::make_shared<octotiger::Subgrid>(initial.fields.layout(),
        initial.fields.cellWidth(), initial.fields.lower())),
    modularNode_(true), modularLocation_(initial.location), is_refined(false),
    child_descendant_count{}, xmin(initial.fields.lower()), dx(initial.fields.cellWidth()),
    number_hydro_exchange_promises(0) {
    subgridPtr_->enableHydro() = std::move(initial.fields);
}

octotiger::HydroSnapshot node_server::hydroSnapshot() const {
    if (!modularNode_ || !subgridPtr_->hydroFields() || modularFlux_)
        throw std::logic_error("Hydro snapshot before reflux or fields not enabled");
    return {modularLocation_, *subgridPtr_->hydroFields()};
}
Real node_server::modularStableStep(Real gamma, Real cfl) const {
    if (!modularNode_ || !subgridPtr_->hydroFields() || modularFlux_)
        throw std::logic_error("Hydro CFL before reflux or fields not enabled");
    return octotiger::hydroStableStep(*subgridPtr_->hydroFields(), gamma, cfl);
}
octotiger::HydroFluxPacket node_server::modularAdvance(
    std::vector<octotiger::HydroSnapshot> const& snapshots,
    octotiger::HydroDomain const& domain, Real gamma, Real stepSize) {
    if (!modularNode_ || !subgridPtr_->hydroFields() || modularFlux_) throw std::logic_error("Invalid modular advance phase");
    modularFlux_ = octotiger::advanceHydroPatch(modularLocation_,
        *subgridPtr_->hydroFields(), snapshots, domain, gamma, stepSize);
    current_time = subgridPtr_->hydroFields()->timeState().time;
    return *modularFlux_;
}
void node_server::modularReflux(std::vector<octotiger::HydroFluxPacket> const& packets,
    octotiger::HydroDomain const& domain, Real gamma) {
    if (!modularNode_ || !modularFlux_) throw std::logic_error("Invalid modular reflux phase");
    octotiger::refluxHydroPatch(*subgridPtr_->hydroFields(), *modularFlux_, packets, domain, gamma);
    modularFlux_.reset();
}
void node_server::modularRefresh(std::vector<octotiger::HydroSnapshot> const& snapshots,
    octotiger::HydroDomain const& domain, Real time) {
    if (!modularNode_ || !subgridPtr_->hydroFields() || modularFlux_) throw std::logic_error("Refresh before reflux barrier");
    octotiger::fillHydroHalo(*subgridPtr_->hydroFields(), snapshots, domain, time);
}

future<octotiger::HydroSnapshot> node_client::hydroSnapshot() const {
    return hpx::async<node_server::hydroSnapshotAction>(get_unmanaged_gid());
}
future<Real> node_client::modularStableStep(Real gamma, Real cfl) const {
    return hpx::async<node_server::modularStableStepAction>(get_unmanaged_gid(), gamma, cfl);
}
future<octotiger::HydroFluxPacket> node_client::modularAdvance(
    std::vector<octotiger::HydroSnapshot> const& snapshots,
    octotiger::HydroDomain const& domain, Real gamma, Real stepSize) const {
    return hpx::async<node_server::modularAdvanceAction>(get_unmanaged_gid(), snapshots, domain, gamma, stepSize);
}
future<void> node_client::modularReflux(std::vector<octotiger::HydroFluxPacket> const& packets,
    octotiger::HydroDomain const& domain, Real gamma) const {
    return hpx::async<node_server::modularRefluxAction>(get_unmanaged_gid(), packets, domain, gamma);
}
future<void> node_client::modularRefresh(std::vector<octotiger::HydroSnapshot> const& snapshots,
    octotiger::HydroDomain const& domain, Real time) const {
    return hpx::async<node_server::modularRefreshAction>(get_unmanaged_gid(), snapshots, domain, time);
}
#endif
