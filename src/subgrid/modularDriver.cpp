// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/subgrid/modularDriver.hpp"
#include "octotiger/subgrid/modularSilo.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/mesh/meshHierarchy.hpp"
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
namespace octotiger {
namespace {
using Directory = std::vector<node_client>;

hydro::Fields initialFields(mesh::BlockLocation const& location, HydroDomain const& domain) {
    Real const blockWidth = std::ldexp(domain.upper - domain.lower, -location.level);
    mesh::PhysicalCoordinates lower{};
    for (int axis = 0; axis < domain.dimensionCount; ++axis)
        lower[axis] = domain.lower + location.coordinates[axis] * blockWidth;
    hydro::Fields fields(mesh::MeshLayout(domain.dimensionCount, INX, 2), blockWidth / INX, lower);
    hydro::HydroSystem system(opts().sod_gamma);
    fields.layout().forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        auto const point = fields.layout().cellCenter(lower, fields.cellWidth(), cell);
        hydro::PrimitiveState primitive{};
        primitive.density() = 1;
        primitive.pressure() = 1;
        if (opts().modularProblem == "sod") {
            bool const left = point[0] < (domain.lower + domain.upper) / 2;
            primitive.density() = left ? 1 : Real(0.125);
            primitive.pressure() = left ? 1 : Real(0.1);
        } else if (opts().modularProblem == "advection") {
            Real phase = 0;
            for (int axis = 0; axis < domain.dimensionCount; ++axis) {
                phase += (point[axis] - domain.lower) / (domain.upper - domain.lower);
                primitive.velocity(axis) = Real(0.2);
            }
            primitive.density() += Real(0.1) * std::sin(2 * std::acos(Real(-1)) * phase);
        } else {
            Real const x = (point[0] - domain.lower) / (domain.upper - domain.lower);
            Real const y = (point[1] - domain.lower) / (domain.upper - domain.lower);
            Real const layer = Real(0.5) * (std::tanh((y-Real(0.25))/Real(0.025)) -
                std::tanh((y-Real(0.75))/Real(0.025)));
            primitive.density() = 1 + layer;
            primitive.pressure() = Real(2.5);
            primitive.velocity(0) = layer - Real(0.5);
            primitive.velocity(1) = Real(0.01) * std::sin(4 * std::acos(Real(-1)) * x);
        }
        fields.values()[index] = system.conservedState(primitive);
    });
    return fields;
}

radiation::Fields initialRadiationFields(mesh::BlockLocation const& location, HydroDomain const& domain) {
    Real const blockWidth = std::ldexp(domain.upper - domain.lower, -location.level);
    mesh::PhysicalCoordinates lower{};
    for (int axis = 0; axis < domain.dimensionCount; ++axis)
        lower[axis] = domain.lower + location.coordinates[axis] * blockWidth;
    radiation::Fields fields(mesh::MeshLayout(domain.dimensionCount, INX, 2), blockWidth / INX, lower);
    fields.layout().forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        auto const point = fields.layout().cellCenter(lower, fields.cellWidth(), cell);
        Real radiusSquared = 0;
        for (int axis = 0; axis < domain.dimensionCount; ++axis) {
            Real distance = point[axis] - (domain.lower + Real(0.25) * (domain.upper - domain.lower));
            if (domain.periodic) distance -= std::round(distance / (domain.upper - domain.lower)) * (domain.upper - domain.lower);
            radiusSquared += (distance / opts().radTestWidth) * (distance / opts().radTestWidth);
        }
        radiation::RadiationSystem::State state{};
        state[0] = opts().radTestBackground + opts().radTestAmplitude * std::exp(-Real(0.5) * radiusSquared);
        if (opts().modularRadiationProblem == "streamingGaussian")
            for (int axis = 0; axis < domain.dimensionCount; ++axis)
                state[axis + 1] = state[0] / std::sqrt(Real(domain.dimensionCount));
        fields.values()[index] = state;
    });
    return fields;
}

Directory createDirectory(mesh::MeshHierarchy const& hierarchy, HydroDomain const& domain,
    std::vector<hpx::id_type> const& localities) {
    Directory result;
    auto locations = hierarchy.leafLocations();
    std::sort(locations.begin(), locations.end(), [](auto const& left, auto const& right) {
        return std::tie(left.level, left.coordinates) < std::tie(right.level, right.coordinates);
    });
    for (auto const& location : locations) {
        TransportSnapshot initial;
        initial.location = location;
        initial.hydroEnabled = opts().hydro;
        initial.radiationEnabled = opts().radiation;
        if (initial.hydroEnabled) initial.hydro = initialFields(location, domain);
        if (initial.radiationEnabled) initial.radiation = initialRadiationFields(location, domain);
        auto id = hpx::new_<node_server>(localities[result.size() % localities.size()],
            std::move(initial)).get();
        result.emplace_back(id);
    }
    return result;
}

std::vector<TransportSnapshot> snapshots(Directory const& directory) {
    std::vector<future<TransportSnapshot>> pending;
    for (auto const& node : directory) pending.push_back(node.transportSnapshot());
    std::vector<TransportSnapshot> result;
    for (auto& item : pending) result.push_back(item.get());
    return result;
}

std::vector<HydroSnapshot> hydroSnapshots(Directory const& directory) {
    std::vector<future<HydroSnapshot>> pending;
    for (auto const& node : directory) pending.push_back(node.hydroSnapshot());
    std::vector<HydroSnapshot> result;
    for (auto& item : pending) result.push_back(item.get());
    return result;
}

std::vector<RadiationSnapshot> radiationSnapshots(Directory const& directory) {
    std::vector<future<RadiationSnapshot>> pending;
    for (auto const& node : directory) pending.push_back(node.radiationSnapshot());
    std::vector<RadiationSnapshot> result;
    for (auto& item : pending) result.push_back(item.get());
    return result;
}

void advanceHydroDirectory(Directory const& directory, HydroDomain const& domain, Real stepSize) {
    auto const old = hydroSnapshots(directory);
    std::vector<future<HydroFluxPacket>> pending;
    for (auto const& node : directory)
        pending.push_back(node.modularAdvance(old, domain, opts().sod_gamma, stepSize));
    std::vector<HydroFluxPacket> fluxes;
    for (auto& item : pending) fluxes.push_back(item.get());
    std::vector<future<void>> corrections;
    for (auto const& node : directory)
        corrections.push_back(node.modularReflux(fluxes, domain, opts().sod_gamma));
    for (auto& item : corrections) item.get();
    auto const current = hydroSnapshots(directory);
    std::vector<future<void>> refreshes;
    for (auto const& node : directory)
        refreshes.push_back(node.modularRefresh(current, domain, current.front().fields.timeState().time));
    for (auto& item : refreshes) item.get();
}

void advanceRadiationDirectory(Directory const& directory, RadiationDomain const& domain, Real stepSize) {
    auto const old = radiationSnapshots(directory);
    Real const reducedLightSpeed = opts().radCRatio * physicalLightSpeed;
    std::vector<future<RadiationFluxPacket>> pending;
    for (auto const& node : directory)
        pending.push_back(node.modularRadiationAdvance(old, domain, reducedLightSpeed, stepSize));
    std::vector<RadiationFluxPacket> fluxes;
    for (auto& item : pending) fluxes.push_back(item.get());
    std::vector<future<void>> corrections;
    for (auto const& node : directory)
        corrections.push_back(node.modularRadiationReflux(fluxes, domain, reducedLightSpeed));
    for (auto& item : corrections) item.get();
    auto const current = radiationSnapshots(directory);
    std::vector<future<void>> refreshes;
    for (auto const& node : directory)
        refreshes.push_back(node.modularRadiationRefresh(current, domain, current.front().fields.timeState().time));
    for (auto& item : refreshes) item.get();
}

void advanceDirectory(Directory const& directory, HydroDomain const& domain, Real stepSize) {
    if (opts().hydro) advanceHydroDirectory(directory, domain, stepSize);
    if (opts().radiation) advanceRadiationDirectory(directory, domain, stepSize);
}

void writeSilo(std::vector<TransportSnapshot> const& patches, int cycle, Real time, int outputIndex) {
    if (opts().disable_output) return;
    std::filesystem::path const directory(opts().data_dir.empty() ? "." : opts().data_dir);
    std::filesystem::create_directories(directory);
    std::ostringstream name;
    name << "modular_" << std::setfill('0') << std::setw(6) << outputIndex << ".silo";
    writeModularSilo(patches, (directory / name.str()).string(), cycle, time);
}
}

void validateModularTransportOptions() {
    if (!opts().modularHydro && !opts().modularTransport) return;
    if ((!opts().hydro && !opts().radiation) || opts().gravity || (opts().hydro && opts().eos != IDEAL) ||
        !opts().restart_filename.empty() || opts().problem != NONE || opts().omega != 0 || opts().omegaX != 0 ||
        opts().omegaY != 0 || opts().reflect_bc || opts().inflow_bc)
        throw std::invalid_argument("Modular driver requires problem.name=NONE, hydro and/or radiation enabled, periodic/outflow boundaries, no rotation/gravity/restart; hydro requires ideal EOS");
    if (opts().min_level < 0 || opts().max_level < std::max(integer(1), opts().min_level) || opts().max_level > 10)
        throw std::invalid_argument("Modular fixed hierarchy requires 0<=minimum<=maximum, 1<=maximum<=10");
    if (opts().hydro && opts().modularProblem != "sod" && opts().modularProblem != "advection" && opts().modularProblem != "kelvinHelmholtz")
        throw std::invalid_argument("Unknown modular hydro initial condition");
    if (opts().hydro && opts().modularProblem == "kelvinHelmholtz" && (opts().dimensionCount != 2 || !opts().periodic))
        throw std::invalid_argument("Kelvin-Helmholtz requires mesh.ndim=2 and periodic boundaries");
    if (!(opts().xscale > 0) || !std::isfinite(opts().xscale) || !(opts().stop_time >= 0) ||
        !std::isfinite(opts().stop_time) || (!opts().disable_output && (!(opts().output_dt > 0) || !std::isfinite(opts().output_dt))))
        throw std::invalid_argument("Invalid modular domain, stop time, or output interval");
    if ((opts().hydro && !(opts().cfl > 0 && opts().cfl <= Real(0.5))) || opts().stop_step < 0 ||
        opts().dt_max != Real(0.333333) || opts().rho_floor != 0 || opts().extra_regrid != 0 ||
        opts().core_refine || opts().accretor_refine != 0 || opts().donor_refine != 0 ||
        opts().grad_rho_refine != -1 || opts().ngrids != -1 ||
        (opts().unigrid && opts().min_level != opts().max_level) || opts().rewrite_silo || opts().idle_rates)
        throw std::invalid_argument("Modular driver requires 0<CFL<=0.5, nonnegative stop step, default legacy dt_max/density floor, no legacy adaptive-refinement or Silo-rewrite/idle-rate controls");
    if (opts().hydro) (void) hydro::HydroSystem(opts().sod_gamma);
    if (opts().radiation) {
        if (!opts().modularRadiationSourceFree || opts().radOpacity != 0 ||
            opts().radiationOpacity.model != "legacy" || opts().rad_implicit ||
            opts().radVelocityTerms || opts().radSubcycling || opts().radEnergyMode != "thermal")
            throw std::invalid_argument("Modular radiation requires explicit radiation.modular.source_free=on, zero constant opacity, legacy opacity model, implicit/velocity_terms/subcycling=off; gas-radiation coupling is not implemented");
        if (opts().code_to_cm != 1 || opts().code_to_s != 1 || opts().code_to_g != 1)
            throw std::invalid_argument("Modular radiation currently requires cgs units (centimeters=seconds=grams=1)");
        if (opts().modularRadiationProblem != "streamingGaussian" && opts().modularRadiationProblem != "isotropicPulse")
            throw std::invalid_argument("Unknown modular radiation initial condition");
        if (!(opts().radCRatio > 0 && opts().radCRatio <= 1) ||
            !(opts().radCfl > 0 && opts().radCfl <= Real(0.5)) ||
            !(opts().radTestWidth > 0) || !std::isfinite(opts().radTestWidth) ||
            !(opts().radTestBackground > 0) || !std::isfinite(opts().radTestBackground) ||
            !(opts().radTestAmplitude >= 0) || !std::isfinite(opts().radTestAmplitude) ||
            !std::isfinite(opts().radTestBackground + opts().radTestAmplitude))
            throw std::invalid_argument("Invalid modular radiation speed, CFL, or Gaussian parameters");
    }
}

Real runModularTransport() {
    validateModularTransportOptions();
    HydroDomain const domain{static_cast<int>(opts().dimensionCount), -opts().xscale, opts().xscale, opts().periodic};
    mesh::MeshHierarchy hierarchy(domain.dimensionCount);
    for (integer level = 0; level < opts().min_level; ++level)
        for (auto const& leaf : hierarchy.leafLocations()) hierarchy.refine(leaf);
    // Fixed AMR benchmark: lower-corner branch beyond the uniform minimum.
    // Dynamic error-driven regridding is deliberately not enabled yet.
    auto branch = hierarchy.rootLocation();
    for (integer level = 0; level < opts().max_level; ++level) {
        if (hierarchy.isLeaf(branch)) hierarchy.refine(branch);
        branch = branch.child(0);
    }
    auto const localities = hpx::find_all_localities();
    auto physical = createDirectory(hierarchy, domain, localities);
    auto shadows = createDirectory(hierarchy.oneLevelCoarser(), domain, localities);
    std::cout << "Modular transport: " << domain.dimensionCount << "D, " << physical.size()
        << " physical leaves, " << shadows.size() << " independently evolved shadow leaves, "
        << localities.size() << " HPX localities; fixed AMR, synchronous snapshot transport\n";
    if (opts().radiation) std::cout << "Radiation: source-free M1; physical c=" << physicalLightSpeed
        << " cm/s, transport cHat=" << opts().radCRatio * physicalLightSpeed
        << " cm/s; hydro coupling disabled\n";
    Real time = 0;
    int cycle = 0, outputIndex = 0;
    Real nextOutput = opts().output_dt;
    Real lastOutputTime = 0;
    writeSilo(snapshots(physical), cycle, time, outputIndex++);
    while (time < opts().stop_time && cycle < opts().stop_step) {
        Real stepSize = opts().stop_time - time;
        if (!opts().disable_output) stepSize = std::min(stepSize, nextOutput - time);
        std::vector<future<Real>> pending;
        for (auto const* directory : {&physical, &shadows}) {
            for (auto const& node : *directory) {
                if (opts().hydro) pending.push_back(node.modularStableStep(opts().sod_gamma, opts().cfl));
                if (opts().radiation) pending.push_back(node.modularRadiationStableStep(
                    opts().radCRatio * physicalLightSpeed, opts().radCfl));
            }
        }
        for (auto& item : pending) stepSize = std::min(stepSize, item.get());
        if (!(stepSize > 0) || !std::isfinite(stepSize) || time + stepSize == time)
            throw std::runtime_error("Modular timestep made no progress");
        advanceDirectory(physical, domain, stepSize);
        advanceDirectory(shadows, domain, stepSize);
        time += stepSize;
        ++cycle;
        auto const current = snapshots(physical);
        Real maximumError = 0, maximumRadiationError = 0;
        if (opts().hydro) {
            auto const errors = hydroLeafErrors(hydroSnapshots(physical), hydroSnapshots(shadows), domain, opts().sod_gamma);
            for (auto const& leaf : errors) maximumError = std::max(maximumError, leaf.error.maximumNormalized);
        }
        if (opts().radiation) {
            auto const errors = radiationLeafErrors(radiationSnapshots(physical), radiationSnapshots(shadows),
                domain, opts().radCRatio * physicalLightSpeed);
            for (auto const& leaf : errors) maximumRadiationError = std::max(maximumRadiationError, leaf.error.maximumNormalized);
        }
        std::cout << "modular step=" << cycle << " time=" << time << " dt=" << stepSize
            << " hydroShadowError=" << maximumError << " radiationShadowError=" << maximumRadiationError << '\n';
        if (!opts().disable_output && time >= nextOutput - 64 * epsilonR * std::max(Real(1), time)) {
            writeSilo(current, cycle, time, outputIndex++);
            lastOutputTime = time;
            nextOutput += opts().output_dt;
        }
    }
    if (!opts().disable_output && time > lastOutputTime)
        writeSilo(snapshots(physical), cycle, time, outputIndex++);
    std::cout << (opts().radiation ? "Modular transport completed at time " : "Modular hydro completed at time ") << time << '\n';
    return time;
}

// Compatibility entry points for the initial hydro-only modular interface.
void validateModularHydroOptions() { validateModularTransportOptions(); }
Real runModularHydro() { return runModularTransport(); }
} // namespace octotiger
#endif
