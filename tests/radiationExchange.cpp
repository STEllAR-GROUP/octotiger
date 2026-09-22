// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/subgrid/radiationExchange.hpp"
#include "octotiger/mesh/meshHierarchy.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {
using namespace octotiger;
using State = radiation::RadiationSystem::State;
constexpr Real physicalLightSpeed = 2.99792458e10;
std::size_t checkCount = 0;
void require(bool condition, std::string const& message) {
    ++checkCount;
    if (!condition) throw std::runtime_error(message);
}
void near(Real actual, Real expected, Real tolerance, std::string const& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        message + ": " + std::to_string(actual) + " != " + std::to_string(expected));
}

RadiationSnapshot makePatch(mesh::BlockLocation location, int cells, bool streaming = false) {
    Real const length = std::ldexp(Real(1), -location.level);
    mesh::PhysicalCoordinates lower{};
    for (int axis = 0; axis < location.dimensionCount; ++axis)
        lower[axis] = location.coordinates[axis] * length;
    radiation::Fields fields(mesh::MeshLayout(location.dimensionCount, cells, 2), length / cells, lower);
    fields.layout().forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        auto const point = fields.layout().cellCenter(lower, fields.cellWidth(), cell);
        State state{};
        state[0] = 1;
        for (int axis = 0; axis < location.dimensionCount; ++axis)
            state[0] += Real(0.05) * std::sin(2 * std::numbers::pi_v<Real> * point[axis]);
        for (int axis = 0; axis < location.dimensionCount; ++axis)
            state[axis+1] = state[0] * (streaming ? Real(1) :
                Real(0.35) + Real(0.1) * std::cos(2 * std::numbers::pi_v<Real> * point[axis])) /
                std::sqrt(Real(location.dimensionCount));
        fields.values()[index] = state;
    });
    return {location, std::move(fields)};
}

State integral(std::vector<RadiationSnapshot> const& patches) {
    State sum{};
    for (auto const& patch : patches)
        patch.fields.layout().forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
            sum += patch.fields.layout().cellMeasure(patch.fields.cellWidth()) * patch.fields.values()[index];
        });
    return sum;
}

void step(std::vector<RadiationSnapshot>& patches, RadiationDomain const& domain, Real speed, Real dt) {
    auto const before = patches;
    std::vector<RadiationFluxPacket> packets;
    for (auto& patch : patches)
        packets.push_back(advanceRadiationPatch(patch.location, patch.fields, before, domain, speed, dt));
    for (std::size_t patch = 0; patch < patches.size(); ++patch)
        refluxRadiationPatch(patches[patch].fields, packets[patch], packets, domain, speed);
    auto const current = patches;
    radiation::RadiationSystem const system(speed);
    for (auto& patch : patches) {
        fillRadiationHalo(patch.fields, current, domain, patch.fields.timeState().time);
        for (auto const& state : patch.fields.values())
            require(system.admissible(state), "radiation realizability after reflux and halo refresh");
    }
}

void tiledEquivalence(int dimensions, bool streaming) {
    mesh::BlockLocation const root{0, {0,0,0}, dimensions};
    RadiationDomain const domain{dimensions, 0, 1, true};
    std::vector<RadiationSnapshot> tiled, single{makePatch(root, 8, streaming)};
    for (int slot = 0; slot < (1 << dimensions); ++slot)
        tiled.push_back(makePatch(root.child(slot), 4, streaming));
    Real const dt = radiationStableStep(single.front().fields, physicalLightSpeed, Real(0.1));
    require(dt < 1e-11, "physical cgs light speed controls timestep");
    for (int iteration = 0; iteration < 3; ++iteration) {
        step(tiled, domain, physicalLightSpeed, dt);
        step(single, domain, physicalLightSpeed, dt);
        for (auto const& patch : tiled)
            patch.fields.layout().forEachInterior([&](mesh::Coordinates cell, std::size_t index) {
                for (int axis = 0; axis < dimensions; ++axis) cell[axis] += 4 * patch.location.coordinates[axis];
                auto const& reference = single.front().fields.atInterior(cell);
                for (int component = 0; component < 4; ++component)
                    near(patch.fields.values()[index][component], reference[component], 3e-13,
                        "nonuniform radiation tiled/single-patch equivalence");
            });
    }
}

void coarseFineAndShadows(int dimensions) {
    RadiationDomain const domain{dimensions, 0, 1, true};
    mesh::MeshHierarchy hierarchy(dimensions);
    auto const children = hierarchy.refine(hierarchy.rootLocation());
    auto const grandchildren = hierarchy.refine(children.back());
    hierarchy.refine(grandchildren.back());
    std::vector<RadiationSnapshot> physical, shadows;
    for (auto const& location : hierarchy.leafLocations()) physical.push_back(makePatch(location, 4));
    for (auto const& location : hierarchy.oneLevelCoarser().leafLocations()) shadows.push_back(makePatch(location, 4));
    auto const initial = integral(physical);
    auto const initialShadow = integral(shadows);
    Real const speed = physicalLightSpeed * Real(0.1);
    Real dt = 1;
    for (auto const& patch : physical) dt = std::min(dt, radiationStableStep(patch.fields, speed, Real(0.1)));
    for (int iteration = 0; iteration < 3; ++iteration) {
        step(physical, domain, speed, dt);
        step(shadows, domain, speed, dt);
        auto const sum = integral(physical), sumShadow = integral(shadows);
        for (int component = 0; component < 4; ++component) {
            near(sum[component], initial[component], 4e-12, "radiation coarse/fine conservation");
            near(sumShadow[component], initialShadow[component], 4e-12, "independent radiation shadow conservation");
        }
    }
    auto const errors = radiationLeafErrors(physical, shadows, domain, speed);
    require(errors.size() == physical.size(), "radiation error for each physical leaf");
    for (std::size_t leaf = 0; leaf < physical.size(); ++leaf) {
        require(errors[leaf].location == physical[leaf].location, "radiation leaf error retains identity");
        require(std::isfinite(errors[leaf].error.maximumNormalized), "finite radiation shadow error");
    }
    auto perturbed = physical;
    perturbed.front().fields.atInterior({0,0,0})[0] *= 2;
    auto const changed = radiationLeafErrors(perturbed, shadows, domain, speed);
    require(changed.front().error.maximumNormalized > errors.front().error.maximumNormalized,
        "radiation shadow cannot be overwritten from fine physical state");
    // Even sub-picosecond cgs time differences must not be silently accepted.
    auto stale = shadows;
    stale.front().fields.timeState().time = 0;
    bool rejected = false;
    try { (void) radiationLeafErrors(physical, stale, domain, speed); }
    catch (std::invalid_argument const&) { rejected = true; }
    require(rejected, "radiation comparison rejects stale tiny physical time");
}

void reducedSpeedAndOutput() {
    mesh::BlockLocation const root{0, {0,0,0}, 2};
    RadiationDomain const domain{2,0,1,true};
    std::vector<RadiationSnapshot> full{makePatch(root,8)}, reduced = full;
    Real const dt = radiationStableStep(full.front().fields, physicalLightSpeed, Real(0.1));
    near(radiationStableStep(reduced.front().fields, physicalLightSpeed / 10, Real(0.1)) / dt,
        10, 1e-12, "reduced light speed changes CFL timestep");
    step(full, domain, physicalLightSpeed, dt);
    step(reduced, domain, physicalLightSpeed / 10, 10*dt);
    auto const& layout = full.front().fields.layout();
    layout.forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
        auto const& state = reduced.front().fields.values()[index];
        auto const flux = radiation::RadiationSystem::toPhysicalFlux(state, physicalLightSpeed);
        for (int component = 0; component < 4; ++component)
            near(state[component], full.front().fields.values()[index][component], 3e-13,
                "same cHat*dt produces same radiation update");
        for (int axis = 0; axis < 3; ++axis)
            near(flux[axis] / physicalLightSpeed, state[axis+1], 1e-14,
                "output physical flux uses physical c, not reduced cHat");
    });
    auto invalid = full;
    invalid.front().fields.atInterior({0,0,0})[0] = -1;
    bool rejected = false;
    try { (void) radiationStableStep(invalid.front().fields, physicalLightSpeed, Real(0.1)); }
    catch (std::exception const&) { rejected = true; }
    require(rejected, "invalid radiation state is rejected");
}
}
int main() {
    try {
        for (int dimensions = 1; dimensions <= 3; ++dimensions) {
            tiledEquivalence(dimensions, false);
            tiledEquivalence(dimensions, true);
            coarseFineAndShadows(dimensions);
        }
        reducedSpeedAndOutput();
        std::cout << "Radiation exchange: " << checkCount << " checks passed\n";
    } catch (std::exception const& error) {
        std::cerr << "Radiation exchange failed: " << error.what() << '\n';
        return 1;
    }
}
