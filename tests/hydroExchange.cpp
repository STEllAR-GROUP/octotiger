// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/subgrid/hydroExchange.hpp"
#include "octotiger/mesh/meshHierarchy.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {
using namespace octotiger;
std::size_t checkCount = 0;

void require(bool condition, std::string const& message) {
    ++checkCount;
    if (!condition) throw std::runtime_error(message);
}
void near(Real actual, Real expected, Real tolerance, std::string const& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        message + ": " + std::to_string(actual) + " != " + std::to_string(expected));
}

hydro::ConservedState smoothState(mesh::PhysicalCoordinates const& position, int dimensions) {
    hydro::PrimitiveState primitive;
    Real wave = 0;
    for (int axis = 0; axis < dimensions; ++axis) {
        wave += std::sin(2 * std::numbers::pi_v<Real> * position[axis]);
        primitive.velocity(axis) = Real(0.2) + Real(0.03) *
            std::cos(2 * std::numbers::pi_v<Real> * position[axis]);
    }
    primitive.density() = 1 + Real(0.05) * wave;
    primitive.pressure() = 1 + Real(0.02) * wave;
    return hydro::HydroSystem(Real(1.4)).conservedState(primitive);
}

HydroSnapshot makePatch(mesh::BlockLocation location, int cells) {
    Real const length = std::ldexp(Real(1), -location.level);
    mesh::PhysicalCoordinates lower{};
    for (int axis = 0; axis < location.dimensionCount; ++axis)
        lower[axis] = location.coordinates[axis] * length;
    hydro::Fields fields(mesh::MeshLayout(location.dimensionCount, cells, 2),
        length / cells, lower);
    fields.layout().forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        fields.values()[index] = smoothState(fields.layout().cellCenter(lower,
            fields.cellWidth(), cell), location.dimensionCount);
    });
    return {location, std::move(fields)};
}

hydro::ConservedState integral(std::vector<HydroSnapshot> const& patches) {
    hydro::ConservedState sum{};
    for (auto const& patch : patches) {
        Real const volume = patch.fields.layout().cellMeasure(patch.fields.cellWidth());
        patch.fields.layout().forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
            sum += volume * patch.fields.values()[index];
        });
    }
    return sum;
}

void step(std::vector<HydroSnapshot>& patches, HydroDomain const& domain, Real dt) {
    auto const before = patches;
    std::vector<HydroFluxPacket> packets;
    for (auto& patch : patches)
        packets.push_back(advanceHydroPatch(patch.location, patch.fields, before,
            domain, Real(1.4), dt));
    for (std::size_t i = 0; i < patches.size(); ++i)
        refluxHydroPatch(patches[i].fields, packets[i], packets, domain, Real(1.4));
    auto const after = patches;
    for (auto& patch : patches)
        fillHydroHalo(patch.fields, after, domain, patch.fields.timeState().time);
}

void tiledEquivalence(int dimensions) {
    HydroDomain const domain{dimensions, 0, 1, true};
    mesh::BlockLocation const root{0, {0, 0, 0}, dimensions};
    std::vector<HydroSnapshot> tiled;
    for (int child = 0; child < (1 << dimensions); ++child)
        tiled.push_back(makePatch(root.child(child), 4));
    std::vector<HydroSnapshot> whole{makePatch(root, 8)};
    Real const dt = Real(0.15) * hydroStableStep(whole.front().fields, Real(1.4), Real(0.3));
    for (int iteration = 0; iteration < 3; ++iteration) {
        step(tiled, domain, dt);
        step(whole, domain, dt);
        for (auto const& patch : tiled)
            patch.fields.layout().forEachInterior([&](mesh::Coordinates cell, std::size_t index) {
                for (int axis = 0; axis < dimensions; ++axis)
                    cell[axis] += 4 * patch.location.coordinates[axis];
                auto const& expected = whole.front().fields.atInterior(cell);
                for (int component = 0; component < 5; ++component)
                    near(patch.fields.values()[index][component], expected[component],
                        2e-13, "nonuniform tiled/single-patch equivalence");
            });
    }
}

void coarseFineConservation(int dimensions) {
    HydroDomain const domain{dimensions, 0, 1, true};
    mesh::MeshHierarchy hierarchy(dimensions);
    auto const children = hierarchy.refine(hierarchy.rootLocation());
    hierarchy.refine(children.back());
    std::vector<HydroSnapshot> patches;
    for (auto const& location : hierarchy.leafLocations()) patches.push_back(makePatch(location, 4));
    auto const initial = integral(patches);
    Real dt = 1;
    for (auto const& patch : patches)
        dt = std::min(dt, hydroStableStep(patch.fields, Real(1.4), Real(0.2)));
    for (int iteration = 0; iteration < 3; ++iteration) {
        step(patches, domain, dt);
        auto const current = integral(patches);
        for (int component = 0; component < 5; ++component)
            near(current[component], initial[component], 3e-12,
                "periodic coarse/fine reflux conservation");
    }
    auto invalid = patches;
    invalid.front().fields.timeState().time += dt;
    bool rejected = false;
    try {
        fillHydroHalo(patches.front().fields, invalid, domain,
            patches.front().fields.timeState().time);
    } catch (std::invalid_argument const&) { rejected = true; }
    require(rejected, "reject mismatched physical time");
    invalid = patches;
    invalid.push_back(invalid.front());
    rejected = false;
    try {
        fillHydroHalo(patches.front().fields, invalid, domain,
            patches.front().fields.timeState().time, true);
    } catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "reject overlapping donor directory");
}

void independentShadows(int dimensions) {
    HydroDomain const domain{dimensions, 0, 1, true};
    mesh::MeshHierarchy hierarchy(dimensions);
    auto const children = hierarchy.refine(hierarchy.rootLocation());
    auto const grandchildren = hierarchy.refine(children.back());
    hierarchy.refine(grandchildren.back());
    auto const shadowTopology = hierarchy.oneLevelCoarser();
    std::vector<HydroSnapshot> physical, shadows;
    for (auto const& location : hierarchy.leafLocations()) physical.push_back(makePatch(location, 4));
    for (auto const& location : shadowTopology.leafLocations()) shadows.push_back(makePatch(location, 4));
    Real const dt = Real(0.0005);
    step(physical, domain, dt);
    step(shadows, domain, dt);
    Real const error = hydroShadowError(physical, shadows, domain, Real(1.4));
    require(std::isfinite(error) && error > 0, "evolved shadow gives finite nonzero error");
    auto const leafErrors = hydroLeafErrors(physical, shadows, domain, Real(1.4));
    require(leafErrors.size() == physical.size(), "one refinement error per physical leaf");
    Real maximum = 0;
    for (std::size_t leaf = 0; leaf < physical.size(); ++leaf) {
        require(leafErrors[leaf].location == physical[leaf].location,
            "refinement error retains leaf identity");
        require(std::isfinite(leafErrors[leaf].error.maximumNormalized),
            "finite leaf-local normalized error");
        maximum = std::max(maximum, leafErrors[leaf].error.maximumNormalized);
    }
    near(maximum, error, 32 * epsilonR, "global error reduces leaf-local errors");

    // Parent projection must reconstruct its halos from current shadow
    // interiors, not consume stale or invalid halo values in a snapshot.
    auto staleHalos = shadows;
    for (auto& patch : staleHalos) {
        auto const extents = patch.fields.layout().extents();
        for (int z = 0; z < extents[2]; ++z)
            for (int y = 0; y < extents[1]; ++y)
                for (int x = 0; x < extents[0]; ++x)
                    if (!patch.fields.layout().isInterior({x,y,z}))
                        patch.fields.atStorage({x,y,z}).density() = std::numeric_limits<Real>::quiet_NaN();
    }
    near(hydroShadowError(physical, staleHalos, domain, Real(1.4)), error,
        32 * epsilonR, "parent uses fresh shadow-interior halo data");

    auto changedPhysical = physical;
    changedPhysical.front().fields.atInterior({0,0,0}).density() *= 2;
    require(hydroShadowError(changedPhysical, shadows, domain, Real(1.4)) > error,
        "physical perturbation is not overwritten into shadow reference");
    auto changedShadow = shadows;
    for (auto& patch : changedShadow)
        patch.fields.layout().forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
            patch.fields.values()[index].density() *= 2;
        });
    require(hydroShadowError(physical, changedShadow, domain, Real(1.4)) > error,
        "comparison uses independent shadow state, including internal parents");

    changedPhysical.front().fields.atInterior({0,0,0}).density() =
        std::numeric_limits<Real>::quiet_NaN();
    bool rejected = false;
    try { (void) hydroShadowError(changedPhysical, shadows, domain, Real(1.4)); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "reject NaN physical state in shadow estimate");
    changedShadow.front().fields.timeState().time += dt;
    rejected = false;
    try { (void) hydroShadowError(physical, changedShadow, domain, Real(1.4)); }
    catch (std::invalid_argument const&) { rejected = true; }
    require(rejected, "reject mismatched shadow time");
}

void genericNonfiniteError(int dimensions) {
    HydroDomain const domain{dimensions, 0, 1, true};
    mesh::BlockLocation const root{0, {0,0,0}, dimensions};
    auto parent = makePatch(root, 4);
    fillHydroHalo(parent.fields, {parent}, domain, 0);
    mesh::ShadowHierarchy<hydro::ConservedState> hierarchy;
    hierarchy.addPatch(root, parent.fields);
    auto fine = makePatch(root.child(0), 4);
    fine.fields.atInterior({0,0,0}).density() = std::numeric_limits<Real>::quiet_NaN();
    std::array<Real,5> tolerance;
    tolerance.fill(Real(0.01));
    bool rejected = false;
    try {
        (void) hierarchy.estimateLeafError(fine.location, fine.fields, tolerance,
            tolerance, [](auto const&) { return true; });
    } catch (std::invalid_argument const&) { rejected = true; }
    require(rejected, "generic error estimator rejects NaNs even with permissive validator");
}
}

int main() {
    try {
        for (int dimensions = 1; dimensions <= 3; ++dimensions) {
            tiledEquivalence(dimensions);
            coarseFineConservation(dimensions);
            independentShadows(dimensions);
            genericNonfiniteError(dimensions);
        }
        std::cout << "Hydro exchange: " << checkCount << " checks passed\n";
    } catch (std::exception const& error) {
        std::cerr << "Hydro exchange failed: " << error.what() << '\n';
        return 1;
    }
}
