// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
// Standalone:
// g++ -std=c++20 -O2 -I. tests/modularPhysics.cpp src/mesh.cpp
//     src/mesh/meshHierarchy.cpp src/physics/finiteVolume.cpp
//     src/hydro/hydroSystem.cpp src/radiation/radiationTransport.cpp
//     src/gravity/gravityFields.cpp src/subgrid/shadowError.cpp
//     src/subgrid/subgrid.cpp src/subgrid/subgridStepper.cpp
//     -o modularPhysicsTests

#include "octotiger/hydro/hydroSystem.hpp"
#include "octotiger/mesh/meshHierarchy.hpp"
#include "octotiger/mesh/shadowHierarchy.hpp"
#include "octotiger/radiation/radiationTransport.hpp"
#include "octotiger/subgrid/subgrid.hpp"
#include "octotiger/subgrid/subgridStepper.hpp"
#include "octotiger/subgrid/shadowError.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {

std::size_t checkCount = 0;

void require(bool condition, std::string const& message) {
    ++checkCount;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(Real value, Real expected, Real tolerance, std::string const& message) {
    ++checkCount;
    if (!std::isfinite(value) || std::abs(value - expected) > tolerance) {
        throw std::runtime_error(message + ": got " + std::to_string(value) +
            ", expected " + std::to_string(expected));
    }
}

Real periodicDistance(Real coordinate, Real center) {
    Real difference = coordinate - center;
    difference -= std::round(difference);
    return difference;
}

template <class State>
State sumInterior(octotiger::mesh::PatchData<State> const& patch) {
    State result{};
    patch.layout().forEachInterior([&](octotiger::mesh::Coordinates const&, std::size_t index) {
        result += patch.values()[index];
    });
    return result;
}

void meshLayoutAndSubgrid() {
    using namespace octotiger;
    for (int dimensionCount = 1; dimensionCount <= 3; ++dimensionCount) {
        mesh::MeshLayout const layout(dimensionCount, 8, 2);
        std::size_t expectedInterior = 1;
        std::size_t expectedStorage = 1;
        for (int axis = 0; axis < dimensionCount; ++axis) {
            expectedInterior *= 8;
            expectedStorage *= 12;
        }
        require(layout.interiorCellCount() == expectedInterior, "collapsed interior allocation");
        require(layout.cellCount() == expectedStorage, "collapsed ghost allocation");
        require(layout.childCount() == (1 << dimensionCount), "active child count");
        near(layout.cellMeasure(Real(0.25)), std::pow(Real(0.25), dimensionCount),
            16 * epsilonR, "dimension-dependent cell measure");
    }
    mesh::MeshLayout const oneDimensional(1, 8, 2);
    require(oneDimensional.extents() == mesh::Coordinates{12, 1, 1}, "1-D storage is 1x1xN");
    require(oneDimensional.index(1, 0, 0) == 1, "x is contiguous");
    mesh::MeshLayout const twoDimensional(2, 8, 2);
    require(twoDimensional.extents() == mesh::Coordinates{12, 12, 1}, "2-D storage is 1xNxN");

    mesh::BlockLocation const root{0, {0, 0, 0}, 2};
    for (int slot = 0; slot < 4; ++slot) {
        mesh::BlockLocation const child = root.child(slot);
        require(child.parent() == root, "block parent round trip");
        require(child.childSlot() == slot, "block child slot round trip");
    }

    Subgrid subgrid(twoDimensional, Real(1) / 8);
    subgrid.enableHydro();
    subgrid.enableRadiation();
    subgrid.initializeIndependentShadow();
    require(subgrid.hydroFields().has_value(), "hydro field set enabled");
    require(subgrid.radiationFields().has_value(), "radiation field set enabled");
    require(subgrid.shadowState().hydro.has_value(), "hydro shadow initialized");
    bool rejectedGravity = false;
    try {
        subgrid.enableGravity();
    } catch (std::invalid_argument const&) {
        rejectedGravity = true;
    }
    require(rejectedGravity, "gravity rejected for mesh.ndim<3");
}

void runtimeSubgridDispatch() {
    using namespace octotiger;
    hydro::HydroSystem const hydroSystem(Real(1.4));
    hydro::PrimitiveState primitive;
    primitive.density() = 1;
    primitive.pressure() = 1;
    hydro::ConservedState const uniformHydro = hydroSystem.conservedState(primitive);
    radiation::RadiationSystem::State const uniformRadiation{1, 0, 0, 0};
    SubgridStepper const stepper(hydroSystem, radiation::RadiationSystem{1});
    for (int dimensionCount = 1; dimensionCount <= 3; ++dimensionCount) {
        mesh::MeshLayout const layout(dimensionCount, 8, 2);
        Subgrid subgrid(layout, Real(1) / 8);
        auto& hydroFields = subgrid.enableHydro();
        auto& radiationFields = subgrid.enableRadiation();
        layout.forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
            hydroFields.values()[index] = uniformHydro;
            radiationFields.values()[index] = uniformRadiation;
        });
        subgrid.initializeIndependentShadow();
        hydro::ConservedState const originalShadow =
            subgrid.shadowState().hydro->atInterior({0, 0, 0});
        subgrid.hydroFields()->atInterior({0, 0, 0}).density() = 2;
        subgrid.initializeIndependentShadow();
        require(subgrid.shadowState().hydro->atInterior({0, 0, 0}).density() ==
                originalShadow.density(),
            "independent shadow cannot be overwritten by a later fine state");
        subgrid.hydroFields()->atInterior({0, 0, 0}) = uniformHydro;

        physics::BoundaryConditions const boundaries =
            physics::BoundaryConditions::periodic(dimensionCount);
        Real const hydroStep = stepper.stableHydroTimestep(subgrid, Real(0.3));
        auto hydroResult = stepper.advanceHydro(subgrid, hydroStep, boundaries);
        require(hydroResult.faceFluxes.size() ==
                static_cast<std::size_t>(dimensionCount),
            "runtime hydro dispatch selected the mesh dimensionality");
        for (int axis = 0; axis < dimensionCount; ++axis) {
            require(hydroResult.faceFluxes[axis].size() == layout.faceCount(axis),
                "runtime hydro dispatch returned face-centered fluxes");
        }

        Real const radiationStep = stepper.stableRadiationTimestep(subgrid, Real(0.3));
        auto radiationResult = stepper.advanceRadiation(subgrid, radiationStep, boundaries);
        require(radiationResult.faceFluxes.size() ==
                static_cast<std::size_t>(dimensionCount),
            "runtime radiation dispatch selected the mesh dimensionality");
        Real const shadowTarget = Real(0.5) * radiationStep;
        stepper.advanceRadiationShadowTo(subgrid, shadowTarget,
            Real(0.3), boundaries);
        near(subgrid.shadowState().radiation->timeState().time, shadowTarget,
            64 * epsilonR, "shadow carries an independent patch-local time");
    }
}

void meshHierarchyTopology() {
    using namespace octotiger;
    for (int dimensionCount = 1; dimensionCount <= 3; ++dimensionCount) {
        mesh::MeshHierarchy hierarchy(dimensionCount);
        mesh::BlockLocation const root = hierarchy.rootLocation();
        auto const children = hierarchy.refine(root);
        require(children.size() == static_cast<std::size_t>(1 << dimensionCount),
            "AMR creates the active-dimensional sibling set");
        require(hierarchy.leafCount() == children.size(),
            "first AMR level has the expected leaf count");
        hierarchy.refine(children.back());
        mesh::MeshHierarchy const shadowTopology = hierarchy.oneLevelCoarser();
        require(shadowTopology.isRefined(root),
            "one-level-coarser shadow retains the shifted refinement");
        require(shadowTopology.leafCount() == children.size(),
            "one-level-coarser shadow is a complete hierarchy");
        auto const removed = hierarchy.coarsen(children.back());
        require(removed.size() == children.size(),
            "AMR coarsening removes the complete sibling set");
        require(hierarchy.oneLevelCoarser().isLeaf(root),
            "coarsened physical hierarchy shifts to a shadow root leaf");
    }
}

void gravityShadowError() {
    using namespace octotiger;
    mesh::MeshLayout const layout(3, 4, 1);
    Subgrid parent(layout, Real(0.25));
    gravity::State const parentState{1, Real(0.1), Real(0.2), Real(0.3)};
    auto& parentFields = parent.enableGravity();
    std::fill(parentFields.values().begin(), parentFields.values().end(),
        parentState);
    parent.initializeIndependentShadow();

    Subgrid leaf(layout, Real(0.125));
    gravity::State const leafState{Real(1.1), Real(0.1), Real(0.2), Real(0.3)};
    auto& leafFields = leaf.enableGravity();
    std::fill(leafFields.values().begin(), leafFields.values().end(), leafState);
    mesh::BlockLocation const leafLocation{0, {0, 0, 0}, 3};
    auto const error = estimateLeafShadowError(leafLocation.child(0), leaf,
        parent,
        ErrorTolerances<hydro::ConservedState>::uniform(Real(1e-8), Real(1e-2)),
        ErrorTolerances<radiation::RadiationSystem::State>::uniform(
            Real(1e-8), Real(1e-2)),
        ErrorTolerances<gravity::State>::uniform(Real(1e-8), Real(1e-2)),
        hydro::HydroSystem{}, radiation::RadiationSystem{1});
    require(error.gravity.has_value() && error.maximumNormalized() > 0,
        "gravity participates in the generic leaf shadow error");
}

void sodShockTube() {
    using namespace octotiger;
    constexpr int cellCount = 192;
    mesh::MeshLayout const layout(1, cellCount, 2);
    hydro::Fields fields(layout, Real(1) / cellCount);
    hydro::HydroSystem const system(Real(1.4));
    layout.forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        Real const x = (cell[0] + Real(0.5)) / cellCount;
        hydro::PrimitiveState primitive;
        primitive.density() = x < Real(0.5) ? Real(1) : Real(0.125);
        primitive.pressure() = x < Real(0.5) ? Real(1) : Real(0.1);
        fields.values()[index] = system.conservedState(primitive);
    });
    hydro::ConservedState const initial = sumInterior(fields);
    hydro::Solver<1> const solver(system, physics::Limiter::VanLeer);
    physics::BoundaryConditions const boundaries;
    Real const finalTime = Real(0.2);
    while (fields.timeState().time < finalTime) {
        Real stepSize = solver.stableTimestep(fields, Real(0.4));
        stepSize = std::min(stepSize, finalTime - fields.timeState().time);
        solver.advance(fields, stepSize, boundaries);
    }
    hydro::ConservedState const final = sumInterior(fields);
    near(final.density(), initial.density(), Real(3e-11) * initial.density(),
        "Sod mass conservation");
    near(final.totalEnergy(), initial.totalEnergy(), Real(3e-11) * initial.totalEnergy(),
        "Sod energy conservation");
    for (auto const& state : fields.values()) {
        if (state.density() != 0) {
            require(system.admissible(state), "Sod positivity");
        }
    }
    Real const densityAtQuarter = fields.atInterior({cellCount / 4, 0, 0}).density();
    Real const densityAtContact = fields.atInterior({cellCount / 2, 0, 0}).density();
    Real const densityAtThreeQuarters = fields.atInterior({3 * cellCount / 4, 0, 0}).density();
    require(densityAtQuarter > Real(0.85), "Sod left state");
    require(densityAtContact > Real(0.35) && densityAtContact < Real(0.55), "Sod star state");
    require(densityAtThreeQuarters > Real(0.2) && densityAtThreeQuarters < Real(0.35), "Sod post-shock state");
}

void kelvinHelmholtz() {
    using namespace octotiger;
    constexpr int cellCount = 48;
    mesh::MeshLayout const layout(2, cellCount, 2);
    hydro::Fields fields(layout, Real(1) / cellCount);
    hydro::HydroSystem const system(Real(1.4));
    layout.forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        Real const x = (cell[0] + Real(0.5)) / cellCount;
        Real const y = (cell[1] + Real(0.5)) / cellCount;
        bool const layer = std::abs(y - Real(0.5)) < Real(0.25);
        Real const interfaceDistance = std::min(std::abs(y - Real(0.25)), std::abs(y - Real(0.75)));
        hydro::PrimitiveState primitive;
        primitive.density() = layer ? Real(2) : Real(1);
        primitive.velocity(0) = layer ? Real(0.5) : Real(-0.5);
        primitive.velocity(1) = Real(0.01) * std::sin(4 * std::numbers::pi_v<Real> * x) *
            std::exp(-interfaceDistance * interfaceDistance / Real(0.01));
        primitive.pressure() = Real(2.5);
        fields.values()[index] = system.conservedState(primitive);
    });
    hydro::ConservedState const initial = sumInterior(fields);
    Real initialTransverseEnergy = 0;
    layout.forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
        auto const& state = fields.values()[index];
        initialTransverseEnergy += Real(0.5) * state.momentum(1) * state.momentum(1) / state.density();
    });
    hydro::Solver<2> const solver(system, physics::Limiter::MinmodTheta, Real(1.3));
    physics::BoundaryConditions const boundaries = physics::BoundaryConditions::periodic(2);
    auto advanceTo = [&](Real targetTime) {
        while (fields.timeState().time < targetTime) {
            Real stepSize = solver.stableTimestep(fields, Real(0.35));
            stepSize = std::min(stepSize, targetTime - fields.timeState().time);
            solver.advance(fields, stepSize, boundaries);
        }
    };
    auto transverseEnergy = [&]() {
        Real result = 0;
        layout.forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
            auto const& state = fields.values()[index];
            result += Real(0.5) * state.momentum(1) * state.momentum(1) / state.density();
        });
        return result;
    };
    advanceTo(Real(0.5));
    Real const settledTransverseEnergy = transverseEnergy();
    advanceTo(Real(1.5));
    hydro::ConservedState const final = sumInterior(fields);
    for (int field = 0; field < hydro::ConservedState::size(); ++field) {
        near(final[field], initial[field], Real(2e-10) * std::max(Real(1), std::abs(initial[field])),
            "Kelvin-Helmholtz periodic conservation");
    }
    Real finalTransverseEnergy = 0;
    layout.forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
        auto const& state = fields.values()[index];
        require(system.admissible(state), "Kelvin-Helmholtz positivity");
        finalTransverseEnergy += Real(0.5) * state.momentum(1) * state.momentum(1) / state.density();
    });
    require(settledTransverseEnergy < initialTransverseEnergy,
        "Kelvin-Helmholtz startup transient is resolved");
    require(finalTransverseEnergy > Real(1.1) * settledTransverseEnergy,
        "Kelvin-Helmholtz transverse mode grows after the startup transient");
}

template <int dimensionCount>
Real radiationStreamingError(int cellCount, Real finalTime) {
    using namespace octotiger;
    mesh::MeshLayout const layout(dimensionCount, cellCount, 2);
    radiation::Fields fields(layout, Real(1) / cellCount);
    radiation::RadiationSystem const system(Real(1));
    Real const inverseRootTwo = Real(1) / std::sqrt(Real(2));
    layout.forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        Real const x = (cell[0] + Real(0.5)) / cellCount;
        Real const y = dimensionCount > 1 ? (cell[1] + Real(0.5)) / cellCount : Real(0);
        Real const dx = periodicDistance(x, Real(0.25));
        Real const dy = dimensionCount > 1 ? periodicDistance(y, Real(0.25)) : Real(0);
        Real const energy = Real(1) + Real(0.1) * std::exp(-(dx * dx + dy * dy) / Real(0.0025));
        radiation::RadiationSystem::State state;
        state[0] = energy;
        state[1] = energy * (dimensionCount > 1 ? inverseRootTwo : Real(1));
        state[2] = dimensionCount > 1 ? energy * inverseRootTwo : Real(0);
        state[3] = 0;
        fields.values()[index] = state;
    });
    auto const initial = sumInterior(fields);
    radiation::Solver<dimensionCount> const solver(system, physics::Limiter::VanLeer);
    physics::BoundaryConditions const boundaries = physics::BoundaryConditions::periodic(dimensionCount);
    while (fields.timeState().time < finalTime) {
        Real stepSize = solver.stableTimestep(fields, Real(0.4));
        stepSize = std::min(stepSize, finalTime - fields.timeState().time);
        solver.advance(fields, stepSize, boundaries);
    }
    auto const final = sumInterior(fields);
    for (int field = 0; field < radiation::RadiationSystem::State::size(); ++field) {
        near(final[field], initial[field], Real(3e-10) * std::max(Real(1), std::abs(initial[field])),
            "radiation periodic conservation");
    }

    Real error = 0;
    layout.forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        Real const x = (cell[0] + Real(0.5)) / cellCount;
        Real const y = dimensionCount > 1 ? (cell[1] + Real(0.5)) / cellCount : Real(0);
        Real const speed = dimensionCount > 1 ? inverseRootTwo : Real(1);
        Real const dx = periodicDistance(x, Real(0.25) + speed * finalTime);
        Real const dy = dimensionCount > 1 ?
            periodicDistance(y, Real(0.25) + speed * finalTime) : Real(0);
        Real const exact = Real(1) + Real(0.1) * std::exp(-(dx * dx + dy * dy) / Real(0.0025));
        auto const& state = fields.values()[index];
        require(system.admissible(state), "radiation realizability");
        error += std::abs(state[0] - exact);
    });
    return error / static_cast<Real>(layout.interiorCellCount());
}

void radiationStreaming() {
    require(radiationStreamingError<1>(128, Real(0.2)) < Real(0.004),
        "1-D radiation streaming pulse");
    require(radiationStreamingError<2>(48, Real(0.12)) < Real(0.003),
        "2-D diagonal radiation streaming pulse");
}

void independentlyEvolvedShadow() {
    using namespace octotiger;
    using State = radiation::RadiationSystem::State;
    constexpr int cellCount = 32;
    mesh::MeshLayout const layout(1, cellCount, 2);
    radiation::Fields parent(layout, Real(1) / cellCount);
    radiation::RadiationSystem const system(Real(1));
    layout.forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        Real const x = (cell[0] + Real(0.5)) / cellCount;
        Real const energy = Real(1) + Real(0.1) *
            std::exp(-std::pow(periodicDistance(x, Real(0.25)), 2) / Real(0.0025));
        parent.values()[index] = State{energy, energy, 0, 0};
    });
    State const initialCenter = parent.atInterior({cellCount / 4, 0, 0});

    mesh::ShadowHierarchy<State> hierarchy;
    mesh::BlockLocation const root{0, {0, 0, 0}, 1};
    hierarchy.addPatch(root, parent);
    radiation::Solver<1> const solver(system, physics::Limiter::VanLeer);
    physics::BoundaryConditions const boundaries = physics::BoundaryConditions::periodic(1);
    Real const targetTime = Real(0.1);
    std::size_t boundarySnapshots = 0;
    hierarchy.advanceSynchronizedTo(targetTime,
        [&](radiation::Fields const& patch) { return solver.stableTimestep(patch, Real(0.4)); },
        [&](mesh::ShadowHierarchy<State>&, mesh::TimeInterval const& interval) {
            require(interval.end > interval.begin,
                "shadow boundary snapshot has a physical-time interval");
            ++boundarySnapshots;
        },
        [&](radiation::Fields& patch, Real stepSize) { solver.advance(patch, stepSize, boundaries); });
    require(boundarySnapshots > 0,
        "complete shadow hierarchy prepared synchronized boundary snapshots");
    require(hierarchy.patch(root).timeState().step > 0, "shadow hierarchy was independently evolved");
    require(hierarchy.patch(root).atInterior({cellCount / 4, 0, 0})[0] != initialCenter[0],
        "shadow state was not a restricted fine copy");

    mesh::BlockLocation const leaf = root.child(0);
    radiation::Fields fine(layout, Real(0.5) / cellCount);
    layout.forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
        Real const x = (cell[0] + Real(0.5)) * fine.cellWidth();
        Real const energy = Real(1) + Real(0.1) *
            std::exp(-std::pow(periodicDistance(x, Real(0.25) + targetTime), 2) / Real(0.0025));
        fine.values()[index] = State{energy, energy, 0, 0};
    });
    fine.timeState().time = targetTime;
    std::array<Real, State::size()> absoluteTolerance{};
    std::array<Real, State::size()> relativeTolerance{};
    absoluteTolerance.fill(Real(1e-8));
    relativeTolerance.fill(Real(1e-2));
    auto const error = hierarchy.estimateLeafError(leaf, fine,
        absoluteTolerance, relativeTolerance,
        [&](State const& state) { return system.admissible(state); });
    require(std::isfinite(error.maximumNormalized) && error.maximumNormalized > 0,
        "coarse-versus-fine shadow error computed");

    Subgrid parentSubgrid(layout, Real(1) / cellCount);
    parentSubgrid.enableRadiation() = hierarchy.patch(root);
    parentSubgrid.initializeIndependentShadow();
    Subgrid leafSubgrid(layout, Real(0.5) / cellCount);
    leafSubgrid.enableRadiation() = fine;
    auto const subgridError = estimateLeafShadowError(leaf, leafSubgrid,
        parentSubgrid,
        ErrorTolerances<hydro::ConservedState>::uniform(Real(1e-8), Real(1e-2)),
        ErrorTolerances<State>::uniform(Real(1e-8), Real(1e-2)),
        ErrorTolerances<gravity::State>::uniform(Real(1e-8), Real(1e-2)),
        hydro::HydroSystem{}, system);
    require(subgridError.radiation.has_value() &&
            !subgridError.hydro.has_value() && !subgridError.gravity.has_value(),
        "leaf error includes exactly the enabled field sets");
    near(subgridError.maximumNormalized(), error.maximumNormalized,
        256 * epsilonR * error.maximumNormalized,
        "subgrid leaf error uses the independently evolved parent shadow");
}

} // namespace

int main() {
    try {
        meshLayoutAndSubgrid();
        runtimeSubgridDispatch();
        meshHierarchyTopology();
        gravityShadowError();
        sodShockTube();
        kelvinHelmholtz();
        radiationStreaming();
        independentlyEvolvedShadow();
        std::cout << "modular physics checks passed: " << checkCount << '\n';
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "modular physics test failed after " << checkCount <<
            " checks: " << error.what() << '\n';
        return 1;
    }
}
