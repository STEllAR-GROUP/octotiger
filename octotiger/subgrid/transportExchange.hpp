// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once
#include "octotiger/mesh/shadowHierarchy.hpp"
#include "octotiger/physics/finiteVolume.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace octotiger {

// Serializable field packets shared by the physical and independent shadow
// directories. A directory contains nonoverlapping leaves at one physical time.
template<class State>
struct FieldSnapshot {
    mesh::BlockLocation location;
    mesh::PatchData<State> fields;
    template<class Archive> void serialize(Archive& archive, unsigned) {
        archive & location & fields;
    }
};

template<class State>
struct FieldFluxPacket {
    mesh::BlockLocation location;
    mesh::MeshLayout layout;
    mesh::PhysicalCoordinates lower{};
    Real cellWidth = 1;
    mesh::TimeInterval interval;
    std::vector<std::vector<State>> fluxes;
    template<class Archive> void serialize(Archive& archive, unsigned) {
        archive & location & layout & lower & cellWidth & interval & fluxes;
    }
};

struct ExchangeDomain {
    int dimensionCount = 3;
    Real lower = -1;
    Real upper = 1;
    bool periodic = false;
    template<class Archive> void serialize(Archive& archive, unsigned) {
        archive & dimensionCount & lower & upper & periodic;
    }
};

template<class State>
struct FieldLeafError {
    mesh::BlockLocation location;
    mesh::ShadowError<State> error;
};

namespace transportExchange {
inline void validateDomain(ExchangeDomain const& domain) {
    if (domain.dimensionCount < 1 || domain.dimensionCount > 3 ||
        !std::isfinite(domain.lower) || !std::isfinite(domain.upper) ||
        !(domain.upper > domain.lower) || !std::isfinite(domain.upper - domain.lower))
        throw std::invalid_argument("Transport domain must have 1-3 dimensions and finite increasing bounds");
}

inline bool sameTime(Real left, Real right) {
    return std::isfinite(left) && std::isfinite(right) &&
        std::abs(left - right) <= 64 * epsilonR *
            std::max(std::abs(left), std::abs(right));
}


template<class State>
void fillHalo(mesh::PatchData<State>& target,
    std::vector<FieldSnapshot<State>> const& sources, ExchangeDomain const& domain,
    Real time, bool includeInterior = false) {
    validateDomain(domain);
    auto const& layout = target.layout();
    if (layout.dimensionCount() != domain.dimensionCount ||
        !(domain.upper > domain.lower) || !sameTime(target.timeState().time, time))
        throw std::invalid_argument("Invalid halo domain or target physical time");
    for (auto const& source : sources) {
        if (source.fields.layout().dimensionCount() != domain.dimensionCount ||
            !sameTime(source.fields.timeState().time, time))
            throw std::invalid_argument("Transport halo snapshot has wrong dimension or physical time");
    }
    auto const extents = layout.extents();
    Real const width = target.cellWidth();
    Real const volume = layout.cellMeasure(width);
    for (int z = 0; z < extents[2]; ++z)
        for (int y = 0; y < extents[1]; ++y)
            for (int x = 0; x < extents[0]; ++x) {
                mesh::Coordinates const storage{x,y,z};
                if (!includeInterior && layout.isInterior(storage)) continue;
                auto cell = storage;
                for (int axis = 0; axis < domain.dimensionCount; ++axis)
                    cell[axis] -= layout.ghostWidth();
                mesh::PhysicalCoordinates low{};
                for (int axis = 0; axis < domain.dimensionCount; ++axis) {
                    low[axis] = target.lower()[axis] + cell[axis] * width;
                    if (domain.periodic) {
                        Real const length = domain.upper - domain.lower;
                        low[axis] -= std::floor((low[axis] - domain.lower) / length) * length;
                    } else {
                        low[axis] = std::clamp(low[axis], domain.lower, domain.upper - width);
                    }
                }
                State sum{};
                Real coverage = 0;
                for (auto const& source : sources) {
                    auto const& patch = source.fields;
                    auto const& sourceLayout = patch.layout();
                    Real const sourceWidth = patch.cellWidth();
                    mesh::Coordinates first{}, last{};
                    bool overlaps = true;
                    for (int axis = 0; axis < domain.dimensionCount; ++axis) {
                        first[axis] = std::max(0, static_cast<int>(std::floor(
                            (low[axis] - patch.lower()[axis]) / sourceWidth)));
                        last[axis] = std::min(sourceLayout.interiorExtent(axis),
                            static_cast<int>(std::ceil((low[axis] + width -
                                patch.lower()[axis]) / sourceWidth)));
                        overlaps = overlaps && first[axis] < last[axis];
                    }
                    for (int axis = domain.dimensionCount; axis < 3; ++axis) last[axis] = 1;
                    if (!overlaps) continue;
                    for (int k = first[2]; k < last[2]; ++k)
                        for (int j = first[1]; j < last[1]; ++j)
                            for (int i = first[0]; i < last[0]; ++i) {
                                mesh::Coordinates const donor{i,j,k};
                                Real overlap = 1;
                                for (int axis = 0; axis < domain.dimensionCount; ++axis) {
                                    Real const donorLow = patch.lower()[axis] + donor[axis] * sourceWidth;
                                    overlap *= std::max(Real(0), std::min(low[axis] + width,
                                        donorLow + sourceWidth) - std::max(low[axis], donorLow));
                                }
                                sum += overlap * patch.atInterior(donor);
                                coverage += overlap;
                            }
                }
                if (std::abs(coverage - volume) > 256 * epsilonR * volume)
                    throw std::runtime_error("Halo directory has a gap, overlap, or unsupported periodic cell");
                target.atStorage(storage) = sum / coverage;
            }
}


template<int dimensions, class State, class System>
FieldFluxPacket<State> advance(mesh::BlockLocation const& location,
    mesh::PatchData<State>& fields, std::vector<FieldSnapshot<State>> const& sources,
    ExchangeDomain const& domain, System const& system, Real stepSize) {
    physics::MusclHancock<System, dimensions> solver{system};
    Real const begin = fields.timeState().time;
    auto result = solver.advanceWithBoundaryUpdater(fields, stepSize,
        [&](mesh::PatchData<State>& target, Real requestedTime) {
            // End-state halos require the global end-state barrier. They must
            // NOT be refreshed using old snapshots mislabeled as new time.
            if (requestedTime == begin) {
                fillHalo(target, sources, domain, requestedTime);
            }
        });
    FieldFluxPacket<State> packet{location, fields.layout(), fields.lower(),
        fields.cellWidth(), result.timeInterval, {}};
    for (auto& flux : result.faceFluxes) packet.fluxes.push_back(std::move(flux));
    return packet;
}
template<class State, class System>
Real stableStep(mesh::PatchData<State> const& fields, System const& system, Real cfl) {
    switch (fields.layout().dimensionCount()) {
    case 1: return physics::MusclHancock<System, 1>(system).stableTimestep(fields, cfl);
    case 2: return physics::MusclHancock<System, 2>(system).stableTimestep(fields, cfl);
    case 3: return physics::MusclHancock<System, 3>(system).stableTimestep(fields, cfl);
    }
    throw std::logic_error("Invalid transport dimensions");
}

template<class State, class System>
FieldFluxPacket<State> advancePatch(mesh::BlockLocation const& location,
    mesh::PatchData<State>& fields, std::vector<FieldSnapshot<State>> const& sources,
    ExchangeDomain const& domain, System const& system, Real stepSize) {
    switch (fields.layout().dimensionCount()) {
    case 1: return advance<1>(location, fields, sources, domain, system, stepSize);
    case 2: return advance<2>(location, fields, sources, domain, system, stepSize);
    case 3: return advance<3>(location, fields, sources, domain, system, stepSize);
    }
    throw std::logic_error("Invalid transport dimensions");
}

template<class State, class System>
void refluxPatch(mesh::PatchData<State>& fields, FieldFluxPacket<State> const& own,
    std::vector<FieldFluxPacket<State>> const& packets, ExchangeDomain const& domain,
    System const& system) {
    validateDomain(domain);
    if (!sameTime(fields.timeState().time, own.interval.end) ||
        !(own.interval.end > own.interval.begin) || !std::isfinite(own.interval.begin) ||
        fields.layout().dimensionCount() != domain.dimensionCount ||
        own.layout.dimensionCount() != domain.dimensionCount || own.lower != fields.lower() ||
        own.cellWidth != fields.cellWidth() ||
        own.layout.interiorExtents() != fields.layout().interiorExtents())
        throw std::invalid_argument("Reflux target geometry or physical time mismatch");
    for (auto const& packet : packets) {
        if (!sameTime(packet.interval.begin, own.interval.begin) ||
            !sameTime(packet.interval.end, own.interval.end))
            throw std::invalid_argument("Reflux physical time interval mismatch");
        if (packet.layout.dimensionCount() != domain.dimensionCount ||
            !(packet.cellWidth > 0) || !std::isfinite(packet.cellWidth) ||
            packet.fluxes.size() != static_cast<std::size_t>(domain.dimensionCount))
            throw std::invalid_argument("Reflux packet geometry mismatch");
        for (int axis = 0; axis < domain.dimensionCount; ++axis)
            if (!std::isfinite(packet.lower[axis]) ||
                packet.fluxes[static_cast<std::size_t>(axis)].size() != packet.layout.faceCount(axis))
                throw std::invalid_argument("Reflux packet has invalid bounds or face count");
    }
    auto const& layout = fields.layout();
    Real const width = fields.cellWidth();
    Real const area = layout.faceMeasure(width);
    Real const planeTolerance = 64 * epsilonR * std::max({std::abs(domain.lower),
        std::abs(domain.upper), domain.upper - domain.lower});
    auto next = fields.values();
    for (int axis = 0; axis < domain.dimensionCount; ++axis) {
        for (int side = 0; side < 2; ++side) {
            Real const plane = fields.lower()[axis] + side * layout.interiorExtent(axis) * width;
            layout.forEachInterior([&](mesh::Coordinates const& cell, std::size_t index) {
                if (cell[axis] != (side ? layout.interiorExtent(axis) - 1 : 0)) return;
                State sum{};
                Real coverage = 0;
                for (auto const& packet : packets) {
                    if (!(packet.cellWidth < width)) continue;
                    Real donorPlane = packet.lower[axis] + (1-side) *
                        packet.layout.interiorExtent(axis) * packet.cellWidth;
                    if (domain.periodic) {
                        Real const length = domain.upper - domain.lower;
                        donorPlane += std::round((plane - donorPlane) / length) * length;
                    }
                    if (std::abs(plane - donorPlane) > planeTolerance) continue;
                    packet.layout.forEachInterior([&](mesh::Coordinates const& donor, std::size_t) {
                        if (donor[axis] != (side ? 0 : packet.layout.interiorExtent(axis)-1)) return;
                        Real overlap = 1;
                        for (int tangent = 0; tangent < domain.dimensionCount; ++tangent) {
                            if (tangent == axis) continue;
                            Real const low = fields.lower()[tangent] + cell[tangent] * width;
                            Real const donorLow = packet.lower[tangent] + donor[tangent] * packet.cellWidth;
                            overlap *= std::max(Real(0), std::min(low+width, donorLow+packet.cellWidth)
                                - std::max(low, donorLow));
                        }
                        auto face = donor;
                        face[axis] = side ? 0 : packet.layout.interiorExtent(axis);
                        sum += overlap * packet.fluxes.at(static_cast<std::size_t>(axis)).at(
                            packet.layout.faceIndex(axis, face));
                        coverage += overlap;
                    });
                }
                if (coverage == 0) return;
                if (std::abs(coverage - area) > 256 * epsilonR * area)
                    throw std::runtime_error("Incomplete fine-face reflux coverage");
                auto face = cell;
                face[axis] = side ? layout.interiorExtent(axis) : 0;
                auto const& old = own.fluxes.at(static_cast<std::size_t>(axis)).at(layout.faceIndex(axis, face));
                next[index] += (side ? -1 : 1) * own.interval.duration() / width * (sum / area - old);
            });
        }
    }
    layout.forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
        if (!system.admissible(next[index]))
            throw std::runtime_error("Reflux produced inadmissible transport state; reduce timestep");
    });
    fields.values().swap(next);
}
template<class State, class System>
std::vector<FieldLeafError<State>> leafErrors(std::vector<FieldSnapshot<State>> const& physical,
    std::vector<FieldSnapshot<State>> const& shadows, ExchangeDomain const& domain,
    System const& system) {
    mesh::ShadowHierarchy<State> hierarchy;
    for (auto const& leaf : physical) {
        if (leaf.location.isRoot()) throw std::invalid_argument("Root leaf has no shadow parent");
        auto const parent = leaf.location.parent();
        if (hierarchy.contains(parent)) continue;
        auto lower = leaf.fields.lower();
        Real const blockWidth = leaf.fields.cellWidth() * leaf.fields.layout().cellsPerActiveDimension();
        for (int axis = 0; axis < domain.dimensionCount; ++axis)
            lower[axis] -= ((leaf.location.childSlot() >> axis) & 1) * blockWidth;
        mesh::PatchData<State> fields(leaf.fields.layout(), leaf.fields.cellWidth() * 2, lower);
        fields.timeState() = leaf.fields.timeState();
        // Internal parent restriction is exclusively from evolved SHADOW
        // leaves, with end-time halos refreshed from that same hierarchy.
        fillHalo(fields, shadows, domain, fields.timeState().time, true);
        hierarchy.addPatch(parent, std::move(fields));
    }
    std::array<Real, State::size()> absolute{}, relative{};
    absolute.fill(Real(1e-8));
    relative.fill(Real(1e-2));
    std::vector<FieldLeafError<State>> result;
    for (auto const& leaf : physical) {
        leaf.fields.layout().forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
            if (!system.admissible(leaf.fields.values()[index]))
                throw std::runtime_error("Invalid physical state in shadow comparison");
        });
        result.push_back({leaf.location, hierarchy.estimateLeafError(leaf.location,
            leaf.fields, absolute, relative, [&](auto const& state) { return system.admissible(state); })});
    }
    return result;
}

} // namespace transportExchange
} // namespace octotiger
