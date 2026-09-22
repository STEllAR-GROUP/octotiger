// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/mesh/meshHierarchy.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace octotiger::mesh {
namespace {

bool locationLess(BlockLocation const& left, BlockLocation const& right) {
    if (left.level != right.level) {
        return left.level < right.level;
    }
    return left.coordinates < right.coordinates;
}

} // namespace

MeshHierarchy::MeshHierarchy(int dimensionCount) :
    dimensionCount_(dimensionCount) {
    if (dimensionCount_ < 1 || dimensionCount_ > maximumDimensionCount) {
        throw std::invalid_argument("Mesh hierarchy dimension count must be 1, 2, or 3");
    }
    BlockLocation const root = rootLocation();
    nodes_.emplace(root, MeshNode{root, false});
}

int MeshHierarchy::dimensionCount() const {
    return dimensionCount_;
}

BlockLocation MeshHierarchy::rootLocation() const {
    return BlockLocation{0, {0, 0, 0}, dimensionCount_};
}

bool MeshHierarchy::contains(BlockLocation const& location) const {
    return nodes_.contains(location);
}

bool MeshHierarchy::isLeaf(BlockLocation const& location) const {
    return !node(location).refined;
}

bool MeshHierarchy::isRefined(BlockLocation const& location) const {
    return node(location).refined;
}

std::size_t MeshHierarchy::nodeCount() const {
    return nodes_.size();
}

std::size_t MeshHierarchy::leafCount() const {
    return static_cast<std::size_t>(std::count_if(nodes_.begin(), nodes_.end(),
        [](auto const& entry) { return !entry.second.refined; }));
}

std::vector<BlockLocation> MeshHierarchy::children(
    BlockLocation const& location) const {
    validateLocation(location);
    std::vector<BlockLocation> result;
    result.reserve(static_cast<std::size_t>(1 << dimensionCount_));
    for (int slot = 0; slot < (1 << dimensionCount_); ++slot) {
        result.push_back(location.child(slot));
    }
    return result;
}

std::vector<BlockLocation> MeshHierarchy::leafLocations() const {
    std::vector<BlockLocation> result;
    result.reserve(leafCount());
    for (auto const& [location, meshNode] : nodes_) {
        if (!meshNode.refined) {
            result.push_back(location);
        }
    }
    std::sort(result.begin(), result.end(), locationLess);
    return result;
}

std::vector<BlockLocation> MeshHierarchy::refine(
    BlockLocation const& location) {
    if (node(location).refined) {
        throw std::logic_error("Cannot refine an AMR node that is already refined");
    }
    std::vector<BlockLocation> const newChildren = children(location);
    for (BlockLocation const& child : newChildren) {
        auto const [iterator, inserted] = nodes_.emplace(child,
            MeshNode{child, false});
        if (!inserted) {
            throw std::logic_error("AMR child already exists");
        }
        (void) iterator;
    }
    node(location).refined = true;
    return newChildren;
}

std::vector<BlockLocation> MeshHierarchy::coarsen(
    BlockLocation const& location) {
    MeshNode& parent = node(location);
    if (!parent.refined) {
        throw std::logic_error("Cannot coarsen an AMR leaf");
    }
    std::vector<BlockLocation> descendants;
    collectDescendants(location, descendants);
    for (BlockLocation const& descendant : descendants) {
        nodes_.erase(descendant);
    }
    parent.refined = false;
    return descendants;
}

MeshHierarchy MeshHierarchy::oneLevelCoarser() const {
    MeshHierarchy result(dimensionCount_);
    buildCoarserBranch(rootLocation(), result);
    return result;
}

MeshNode const& MeshHierarchy::node(BlockLocation const& location) const {
    validateLocation(location);
    auto const iterator = nodes_.find(location);
    if (iterator == nodes_.end()) {
        throw std::out_of_range("AMR node does not exist");
    }
    return iterator->second;
}

MeshNode& MeshHierarchy::node(BlockLocation const& location) {
    return const_cast<MeshNode&>(std::as_const(*this).node(location));
}

void MeshHierarchy::validateLocation(BlockLocation const& location) const {
    if (location.dimensionCount != dimensionCount_ || location.level < 0 ||
        location.level >= std::numeric_limits<unsigned>::digits - 1) {
        throw std::invalid_argument("AMR location has the wrong dimensionality or level");
    }
    unsigned const locationsPerAxis = 1u << location.level;
    for (int axis = 0; axis < maximumDimensionCount; ++axis) {
        if (axis < dimensionCount_) {
            if (location.coordinates[axis] < 0 ||
                static_cast<unsigned>(location.coordinates[axis]) >= locationsPerAxis) {
                throw std::invalid_argument("AMR block coordinate is outside its level");
            }
        } else if (location.coordinates[axis] != 0) {
            throw std::invalid_argument("Inactive AMR coordinates must be zero");
        }
    }
}

void MeshHierarchy::collectDescendants(BlockLocation const& location,
    std::vector<BlockLocation>& descendants) const {
    for (BlockLocation const& child : children(location)) {
        MeshNode const& childNode = node(child);
        if (childNode.refined) {
            collectDescendants(child, descendants);
        }
        descendants.push_back(child);
    }
}

void MeshHierarchy::buildCoarserBranch(BlockLocation const& location,
    MeshHierarchy& result) const {
    MeshNode const& physicalNode = node(location);
    if (!physicalNode.refined) {
        return;
    }
    bool hasRefinedChild = false;
    for (BlockLocation const& child : children(location)) {
        hasRefinedChild = hasRefinedChild || node(child).refined;
    }
    if (!hasRefinedChild) {
        return;
    }
    result.refine(location);
    for (BlockLocation const& child : children(location)) {
        buildCoarserBranch(child, result);
    }
}

} // namespace octotiger::mesh
