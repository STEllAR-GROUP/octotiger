// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/mesh.hpp"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace octotiger::mesh {

struct MeshNode {
    BlockLocation location;
    bool refined = false;
};

// Dimension-aware AMR topology. Refinement always creates the complete active
// sibling set: 2 children in 1-D, 4 in 2-D, and 8 in 3-D.
class MeshHierarchy {
public:
    explicit MeshHierarchy(int dimensionCount);

    int dimensionCount() const;
    BlockLocation rootLocation() const;
    bool contains(BlockLocation const& location) const;
    bool isLeaf(BlockLocation const& location) const;
    bool isRefined(BlockLocation const& location) const;
    std::size_t nodeCount() const;
    std::size_t leafCount() const;
    std::vector<BlockLocation> children(
        BlockLocation const& location) const;
    std::vector<BlockLocation> leafLocations() const;

    std::vector<BlockLocation> refine(BlockLocation const& location);
    std::vector<BlockLocation> coarsen(BlockLocation const& location);

    // Remove the finest level everywhere while retaining a proper AMR tree.
    // A shadow node is refined exactly when at least one corresponding child
    // in the physical hierarchy is refined. This is the topology obtained by
    // evolving a complete copy of the hierarchy one resolution level coarser.
    [[nodiscard]] MeshHierarchy oneLevelCoarser() const;

private:
    int dimensionCount_;
    std::unordered_map<BlockLocation, MeshNode, BlockLocationHash> nodes_;

    MeshNode const& node(BlockLocation const& location) const;
    MeshNode& node(BlockLocation const& location);
    void validateLocation(BlockLocation const& location) const;
    void collectDescendants(BlockLocation const& location,
        std::vector<BlockLocation>& descendants) const;
    void buildCoarserBranch(BlockLocation const& location,
        MeshHierarchy& result) const;
};

} // namespace octotiger::mesh
