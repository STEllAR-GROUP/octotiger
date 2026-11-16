// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "octotiger/defs.hpp"

namespace octotiger::multigrid {

// Numerical port of dmarce1/scorpio b39beb91, src/multigrid and array3d.h.
// Interior indices are 1..n; 0 and n+1 are ghosts. The z index is contiguous.
// Geometry, indices, and transfer sizes are internal preconditions checked by
// assertions, which are disabled when NDEBUG is defined.
struct Field
{
    int n = 0;
    std::vector<double> values;

    Field() = default;
    explicit Field(int interiorSize);
    double& operator()(int i, int j, int k);
    double operator()(int i, int j, int k) const;
    void fill(double value);
    double laplacianNumerator(int i, int j, int k) const;
    double octetAverage(int i, int j, int k) const;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & n & values;
    }
};

class Block
{
public:
    int n;
    double dx;
    Field phi;
    Field source;
    Field correction;
    Field correctionRhs;
    std::array<Field, NDIM> force;
    // OctoTiger refines all eight children or none. Covered parent cells are
    // solver storage; a leaf owns every interior cell as physical solution.
    bool refined = false;

    Block(int interiorSize, double cellWidth);

    // Preserve received child corrections in a refined block. Initialize leaf
    // interiors and homogeneous correction ghosts; exchange other ghosts afterward.
    void beginDown();
    void saveCorrectionRhs();
    // One complete local red/black pair. Halo snapshots stay fixed in this call.
    void relax();
    void retireCorrection();
    void computeForces();
    // Call after forces have been computed and coarse AMR faces overwritten.
    double physicalResidual(int i, int j, int k) const;
    double residualNorm() const;
    double residualMaximum() const;

    // Vectors use z-contiguous ordering, with (n/2)^3 entries.
    std::vector<double> restrictField(Field const& field) const;
    std::vector<double> childValues(Field const& field, int octant) const;
    void setChild(Field& field, int octant, std::vector<double> const& values) const;
    void replaceCorrectionMean(std::vector<double> const& parentValues);

    // Logical coordinates are in coarse cell-index units: an integer denotes
    // that cell's center. The 27-point polynomial is Scorpio's phi interpolation.
    static double quadraticInterpolate(
        Field const& field, std::array<double, NDIM> const& logicalPosition);
    // Sample only; the caller assigns the returned value to its destination.
    static double sampleNearestCell(Field const& field, std::array<double, NDIM> const& logicalPosition);

    // side=0 lower, side=1 upper. Positive-axis orientation on both sides.
    // Jumps: half the sum of four fine phi differences (coarse dx scaling).
    // Forces: quarter the sum of four physical fine accelerations.
    // The tangential dimensions are (axis+1)%NDIM and (axis+2)%NDIM, second fastest.
    std::vector<double> coarseFaceJumps(int axis, int side) const;
    std::vector<double> coarseFaceForces(int axis, int side) const;
    // faceIndex is the right cell's index, 1 or n+1; t0/t1 are zero-based offsets
    // of a half-face patch on an unrefined block's boundary.
    void adjustFluxDefect(
        int axis, int faceIndex, int t0, int t1, std::vector<double> const& jumps);
    void setRestrictedForce(
        int axis, int faceIndex, int t0, int t1, std::vector<double> const& forces);
};

}    // namespace octotiger::multigrid
