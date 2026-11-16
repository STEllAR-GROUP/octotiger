// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <array>
#include <vector>

#include "octotiger/multigrid/block.hpp"

namespace octotiger::multigrid {

// Matrix-free finite-volume A u = a u + div(F), F = -b grad(u).
// All residuals/RHS values have physical units; no Poisson diagonal normalization.
// Face storage uses the right cell's index, 1..n+1 along its normal axis.
class ReactionDiffusionBlock
{
public:
    int n;
    double dx;
    bool refined = false;
    Field solution;
    Field rhs;
    Field reaction;
    Field residual;
    Field saved;
    Field fluxDefect;
    std::array<Field, NDIM> diffusion;
    // Timestep-integrated outward conductance dt*H (units length), on local
    // boundary faces only. The additional outward flux is dt*H*u_inside.
    // Physical-wall diffusion is set separately by the caller, normally zero.
    std::array<Field, NDIM> wallConductance;
    std::array<Field, NDIM> flux;

    ReactionDiffusionBlock(int interiorSize, double cellWidth);
    double applyLocal(int i, int j, int k) const;
    double fluxDivergence(int i, int j, int k) const;
    double diagonal(int i, int j, int k) const;
    void computeFluxes();
    // Call after replacing coarse interface fluxes by area-averaged fine fluxes.
    void updateFluxDefect();
    void computeResidual();
    // One local red/black pair with fixed halo and interface-defect snapshots.
    void relax(double omega = 1.0);
    void saveSolution();
    // After solution=R(u_h), saved=solution, and flux reconciliation on this level.
    // The supplied field is the volume-restricted fine physical residual.
    void formCoarseRhs(Field const& restrictedResidual);
    std::vector<double> restrictField(Field const& field) const;
    std::vector<double> childValues(Field const& field, int octant) const;
    std::vector<double> childCorrection(int octant) const;
    void setChild(Field& field, int octant, std::vector<double> const& values) const;
    // Add P(u_H - R u_h), with piecewise-constant injection into each octet.
    void addCoarseCorrection(std::vector<double> const& values);
    std::vector<double> coarseFaceFluxes(int axis, int side) const;
    void setRestrictedFlux(int axis, int faceIndex, int t0, int t1,
        std::vector<double> const& values);
    // Restrict the coefficients on aligned faces; payload normal index is fastest.
    std::vector<double> restrictFaceCoefficients(int axis) const;
    void setChildFaceCoefficients(int axis, int octant, std::vector<double> const& values);
    // Same face layout and area weighting as diffusion; no dx rescaling.
    std::vector<double> restrictWallConductance(int axis) const;
    void setChildWallConductance(int axis, int octant, std::vector<double> const& values);
};

}    // namespace octotiger::multigrid
