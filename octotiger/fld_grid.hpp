// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "octotiger/defs.hpp"
#include "octotiger/fld_closure.hpp"
#include "octotiger/fld_opacity.hpp"
#include "octotiger/fld_stage_result.hpp"
#include "octotiger/multigrid/reaction_diffusion.hpp"

namespace octotiger::fld {

struct Parameters : OpacityParameters
{
    double cv = 1.0;
    double c = 1.0;
    double ar = 1.0;
};

}    // namespace octotiger::fld

// Owns the implicit-stage scratch state. The shared hydro-layout energy vector
// is changed only by commit(), after global nonlinear convergence.
class fld_grid
{
public:
    using Field = octotiger::multigrid::Field;
    std::shared_ptr<std::vector<real>> sharedEnergy;
    octotiger::multigrid::ReactionDiffusionBlock linear;
    Field energy;
    Field density;
    Field capacity;
    // Additive composition moments: rho * sum X Z/A and rho * sum X Z^2/A.
    // Restrict these densities, then divide by restricted rho for opacity.
    Field electronDensity;
    Field ionChargeSquaredDensity;
    Field temperature;
    Field energyStar;
    Field thermalStar;
    Field opacity; // Planck absorption coefficient, in inverse code length.
    Field extinction; // Rosseland total extinction, in inverse code length.
    Field diffusionCell;
    Field savedEnergy;
    Field savedTemperature;
    Field proposedEnergy;
    Field proposedTemperature;
    octotiger::fld::Parameters parameters;
    double dt = 0.0;
    size_t regularizedCells = 0;

    fld_grid(std::shared_ptr<std::vector<real>> radiationEnergy, double cellWidth);
    void initialize(double timestep, octotiger::fld::Parameters const& values,
        std::vector<real> const& rho, std::vector<real> const& internalEnergy,
        std::vector<real> const& cellCapacity = {},
        std::vector<real> const& cellElectronFraction = {},
        std::vector<real> const& cellIonChargeSquared = {});
    // Refresh ghosts of energy, density, temperature, capacity and composition.
    void refreshCellCoefficients();
    void refreshFaceCoefficients();
    // Override a fine/coarse boundary using its actual center separation. The
    // stored b includes dx/distance, so the block's standard face stencil applies.
    void refreshBoundaryFaceCoefficients(int axis, int side, double neighborWidthRatio);
    // Physical walls have no neighbor diffusion stencil. Vacuum supplies a
    // positive, lagged modified-Marshak conductance through the half-cell.
    // The mask uses 2*axis+side; interior one-sided derivatives avoid mirrored
    // physical ghosts in every affected direction, including edges/corners.
    void refreshPhysicalBoundaryFaceCoefficients(int axis, int side, bool vacuum,
        std::array<bool, NFACE> const& physicalBoundaries);
    // Freeze opacity and face diffusivity, and eliminate the linearized material
    // temperature equation. The resulting scalar operator has positive reaction.
    void prepareLinearization();
    // Save an immutable feasible Newton/Picard proposal. Project negative
    // linear radiation iterates to zero, then back substitute the material
    // row. The caller still checks the full nonlinear residual before commit.
    double prepareUpdate();
    void applyUpdate(double damping);
    // Requires current coefficients, solution=energy, and reconciled fluxes.
    // Returns integral Schur-scaled radiation residual^2, scaled material
    // residual^2, scale^2, max scaled residual. This invertible local row scaling
    // removes k*roundoff amplification in stiff equilibrium; their sum is the
    // unscaled total-energy residual. Covered blocks are excluded by the caller.
    octotiger::fld::NonlinearResidual computeNonlinearResidual() const;
    // Scale the frozen linear residual by each proposed radiation/material
    // reservoir, so global L2 convergence cannot hide errors in cold cells.
    double computeLinearResidualMaximum() const;
    std::vector<real> materialEnergyIncrement() const;
    // Return the material internal-energy increment in H_N3 layout. It balances
    // the accepted radiation change and reconciled diffusion flux exactly.
    std::vector<real> commit();
};
