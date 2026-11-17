// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

namespace octotiger::multigrid {

// Shared by the distributed adapter and the numerical fixture. Each callback
// must finish the selected level before returning. A negative level selects
// every block. Adapters map these stages into their own action identifiers.
enum class PoissonStage : int {
    initialize, restrict_residual, adjust_rhs, halo_down, halo_up,
    smooth, prolong, retire
};

template <class Run>
void poissonCycle(int deepest, Run&& run) {
    constexpr int preSweeps = 2;
    constexpr int postSweeps = 4;
    run(PoissonStage::initialize, -1);
    for (int level = deepest; level >= 0; --level) {
        run(PoissonStage::restrict_residual, level);
        run(PoissonStage::adjust_rhs, level);
        for (int sweep = 0; sweep < preSweeps; ++sweep) {
            run(PoissonStage::halo_down, level);
            run(PoissonStage::smooth, level);
        }
        // Restriction and coarse/fine fluxes use the remaining residual, so
        // they must see neighbors after the last smoothing sweep.
        run(PoissonStage::halo_down, level);
    }
    for (int level = 0; level <= deepest; ++level) {
        if (level > 0) {
            // Trilinear prolongation needs refreshed parent edges and corners.
            run(PoissonStage::halo_up, level - 1);
            run(PoissonStage::prolong, level);
        }
        for (int sweep = 0; sweep < postSweeps; ++sweep) {
            run(PoissonStage::halo_up, level);
            run(PoissonStage::smooth, level);
        }
    }
    run(PoissonStage::retire, -1);
}

} // namespace octotiger::multigrid
