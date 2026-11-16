// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <array>
#include <limits>

#include "octotiger/defs.hpp"

namespace octotiger::fld {

struct NonlinearResidual {
    double radiationSquared = 0.0;
    double materialSquared = 0.0;
    double scaleSquared = 0.0;
    double maximum = 0.0;

    void accumulate(NonlinearResidual const& other) {
        radiationSquared += other.radiationSquared;
        materialSquared += other.materialSquared;
        scaleSquared += other.scaleSquared;
        maximum = std::max(maximum, other.maximum);
    }
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & radiationSquared & materialSquared & scaleSquared & maximum;
    }
};

struct Diagnostics {
    double radiationEnergy = 0.0;
    double thermalEnergy = 0.0;
    double initialRadiationEnergy = 0.0;
    double initialThermalEnergy = 0.0;
    double minimumEnergy = std::numeric_limits<double>::infinity();
    double maximumEnergy = 0.0;
    double minimumTemperature = std::numeric_limits<double>::infinity();
    double unsharedBlocks = 0.0;
    double gasTotalEnergy = 0.0;
    double initialGasTotalEnergy = 0.0;
    double tauThermalEnergy = 0.0;
    double opacityRegularizedCells = 0.0;
    double coldEnergy = 0.0;
    double atmosphereThermalEnergy = 0.0;
    double atmosphereRadiationEnergy = 0.0;
    double escapedDiffusionEnergy = 0.0;
    std::array<double, NFACE> faceEscape{};

    void accumulate(Diagnostics const& other) {
        radiationEnergy += other.radiationEnergy;
        thermalEnergy += other.thermalEnergy;
        initialRadiationEnergy += other.initialRadiationEnergy;
        initialThermalEnergy += other.initialThermalEnergy;
        minimumEnergy = std::min(minimumEnergy, other.minimumEnergy);
        maximumEnergy = std::max(maximumEnergy, other.maximumEnergy);
        minimumTemperature = std::min(minimumTemperature, other.minimumTemperature);
        unsharedBlocks += other.unsharedBlocks;
        gasTotalEnergy += other.gasTotalEnergy;
        initialGasTotalEnergy += other.initialGasTotalEnergy;
        tauThermalEnergy += other.tauThermalEnergy;
        opacityRegularizedCells += other.opacityRegularizedCells;
        coldEnergy += other.coldEnergy;
        atmosphereThermalEnergy += other.atmosphereThermalEnergy;
        atmosphereRadiationEnergy += other.atmosphereRadiationEnergy;
        escapedDiffusionEnergy += other.escapedDiffusionEnergy;
        for (int face = 0; face < NFACE; ++face) {
            faceEscape[face] += other.faceEscape[face];
        }
    }
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & radiationEnergy & thermalEnergy & initialRadiationEnergy & initialThermalEnergy;
        archive & minimumEnergy & maximumEnergy & minimumTemperature & unsharedBlocks;
        archive & gasTotalEnergy & initialGasTotalEnergy & tauThermalEnergy & opacityRegularizedCells;
        archive & coldEnergy & atmosphereThermalEnergy & atmosphereRadiationEnergy;
        archive & escapedDiffusionEnergy;
        for (auto& escaped : faceEscape) {
            archive & escaped;
        }
    }
};

// Each stage fills only the fields it reduces. Named fields avoid a shared
// positional packet whose meaning and length depend on the caller's stage.
struct StageResult {
    int deepestLevel = 0;
    double leaves = 0.0;
    double dualEnergyCells = 0.0;
    double linearResidualSquared = 0.0;
    double linearRhsSquared = 0.0;
    double linearMaximum = 0.0;
    double updateLimit = 1.0;
    NonlinearResidual nonlinear;
    Diagnostics diagnostics;

    void accumulate(StageResult const& other) {
        deepestLevel = std::max(deepestLevel, other.deepestLevel);
        leaves += other.leaves;
        dualEnergyCells += other.dualEnergyCells;
        linearResidualSquared += other.linearResidualSquared;
        linearRhsSquared += other.linearRhsSquared;
        linearMaximum = std::max(linearMaximum, other.linearMaximum);
        updateLimit = std::min(updateLimit, other.updateLimit);
        nonlinear.accumulate(other.nonlinear);
        diagnostics.accumulate(other.diagnostics);
    }
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & deepestLevel & leaves & dualEnergyCells;
        archive & linearResidualSquared & linearRhsSquared & linearMaximum & updateLimit;
        archive & nonlinear & diagnostics;
    }
};

} // namespace octotiger::fld
