// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace radiation {

// Dedicated, transient tree messages. No M1 halo/reflux channel is reused.
// Coefficients and nonlinear/V-cycle scratch stay at the reference solve owner.
struct fld_runtime_cell {
    std::array<double, 3> center{}, flux{};
    double width = 0, density = 0, mmw = 0, hydrogen = 0, metals = 0;
    double energy = 0, internal_energy = 0;
    double boundary_energy = 0, source_energy = 0;
    template<class Archive> void serialize(Archive& arc, unsigned) {
        arc & center & flux & width & density & mmw & hydrogen & metals;
        arc & energy & internal_energy & boundary_energy & source_energy;
    }
};

struct fld_packet {
    std::vector<fld_runtime_cell> cells;
    std::string error;
    template<class Archive> void serialize(Archive& arc, unsigned) {
        arc & cells & error;
    }
};

} // namespace radiation
