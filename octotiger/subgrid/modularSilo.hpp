// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once
#include "octotiger/subgrid/transportRuntime.hpp"
#include <string>
namespace octotiger {
// Common hydro/radiation visualization writer; refuses existing files.
// Exports radiation (E,F=cQ) in physical cgs units, never reduced-speed cHat*Q.
void writeModularSilo(std::vector<TransportSnapshot> const& patches,
    std::string const& filename, int cycle, Real time);
}
