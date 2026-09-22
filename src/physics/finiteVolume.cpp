// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/physics/finiteVolume.hpp"

namespace octotiger::physics {

std::string_view finiteVolumeSchemeName() {
    return "unsplit MUSCL-Hancock with face-centered fluxes";
}

} // namespace octotiger::physics
