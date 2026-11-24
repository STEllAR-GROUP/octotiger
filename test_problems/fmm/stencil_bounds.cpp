// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <cstddef>
#include <iostream>
#include <vector>

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/multiindex.hpp"

int main() {
    using namespace octotiger::fmm;

    int const side = STENCIL_MAX - STENCIL_MIN + 1;
    if (NUMBER_P2P_BLOCKS != side || NUMBER_MULTIPOLE_BLOCKS != side) {
        std::cerr << "Stencil launch dimensions do not cover the offset range\n";
        return 1;
    }
    std::vector<bool> visited(FULL_STENCIL_SIZE, false);
    size_t count = 0;
    for (int x = STENCIL_MIN; x <= STENCIL_MAX; ++x) {
        for (int y = STENCIL_MIN; y <= STENCIL_MAX; ++y) {
            for (int z = STENCIL_MIN; z <= STENCIL_MAX; ++z) {
                size_t const index = (x - STENCIL_MIN) * STENCIL_INX * STENCIL_INX
                    + (y - STENCIL_MIN) * STENCIL_INX + (z - STENCIL_MIN);
                if (index >= visited.size() || visited[index]) {
                    std::cerr << "Stencil offset " << x << ',' << y << ',' << z
                              << " has an invalid or aliased index " << index << '\n';
                    return 1;
                }
                visited[index] = true;
                ++count;

                for (int corner = 0; corner < 8; ++corner) {
                    multiindex<> const partner(
                        INX + ((corner & 1) ? INX - 1 : 0) + x,
                        INX + ((corner & 2) ? INX - 1 : 0) + y,
                        INX + ((corner & 4) ? INX - 1 : 0) + z);
                    auto const flat = to_flat_index_padded(partner);
                    if (flat < 0 || static_cast<size_t>(flat) >= ENTRIES) {
                        std::cerr << "Stencil partner lies outside the padded grid\n";
                        return 1;
                    }
                }
            }
        }
    }
    if (count != visited.size()) {
        std::cerr << "Stencil offsets do not cover the allocated mask\n";
        return 1;
    }
    std::cout << "theta minimum " << THETA_FLOOR << ": " << count
              << " stencil indices and padded-grid corner bounds passed\n";
}
