//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/common_kernel/interactions_iterators.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace octotiger {
namespace fmm {
    multiindex<> get_padding_real_size(const geo::direction& dir) {
        multiindex<> size;
        if (dir[0] == -1 || dir[0] == 1)
            size.x = STENCIL_MAX;
        else
            size.x = INNER_CELLS_PER_DIRECTION;
        if (dir[1] == -1 || dir[1] == 1)
            size.y = STENCIL_MAX;
        else
            size.y = INNER_CELLS_PER_DIRECTION;
        if (dir[2] == -1 || dir[2] == 1)
            size.z = STENCIL_MAX;
        else
            size.z = INNER_CELLS_PER_DIRECTION;
        return size;
    }
    multiindex<> get_padding_start_indices(const geo::direction& dir) {
        multiindex<> start_index;
        if (dir[0] == -1)
            start_index.x = INNER_CELLS_PER_DIRECTION - STENCIL_MAX;
        else
            start_index.x = 0;
        if (dir[1] == -1)
            start_index.y = INNER_CELLS_PER_DIRECTION - STENCIL_MAX;
        else
            start_index.y = 0;
        if (dir[2] == -1)
            start_index.z = INNER_CELLS_PER_DIRECTION - STENCIL_MAX;
        else
            start_index.z = 0;
        return start_index;

    }
    multiindex<> get_padding_end_indices(const geo::direction& dir) {
        multiindex<> end_index;
        if (dir[0] == 1)
            end_index.x = STENCIL_MAX;
        else
            end_index.x = INNER_CELLS_PER_DIRECTION;
        if (dir[1] == 1)
            end_index.y = STENCIL_MAX;
        else
            end_index.y = INNER_CELLS_PER_DIRECTION;
        if (dir[2] == 1)
            end_index.z = STENCIL_MAX;
        else
            end_index.z = INNER_CELLS_PER_DIRECTION;
        return end_index;
    }
    bool expansion_comparator(const expansion& ref, const expansion& mine) {
        if (ref.size() != mine.size()) {
            std::cout << "size of expansion doesn't match" << std::endl;
            return false;
        }
        for (size_t i = 0; i < mine.size(); i++) {
            if (std::abs(ref[i] - mine[i]) >= 10000.0 * std::numeric_limits<real>::epsilon()) {
                std::cout << "error: index padded: " << i << ", mine[" << i << "] != ref[" << i
                          << "] <=> " << mine[i] << " != " << ref[i] << ", "
                          << std::abs(ref[i] - mine[i])
                          << " >= " << 1000.0 * std::numeric_limits<real>::epsilon() << std::endl;
                return false;
            }
        }
        return true;
    }

    bool space_vector_comparator(const space_vector& ref, const space_vector& mine) {
        for (size_t i = 0; i < mine.size(); i++) {
            if (std::abs(ref[i] - mine[i]) >= 10000.0 * std::numeric_limits<real>::epsilon()) {
                std::cout << "error: index padded: " << i << ", mine[" << i << "] != ref[" << i
                          << "] <=> " << mine[i] << " != " << ref[i] << ", "
                          << std::abs(ref[i] - mine[i])
                          << " >= " << 1000.0 * std::numeric_limits<real>::epsilon() << std::endl;
                return false;
            }
        }
        return true;
    }

}    // namespace fmm
}    // namespace octotiger
