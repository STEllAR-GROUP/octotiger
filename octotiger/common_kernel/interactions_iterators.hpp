//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/geometry.hpp"
//#include "octotiger/interaction_types.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/taylor.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    // meant to iterate the input data structure
    template <typename F>
    void iterate_inner_cells_padded(const F& f) {
        for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
            for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                    const multiindex<> m(i0 + INNER_CELLS_PADDING_DEPTH,
                        i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                    const size_t inner_flat_index = to_flat_index_padded(m);
                    const multiindex<> m_unpadded(i0, i1, i2);
                    const size_t inner_flat_index_unpadded =
                        to_inner_flat_index_not_padded(m_unpadded);
                    f(m, inner_flat_index, m_unpadded, inner_flat_index_unpadded);
                }
            }
        }
    }

    multiindex<> get_padding_real_size(const geo::direction& dir); 
    multiindex<> get_padding_start_indices(const geo::direction& dir); 
    multiindex<> get_padding_end_indices(const geo::direction& dir); 

    template <typename F>
    void iterate_padding(const geo::direction& dir, const F& f) {
        multiindex<> start_index = get_padding_start_indices(dir);
        multiindex<> end_index = get_padding_end_indices(dir);
        multiindex<> size = get_padding_real_size(dir);
        assert(end_index.x - start_index.x == size.x);
        assert(end_index.y - start_index.y == size.y);
        assert(end_index.z - start_index.z == size.z);
        for (size_t i0 = start_index.x; i0 < end_index.x; i0++) {
            for (size_t i1 = start_index.y; i1 < end_index.y; i1++) {
                for (size_t i2 = start_index.z; i2 < end_index.z; i2++) {
                    // shift to central cube and apply dir-based offset
                    const multiindex<> m_unpadded(i0, i1, i2);
                    const multiindex<> m_padding_index(
                        i0 - start_index.x, i1 - start_index.y, i2 - start_index.z);
                    const size_t inner_flat_index_unpadded =
                        to_inner_flat_index_not_padded(m_unpadded);
                    const size_t flat_index_padding = m_padding_index.x * (size.y * size.z) +
                        m_padding_index.y * size.z + m_padding_index.z;
                    f(m_padding_index, flat_index_padding, m_unpadded, inner_flat_index_unpadded);
                }
            }
        }
    }

    template <typename F>
    void iterate_inner_cells_padding(const geo::direction& dir, const F& f) {
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

        multiindex<> padded_start(0, 0, 0);
        if (dir[0] == 1)
            padded_start.x = INNER_CELLS_PER_DIRECTION + INNER_CELLS_PER_DIRECTION;
        else if (dir[0] == 0)
            padded_start.x = INNER_CELLS_PER_DIRECTION;
        if (dir[1] == 1)
            padded_start.y = INNER_CELLS_PER_DIRECTION + INNER_CELLS_PER_DIRECTION;
        else if (dir[1] == 0)
            padded_start.y = INNER_CELLS_PER_DIRECTION;
        if (dir[2] == 1)
            padded_start.z = INNER_CELLS_PER_DIRECTION + INNER_CELLS_PER_DIRECTION;
        else if (dir[2] == 0)
            padded_start.z = INNER_CELLS_PER_DIRECTION;
        for (size_t i0 = start_index.x; i0 < end_index.x; i0++) {
            for (size_t i1 = start_index.y; i1 < end_index.y; i1++) {
                for (size_t i2 = start_index.z; i2 < end_index.z; i2++) {
                    // shift to central cube and apply dir-based offset
                    const multiindex<> m(
                        i0 + padded_start.x, i1 + padded_start.y, i2 + padded_start.z);
                    const size_t inner_flat_index = to_flat_index_padded(m);
                    const multiindex<> m_unpadded(i0, i1, i2);

                    // Unpadded index is correct now
                    const size_t inner_flat_index_unpadded =
                        to_inner_flat_index_not_padded(m_unpadded);
                    // std::cout << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
                    // std::cout << m << std::endl;
                    // std::cout << m_unpadded << std::endl;
                    // std::cin.get();
                    f(m, inner_flat_index, m_unpadded, inner_flat_index_unpadded);
                }
            }
        }
    }

}    // namespace fmm
}    // namespace octotiger
