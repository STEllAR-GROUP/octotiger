//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/cuda_util/cuda_global_def.hpp"
#include "octotiger/defs.hpp"

#include <cmath>
#include <cstddef>
//#include <iostream>
#include <vector>

namespace octotiger {
namespace fmm {
    // is template to allow for vectorization
    template <typename T = int32_t>
    class multiindex
    {
    public:
        T x;
        T y;
        T z;

        CUDA_GLOBAL_METHOD multiindex(T x, T y, T z)
          : x(x)
          , y(y)
          , z(z) {
        }

        template <typename U>
        CUDA_GLOBAL_METHOD multiindex(const multiindex<U>& other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        CUDA_GLOBAL_METHOD multiindex() {
            /* do not initialise anything in here! If a value is touched in the default
            constructor, we cannot use the class within cuda constant memory (and we want to) */
        }

        // // remove when vectorization is fully enabled
        // multiindex(size_t x, size_t y, size_t z)
        //   : x(x)
        //   , y(y)
        //   , z(z) {}

        CUDA_GLOBAL_METHOD inline double length() const {
            return sqrt(static_cast<double>(x * x + y * y + z * z));
        }

        CUDA_GLOBAL_METHOD inline bool compare(multiindex& other) {
            if (this->x == other.x && this->y == other.y && this->z == other.z) {
                return true;
            } else {
                return false;
            }
        }

        CUDA_GLOBAL_METHOD inline bool operator == (const multiindex& other) const {
            if (this->x == other.x && this->y == other.y && this->z == other.z) {
                return true;
            } else {
                return false;
            }
        }

        // set this multiindex to the next coarser level index
        CUDA_GLOBAL_METHOD void transform_coarse() {
            const T patch_size = static_cast<typename T::value_type>(INX);
            const T subtract = static_cast<typename T::value_type>(INX / 2);

            x = ((x + patch_size) >> 1) - subtract;
            y = ((y + patch_size) >> 1) - subtract;
            z = ((z + patch_size) >> 1) - subtract;
        }
    };

    CUDA_GLOBAL_METHOD inline multiindex<> flat_index_to_multiindex_not_padded(
        size_t flat_index) {
        size_t x = flat_index / (INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION);
        flat_index %= (INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION);
        size_t y = flat_index / INNER_CELLS_PER_DIRECTION;
        flat_index %= INNER_CELLS_PER_DIRECTION;
        size_t z = flat_index;
        multiindex<> m(x, y, z);
        return m;
    }

    CUDA_GLOBAL_METHOD inline multiindex<> flat_index_to_multiindex_padded(size_t flat_index) {
        size_t x = flat_index / (PADDED_STRIDE * PADDED_STRIDE);
        flat_index %= (PADDED_STRIDE * PADDED_STRIDE);
        size_t y = flat_index / PADDED_STRIDE;
        flat_index %= PADDED_STRIDE;
        size_t z = flat_index;
        multiindex<> m(x, y, z);
        return m;
    }

    template <typename T>
    CUDA_GLOBAL_METHOD inline T to_flat_index_padded(const multiindex<T>& m) {
      return (m.x - PADDING_OFFSET) * PADDED_STRIDE * PADDED_STRIDE + (m.y - PADDING_OFFSET) *
      PADDED_STRIDE + (m.z - PADDING_OFFSET);
    }

    /** are only valid for single cell! (no padding)
     * Note: for m2m_int_vector and integer
     * Note: returns uint32_t vector because of Vc limitation */
    template <typename T>
    CUDA_GLOBAL_METHOD inline T to_inner_flat_index_not_padded(const multiindex<T>& m) {
        return m.x * INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION +
            m.y * INNER_CELLS_PER_DIRECTION + m.z;
    }

    // This specialization is only required on cuda devices since T::value_type is not supported!
    template <>
    CUDA_GLOBAL_METHOD inline void multiindex<int32_t>::transform_coarse() {
        const int32_t patch_size = static_cast<int32_t>(INX);
        const int32_t subtract = static_cast<int32_t>(INX / 2);
        x = ((x + patch_size) >> 1) - subtract;
        y = ((y + patch_size) >> 1) - subtract;
        z = ((z + patch_size) >> 1) - subtract;
    }

    CUDA_GLOBAL_METHOD inline int32_t distance_squared_reciprocal(
        const multiindex<>& i, const multiindex<>& j) {
        return ((i.x - j.x) * (i.x - j.x) + (i.y - j.y) * (i.y - j.y) + (i.z - j.z) * (i.z - j.z));
    }

    struct two_phase_stencil
    {
        std::vector<multiindex<>> stencil_elements;
        std::vector<bool> stencil_phase_indicator;
    };
}    // namespace fmm
}    // namespace octotiger

template <typename T>
std::ostream& operator<<(std::ostream& os, const octotiger::fmm::multiindex<T>& m) {
    return os << m.x << ", " << m.y << ", " << m.z;
}
