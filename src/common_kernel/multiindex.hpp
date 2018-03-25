#pragma once

#include "../cuda_util/cuda_global_def.hpp"
#include "defs.hpp"
#include "interaction_constants.hpp"

#include <cmath>
#include <iostream>
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

        CUDA_CALLABLE_METHOD multiindex(T x, T y, T z)
          : x(x)
          , y(y)
          , z(z) {
            // std::cout << "x arg: " << x << std::endl;
            // std::cout << "this->x: " << this->x << std::endl;
        }

        template <typename U>
        CUDA_CALLABLE_METHOD multiindex(const multiindex<U>& other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }

        // // remove when vectorization is fully enabled
        // multiindex(size_t x, size_t y, size_t z)
        //   : x(x)
        //   , y(y)
        //   , z(z) {}

        CUDA_CALLABLE_METHOD inline double length() const {
            return sqrt(static_cast<double>(x * x + y * y + z * z));
        }

        CUDA_CALLABLE_METHOD inline bool compare(multiindex& other) {
            if (this->x == other.x && this->y == other.y && this->z == other.z) {
                return true;
            } else {
                return false;
            }
        }

        // set this multiindex to the next coarser level index
        CUDA_CALLABLE_METHOD void transform_coarse() {
            const T patch_size = static_cast<typename T::value_type>(INX);
            const T subtract = static_cast<typename T::value_type>(INX / 2);

            x = ((x + patch_size) >> 1) - subtract;
            y = ((y + patch_size) >> 1) - subtract;
            z = ((z + patch_size) >> 1) - subtract;
        }
    };

    CUDA_CALLABLE_METHOD inline multiindex<> flat_index_to_multiindex_not_padded(
        size_t flat_index) {
        size_t x = flat_index / (INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION);
        flat_index %= (INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION);
        size_t y = flat_index / INNER_CELLS_PER_DIRECTION;
        flat_index %= INNER_CELLS_PER_DIRECTION;
        size_t z = flat_index;
        multiindex<> m(x, y, z);
        return m;
    }

    CUDA_CALLABLE_METHOD inline multiindex<> flat_index_to_multiindex_padded(size_t flat_index) {
        size_t x = flat_index / (PADDED_STRIDE * PADDED_STRIDE);
        flat_index %= (PADDED_STRIDE * PADDED_STRIDE);
        size_t y = flat_index / PADDED_STRIDE;
        flat_index %= PADDED_STRIDE;
        size_t z = flat_index;
        multiindex<> m(x, y, z);
        return m;
    }

    template <typename T>
    CUDA_CALLABLE_METHOD inline T to_flat_index_padded(const multiindex<T>& m) {
        return m.x * PADDED_STRIDE * PADDED_STRIDE + m.y * PADDED_STRIDE + m.z;
    }

    /** are only valid for single cell! (no padding)
     * Note: for m2m_int_vector and integer
     * Note: returns uint32_t vector because of Vc limitation */
    template <typename T>
    CUDA_CALLABLE_METHOD inline T to_inner_flat_index_not_padded(const multiindex<T>& m) {
        return m.x * INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION +
            m.y * INNER_CELLS_PER_DIRECTION + m.z;
    }

    // This specialization is only required on cuda devices since T::value_type is not supported!
    template <>
    CUDA_CALLABLE_METHOD inline void multiindex<int32_t>::transform_coarse() {
        const int32_t patch_size = static_cast<int32_t>(INX);
        const int32_t subtract = static_cast<int32_t>(INX / 2);
        x = ((x + patch_size) >> 1) - subtract;
        y = ((y + patch_size) >> 1) - subtract;
        z = ((z + patch_size) >> 1) - subtract;
    }

    CUDA_CALLABLE_METHOD inline int32_t distance_squared_reciprocal(
        const multiindex<>& i, const multiindex<>& j) {
        return (sqr(i.x - j.x) + sqr(i.y - j.y) + sqr(i.z - j.z));
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
