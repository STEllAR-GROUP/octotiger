#pragma once

#include "../cuda_util/cuda_helper.hpp"
#include "defs.hpp"

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
