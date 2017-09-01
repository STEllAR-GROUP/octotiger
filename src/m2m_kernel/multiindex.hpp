#pragma once

#include "defs.hpp"

#include <cmath>
#include <iostream>

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

        multiindex(T x, T y, T z)
          : x(x)
          , y(y)
          , z(z) {
            // std::cout << "x arg: " << x << std::endl;
            // std::cout << "this->x: " << this->x << std::endl;
        }

        template <typename U>
        multiindex(const multiindex<U>& other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }

        // // remove when vectorization is fully enabled
        // multiindex(size_t x, size_t y, size_t z)
        //   : x(x)
        //   , y(y)
        //   , z(z) {}

        inline double length() const {
            return sqrt(static_cast<double>(x * x + y * y + z * z));
        }

        inline bool compare(multiindex& other) {
            if (this->x == other.x && this->y == other.y && this->z == other.z) {
                return true;
            } else {
                return false;
            }
        }

        // set this multiindex to the next coarser level index
        void transform_coarse() {
            const T patch_size = static_cast<typename T::value_type>(INX);
            const T subtract = static_cast<typename T::value_type>(INX / 2);

            x = ((x + patch_size) >> 1) - subtract;
            y = ((y + patch_size) >> 1) - subtract;
            z = ((z + patch_size) >> 1) - subtract;
        }
    };

}    // namespace fmm
}    // namespace octotiger

template <typename T>
std::ostream& operator<<(std::ostream& os, const octotiger::fmm::multiindex<T>& m) {
    return os << m.x << ", " << m.y << ", " << m.z;
}
