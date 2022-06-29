//  Copyright (c) 2022 Srinivas Yadav
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#pragma once

#include <experimental/simd>

namespace SIMD_NAMESPACE = std::experimental;
using host_simd_t = std::experimental::native_simd<double>;
using host_simd_mask_t = std::experimental::native_simd_mask<double>;

using device_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using device_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;

namespace std{
    namespace experimental {
        template <typename Vector, typename Mask>
        inline Vector choose(const Mask& m, const Vector& tr, const Vector& fl)
        {
            Vector res;
            where(m, res) = tr;
            where(!m, res) = fl;
            return res;
        }
    }
}
