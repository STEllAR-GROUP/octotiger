//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/common_kernel/multiindex.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/real.hpp"

#include <cmath>

namespace octotiger {
namespace fmm {
        namespace detail {

            // calculates 1/distance between i and j
            inline real reciprocal_distance(const integer i0, const integer i1, const integer i2,
                const integer j0, const integer j1, const integer j2) {
                real tmp = (sqr(i0 - j0) + sqr(i1 - j1) + sqr(i2 - j2));
                // protect against sqrt(0)
                if (tmp > 0.0) {    // TODO: remove this branch
                    return 1.0 /
                        (std::sqrt(tmp));    // TODO: use squared theta instead for comparison
                } else {
                    return 1.0e+10;    // dummy value
                }
            }

            // calculates 1/distance between i and j
            template<class T>
            inline T distance_squared_reciprocal(
                const multiindex<T>& i, const multiindex<T>& j) {
                return (sqr(i.x - j.x) + sqr(i.y - j.y) + sqr(i.z - j.z));
            }
            template <class T>
            inline T distance_squared_reciprocal(
                const T i0, const T i1, const T i2, const T j0, const T j1, const T j2) {
            return (sqr(i0 - j0) + sqr(i1 - j1) + sqr(i2 - j2));
            }
        }    // namespace detail
}    // namespace fmm
}    // namespace octotiger
