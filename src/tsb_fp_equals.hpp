///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_A983ED3B_3C3B_42AF_8057_C5E91CA3242B)
#define TSB_A983ED3B_3C3B_42AF_8057_C5E91CA3242B

#include <type_traits>
#include <limits>

namespace tsb
{

template <typename T>
constexpr bool fp_equals(
    T x, T y, T epsilon = std::numeric_limits<T>::epsilon()
    ) noexcept
{
    static_assert(
        std::is_floating_point<T>::value
      , "T must be a floating point type."
    );
    return ( ((x + epsilon >= y) && (x - epsilon <= y))
           ? true
           : false);
}

} // tsb

#endif // TSB_A983ED3B_3C3B_42AF_8057_C5E91CA3242B

