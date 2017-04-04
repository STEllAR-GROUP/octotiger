///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(OCTOTIGER_63801C03_BBA2_4F5E_A30C_4387A73F8AE2)
#define OCTOTIGER_63801C03_BBA2_4F5E_A30C_4387A73F8AE2

#include <type_traits>
#include <limits>

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

#endif // OCTOTIGER_63801C03_BBA2_4F5E_A30C_4387A73F8AE2

