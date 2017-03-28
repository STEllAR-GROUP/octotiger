//  Copyright (c) 2017 Bryce Adelstein Lelbach 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt) 
#ifndef HPX_UTIL_ARRAY_HPP
#define HPX_UTIL_ARRAY_HPP

#include <cstdint>
#include <iterator>
#include <algorithm>
#include <type_traits>

#include <hpx/runtime/serialization/array.hpp>

namespace hpx { namespace util
{

// C++17, 23.3.7 [array]

template <typename T, std::size_t N >
struct array
{
    ///////////////////////////////////////////////////////////////////////////
    // Types

    using value_type             = T;
    using pointer                = T*;
    using const_pointer          = T const*;
    using reference              = T&;
    using const_reference        = T const&;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using iterator               = T*;
    using const_iterator         = T const*;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    ///////////////////////////////////////////////////////////////////////////

    // Aggregate type: no explicit construct/copy/destroy.

    void fill(T const& u)
    {
        std::fill(begin(), end(), u);
    }

    // TODO void swap(array& other) noexcept(std::is_nothrow_swappable_v<T>)
    // ^ is_nothrow_swappable_v is C++17 only
    void swap(array& other) noexcept
    {
        std::swap(A, other.A);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Iterators.

    constexpr iterator       begin()         noexcept { return A; }
    constexpr const_iterator begin()   const noexcept { return A; }
    constexpr iterator       end()           noexcept { return A + size(); }
    constexpr const_iterator end()     const noexcept { return A + size(); }

    constexpr iterator       rbegin()        noexcept { return {end()}; }
    constexpr const_iterator rbegin()  const noexcept { return {end()}; }
    constexpr iterator       rend()          noexcept { return {begin()}; }
    constexpr const_iterator rend()    const noexcept { return {begin()}; }

    constexpr const_iterator cbegin()  const noexcept { return begin(); }
    constexpr const_iterator cend()    const noexcept { return end(); }
    constexpr const_iterator crbegin() const noexcept { return rbegin(); }
    constexpr const_iterator crend()   const noexcept { return rend(); }

    ///////////////////////////////////////////////////////////////////////////
    // Capacity.

    constexpr bool      empty()    const noexcept { return 0 == size(); }
    constexpr size_type size()     const noexcept { return N; }
    constexpr size_type max_size() const noexcept { return size(); }

    ///////////////////////////////////////////////////////////////////////////
    // Element access.

    // noexcept as a conforming extension.
    constexpr reference       operator[](size_type i)       noexcept { return A[i]; }
    constexpr const_reference operator[](size_type i) const noexcept { return A[i]; }

    // TODO constexpr reference       at(size_type n);
    // TODO constexpr const_reference at(size_type n) const;

    constexpr reference       front()       noexcept { return A[0]; }
    constexpr const_reference front() const noexcept { return A[0]; }
    constexpr reference       back()        noexcept { return A[size()-1]; }
    constexpr const_reference back()  const noexcept { return A[size()-1]; }

    constexpr pointer       data()       noexcept { return A; }
    constexpr const_pointer data() const noexcept { return A; }

    ///////////////////////////////////////////////////////////////////////////
    // Serialization support (conforming extension).

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & hpx::serialization::make_array(A, N);
    }

    ///////////////////////////////////////////////////////////////////////////

    // Notionally private.
    value_type A[N];
};

}} // hpx::util

#endif // HPX_UTIL_ARRAY_HPP

