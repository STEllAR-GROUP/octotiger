///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_CC1C4B22_3289_48FD_AE9B_BB88B3928D01)
#define TSB_CC1C4B22_3289_48FD_AE9B_BB88B3928D01

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <memory>
#include <type_traits> 

#include "vectorization_and_assumption_hints.hpp"

namespace tsb
{

template <typename T>
inline constexpr typename std::enable_if<
    std::is_unsigned<T>::value
  , bool
>::type is_power_of_2(T t)
{
    return bool((t != 0) && !(t & (t - 1)));
}

template <typename T>
inline constexpr typename std::enable_if<
    std::is_signed<T>::value
  , bool
>::type is_power_of_2(T t)
{
    return is_power_of_2(std::uintptr_t(t)); 
}

///////////////////////////////////////////////////////////////////////////////

inline void* align_ptr(
    void* ptr
  , std::ptrdiff_t alignment
  , std::ptrdiff_t size
  , std::ptrdiff_t space
    ) noexcept
{
    if (0 == alignment)
    {
        if (size > space)
            return nullptr;
        else
            return ptr;
    }

    auto const start = reinterpret_cast<std::uintptr_t>(ptr);

    auto aligned = start; 
    while (0 != (aligned % alignment))
        ++aligned;

    auto const diff = aligned - start;

    if ((size + diff) > space)
        return nullptr;
    else
    {
        space -= diff;
        return ptr = reinterpret_cast<void*>(aligned);
    }
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct aligned_array_ptr;

template <typename T>
inline aligned_array_ptr<T> make_aligned_array_posix_memalign(
    std::ptrdiff_t alignment
  , std::ptrdiff_t size
    ) noexcept
{
    BOOST_ASSUME(true == is_power_of_2(alignment));

    void* p = 0;
    int const r = ::posix_memalign(&p, alignment, size * sizeof(T));
    BOOST_ASSUME(0 == r);

    BOOST_ASSUME_ALIGNED(p, 2 * sizeof(void*));

    return aligned_array_ptr<T>(
        alignment, size, reinterpret_cast<T*>(p), reinterpret_cast<T*>(p)
    );
}

template <typename T>
inline aligned_array_ptr<T> make_aligned_array_overallocate(
    std::ptrdiff_t alignment
  , std::ptrdiff_t size
    ) noexcept
{
    auto const space = (size * sizeof(T)) + alignment;

    void* p = std::malloc(space);
    BOOST_ASSUME(p);

    void* ap = align_ptr(p, alignment, size * sizeof(T), space);
    BOOST_ASSUME(ap);

    BOOST_ASSUME_ALIGNED(ap, 2 * sizeof(void*));

    return aligned_array_ptr<T>(
        alignment, size, reinterpret_cast<T*>(p), reinterpret_cast<T*>(ap)
    );
}

template <typename T>
inline aligned_array_ptr<T> make_aligned_array(
    std::ptrdiff_t alignment
  , std::ptrdiff_t size
    ) noexcept
{
    BOOST_ASSUME(0 == (alignment % (2 * sizeof(void*))));

    if (is_power_of_2(alignment))
        return make_aligned_array_posix_memalign<T>(alignment, size);
    else
        return make_aligned_array_overallocate<T>(alignment, size);
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct free_deleter
{
    void operator()(T* p) const noexcept
    {
        std::free(p);
    }
};

template <typename T>
struct aligned_array_ptr
{
    static_assert(true == std::is_trivially_constructible<T>::value,
                  "T must be TriviallyConstructible.");
    static_assert(true == std::is_trivially_constructible<T>::value,
                  "T must be TriviallyDestructible.");

    using size_type  = std::ptrdiff_t;
    using value_type = T;
    using pointer    = T*;
    using reference  = T&;

  private:
    size_type alignment_;
    size_type size_;
    std::unique_ptr<T[], free_deleter<T> > true_ptr_;
    T* aligned_ptr_;

    explicit aligned_array_ptr(
        size_type alignment
      , size_type size
      , T* true_ptr
      , T* aligned_array_ptr
        )
      : alignment_(alignment)
      , size_(size)
      , true_ptr_(true_ptr, free_deleter<T>())
      , aligned_ptr_(aligned_array_ptr)
    {}

    friend aligned_array_ptr<T> make_aligned_array_posix_memalign<T>(
        size_type alignment
      , size_type size
        ) noexcept;

    friend aligned_array_ptr<T> make_aligned_array_overallocate<T>(
        size_type alignment
      , size_type size
        ) noexcept;

  public:
    constexpr aligned_array_ptr() noexcept
      : alignment_(0)
      , size_(0)
      , true_ptr_(nullptr, free_deleter<T>())
      , aligned_ptr_(nullptr)
    {}

    aligned_array_ptr(aligned_array_ptr&& other) noexcept
      : alignment_(std::move(other.alignment_))
      , size_(std::move(other.size_))
      , true_ptr_(std::move(other.true_ptr_))
      , aligned_ptr_(std::move(other.aligned_ptr_))
    {}

    aligned_array_ptr& operator=(aligned_array_ptr&& other) noexcept
    {
        std::swap(alignment_,   other.alignment_);
        std::swap(size_,        other.size_);
        std::swap(true_ptr_,    other.true_ptr_);
        std::swap(aligned_ptr_, other.aligned_ptr_);
    }

    reference operator[](size_type i) const noexcept
    {
        return aligned_ptr_[i];
    }

    reference operator*() const noexcept
    {
        return *aligned_ptr_;
    }

    pointer operator->() const noexcept
    {
        return aligned_ptr_;
    }

    pointer get() const noexcept
    {
        return aligned_ptr_;
    }

    size_type constexpr alignment() const noexcept
    {
        return alignment_;
    }

    size_type constexpr size() const noexcept
    {
        return size_;
    }
};

} // tsb

#endif // TSB_CC1C4B22_3289_48FD_AE9B_BB88B3928D01


