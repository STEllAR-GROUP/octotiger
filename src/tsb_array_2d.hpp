///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_56BBC442_B51C_4A22_AA8B_AE4F20B7E7C7)
#define TSB_56BBC442_B51C_4A22_AA8B_AE4F20B7E7C7

#include <cstddef>

#include "tsb_placeholder.hpp"
#include "tsb_make_aligned_array.hpp"

namespace tsb
{

struct layout_left_2d
{
    using size_type = std::ptrdiff_t;

  private:
    size_type nx_, ny_;
    size_type px_, py_;

  public:
    constexpr layout_left_2d() noexcept
      : nx_(0), ny_(0), px_(0), py_(0)
    {}

    constexpr layout_left_2d(
        size_type nx, size_type ny
        ) noexcept
      : nx_(nx), ny_(ny), px_(0), py_(0)
    {}

    constexpr layout_left_2d(
        size_type nx, size_type ny
      , size_type px, size_type py
        ) noexcept
      : nx_(nx), ny_(ny), px_(px), py_(py)
    {}

    constexpr size_type operator()(
        size_type i, size_type j
        ) const noexcept
    {
        return stride_x() * i + stride_y() * j;
    }
    constexpr size_type operator()(
        placeholder, size_type j
        ) const noexcept
    {
        return stride_y() * j;
    }
    constexpr size_type operator()(
        size_type i, placeholder
        ) const noexcept
    {
        return stride_x() * i;
    }

    constexpr size_type stride_x() const noexcept
    {
        return 1;
    }
    constexpr size_type stride_y() const noexcept
    {
        return (nx_ + px_);
    }

    constexpr size_type nx() const noexcept
    {
        return nx_;
    }
    constexpr size_type ny() const noexcept
    {
        return ny_;
    }

    constexpr size_type span() const noexcept
    {
        return (nx_ * ny_) + stride_x() * px_  + stride_y() * py_;  
    }
};

struct layout_right_2d
{
    using size_type = std::ptrdiff_t;

  private:
    size_type nx_, ny_;
    size_type px_, py_;

  public:
    constexpr layout_right_2d() noexcept
      : nx_(0), ny_(0), px_(0), py_(0)
    {}

    constexpr layout_right_2d(
        size_type nx, size_type ny
        ) noexcept
      : nx_(nx), ny_(ny), px_(0), py_(0)
    {}

    constexpr layout_right_2d(
        size_type nx, size_type ny
      , size_type px, size_type py
        ) noexcept
      : nx_(nx), ny_(ny), px_(px), py_(py)
    {}

    constexpr size_type operator()(
        size_type i, size_type j
        ) const noexcept
    {
        return stride_x() * i + stride_y() * j;
    }
    constexpr size_type operator()(
        placeholder, size_type j
        ) const noexcept
    {
        return stride_y() * j;
    }
    constexpr size_type operator()(
        size_type i, placeholder
        ) const noexcept
    {
        return stride_x() * i;
    }

    constexpr size_type stride_x() const noexcept
    {
        return (ny_ + py_); 
    }
    constexpr size_type stride_y() const noexcept
    {
        return 1; 
    }

    constexpr size_type nx() const noexcept
    {
        return nx_;
    }
    constexpr size_type ny() const noexcept
    {
        return ny_;
    }

    constexpr size_type span() const noexcept
    {
        return (nx_ * ny_) + stride_x() * px_ + stride_y() * py_;  
    }
};

template <typename T, typename Layout>
struct array_2d
{
    using layout = Layout;

    using size_type       = typename layout::size_type;
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = T const*;
    using reference       = T&;
    using const_reference = T const&;

  private:
    layout layout_;
    decltype(make_aligned_array<T>(2 * sizeof(void*), 0)) data_;

  public:
    constexpr array_2d() noexcept : layout_(), data_() {}

    array_2d(
        size_type alignment
      , size_type nx, size_type ny
        ) noexcept
      : layout_(), data_()
    {
        resize(alignment, nx, ny);
    }

    array_2d(
        size_type alignment
      , size_type nx, size_type ny
      , size_type px, size_type py
        ) noexcept
      : layout_(), data_()
    {
        resize(alignment, nx, ny, px, py);
    }

    void resize(
        size_type alignment
      , size_type nx, size_type ny
        ) noexcept
    {
        layout_ = layout(nx, ny);
        data_   = make_aligned_array<T>(alignment, layout_.span());
    }

    void resize(
        size_type alignment
      , size_type nx, size_type ny
      , size_type px, size_type py
        ) noexcept
    {
        layout_ = layout(nx, ny, px, py);
        data_   = make_aligned_array<T>(alignment, layout_.span());
    }

    const_pointer data() const noexcept
    {
        return data_.get();
    }
    pointer data() noexcept
    {
        return data_.get();
    }

    const_reference operator()(
        size_type i, size_type j
        ) const noexcept
    {
        return data_[layout_(i, j)];
    }
    reference operator()(
        size_type i, size_type j
        ) noexcept
    {
        return data_[layout_(i, j)];
    }

    const_pointer operator()(
        placeholder p, size_type j
        ) const noexcept
    {
        return &data_[layout_(p, j)];
    }
    pointer operator()(
        placeholder p, size_type j
        ) noexcept
    {
        return &data_[layout_(p, j)];
    }

    const_pointer operator()(
        size_type i, placeholder p
        ) const noexcept
    {
        return &data_[layout_(i, p)];
    }
    pointer operator()(
        size_type i, placeholder p
        ) noexcept
    {
        return &data_[layout_(i, p)];
    }

    constexpr size_type stride_x() const noexcept
    {
        return layout_.stride_x();
    }
    constexpr size_type stride_y() const noexcept
    {
        return layout_.stride_y();
    }

    constexpr size_type nx() const noexcept
    {
        return layout_.nx();
    }
    constexpr size_type ny() const noexcept
    {
        return layout_.ny();
    }
    constexpr size_type span() const noexcept
    {
        return layout_.span();
    }
};

} // tsb

#endif // TSB_56BBC442_B51C_4A22_AA8B_AE4F20B7E7C7

