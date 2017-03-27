///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_6594A0DA_8E9E_4B31_A32F_38EFDF13289E)
#define TSB_6594A0DA_8E9E_4B31_A32F_38EFDF13289E

#include <cstddef>

#include "tsb_placeholder.hpp"
#include "tsb_make_aligned_array.hpp"

namespace tsb
{

struct placeholder {};

constexpr placeholder _ {}; // This does not make me a bad person.

struct layout_left
{
    using size_type = std::ptrdiff_t;

  private:
    size_type nx_, ny_, nz_;
    size_type px_, py_, pz_;

  public:
    constexpr layout_left() noexcept
      : nx_(0), ny_(0), nz_(0), px_(0), py_(0), pz_(0)
    {}

    constexpr layout_left(
        size_type nx, size_type ny, size_type nz
        ) noexcept
      : nx_(nx), ny_(ny), nz_(nz), px_(0), py_(0), pz_(0)
    {}

    constexpr layout_left(
        size_type nx, size_type ny, size_type nz
      , size_type px, size_type py, size_type pz
        ) noexcept
      : nx_(nx), ny_(ny), nz_(nz), px_(px), py_(py), pz_(pz)
    {}

    constexpr size_type operator()(
        size_type i, size_type j, size_type k
        ) const noexcept
    {
        return stride_x() * i + stride_y() * j + stride_z() * k;
    }
    constexpr size_type operator()(
        placeholder, size_type j, size_type k
        ) const noexcept
    {
        return stride_y() * j + stride_z() * k;
    }
    constexpr size_type operator()(
        size_type i, placeholder, size_type k
        ) const noexcept
    {
        return stride_x() * i + stride_z() * k;
    }
    constexpr size_type operator()(
        size_type i, size_type j, placeholder
        ) const noexcept
    {
        return stride_x() * i + stride_y() * j;
    }    

    constexpr size_type stride_x() const noexcept
    {
        return 1;
    }
    constexpr size_type stride_y() const noexcept
    {
        return (nx_ + px_);
    }
    constexpr size_type stride_z() const noexcept
    {
        return (((nx_ + px_) * ny_) + py_);
    }

    constexpr size_type nx() const noexcept
    {
        return nx_;
    }
    constexpr size_type ny() const noexcept
    {
        return ny_;
    }
    constexpr size_type nz() const noexcept
    {
        return nz_;
    }

    constexpr size_type span() const noexcept
    {
        return (nx_ * ny_ * nz_)
             + stride_x() * px_  
             + stride_y() * py_ 
             + stride_z() * pz_
            ;  
    }
};

struct layout_right
{
    using size_type = std::ptrdiff_t;

  private:
    size_type nx_, ny_, nz_;
    size_type px_, py_, pz_;

  public:
    constexpr layout_right() noexcept
      : nx_(0), ny_(0), nz_(0), px_(0), py_(0), pz_(0)
    {}

    constexpr layout_right(
        size_type nx, size_type ny, size_type nz
        ) noexcept
      : nx_(nx), ny_(ny), nz_(nz), px_(0), py_(0), pz_(0)
    {}

    constexpr layout_right(
        size_type nx, size_type ny, size_type nz
      , size_type px, size_type py, size_type pz
        ) noexcept
      : nx_(nx), ny_(ny), nz_(nz), px_(px), py_(py), pz_(pz)
    {}

    constexpr size_type operator()(
        size_type i, size_type j, size_type k
        ) const noexcept
    {
        return stride_x() * i + stride_y() * j + stride_z() * k;
    }
    constexpr size_type operator()(
        placeholder, size_type j, size_type k
        ) const noexcept
    {
        return stride_y() * j + stride_z() * k;
    }
    constexpr size_type operator()(
        size_type i, placeholder, size_type k
        ) const noexcept
    {
        return stride_x() * i + stride_z() * k;
    }
    constexpr size_type operator()(
        size_type i, size_type j, placeholder
        ) const noexcept
    {
        return stride_x() * i + stride_y() * j;
    }    

    constexpr size_type stride_x() const noexcept
    {
        return (((nz_ + pz_) * ny_) + py_);
    }
    constexpr size_type stride_y() const noexcept
    {
        return (nz_ + pz_);
    }
    constexpr size_type stride_z() const noexcept
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
    constexpr size_type nz() const noexcept
    {
        return nz_;
    }

    constexpr size_type span() const noexcept
    {
        return (nx_ * ny_ * nz_)
             + stride_x() * px_  
             + stride_y() * py_ 
             + stride_z() * pz_
            ;  
    }
};

template <typename T, typename Layout>
struct array3d
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
    constexpr array3d() noexcept : layout_(), data_() {}

    array3d(
        size_type alignment
      , size_type nx, size_type ny, size_type nz
        ) noexcept
      : layout_(), data_()
    {
        resize(alignment, nx, ny, nz);
    }

    array3d(
        size_type alignment
      , size_type nx, size_type ny, size_type nz
      , size_type px, size_type py, size_type pz
        ) noexcept
      : layout_(), data_()
    {
        resize(alignment, nx, ny, nz, px, py, pz);
    }

    void resize(
        size_type alignment
      , size_type nx, size_type ny, size_type nz
        ) noexcept
    {
        layout_ = layout(nx, ny, nz);
        data_   = make_aligned_array<T>(alignment, layout_.span());
    }

    void resize(
        size_type alignment
      , size_type nx, size_type ny, size_type nz
      , size_type px, size_type py, size_type pz
        ) noexcept
    {
        layout_ = layout(nx, ny, nz, px, py, pz);
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
        size_type i, size_type j, size_type k
        ) const noexcept
    {
        return data_[layout_(i, j, k)];
    }
    reference operator()(
        size_type i, size_type j, size_type k
        ) noexcept
    {
        return data_[layout_(i, j, k)];
    }

    const_pointer operator()(
        placeholder p, size_type j, size_type k
        ) const noexcept
    {
        return &data_[layout_(p, j, k)];
    }
    pointer operator()(
        placeholder p, size_type j, size_type k
        ) noexcept
    {
        return &data_[layout_(p, j, k)];
    }

    const_pointer operator()(
        size_type i, placeholder p, size_type k
        ) const noexcept
    {
        return &data_[layout_(i, p, k)];
    }
    pointer operator()(
        size_type i, placeholder p, size_type k
        ) noexcept
    {
        return &data_[layout_(i, p, k)];
    }

    const_pointer operator()(
        size_type i, size_type j, placeholder p
        ) const noexcept
    {
        return &data_[layout_(i, j, p)];
    }
    pointer operator()(
        size_type i, size_type j, placeholder p
        ) noexcept
    {
        return &data_[layout_(i, j, p)];
    }

    constexpr size_type stride_x() const noexcept
    {
        return layout_.stride_x();
    }
    constexpr size_type stride_y() const noexcept
    {
        return layout_.stride_y();
    }
    constexpr size_type stride_z() const noexcept
    {
        return layout_.stride_z();
    }

    constexpr size_type nx() const noexcept
    {
        return layout_.nx();
    }
    constexpr size_type ny() const noexcept
    {
        return layout_.ny();
    }
    constexpr size_type nz() const noexcept
    {
        return layout_.nz();
    }

    constexpr size_type span() const noexcept
    {
        return layout_.span();
    }
};

} // tsb

#endif // TSB_6594A0DA_8E9E_4B31_A32F_38EFDF13289E

