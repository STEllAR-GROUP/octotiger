/*
 * SubArray.hpp
 *
 *  Created on: Feb 13, 2026
 *      Author: dmarce1
 */

#ifndef INCLUDE_BOX_HPP_
#define INCLUDE_BOX_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

template <int64_t>
struct BoxIterator;

template <int64_t D>
class Box {
    using Int = int64_t;
    std::array<Int, D> begin_{};
    std::array<Int, D> end_{};

public:
    constexpr Box() = default;
    constexpr Box(Box const&) = default;
    constexpr Box(Box&&) = default;
    template <std::integral T1, std::integral T2>
    constexpr Box(std::array<T1, D> const& b, std::array<T2, D> const& e) {
        std::copy_n(b.begin(), D, begin_.begin());
        std::copy_n(e.begin(), D, end_.begin());
    }
    template <std::integral T>
    constexpr Box(std::array<T, D> const& e) {
        begin_.fill(0);
        std::copy_n(e.begin(), D, end_.begin());
    }
    constexpr Box(Int e) {
        begin_.fill(0);
        end_.fill(e);
    }
    constexpr Box(Int b, Int e) {
        begin_.fill(b);
        end_.fill(e);
    }
    Box& operator=(Box const&) = default;
    Box& operator=(Box&&) = default;
    auto const& begin() const {
        return begin_;
    }
    auto const& end() const {
        return end_;
    }
    constexpr auto span() const {
        std::array<Int, D> s;
        for (Int d = 0; d < D; ++d) {
            s[d] = span(d);
        }
        return s;
    }
    template <std::integral T>
    bool contains(std::array<T, D> const& point) const {
        for (Int d = 0; d < D; ++d) {
            if (point[d] < begin_[d]) return false;
            if (point[d] >= end_[d]) return false;
        }
        return true;
    }
    bool contains(Box const& other) const {
        if (!contains(other.begin_)) return false;
        if (!contains(other.end_)) return false;
        return true;
    }
    bool intersects(Box const& A) const {
        for (Int d = 0; d < D; ++d) {
            if (begin_[d] >= A.end_[d]) return false;
            if (end_[d] <= A.begin_[d]) return false;
        }
        return true;
    }
    bool operator==(Box const& other) const {
        if (begin_ != other.begin_) return false;
        if (end_ != other.end_) return false;
        return true;
    }
    bool operator!=(Box const& other) const {
        return !(*this == other);
    }
    constexpr Int span(Int d) const {
        return std::max(end_[d] - begin_[d], Int(0));
    }
    constexpr Int volume() const {
        Int v = 1;
        for (Int d = 0; d < D; ++d) {
            v *= span(d);
        }
        return v;
    }
    [[nodiscard]] constexpr Box expand(Int n) const {
        Box b;
        for (Int d = 0; d < D; ++d) {
            b.begin_[d] -= n;
            b.end_[d] += n;
        }
        return b;
    }
    [[nodiscard]] constexpr Box scale(Int n) const {
        Box b;
        for (Int d = 0; d < D; ++d) {
            b.begin_[d] *= n;
            b.end_[d] *= n;
        }
        return b;
    }
    [[nodiscard]] constexpr Box shrink(Int n) const {
        return expand(-n);
    }
    template <std::integral T>
    [[nodiscard]] constexpr Box shift(std::array<T, D> const& n) const {
        Box b;
        for (Int d = 0; d < D; ++d) {
            b.begin_[d] = begin_[d] + n[d];
            b.end_[d] = end_[d] + n[d];
        }
        return b;
    }
    [[nodiscard]] constexpr Box shift(Int d, Int n) const {
        std::array<Int, D> s;
        for (Int i = 0; i < D; ++i) {
            s[i] = (i == d) ? n : 0;
        }
        return shift(s);
    }
    [[nodiscard]] constexpr Box slice(Int d, Int n) const {
        auto result = *this;
        result.begin_[d] = n;
        result.end_[d] = n + 1;
        return result;
    }
    template <std::integral T>
    [[nodiscard]] constexpr auto flatten(std::array<T, D> const& I) const {
        auto i = I[0] - begin_[0];
        for (int d = 1; d < D; d++) {
            i = span(d) * i + (I[d] - begin_[d]);
        }
        return i;
    }
    void serialize(auto& arc, unsigned) {
        arc & begin_;
        arc & end_;
    }
    [[nodiscard]] friend constexpr Box bounding(Box const& A, Box const& B) {
        Box I;
        for (Int d = 0; d < D; ++d) {
            I.begin_[d] = std::min(A.begin_[d], B.begin_[d]);
            I.end_[d] = std::max(A.end_[d], B.end_[d]);
        }
        return I;
    }
    [[nodiscard]] friend constexpr Box intersection(Box const& A, Box const& B) {
        Box I;
        for (Int d = 0; d < D; ++d) {
            I.begin_[d] = std::max(A.begin_[d], B.begin_[d]);
            I.end_[d] = std::min(A.end_[d], B.end_[d]);
        }
        return I;
    }
    friend BoxIterator<D>;
};

template <int64_t D>
struct BoxIterator : std::array<int64_t, D> {
    using base_type = std::array<int64_t, D>;
    BoxIterator(Box<D> const& inner, Box<D> const& outer)
      : base_type(inner.begin())
      , done(false)
      , inner_(inner)
      , outer_(outer)
      , idx_(static_cast<base_type&>(*this)) {}
    BoxIterator(Box<D> const& box)
      : base_type(box.begin())
      , done(false)
      , inner_(box)
      , outer_(box)
      , idx_(static_cast<base_type&>(*this)) {}
    BoxIterator(BoxIterator const& other)
      : base_type(other.idx_)
      , done(other.done)
      , inner_(other.inner_)
      , outer_(other.outer_)
      , idx_(static_cast<base_type&>(*this)) {}
    BoxIterator& operator++() {
        for (int64_t d = D - 1; d >= 0; d--) {
            if (++idx_[d] < inner_.end_[d]) return *this;
            idx_[d] = inner_.begin_[d];
        }
        done = true;
        return *this;
    }
    BoxIterator& operator+=(int n) {
        for (int64_t d = D - 1; d >= 0; d--) {
            idx_[d] += n;
            if (idx_[d] < inner_.end_[d]) return *this;
            idx_[d] = inner_.begin_[d];
        }
        done = true;
        return *this;
    }
    bool end() const {
        return done;
    }
    void reset() {
        done = false;
        idx_ = inner_.begin();
    }
    int64_t operator[](std::integral auto i) const {
        return idx_[i];
    }
    operator int64_t() const {
        return outer_.flatten(idx_);
    }

private:
    bool done;
    Box<D> const& inner_;
    Box<D> const& outer_;
    std::array<int64_t, D>& idx_;
};

#endif /* INCLUDE_BOX_HPP_ */
