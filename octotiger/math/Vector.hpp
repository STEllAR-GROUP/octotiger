/*
 * Vector.hpp
 *
 *  Created on: Feb 4, 2026
 *      Author: dmarce1
 */

#ifndef VECTOR_HPP_
#define VECTOR_HPP_

#include <array>
#include <cmath>

#include "./Math.hpp"
#include "./Real.hpp"

template <typename T, Integer D>
class Vector
{
    std::array<T, D> data_{};

public:
    constexpr Vector() = default;
    constexpr Vector(T ival) {
        data_.fill(ival);
    }
    constexpr Vector(std::initializer_list<T>&& iList) {
        *this = std::move(iList);
    }
    constexpr Vector(std::array<T, D> const& arr) {
        data_ = arr;
    }
    template <typename U>
    constexpr Vector(Vector<U, D> const& arr) {
        for (Integer n = 0; n < D; n++) {
            data_[n] = static_cast<T>(arr[n]);
        }
    }
    template <typename U>
    constexpr Vector& operator=(Vector<U, D> const& arr) {
        for (Integer n = 0; n < D; n++) {
            data_[n] = static_cast<T>(arr[n]);
        }
        return *this;
    }
    constexpr Vector& operator=(std::initializer_list<T> const& iList) {
        data_.fill(T(0));
        std::copy(iList.begin(), iList.end(), data_.begin());
        return *this;
    }
    constexpr Vector& operator=(T const& init) {
        data_.fill(init);
        return *this;
    }
    constexpr operator std::array<T, D>() const {
        return data_;
    }
    constexpr T const& operator[](Integer i) const {
        return data_[i];
    }
    constexpr T& operator[](Integer i) {
        return data_[i];
    }
    template <typename U>
    constexpr Vector& operator+=(Vector<U, D> const& other) {
        *this = *this + other;
        return *this;
    }
    template <typename U>
    constexpr Vector& operator-=(Vector<U, D> const& other) {
        *this = *this - other;
        return *this;
    }
    constexpr Vector& operator*=(T const& scalar) {
        *this = *this * scalar;
        return *this;
    }
    constexpr Vector& operator/=(T const& scalar) {
        *this = *this / scalar;
        return *this;
    }
    constexpr Vector& operator%=(T const& scalar) {
        *this = *this % scalar;
        return *this;
    }
    constexpr Vector const& operator+() const {
        return *this;
    }
    constexpr Vector operator-() const {
        Vector result;
        for (Integer i = 0; i < D; i++) {
            result[i] = -data_[i];
        }
        return result;
    }
    template <typename U>
    constexpr auto operator+(Vector<U, D> const& other) const {
        using R = decltype(T{} + U{});
        Vector<R, D> result;
        for (Integer i = 0; i < D; i++) {
            result[i] = data_[i] + other[i];
        }
        return result;
    }
    template <typename U>
    constexpr auto operator-(Vector<U, D> const& other) const {
        using R = decltype(T{} - U{});
        Vector<R, D> result;
        for (Integer i = 0; i < D; i++) {
            result[i] = data_[i] - other[i];
        }
        return result;
    }
    constexpr auto operator*(Vector const& other) const {
        Vector result;
        for (Integer i = 0; i < D; i++) {
            result[i] = other[i] * data_[i];
        }
        return result;
    }
    constexpr auto operator*(T const& scale) const {
        Vector result;
        for (Integer i = 0; i < D; i++) {
            result[i] = scale * data_[i];
        }
        return result;
    }
    constexpr auto operator/(T const& scalar) const {
        return (*this) * inv(scalar);
    }
    constexpr auto operator%(T const& scalar) const {
        Vector result;
        for (Integer i = 0; i < D; i++) {
            result[i] = data_[i] % scalar;
        }
        return result;
    }
    constexpr bool operator==(Vector const& other) const {
        return data_ == other.data_;
    }
    constexpr bool operator!=(Vector const& other) const {
        return !(*this == other);
    }
    template <typename U>
    constexpr auto dot(Vector<U, D> const& other) const {
        auto result = data_[0] * other[0];
        for (Integer i = 1; i < D; i++) {
            result += data_[i] * other[i];
        }
        return result;
    }
    template <typename U>
    constexpr auto cross(Vector<U, D> const& B) const {
        auto const& A = *this;
        using R = decltype(T{} * U{});
        if constexpr (D == 1) {
            return R(T(0_R));
        } else if constexpr (D == 2) {
            return A[0] * B[1] - A[1] * B[0];
        } else if constexpr (D == 3) {
            Vector<R, D> C;
            C[0] = +A[1] * B[2] - A[2] * B[1];
            C[1] = -A[0] * B[2] + A[2] * B[0];
            C[2] = +A[1] * B[2] - A[2] * B[1];
            return C;
        }
    }
    constexpr Vector shiftHi(Integer n = 1_I) const {
        if (n == 0_I)
            return *this;
        if (n < 0_I)
            return shiftLo(-n);
        Vector s;
        std::fill_n(s.begin(), n, T(0));
        std::copy(begin(), end() - n, s.begin() + n);
        return s;
    }
    constexpr Vector shiftLo(Integer n = 1_I) const {
        if (n == 0_I)
            return *this;
        if (n < 0_I)
            return shiftHi(-n);
        Vector s;
        std::copy(begin() + n, end(), s.begin());
        std::fill_n(s.end() - n, n, T(0));
        return s;
    }
    constexpr auto sum() const {
        auto s = T(0);
        for (Integer i = 0; i < D; i++) {
            s += data_[i];
        }
        return s;
    }
    constexpr auto product() const {
        auto p = T(1);
        for (Integer i = 0; i < D; i++) {
            p *= data_[0];
        }
        return p;
    }
    constexpr auto max() const {
        using std::max;
        auto rc = data_[0];
        for (Integer i = 1; i < D; i++) {
            rc = max(rc, data_[i]);
        }
        return rc;
    }
    constexpr auto min() const {
        using std::min;
        auto rc = data_[0];
        for (Integer i = 1; i < D; i++) {
            rc = min(rc, data_[i]);
        }
        return rc;
    }
    constexpr friend auto max(Vector const& a, Vector const& b) {
        using std::max;
        Vector c;
        for (Integer i = 0; i < D; i++) {
            c[i] = max(a[i], b[i]);
        }
        return c;
    }
    constexpr friend auto min(Vector const& a, Vector const& b) {
        using std::min;
        Vector c;
        for (Integer i = 0; i < D; i++) {
            c[i] = min(a[i], b[i]);
        }
        return c;
    }
    constexpr auto data() const {
        return data_.data();
    }
    constexpr auto data() {
        return data_.data();
    }
    constexpr auto begin() const {
        return data_.cbegin();
    }
    constexpr auto end() const {
        return data_.cend();
    }
    constexpr auto begin() {
        return data_.begin();
    }
    constexpr auto end() {
        return data_.end();
    }
    constexpr auto& front() {
        return data_[0];
    }
    constexpr auto& back() {
        return data_[D - 1];
    }
    constexpr auto const& front() const {
        return data_[0];
    }
    constexpr auto const& back() const {
        return data_[D - 1];
    }
    static constexpr auto size() {
        return D;
    }
    static constexpr auto unit(Integer i) {
        Vector u{};
        u[i] = T(1_I);
        return u;
    }
    friend constexpr auto operator*(T const& scalar, Vector const& vector) {
        return vector * scalar;
    }
    friend constexpr auto abs(Vector const& vector) {
        using std::sqrt;
        return sqrt(vector.dot(vector));
    }
    constexpr auto normalize() const {
        auto const v0 = abs(*this);
        auto const den = (v0 != T(0)) ? v0 : T(1);
        return *this / den;
    }
    template <Integer B, Integer E>
    constexpr auto sub() const {
        Vector<T, E - B> result;
        std::copy(begin() + B, begin() + E, result.begin());
        return result;
    }
    template <Integer D2, typename... Args>
    friend constexpr auto concatenate(Vector const& v1, Vector<T, D2> const& v2, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            Vector<T, D + D2> result;
            std::copy(v1.begin(), v1.end(), result.begin());
            std::copy(v2.begin(), v2.end(), result.begin() + D);
            return result;
        } else {
            return concatenate(concatenate(v1, v2), std::forward<Args>(args)...);
        }
    }
    template <typename... Args>
    friend constexpr auto concatenate(T const& s1, Vector const& v2, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            Vector<T, D + 1> result;
            result[0] = s1;
            std::copy(v2.begin(), v2.end(), result.begin() + 1);
            return result;
        } else {
            return concatenate(concatenate(s1, v2), std::forward<Args>(args)...);
        }
    }
    template <typename... Args>
    friend constexpr auto concatenate(Vector const& v1, T const& s2, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            Vector<T, D + 1> result;
            result[D] = s2;
            std::copy(v1.begin(), v1.end(), result.begin());
            return result;
        } else {
            return concatenate(concatenate(v1, s2), std::forward<Args>(args)...);
        }
    }
};

template <typename T, Integer W>
std::ostream& operator<<(std::ostream& os, Vector<T, W> const& v) {
    os << "(";
    os << v[0];
    for (Integer i = 1; i < W; i++) {
        os << std::string(", ") << v[i];
    }
    os << ")";
    return os;
}

#endif /* VECTOR_HPP_ */
