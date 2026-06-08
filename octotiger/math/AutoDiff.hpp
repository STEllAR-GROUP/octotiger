/*
 * AutoDiff.hpp
 *
 *  Created on: Feb 26, 2026
 *      Author: dmarce1
 */

#ifndef INCLUDE_AUTODIFF_HPP_
#define INCLUDE_AUTODIFF_HPP_

#include "Debug.hpp"
#include "Integer.hpp"
#include "Vector.hpp"

#include <array>
#include <ostream>

template <typename T, Integer N = 1>
struct AutoDiff
{
    inline constexpr AutoDiff() noexcept = default;
    inline constexpr AutoDiff(T constant) noexcept {
        v_ = constant;
        d_ = 0_R;
    }
    inline constexpr AutoDiff(AutoDiff const&) noexcept = default;
    inline constexpr AutoDiff(AutoDiff&&) noexcept = default;
    inline constexpr AutoDiff& operator=(AutoDiff const&) noexcept = default;
    inline constexpr AutoDiff& operator=(AutoDiff&&) noexcept = default;
    inline constexpr T get(Integer i) const noexcept {
        return d_[i];
    }
    inline constexpr T get() const noexcept {
        return v_;
    }
    inline constexpr AutoDiff& operator+=(AutoDiff const& A) noexcept {
        v_ += A.v_;
        d_ += A.d_;
        return *this;
    }
    inline constexpr AutoDiff& operator+=(T const& a) noexcept {
        v_ += a;
        return *this;
    }
    inline constexpr AutoDiff& operator-=(AutoDiff const& A) noexcept {
        v_ -= A.v_;
        d_ -= A.d_;
        return *this;
    }
    inline constexpr AutoDiff& operator-=(T const& a) noexcept {
        v_ -= a;
        return *this;
    }
    inline constexpr AutoDiff& operator*=(AutoDiff const& A) noexcept {
        d_ = A.v_ * d_ + A.d_ * v_;
        v_ = A.v_ * v_;
        return *this;
    }
    inline constexpr AutoDiff& operator*=(T const& a) noexcept {
        d_ *= a;
        v_ *= a;
        return *this;
    }
    inline constexpr AutoDiff& operator/=(AutoDiff const& B) noexcept {
        auto const iB = inv(B.v_);
        d_ = sqr(iB) * (B.v_ * d_ - B.d_ * v_);
        v_ *= iB;
        return *this;
    }
    inline constexpr AutoDiff& operator/=(T const& b) noexcept {
        *this *= inv(b);
        return *this;
    }
    friend inline constexpr AutoDiff operator+(AutoDiff const& A) noexcept {
        return A;
    }
    friend inline constexpr AutoDiff operator-(AutoDiff const& A) noexcept {
        AutoDiff C;
        C.v_ = -A.v_;
        C.d_ = -A.d_;
        return C;
    }
    friend inline constexpr AutoDiff operator+(AutoDiff const& A, AutoDiff const& B) noexcept {
        AutoDiff C;
        C.v_ = A.v_ + B.v_;
        C.d_ = A.d_ + B.d_;
        return C;
    }
    friend inline constexpr AutoDiff operator+(T const& a, AutoDiff const& B) noexcept {
        AutoDiff C;
        C.v_ = a + B.v_;
        C.d_ = B.d_;
        return C;
    }
    friend inline constexpr AutoDiff operator+(AutoDiff const& A, T const& b) noexcept {
        AutoDiff C;
        C.v_ = A.v_ + b;
        C.d_ = A.d_;
        return C;
    }
    friend inline constexpr AutoDiff operator-(AutoDiff const& A, AutoDiff const& B) noexcept {
        AutoDiff C;
        C.v_ = A.v_ - B.v_;
        C.d_ = A.d_ - B.d_;
        return C;
    }
    friend inline constexpr AutoDiff operator-(T const& a, AutoDiff const& B) noexcept {
        AutoDiff C;
        C.v_ = a - B.v_;
        C.d_ = -B.d_;
        return C;
    }
    friend inline constexpr AutoDiff operator-(AutoDiff const& A, T const& b) noexcept {
        AutoDiff C;
        C.v_ = A.v_ - b;
        C.d_ = A.d_;
        return C;
    }
    friend inline constexpr AutoDiff operator*(AutoDiff const& A, AutoDiff const& B) noexcept {
        AutoDiff C;
        C.v_ = A.v_ * B.v_;
        C.d_ = A.v_ * B.d_ + A.d_ * B.v_;
        return C;
    }
    friend inline constexpr AutoDiff operator*(T const& a, AutoDiff const& B) noexcept {
        AutoDiff C;
        C.v_ = a * B.v_;
        C.d_ = a * B.d_;
        return C;
    }
    friend inline constexpr AutoDiff operator*(AutoDiff const& A, T const& b) noexcept {
        AutoDiff C;
        C.v_ = A.v_ * b;
        C.d_ = A.d_ * b;
        return C;
    }
    friend inline constexpr AutoDiff operator/(AutoDiff const& A, AutoDiff const& B) noexcept {
        AutoDiff C;
        auto const iB = inv(B.v_);
        C.v_ = A.v_ * iB;
        C.d_ = sqr(iB) * (B.v_ * A.d_ - B.d_ * A.v_);
        return C;
    }
    friend inline constexpr AutoDiff operator/(T const& a, AutoDiff const& B) noexcept {
        AutoDiff C;
        auto const iB = inv(B.v_);
        C.v_ = a * iB;
        C.d_ = -a * sqr(iB) * B.d_;
        return C;
    }
    friend inline constexpr AutoDiff operator/(AutoDiff const& A, T const& b) noexcept {
        AutoDiff C;
        auto const ib = inv(b);
        C.v_ = ib * A.v_;
        C.d_ = ib * A.d_;
        return C;
    }
    friend inline constexpr AutoDiff max(AutoDiff const& A, AutoDiff const& B) noexcept {
        using std::max;
        AutoDiff C;
        auto const gt = A.v_ > B.v_;
        C.v_ = select(gt, A.v_, B.v_);
        for (Integer n = 0; n < N; n++) {
            C.d_[n] = select(gt, A.d_[n], B.d_[n]);
        }

        return C;
    }
    friend inline constexpr AutoDiff max(T const& a, AutoDiff const& B) noexcept {
        using std::max;
        AutoDiff C;
        auto const gt = a > B.v_;
        C.v_ = select(gt, a, B.v_);
        for (Integer n = 0; n < N; n++) {
            C.d_[n] = select(gt, 0_R, B.d_[n]);
        }
        return C;
    }
    friend inline constexpr AutoDiff max(AutoDiff const& A, T const& b) noexcept {
        using std::max;
        AutoDiff C;
        auto const gt = A.v_ > b;
        C.v_ = select(gt, A.v_, b);
        for (Integer n = 0; n < N; n++) {
            C.d_[n] = select(gt, A.d_[n], 0_R);
        }
        return C;
    }
    friend inline constexpr AutoDiff min(AutoDiff const& A, AutoDiff const& B) noexcept {
        using std::min;
        AutoDiff C;
        auto const eq = A.v_ == B.v_;
        auto const lt = A.v_ < B.v_;
        C.v_ = select(lt, A.v_, B.v_);
        for (Integer n = 0; n < N; n++) {
            C.d_[n] = select(lt, A.d_[n], B.d_[n]);
        }
        return C;
    }
    friend inline constexpr AutoDiff min(T const& a, AutoDiff const& B) noexcept {
        using std::min;
        AutoDiff C;
        auto const lt = a < B.v_;
        C.v_ = select(lt, a, B.v_);
        for (Integer n = 0; n < N; n++) {
            C.d_[n] = select(lt, 0_R, B.d_[n]);
        }
        return C;
    }
    friend inline constexpr AutoDiff min(AutoDiff const& A, T const& b) noexcept {
        using std::min;
        AutoDiff C;
        auto const lt = A.v_ < b;
        C.v_ = select(lt, A.v_, b);
        for (Integer n = 0; n < N; n++) {
            C.d_[n] = select(lt, A.d_[n], 0_R);
        }
        return C;
    }
    template <typename U>
    friend inline constexpr AutoDiff pow(AutoDiff const& A, U const& b) noexcept {
        using std::pow;
        AutoDiff B;
        auto const Abm1 = pow(A.v_, b - U(1_R));
        B.v_ = Abm1 * A.v_;
        B.d_ = b * Abm1;
        return B;
    }
    friend inline constexpr AutoDiff exp(AutoDiff const& A) noexcept {
        using std::exp;
        AutoDiff eA;
        eA.v_ = exp(A.v_);
        eA.d_ = eA.v_ * A.d_;
        return eA;
    }
    friend inline constexpr AutoDiff sqr(AutoDiff const& A) noexcept {
        AutoDiff A2;
        A2.v_ = sqr(A.v_);
        A2.d_ = 2_R * A.v_ * A.d_;
        return A2;
    }
    friend inline constexpr AutoDiff sqrt(AutoDiff const& A) noexcept {
        using std::sqrt;
        AutoDiff r2A;
        r2A.v_ = sqrt(A.v_);
        r2A.d_ = 0.5_R * inv(r2A.v_) * A.d_;
        return r2A;
    }
    friend inline constexpr AutoDiff inv(AutoDiff const& A) noexcept {
        AutoDiff iA;
        iA.v_ = inv(A.v_);
        iA.d_ = -sqr(iA.v_) * A.d_;
        return iA;
    }
    static inline constexpr AutoDiff genVar(T const& value, Integer i = 0) {
        AutoDiff ad;
        ad.v_ = value;
        ad.d_ = Vector<T, N>::unit(i);
        return ad;
    }
    template <typename Boolean>
    friend constexpr auto select(Boolean const& f, AutoDiff const& A, AutoDiff const& B) {
        return select(f, A.v_, B.v_);
    }
    template <typename Boolean>
    friend constexpr auto select(Boolean const& f, T const& a, AutoDiff const& B) {
        return select(f, a, B.v_);
    }
    template <typename Boolean>
    friend constexpr auto select(Boolean const& f, AutoDiff const& A, T const& b) {
        return select(f, A.v_, b);
    }
    //	friend std::ostream &operator<<(std::ostream &os, AutoDiff const &v) {
    //		os << "(v: " << v.v_ << " dv/dx: " << v.d_ << ")";
    //		return os;
    //	}
    friend std::ostream& operator<<(std::ostream& os, AutoDiff const& v) {
        os << v.v_;
        return os;
    }

private:
    T v_;
    Vector<T, N> d_;
};

template <typename T, Integer N>
constexpr auto cvtVector2AutoVector(Vector<T, N> const& v) {
    Vector<AutoDiff<T, N>, N> u;
    for (Integer n = 0; n < N; n++) {
        u[n] = AutoDiff<T, N>::genVar(v[n], n);
    }
    return u;
};

template <typename T, Integer N>
constexpr auto cvtAutoVector2Vector(Vector<AutoDiff<T, N>, N> const& u) {
    Vector<T, N> v;
    for (Integer n = 0; n < N; n++) {
        v[n] = u[n].get();
    }
    return v;
};

#endif /* INCLUDE_AUTODIFF_HPP_ */
