//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This code should be used for counting the number of executed floating point
// operations only. Don't use this for performance evaluations.

// This code was partially adopted from Boost:
//
//  Copyright John Maddock 2006.

#if !defined(COUNTING_DOUBLE_HPP)
#define COUNTING_DOUBLE_HPP

#if defined(OCTOTIGER_HAVE_OPERATIONS_COUNT)

#include <boost/config.hpp>
#include <boost/atomic.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/special_functions/modf.hpp>
#include <boost/math/special_functions/modf.hpp>

#include "real.hpp"

#include <cstddef>
#include <cmath>
#include <ostream>
#include <istream>
#include <iostream>

typedef real_type counting_double_base_type;

class counting_double
{
public:
    static boost::atomic<std::size_t> additions;
    static boost::atomic<std::size_t> subtractions;
    static boost::atomic<std::size_t> multiplications;
    static boost::atomic<std::size_t> divisions;
    static boost::atomic<std::size_t> transcendentals;
    static boost::atomic<std::size_t> miscellaneous;

    static std::size_t additions_overall;
    static std::size_t subtractions_overall;
    static std::size_t multiplications_overall;
    static std::size_t divisions_overall;
    static std::size_t transcendentals_overall;
    static std::size_t miscellaneous_overall;

    static void report_counters() {
        std::cout << "   Additions: "         << counting_double::additions.load() << '\n';
        std::cout << "   Subtractions: "      << counting_double::subtractions.load() << '\n';
        std::cout << "   Multiplications: "   << counting_double::multiplications.load() << '\n';
        std::cout << "   Divisions: "         << counting_double::divisions.load() << '\n';
        std::cout << "   Transcendentals: "         << counting_double::transcendentals.load() << '\n';
        std::cout << "   Miscellaneous: "         << counting_double::miscellaneous.load() << '\n';
        size_t total = counting_double::additions.load() + counting_double::subtractions.load() + counting_double::multiplications.load() + counting_double::divisions.load() + counting_double::transcendentals.load() + counting_double::miscellaneous.load();
        std::cout << "   == Total Step Flop: "         << total << '\n';
    }
    
    static void reset_counters() {
        // context is single thread
        additions_overall += counting_double::additions.load();
        subtractions_overall += counting_double::subtractions.load();
        multiplications_overall += counting_double::multiplications.load();
        divisions_overall += counting_double::divisions.load();
        transcendentals_overall += counting_double::transcendentals.load();
        miscellaneous_overall += counting_double::miscellaneous.load();
        // lost update shouldn't be possible as context is single-threaded
        counting_double::additions.store(0);
        counting_double::subtractions.store(0);
        counting_double::multiplications.store(0);
        counting_double::divisions.store(0);
        counting_double::transcendentals.store(0);
        counting_double::miscellaneous.store(0);
    }

    static void report_counters_overall() {
        // make sure you catch the last remaining flops after the last step
        counting_double::reset_counters();
        // context is single thread
        std::cout << "   Additions: "         << additions_overall << '\n';
        std::cout << "   Subtractions: "      << subtractions_overall << '\n';
        std::cout << "   Multiplications: "   << multiplications_overall << '\n';
        std::cout << "   Divisions: "         << divisions_overall << '\n';
        std::cout << "   Transcendentals: "   << transcendentals_overall << '\n';
        std::cout << "   Miscellaneous: "     << miscellaneous_overall << '\n';
        size_t total = additions_overall + subtractions_overall + multiplications_overall + divisions_overall + transcendentals_overall + miscellaneous_overall;
        std::cout << "   == Total Application Flop: "         << total << '\n';
    }

    // Constructors:
    constexpr counting_double()
      : m_value(0) {}
    constexpr counting_double(char c)
      : m_value(c) {}
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
    constexpr counting_double(wchar_t c)
      : m_value(c) {}
#endif
    constexpr counting_double(unsigned char c)
      : m_value(c) {}
    constexpr counting_double(signed char c)
      : m_value(c) {}
    constexpr counting_double(unsigned short c)
      : m_value(c) {}
    constexpr counting_double(short c)
      : m_value(c) {}
    constexpr counting_double(unsigned int c)
      : m_value(c) {}
    constexpr counting_double(int c)
      : m_value(c) {}
    constexpr counting_double(unsigned long c)
      : m_value(c) {}
    constexpr counting_double(long c)
      : m_value(c) {}
#if defined(BOOST_HAS_LONG_LONG)
    constexpr counting_double(boost::ulong_long_type c)
      : m_value(static_cast<counting_double_base_type>(c)) {}
    constexpr counting_double(boost::long_long_type c)
      : m_value(static_cast<counting_double_base_type>(c)) {}
#elif defined(BOOST_HAS_MS_INT64)
    constexpr counting_double(unsigned __int64 c)
      : m_value(static_cast<counting_double_base_type>(c)) {}
    constexpr counting_double(__int64 c)
      : m_value(static_cast<counting_double_base_type>(c)) {}
#endif
    constexpr counting_double(float c)
      : m_value(c) {}
    constexpr counting_double(double c)
      : m_value(c) {}
    constexpr counting_double(long double c)
      : m_value(c) {}

    // Assignment:
    counting_double& operator=(char c) {
        m_value = c;
        return *this;
    }
    counting_double& operator=(unsigned char c) {
        m_value = c;
        return *this;
    }
    counting_double& operator=(signed char c) {
        m_value = c;
        return *this;
    }
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
    counting_double& operator=(wchar_t c) {
        m_value = c;
        return *this;
    }
#endif
    counting_double& operator=(short c) {
        m_value = c;
        return *this;
    }
    counting_double& operator=(unsigned short c) {
        m_value = c;
        return *this;
    }
    counting_double& operator=(int c) {
        m_value = c;
        return *this;
    }
    counting_double& operator=(unsigned int c) {
        m_value = c;
        return *this;
    }
    counting_double& operator=(long c) {
        m_value = c;
        return *this;
    }
    counting_double& operator=(unsigned long c) {
        m_value = c;
        return *this;
    }
#ifdef BOOST_HAS_LONG_LONG
    counting_double& operator=(boost::long_long_type c) {
        m_value = static_cast<counting_double_base_type>(c);
        return *this;
    }
    counting_double& operator=(boost::ulong_long_type c) {
        m_value = static_cast<counting_double_base_type>(c);
        return *this;
    }
#endif
    counting_double& operator=(float c) {
        m_value = c;
        return *this;
    }
    counting_double& operator=(double c) {
        m_value = c;
        return *this;
    }
    counting_double& operator=(long double c) {
        m_value = c;
        return *this;
    }

    // Access:
    constexpr counting_double_base_type value() const {
        return m_value;
    }

    // Member arithmetic:
    counting_double& operator+=(const counting_double& other) {
        m_value += other.value();
        ++additions;
        return *this;
    }
    counting_double& operator-=(const counting_double& other) {
        m_value -= other.value();
        ++subtractions;
        return *this;
    }
    counting_double& operator*=(const counting_double& other) {
        m_value *= other.value();
        ++multiplications;
        return *this;
    }
    counting_double& operator/=(const counting_double& other) {
        m_value /= other.value();
        ++divisions;
        return *this;
    }

    counting_double operator-() const {
        ++subtractions; // check
        return counting_double(-m_value);
    }
    constexpr counting_double const& operator+() const {
        return *this;
    }
    counting_double& operator++() {
        ++m_value;
        ++additions;
        return *this;
    }
    counting_double& operator--() {
        --m_value;
        ++subtractions;
        return *this;
    }

    constexpr operator counting_double_base_type() const {
        return value();
    }

    template <typename Archive>
    void serialize(Archive & ar, const unsigned) {
        ar & m_value;
    }

private:
    counting_double_base_type m_value;
};

// Non-member arithmetic:
inline counting_double operator+(const counting_double& a, const counting_double& b) {
    counting_double result(a);
    result += b;
    return result;
}
inline counting_double operator-(const counting_double& a, const counting_double& b) {
    counting_double result(a);
    result -= b;
    return result;
}
inline counting_double operator*(const counting_double& a, const counting_double& b) {
    counting_double result(a);
    result *= b;
    return result;
}
inline counting_double operator/(const counting_double& a, const counting_double& b) {
    counting_double result(a);
    result /= b;
    return result;
}

inline counting_double operator+(const counting_double& a, const counting_double_base_type& b) {
    counting_double result(a);
    result += counting_double(b);
    return result;
}
inline counting_double operator-(const counting_double& a, const counting_double_base_type& b) {
    counting_double result(a);
    result -= counting_double(b);
    return result;
}
inline counting_double operator*(const counting_double& a, const counting_double_base_type& b) {
    counting_double result(a);
    result *= counting_double(b);
    return result;
}
inline counting_double operator/(const counting_double& a, const counting_double_base_type& b) {
    counting_double result(a);
    result /= counting_double(b);
    return result;
}

inline counting_double operator+(const counting_double_base_type& a, const counting_double& b) {
    counting_double result(a);
    result += b;
    return result;
}
inline counting_double operator-(const counting_double_base_type& a, const counting_double& b) {
    counting_double result(a);
    result -= b;
    return result;
}
inline counting_double operator*(const counting_double_base_type& a, const counting_double& b) {
    counting_double result(a);
    result *= b;
    return result;
}
inline counting_double operator/(const counting_double_base_type& a, const counting_double& b) {
    counting_double result(a);
    result /= b;
    return result;
}

// Comparison:
constexpr inline bool operator==(const counting_double& a, const counting_double& b) {
    return a.value() == b.value();
}
constexpr inline bool operator!=(const counting_double& a, const counting_double& b) {
    return a.value() != b.value();
}
constexpr inline bool operator<(const counting_double& a, const counting_double& b) {
    return a.value() < b.value();
}
constexpr inline bool operator<=(const counting_double& a, const counting_double& b) {
    return a.value() <= b.value();
}
constexpr inline bool operator>(const counting_double& a, const counting_double& b) {
    return a.value() > b.value();
}
constexpr inline bool operator>=(const counting_double& a, const counting_double& b) {
    return a.value() >= b.value();
}

constexpr inline bool operator==(const counting_double& a, const counting_double_base_type& b) {
    return a.value() == b;
}
constexpr inline bool operator!=(const counting_double& a, const counting_double_base_type& b) {
    return a.value() != b;
}
constexpr inline bool operator<(const counting_double& a, const counting_double_base_type& b) {
    return a.value() < b;
}
constexpr inline bool operator<=(const counting_double& a, const counting_double_base_type& b) {
    return a.value() <= b;
}
constexpr inline bool operator>(const counting_double& a, const counting_double_base_type& b) {
    return a.value() > b;
}
constexpr inline bool operator>=(const counting_double& a, const counting_double_base_type& b) {
    return a.value() >= b;
}

constexpr inline bool operator==(const counting_double_base_type& a, const counting_double& b) {
    return a == b.value();
}
constexpr inline bool operator!=(const counting_double_base_type& a, const counting_double& b) {
    return a != b.value();
}
constexpr inline bool operator<(const counting_double_base_type& a, const counting_double& b) {
    return a < b.value();
}
constexpr inline bool operator<=(const counting_double_base_type& a, const counting_double& b) {
    return a <= b.value();
}
constexpr inline bool operator>(const counting_double_base_type& a, const counting_double& b) {
    return a > b.value();
}
constexpr inline bool operator>=(const counting_double_base_type& a, const counting_double& b) {
    return a >= b.value();
}

// Non-member functions:
inline counting_double acos(counting_double a) {
    ++counting_double::transcendentals;
    return std::acos(a.value());
}
inline counting_double cos(counting_double a) {
    ++counting_double::transcendentals;
    return std::cos(a.value());
}
inline counting_double asin(counting_double a) {
    ++counting_double::transcendentals;
    return std::asin(a.value());
}
inline counting_double atan(counting_double a) {
    ++counting_double::transcendentals;
    return std::atan(a.value());
}
inline counting_double atan2(counting_double a, counting_double b) {
    ++counting_double::transcendentals;
    return std::atan2(a.value(), b.value());
}
inline counting_double ceil(counting_double a) {
    ++counting_double::miscellaneous;
    return std::ceil(a.value());
}
// I've seen std::fmod(long double) crash on some platforms
// so use fmodl instead:
inline counting_double fmod(counting_double a, counting_double b) {
    ++counting_double::miscellaneous;
    return fmodl(a.value(), b.value());
}
inline counting_double cosh(counting_double a) {
    ++counting_double::transcendentals;
    return std::cosh(a.value());
}
inline counting_double exp(counting_double a) {
    ++counting_double::transcendentals;
    return std::exp(a.value());
}
inline counting_double fabs(counting_double a) {
    ++counting_double::miscellaneous;
    return std::fabs(a.value());
}
inline counting_double abs(counting_double a) {
    ++counting_double::miscellaneous;
    return std::abs(a.value());
}
inline counting_double floor(counting_double a) {
    ++counting_double::miscellaneous;
    return std::floor(a.value());
}
inline counting_double modf(counting_double a, counting_double* ipart) {
    ++counting_double::miscellaneous;
    counting_double_base_type ip;
    counting_double_base_type result = std::modf(a.value(), &ip);
    *ipart = ip;
    return result;
}
inline counting_double frexp(counting_double a, int* expon) {
    ++counting_double::transcendentals;
    return std::frexp(a.value(), expon);
}
inline counting_double ldexp(counting_double a, int expon) {
    ++counting_double::transcendentals;
    return std::ldexp(a.value(), expon);
}
inline counting_double log(counting_double a) {
    ++counting_double::transcendentals;
    return std::log(a.value());
}
inline counting_double log10(counting_double a) {
    ++counting_double::transcendentals;
    return std::log10(a.value());
}
inline counting_double tan(counting_double a) {
    ++counting_double::transcendentals;
    return std::tan(a.value());
}
inline counting_double pow(counting_double a, counting_double b) {
    ++counting_double::transcendentals;
    return std::pow(a.value(), b.value());
}
inline counting_double pow(counting_double a, counting_double_base_type b) {
    ++counting_double::transcendentals;
    return std::pow(a.value(), b);
}
inline counting_double pow(counting_double a, int b) {
    ++counting_double::transcendentals;
    return std::pow(a.value(), b);
}
inline counting_double sin(counting_double a) {
    ++counting_double::transcendentals;
    return std::sin(a.value());
}
inline counting_double sinh(counting_double a) {
    ++counting_double::transcendentals;
    return std::sinh(a.value());
}
inline counting_double sqrt(counting_double a) {
    ++counting_double::transcendentals;
    return std::sqrt(a.value());
}
inline counting_double tanh(counting_double a) {
    ++counting_double::transcendentals;
    return std::tanh(a.value());
}
inline counting_double max(counting_double a, counting_double b) {
    ++counting_double::miscellaneous;
    return std::max(a.value(), b.value());
}
inline counting_double min(counting_double a, counting_double b) {
    ++counting_double::miscellaneous;
    return std::min(a.value(), b.value());
}

// Conversion and truncation routines:
template <class Policy>
inline int iround(const counting_double& v, const Policy& pol) {
    ++counting_double::miscellaneous;
    return boost::math::iround(v.value(), pol);
}
inline int iround(const counting_double& v) {
    ++counting_double::miscellaneous;
    return boost::math::iround(v.value(), boost::math::policies::policy<>());
}
template <class Policy>
inline long lround(const counting_double& v, const Policy& pol) {
    ++counting_double::miscellaneous;
    return boost::math::lround(v.value(), pol);
}
inline long lround(const counting_double& v) {
    ++counting_double::miscellaneous;
    return boost::math::lround(v.value(), boost::math::policies::policy<>());
}

#ifdef BOOST_HAS_LONG_LONG
template <class Policy>
inline boost::long_long_type llround(const counting_double& v, const Policy& pol) {
    ++counting_double::miscellaneous;
    return boost::math::llround(v.value(), pol);
}
inline boost::long_long_type llround(const counting_double& v) {
    ++counting_double::miscellaneous;
    return boost::math::llround(v.value(), boost::math::policies::policy<>());
}
#endif

template <class Policy>
inline int itrunc(const counting_double& v, const Policy& pol) {
    ++counting_double::miscellaneous;
    return boost::math::itrunc(v.value(), pol);
}
inline int itrunc(const counting_double& v) {
    ++counting_double::miscellaneous;
    return boost::math::itrunc(v.value(), boost::math::policies::policy<>());
}
template <class Policy>
inline long ltrunc(const counting_double& v, const Policy& pol) {
    ++counting_double::miscellaneous;
    return boost::math::ltrunc(v.value(), pol);
}
inline long ltrunc(const counting_double& v) {
    ++counting_double::miscellaneous;
    return boost::math::ltrunc(v.value(), boost::math::policies::policy<>());
}

#ifdef BOOST_HAS_LONG_LONG
template <class Policy>
inline boost::long_long_type lltrunc(const counting_double& v, const Policy& pol) {
    ++counting_double::miscellaneous;
    return boost::math::lltrunc(v.value(), pol);
}
inline boost::long_long_type lltrunc(const counting_double& v) {
    ++counting_double::miscellaneous;
    return boost::math::lltrunc(v.value(), boost::math::policies::policy<>());
}
#endif

// Streaming:
template <class charT, class traits>
inline std::basic_ostream<charT, traits>& operator<<(
    std::basic_ostream<charT, traits>& os, const counting_double& a) {
    return os << a.value();
}
template <class charT, class traits>
inline std::basic_istream<charT, traits>& operator>>(
    std::basic_istream<charT, traits>& is, counting_double& a) {
    counting_double_base_type v;
    is >> v;
    a = v;
    return is;
}

using real = counting_double;

#endif
#endif
