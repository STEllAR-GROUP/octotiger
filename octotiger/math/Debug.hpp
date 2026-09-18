/*
 * Debug.hpp
 *
 *  Created on: Feb 26, 2026
 *      Author: dmarce1
 */

#ifndef INCLUDE_DEBUG123_HPP_
#define INCLUDE_DEBUG123_HPP_

#include "./IO.hpp"
#include "./Real.hpp"

#include <cfenv>
#include <initializer_list>
#include <iostream>
#include <sstream>
#define BOOST_STACKTRACE_USE_ADDR2LINE
// #define BOOST_STACKTRACE_USE_BACKTRACE
// Keep the numerical headers usable in standalone tests without Boost.
// The full application build supplies Boost and retains stack traces.
#if __has_include(<boost/stacktrace.hpp>)
#include <boost/stacktrace.hpp>
#define OCTOTIGER_DEBUG_HAS_STACKTRACE 1
#endif

#if defined(__GLIBC__) || defined(__linux__)
extern "C" {
int fegetexcept();
int feenableexcept(int);
int fedisableexcept(int);
}
#define HAS_FEENABLEEXCEPT 1
#else
#define HAS_FEENABLEEXCEPT 0
#endif

enum class Fpe : int
{
    invalid = FE_INVALID,
    divByZero = FE_DIVBYZERO,
    overflow = FE_OVERFLOW,
    underflow = FE_UNDERFLOW,
    inexact = FE_INEXACT
};

constexpr int toMask(std::initializer_list<Fpe> ex) noexcept {
    int mask = 0;
    for (auto const e : ex) {
        mask |= static_cast<int>(e);
    }
    return mask;
}

class FpeGuard
{
public:
    explicit FpeGuard(int mask = (FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID)) noexcept {
#if HAS_FEENABLEEXCEPT
        previousEnabled = fegetexcept();
        if (previousEnabled < 0) {
            return;
        }
        enabledHere = (mask & FE_ALL_EXCEPT) & ~previousEnabled;
        if (enabledHere != 0) {
            // Pending masked exceptions belong to the caller. Clear them
            // before unmasking, then restore them after masking on exit.
            std::fegetexceptflag(&previousFlags, enabledHere);
            std::feclearexcept(enabledHere);
            feenableexcept(enabledHere);
        }
#else
        (void)mask;
#endif
    }

    FpeGuard(FpeGuard const&) = delete;
    FpeGuard(FpeGuard&&) = delete;
    FpeGuard& operator=(FpeGuard const&) = delete;
    FpeGuard& operator=(FpeGuard&&) = delete;

    ~FpeGuard() noexcept {
#if HAS_FEENABLEEXCEPT
        if (previousEnabled < 0) {
            return;
        }
        int const nowEnabled = fegetexcept();
        int const toDisable = nowEnabled & ~previousEnabled;
        int const toEnable = previousEnabled & ~nowEnabled;
        // Ordinary nested guards find identical masks and perform no writes.
        if (toDisable != 0) {
            fedisableexcept(toDisable);
        }
        if (toEnable != 0) {
            // Do not activate stale exceptions if code inside changed masks.
            std::feclearexcept(toEnable);
            feenableexcept(toEnable);
        }
        if (enabledHere != 0) {
            std::fesetexceptflag(&previousFlags, enabledHere);
        }
#endif
    }

private:
    int previousEnabled = -1;
    int enabledHere = 0;
    std::fexcept_t previousFlags{};
};

#ifdef NDEBUG
#define ASSERT_RANGE(l, v, u)
#else
#define ASSERT_RANGE(l, v, u)          \
    if (!std::is_constant_evaluated()) \
    assertRange(l, v, u, #v, __FILE__, __LINE__)
#endif

#ifdef NDEBUG
#define ASSERT_NONZERO(v)
#else
#define ASSERT_NONZERO(v)              \
    if (!std::is_constant_evaluated()) \
    assertNonzero(v, #v, __FILE__, __LINE__)
#endif

#ifdef NDEBUG
#define ASSERT_POSITIVE(v)
#else
#define ASSERT_POSITIVE(v)             \
    if (!std::is_constant_evaluated()) \
    assertPositive(v, #v, __FILE__, __LINE__)
#endif

#ifdef NDEBUG
#define ASSERT_NONNEGATIVE(v)
#else
#define ASSERT_NONNEGATIVE(v)          \
    if (!std::is_constant_evaluated()) \
    assertNonNegative(v, #v, __FILE__, __LINE__)
#endif

void assertRange(Real const& lo, auto const& var, Real const& hi, char const* expr,
    char const* filename, Integer line) {
    FpeGuard fpeGuard{};
    if (!(lo <= var && var <= hi)) {
        std::ostringstream os;
        os << "Out of range: " << expr << " = " << var;
        os << "  Correct range: (" << lo << ", " << hi << ")";
        os << "  File: " << filename;
        os << "  Line: " << line;
        throw std::runtime_error(os.str());
    }
}

inline void expectUnreachable() {
    throw std::logic_error("Reached unreachable code.");
}

void assertNonzero(auto const& var, char const* expr, char const* filename, Integer line) {
    FpeGuard fpeGuard{};
    if (var == decltype(var)(0)) {
        std::ostringstream os;
        os << "Zero when non-zero expected: " << expr << " = " << var;
        os << "  File: " << filename;
        os << "  Line: " << line;
#ifdef OCTOTIGER_DEBUG_HAS_STACKTRACE
        std::cout << boost::stacktrace::stacktrace();
#endif
        throw std::runtime_error(os.str());
    }
}

void assertPositive(auto const& var, char const* expr, char const* filename, Integer line) {
    FpeGuard fpeGuard{};
    if (!(var > 0_R)) {
        std::ostringstream os;
        os << "Non-positive when positive expected: " << expr << " = " << var;
        os << "  File: " << filename;
        os << "  Line: " << line;
        throw std::runtime_error(os.str());
    }
}

void assertNonNegative(auto const& var, char const* expr, char const* filename, Integer line) {
    FpeGuard fpeGuard{};
    if (!(var >= 0_R)) {
        std::ostringstream os;
        os << "Negative when non-negative expected: " << expr << " = " << var;
        os << "  File: " << filename;
        os << "  Line: " << line;
        throw std::runtime_error(os.str());
    }
}

// These contracts assume finite arguments and check only sign or range.
inline constexpr auto __expectPositive(auto&& value, char const* file, int line) {
    if (!(value > 0_R))
        throw std::runtime_error(
            print2string("Expected v > 0, got v = %e. %s:%i\n", value, file, line));
    return value;
}

inline constexpr auto __expectNegative(auto&& value, char const* file, int line) {
    if (!(value < 0_R))
        throw std::runtime_error(
            print2string("Expected v < 0, got v = %e. %s:%i\n", value, file, line));
    return value;
}

inline constexpr auto __expectNonPositive(auto&& value, char const* file, int line) {
    if (!(value <= 0_R))
        throw std::runtime_error(
            print2string("Expected v <= 0, got v = %e. %s:%i\n", value, file, line));
    return value;
}

inline constexpr auto __expectNonNegative(auto&& value, char const* file, int line) {
    if (!(value >= 0_R))
        throw std::runtime_error(
            print2string("Expected v >= 0, got v = %e. %s:%i\n", value, file, line));
    return value;
}

inline constexpr auto __expectNonZero(auto&& value, char const* file, int line) {
    if (value == 0_R)
        throw std::runtime_error(
            print2string("Expected v != 0, got v = %e. %s:%i\n", value, file, line));
    return value;
}

inline constexpr auto __expectRange(auto a, auto&& value, auto b, char const* file, int line) {
    if (!(value >= a))
        throw std::runtime_error(
            print2string("Expected %e <= v got v = %e, %e too little.   %s:%i\n", a, value,
                a - value, file, line));
    else if (!(value <= b))
        throw std::runtime_error(print2string("Expected v <= %e got v = %e, %e too much.   %s:%i\n",
            b, value, value - b, file, line));

    return value;
}

#ifndef NDEBUG
#define expectPositive(v) __expectPositive((v), __FILE__, __LINE__)
#define expectNegative(v) __expectNegative((v), __FILE__, __LINE__)
#define expectNonPositive(v) __expectNonPositive((v), __FILE__, __LINE__)
#define expectNonNegative(v) __expectNonNegative((v), __FILE__, __LINE__)
#define expectNonZero(v) __expectNonZero((v), __FILE__, __LINE__)
#define expectRange(a, v, b) __expectRange((a), (v), (b), __FILE__, __LINE__)
#else
#define expectPositive(v) (v)
#define expectNegative(v) (v)
#define expectNonPositive(v) (v)
#define expectNonNegative(v) (v)
#define expectNonZero(v) (v)
#define expectRange(a, v, b) (v)
#endif

#define INVERSE(x) (1_R / expectNonZero((x)))
#define SQRT(x) (std::sqrt(expectNonNegative((x))))
#define POWER(x, y) (std::pow((x), expectNonNegative((y))))

#endif /* INCLUDE_DEBUG_HPP_ */
