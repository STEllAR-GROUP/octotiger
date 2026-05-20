/*
 * Debug.hpp
 *
 *  Created on: Feb 26, 2026
 *      Author: dmarce1
 */

#ifndef INCLUDE_DEBUG123_HPP_
#define INCLUDE_DEBUG123_HPP_


#include "./Real.hpp"

#include <cfenv>
#include <sstream>

#ifdef NDEBUG
#define ASSERT_RANGE(l, v, u)
#else
#define ASSERT_RANGE(l, v, u)                                                                                                              \
	if (!std::is_constant_evaluated()) assertRange(l, v, u, __FILE__, __LINE__)
#endif

#ifdef NDEBUG
#define ASSERT_NONZERO(v)
#else
#define ASSERT_NONZERO(v)                                                                                                                  \
	if (!std::is_constant_evaluated()) assertNonzero(v, __FILE__, __LINE__)
#endif

#ifdef NDEBUG
#define ASSERT_POSITIVE(v)
#else
#define ASSERT_POSITIVE(v)                                                                                                                 \
	if (!std::is_constant_evaluated()) assertPositive(v, __FILE__, __LINE__)
#endif

#ifdef NDEBUG
#define ASSERT_NONNEGATIVE(v)
#else
#define ASSERT_NONNEGATIVE(v)                                                                                                              \
	if (!std::is_constant_evaluated()) assertNonNegative(v, __FILE__, __LINE__)
#endif

void assertRange(Real const &lo, auto const &var, Real const &hi, char const *filename, Integer line) {
	if ((lo > var) || (var > hi)) {
		std::ostringstream os;
		os << "Out of range: " << var;
		os << "  Correct range: (" << lo << ", " << hi << ")";
		os << "  File: " << filename;
		os << "  Line: " << line;
		//		throw std::runtime_error(os.str());
		abort();
	}
}

void assertNonzero(auto const &var, char const *filename, Integer line) {
	if (var == 0_R) {
		std::ostringstream os;
		os << "Zero when non-zero expected: " << var;
		os << "  File: " << filename;
		os << "  Line: " << line;
		//		throw std::runtime_error(os.str());
		abort();
	}
}

void assertPositive(auto const &var, char const *filename, Integer line) {
	if (var == 0_R) {
		std::ostringstream os;
		os << "Non-positive when positive expected: " << var;
		os << "  File: " << filename;
		os << "  Line: " << line;
		//		throw std::runtime_error(os.str());
		abort();
	}
}

void assertNonNegative(auto const &var, char const *filename, Integer line) {
	if (!all(var != 0_R)) {
		std::ostringstream os;
		os << "Negative when non-negative expected: " << var;
		os << "  File: " << filename;
		os << "  Line: " << line;
		//		throw std::runtime_error(os.str());
		abort();
	}
}

#if defined(__GLIBC__) || defined(__linux__)
extern "C" {
int feenableexcept(int);
int fedisableexcept(int);
}
#define HAS_FEENABLEEXCEPT 1
#else
#define HAS_FEENABLEEXCEPT 0
#endif

enum class Fpe : int {
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

class FpeGuard {
public:
	explicit FpeGuard(int mask = (FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID)) noexcept :
		mask_(mask) {
#if HAS_FEENABLEEXCEPT
		prevEnabled_ = feenableexcept(0);
		feenableexcept(mask_);
#else
		(void)enable;
		prevEnabled_ = 0;
#endif
	}
	~FpeGuard() noexcept {
#if HAS_FEENABLEEXCEPT
		int const nowEnabled = feenableexcept(0);
		int const toDisable = nowEnabled & ~prevEnabled_;
		int const toEnable = prevEnabled_ & ~nowEnabled;

		if (toDisable) {
			fedisableexcept(toDisable);
		}
		if (toEnable) {
			feenableexcept(toEnable);
		}
#endif
	}

private:
	int mask_ = 0;
	int prevEnabled_ = 0;
};

#endif /* INCLUDE_DEBUG_HPP_ */
