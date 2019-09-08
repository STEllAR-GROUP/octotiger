//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SRC_RADIATION_SAFE_MATH_HPP_
#define SRC_RADIATION_SAFE_MATH_HPP_

#include <cmath>
#include <cstdio>
#include <cstdlib>

template<class T, class U>
inline T safe_power(const T& a, const U& b, const char* file, const int line) {
#ifdef SAFE_MATH_ON
	if (a >= T(0)) {
#endif
		return std::pow(a, T(b));
#ifdef SAFE_MATH_ON
	} else {
		printf("Power of a negative. File:%s Line:%i\n", file, line);
		abort();
	}
#endif
}

template<class T>
inline T safe_inverse(const T& a, const char* file, const int line) {
#ifdef SAFE_MATH_ON
	if (a != T(0)) {
#endif
		return T(1) / a;
#ifdef SAFE_MATH_ON
	} else {
		printf("Divide by zero. File:%s Line:%i\n", file, line);
		abort();
	}
#endif
}

template<class T>
inline T safe_sqrt(const T& a, const char* file, const int line) {
#ifdef SAFE_MATH_ON
	if (a >= T(0)) {
#endif
		return std::sqrt(a);
#ifdef SAFE_MATH_ON
	} else {
		printf("Square root of a negative = %e. File:%s Line:%i\n", (double) a, file, line);
		abort();
	}
#endif
}

#define INVERSE( a ) safe_inverse(a,__FILE__,__LINE__)
#define SQRT( a ) safe_sqrt(a, __FILE__, __LINE__)
#define POWER( a, b ) safe_power(a, b, __FILE__, __LINE__)

#endif /* SRC_RADIATION_SAFE_MATH_HPP_ */
