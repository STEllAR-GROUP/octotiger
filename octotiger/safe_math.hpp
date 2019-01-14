/*
 * safe_math.hpp
 *
 *  Created on: Sep 25, 2018
 *      Author: dmarce1
 */

#ifndef SRC_RADIATION_SAFE_MATH_HPP_
#define SRC_RADIATION_SAFE_MATH_HPP_

#include <cmath>
#include <cstdio>
#include <cstdlib>

template<class T, class U>
inline T safe_power(const T& a, const U& b, const char* file, const int line) {
	if (a >= T(0)) {
		return std::pow(a,T(b));
	} else {
		printf( "Power of a negative. File:%s Line:%i\n", file, line);
		abort();
	}
}

template<class T>
inline T safe_inverse(const T& a, const char* file, const int line) {
	if (a != T(0)) {
		return T(1) / a;
	} else {
		printf( "Divide by zero. File:%s Line:%i\n", file, line);
		abort();
	}
}


template<class T>
inline T safe_sqrt(const T& a, const char* file, const int line) {
	if (a >= T(0)) {
		return std::sqrt(a);
	} else {
		printf( "Square root of a negative = %e. File:%s Line:%i\n", (double) a, file, line);
		abort();
	}
}



#define INVERSE( a ) safe_inverse(a,__FILE__,__LINE__)
#define SQRT( a ) safe_sqrt(a, __FILE__, __LINE__)
#define POWER( a, b ) safe_power(a, b, __FILE__, __LINE__)



#endif /* SRC_RADIATION_SAFE_MATH_HPP_ */
