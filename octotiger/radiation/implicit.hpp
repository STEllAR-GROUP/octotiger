/*
 * implicit.hpp
 *
 *  Created on: Sep 25, 2018
 *      Author: dmarce1
 */

#ifndef SRC_RADIATION_IMPLICIT_HPP_
#define SRC_RADIATION_IMPLICIT_HPP_

#include "octotiger/physcon.hpp"
#include "octotiger/real.hpp"
#include "octotiger/safe_math.hpp"
#include "octotiger/space_vector.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

template<class T>
T light_f(const T& E, const T& F) {
	const T c = physcon().c;
	return F * INVERSE(c * E);
}

template<class T>
T light_f(const T& E, const T& F0, const T& F1) {
	const T c = physcon().c;
	return SQRT(F0*F0+F1*F1) * INVERSE(c * E);
}

template<class T>
T light_f(const T& E, const T& F0, const T& F1, const T& F2) {
	const T c = physcon().c;
	return SQRT(F0*F0+F1*F1+F2*F2) * INVERSE(c * E);
}

template<class T>
T light_f(const T& E, const T& F, int line, const char* file) {
	T f = light_f(E,F);
	if (T(f) > T(1) + 2*std::numeric_limits < T >::round_error()) {
		printf("light_f computation failed in %s line %i. f-1 is %e\n", file, line, (double)(T(f) - T(1)));
		printf( "%e %e\n", (double) E, (double) F);
		abort();
	} else if (T(f) < T(0)) {
		printf("light_f computation failed in %s line %i. f is %e\n", file, line, T(f));
		printf( "%e %e\n", (double) E, (double) F);
		abort();
	}
	f = std::min(f,T(1));
	return f;
}


#define LIGHT_F1(e,f1) light_f((e),(f1),__LINE__,__FILE__)
#define LIGHT_F2(e,f1,f2) light_f((e),(SQRT((f1)*(f1) + (f2)*(f2))),__LINE__,__FILE__)
#define LIGHT_F3(e,f1,f2,f3) light_f((e),(SQRT((f1)*(f1) + (f2)*(f2) + (f3)*(f3))),__LINE__,__FILE__)


#endif /* SRC_RADIATION_IMPLICIT_HPP_ */
