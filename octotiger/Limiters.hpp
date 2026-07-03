#pragma once

#include "octotiger/defs.hpp"
#include "octotiger/math/Real.hpp"

inline Real minmod(Real leftSlope, Real rightSlope) {
	using namespace std;
	auto const smallestSlope = 0.5_R * min(abs(leftSlope), abs(rightSlope));
	return copysign(smallestSlope, leftSlope) + copysign(smallestSlope, rightSlope);
}

inline auto minmod(auto const &a, auto b) {
	auto const N = b.size();
	for (unsigned n = 0; n < N; n++) {
		b[n] = minmod(a[n], b[n]);
	}
	return b;
}

inline auto minmod(auto a, auto b, Real θ) {
	return minmod(0.5_R * (a + b), θ * minmod(a, b));
}
