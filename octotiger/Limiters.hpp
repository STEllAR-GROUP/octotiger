#pragma once

#include "octotiger/defs.hpp"
#include "octotiger/astrolib/Real.hpp"

inline Real vanleer(Real a, Real b) {
	using namespace std;
	auto const num = a * abs(b) + b * abs(a);
	auto const den = abs(a) + abs(b);
	return num / (den + tiny_R);
}

inline auto vanleer(auto const &a, auto b) {
	auto const N = b.size();
	for (unsigned n = 0; n < N; n++) {
		b[n] = vanleer(a[n], b[n]);
	}
	return b;
}

inline Real minmod(Real a, Real b) {
	using namespace std;
	auto const c = 0.5_R * min(abs(a), abs(b));
	return copysign(c, a) + copysign(c, b);
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
