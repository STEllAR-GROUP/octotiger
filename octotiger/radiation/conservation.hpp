// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#ifndef OCTOTIGER_RADIATION_CONSERVATION_HPP
#define OCTOTIGER_RADIATION_CONSERVATION_HPP

#include "octotiger/math/Debug.hpp"

#include <array>
#include <iomanip>
#include <limits>
#include <ostream>

namespace radiationConservation {

// Code units throughout; the three flux moments are physical F, not F/c.
using Moments = std::array<long double, 4>;

struct Totals {
	long double volume = 0;
	Moments value{};
	Moments boundary{}; // Signed outward transport integrated over the interval.
	Moments source{};   // Actual local source increments integrated over volume.

	Totals &operator+=(const Totals &other) {
		FpeGuard fpeGuard{};
		volume += other.volume;
		for (int f = 0; f < 4; ++f) {
			value[f] += other.value[f];
			boundary[f] += other.boundary[f];
			source[f] += other.source[f];
		}
		return *this;
	}

	template <class Archive> void serialize(Archive &arc, unsigned) {
		arc & volume;
		for (int f = 0; f < 4; ++f) {
			arc & value[f];
			arc & boundary[f];
			arc & source[f];
		}
	}
};

// The driver owns cumulative budgets. Mesh-local intervals are drained before
// regridding/migration; diagnostic state never changes the checkpoint format.
struct Ledger {
	Moments boundary{};
	Moments source{};

	Totals consume(Totals interval) {
		FpeGuard fpeGuard{};
		for (int f = 0; f < 4; ++f) {
			boundary[f] += interval.boundary[f];
			source[f] += interval.source[f];
		}
		interval.boundary = boundary;
		interval.source = source;
		return interval;
	}
};

inline void writeCsvHeader(std::ostream &out) {
	out << "t,volume,er,fx,fy,fz,er_boundary,fx_boundary,fy_boundary,fz_boundary,"
		   "er_source,fx_source,fy_source,fz_source\n";
}

inline void writeCsvRow(std::ostream &out, double time, const Totals &sample) {
	FpeGuard fpeGuard{};
	out << std::scientific << std::setprecision(std::numeric_limits<long double>::max_digits10)
		<< time << ',' << sample.volume;
	for (auto value : sample.value) out << ',' << value;
	for (auto value : sample.boundary) out << ',' << value;
	for (auto value : sample.source) out << ',' << value;
	out << '\n';
}

} // namespace radiationConservation

#endif
