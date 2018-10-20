#include "defs.hpp"
#include "grid.hpp"
#ifdef DO_OUTPUT
#include <silo.h>
#endif
#include <atomic>
#include <ios>
#include <fstream>
#include <thread>
#include <cmath>
#include "physcon.hpp"
#include "options.hpp"

extern options opts;
#include "radiation/rad_grid.hpp"

#include <hpx/include/lcos.hpp>

namespace hpx {
using mutex = hpx::lcos::local::spinlock;
}

std::vector<std::vector<real>>& TLS_V();

#include <unordered_map>




inline bool float_eq(xpoint_type a, xpoint_type b) {
	constexpr static xpoint_type eps = 0.00000011920928955078125; // std::pow(xpoint_type(2), -23);
// 	const xpoint_type eps = std::pow(xpoint_type(2), -23);
	return std::abs(a - b) < eps;
}

bool grid::xpoint_eq(const xpoint& a, const xpoint& b) {
	bool rc = true;
	for (integer d = 0; d != NDIM; ++d) {
		if (!float_eq(a[d], b[d])) {
			rc = false;
			break;
		}
	}
	return rc;
}

bool grid::node_point::operator==(const node_point& other) const {
	return xpoint_eq(other.pt, pt);
}

bool grid::node_point::operator<(const node_point& other) const {
	bool rc = false;
	for (integer d = 0; d != NDIM; ++d) {
		if (!float_eq(pt[d], other.pt[d])) {
			rc = (pt[d] < other.pt[d]);
			break;
		}
	}
	return rc;
}

const std::array<const char*,OUTPUT_COUNT>& grid::field_names(  ) {
	static std::array<const char*,OUTPUT_COUNT> field_names;
	if( opts.radiation) {
		field_names = {{"rho", "egas", "sx", "sy", "sz", "tau", "pot", "zx", "zy", "zz", "primary_core", "primary_envelope", "secondary_core",
			"secondary_envelope", "vacuum", "er", "fx", "fy", "fz", "phi", "gx", "gy", "gz", "vx", "vy", "vz", "eint",
			"zzs", "roche"}};
	} else {
		field_names = {{"rho", "egas", "sx", "sy", "sz", "tau", "pot", "zx", "zy", "zz", "primary_core",
			"primary_envelope", "secondary_core", "secondary_envelope", "vacuum", "phi", "gx", "gy", "gz", "vx", "vy", "vz", "eint",
			"zzs", "roche"}};
	}
	return field_names;
}
