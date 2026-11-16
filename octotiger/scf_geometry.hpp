#ifndef OCTOTIGER_SCF_GEOMETRY_HPP
#define OCTOTIGER_SCF_GEOMETRY_HPP

#include "octotiger/real.hpp"

namespace octotiger::scf {

// The line coordinate is only a projection. Retain the actual cell position
// when evaluating the centrifugal potential used by the SCF density update.
struct SamplePosition {
	real x = 0.0;
	real y = 0.0;
	real z = 0.0;

	real cylindrical_radius_squared(real com) const {
		real const offset_x = x - com;
		return offset_x * offset_x + y * y;
	}

	real effective_potential(real gravitational_potential, real omega, real com) const {
		return gravitational_potential - 0.5 * omega * omega * cylindrical_radius_squared(com);
	}

	template<class Archive>
	void serialize(Archive &archive, unsigned) {
		archive & x & y & z;
	}
};

}

#endif
