// Shared timestep diagnostics for hydro and radiation integration.
#pragma once
#include <limits>
#include <algorithm>
#include <vector>

struct timestep_t {
	double a = 0;
	double x = 0, y = 0, z = 0;
	double dt = std::numeric_limits<double>::max();
	// Reduced independently from dt: its winning leaf can be different.
	double radiation_dt = std::numeric_limits<double>::max();
	int dim = -1;
	std::vector<double> ur;
	std::vector<double> ul;
    void reduce(timestep_t const& other) {
        double const limit=std::min(radiation_dt,other.radiation_dt);
        if (other.dt<dt) *this=other;
        radiation_dt=limit;
    }
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & a;
		arc & x;
		arc & y;
		arc & z;
		arc & dim;
		arc & dt;
		arc & radiation_dt;
		arc & ur;
		arc & ul;
	}
};
