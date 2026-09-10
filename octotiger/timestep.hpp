// Shared timestep diagnostics for hydro and radiation integration.
#pragma once
#include <limits>
#include <vector>

struct timestep_t {
	double a = 0;
	double x = 0, y = 0, z = 0;
	double dt = std::numeric_limits<double>::max();
	int dim = -1;
	std::vector<double> ur;
	std::vector<double> ul;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & a;
		arc & x;
		arc & y;
		arc & z;
		arc & dim;
		arc & dt;
		arc & ur;
		arc & ul;
	}
};
