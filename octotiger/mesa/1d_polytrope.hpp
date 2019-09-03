//Copyright (c) 2019 Dominic C. Marcello

#ifndef _1D_POLYTROPE_HPP_
#define _1D_POLYTROPE_HPP_

#include <functional>

struct spherical_polytrope {
	std::function<double(double)> rho_of_r;
	std::function<double(double)> p_of_r;
	double p_scale;
	double rho_scale;
};

spherical_polytrope make_1d_spherical_polytrope(
		const std::function<double(double)>& d_of_h_norm, const std::function<double(double)>& p_of_d_norm, double mass,
		double radius);

#endif /* 1D_POLYTROPE_HPP_ */
