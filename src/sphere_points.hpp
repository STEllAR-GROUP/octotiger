/*
 * sphere_sphere_points.hpp
 *
 *  Created on: Jun 4, 2016
 *      Author: dmarce1
 */

#ifndef SPHERE_POINTS_HPP_
#define SPHERE_POINTS_HPP_

/*
 * main.cpp
 *
 *  Created on: May 24, 2016
 *      Author: dmarce1
 */

#include "defs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <array>
#include <vector>
#include "geometry.hpp"

struct sphere_point {
	real nx;
	real ny;
	real nz;
	real wx;
	real wy;
	real wz;
	real dA;
	real dl;
	geo::octant get_octant() const;
	void normalize();
	real norm() const;
	sphere_point operator/(real den) const;
};

sphere_point operator+(const sphere_point& pt1, const sphere_point& pt2);

sphere_point operator-(const sphere_point& pt1, const sphere_point& pt2);

struct triangle {
	sphere_point A, B, C;
	void normalize();
	real area() const;
	void split_triangle(std::vector<sphere_point>& sphere_points, int this_lev, int max_lev);
};

std::vector<sphere_point> generate_sphere_points(int nlev);

#endif /* SPHERE_POINTS_HPP_ */
