/*
 * rad_grid.hpp
 *
 *  Created on: May 20, 2016
 *      Author: dmarce1
 */

#ifndef RAD_GRID_HPP_
#define RAD_GRID_HPP_

#include "defs.hpp"

#define R_BW 1
#define R_NX (INX+R_BW)
#define R_N3 (R_NX*R_NX*R_NX)


class rad_grid {
private:
	real dx;
	std::vector<real> source;

public:
	rad_grid(real dx);
};

#endif /* RAD_GRID_HPP_ */
