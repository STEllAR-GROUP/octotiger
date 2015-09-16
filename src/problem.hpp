/*
 * problem.hpp
 *
 *  Created on: May 29, 2015
 *      Author: dmarce1
 */

#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include "defs.hpp"
#include <vector>

std::vector<real> sod_shock_tube(real,real,real);
std::vector<real> star(real,real,real);
std::vector<real> equal_mass_binary(real,real,real);
std::vector<real> solid_sphere(real,real,real);



const auto problem = equal_mass_binary;

#endif /* PROBLEM_HPP_ */
