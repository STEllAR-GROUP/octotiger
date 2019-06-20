/*
 * rotating_star.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: dmarce1
 */

#ifndef ROTATING_STAR_ROTATING_STAR_HPP_
#define ROTATING_STAR_ROTATING_STAR_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/real.hpp"

#include <vector>

OCTOTIGER_EXPORT std::vector<real> rotating_star(real x, real y, real z, real);

OCTOTIGER_EXPORT std::vector<real> rotating_star_a(real x, real y, real z, real);

#endif /* ROTATING_STAR_ROTATING_STAR_HPP_ */
