/*
 * roe.hpp
 *
 *  Created on: May 28, 2015
 *      Author: dmarce1
 */

#ifndef ROE_HPP_
#define ROE_HPP_

#include "defs.hpp"
#include "space_vector.hpp"

constexpr real de_switch2 = real(1.0e-3);
constexpr real de_switch1 = real(1.0e-1);

real roe_fluxes(std::array<std::vector<real>, NF>& F, std::array<std::vector<real>, NF>& UL,
		std::array<std::vector<real>, NF>& UR,  const std::vector<space_vector>& X, real omega, integer dimension);

#endif /* ROE_HPP_ */
