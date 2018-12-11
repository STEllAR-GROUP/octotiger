/*
 * implicit.hpp
 *
 *  Created on: Sep 25, 2018
 *      Author: dmarce1
 */

#ifndef SRC_RADIATION_IMPLICIT_HPP_
#define SRC_RADIATION_IMPLICIT_HPP_

std::pair<real,space_vector> implicit_radiation_step_2nd_order(real E0, real& e0, const space_vector& F0, const space_vector& u0, real rho, real mmw, real X, real Z,  real dt);




#endif /* SRC_RADIATION_IMPLICIT_HPP_ */
