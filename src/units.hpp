/*
 * units.hpp
 *
 *  Created on: Oct 25, 2016
 *      Author: dmarce1
 */

#ifndef UNITS_HPP_
#define UNITS_HPP_


#include "real.hpp"


real get_pressure_factor();
real get_force_factor();
real get_energy_factor();
real get_length_factor();
real get_length_factor();
real get_mass_factor();
real get_time_factor();
void set_length_factor(real);
void set_mass_factor(real);
void set_time_factor(real);

#endif /* UNITS_HPP_ */
