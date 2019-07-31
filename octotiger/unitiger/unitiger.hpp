/*
 * unitiger.hpp
 *
 *  Created on: Jul 29, 2019
 *      Author: dmarce1
 */

#ifndef UNITIGER_HPP_
#define UNITIGER_HPP_



#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <cassert>
#include <limits>
#include <silo.h>

#define THETA 1.3
using namespace std;

static constexpr double tmax = 1e-4;

#ifdef OCTOTIGER_GRIDDIM
#include "octotiger/hydro_defs.hpp"
#else
//constexpr int NDIM = 2;

constexpr auto rho_i = 0;
constexpr auto egas_i = 1;
constexpr auto sx_i = 2;
constexpr auto sy_i = 3;
constexpr auto sz_i = 4;
constexpr auto tau_i = 4;
#endif
constexpr int ORDER = 3;

static constexpr char *field_names[] = { "rho", "egas", "sx", "sy", "sz", "tau" };




#define FGAMMA (7.0/5.0)


#endif /* UNITIGER_HPP_ */
