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

using namespace std;

static constexpr double tmax = 1.0e-1;

#ifdef OCTOTIGER_GRIDDIM
#include "octotiger/hydro_defs.hpp"
#else
//constexpr int NDIM = 2;

#endif
constexpr int ORDER = 3;

static constexpr char const *field_names3[] = { "rho", "egas", "tau", "pot", "sx", "sy", "sz", "zx", "zy", "zz", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5" };
static constexpr char const *field_names2[] = { "rho", "egas", "tau", "pot", "sx", "sy", "zz", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5" };
static constexpr char const *field_names1[] = { "rho", "egas", "tau", "pot", "sx", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5" };

#define FGAMMA (7.0/5.0)

#endif /* UNITIGER_HPP_ */
