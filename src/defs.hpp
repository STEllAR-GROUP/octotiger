/*
 * types.hpp
 *
 *  Created on: May 26, 2015
 *      Author: dmarce1
 */


#ifndef TYPES444_HPP_

//#define SCF
//#define SILO

#ifdef MINI_HPX
#include "./hpx_lite/hpx_lite.hpp"
#else
#include <hpx/hpx.hpp>
#endif

typedef double real;
typedef long long int integer;
#define TYPES444_HPP_

typedef char byte;

enum gsolve_type {
	RHO, DRHODT
};

#ifndef MINI_HPX
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/set.hpp>
#include <hpx/runtime/serialization/array.hpp>
#include <hpx/runtime/serialization/vector.hpp>

namespace hpx {
using mutex = hpx::lcos::local::spinlock;
}

#endif

#define USE_ROTATING_FRAME

const real DEFAULT_OMEGA = 0.142194022;

const integer MAX_LEVEL = 4;
enum boundary_type {OUTFLOW, REFLECT};

const integer NDIM = 3;

const integer NSPECIES = 2;

const integer HBW = 2;
const integer GBW = 2;
const integer INX = 8;
const integer HNX = 2 * HBW + INX;
const integer GNX = 2 * GBW + INX;
const integer HN3 = HNX * HNX * HNX;
const integer NF = 12;
const integer NDIR = 27;
const integer DNX = HNX * HNX;
const integer DNY = HNX;
const integer DNZ = 1;
const integer DN[NDIM] = { HNX * HNX, HNX, 1 };



const integer rho_i = 0;
const integer egas_i = 1;
const integer sx_i = 2;
const integer sy_i = 3;
const integer sz_i = 4;
const integer tau_i = 5;
const integer pot_i = 6;
const integer zx_i = 7;
const integer zy_i = 8;
const integer zz_i = 9;
const integer acc_i = 10;
const integer don_i = 11;

const integer vx_i = sx_i;
const integer vy_i = sy_i;
const integer vz_i = sz_i;
const integer h_i = egas_i;

const integer XDIM = 0;
const integer YDIM = 1;
const integer ZDIM = 2;

const integer FXM = 0;
const integer FXP = 1;
const integer FYM = 2;
const integer FYP = 3;
const integer FZM = 4;
const integer FZP = 5;

const integer NFACE = 2 * NDIM;
const integer NVERTEX = 8;
const integer NCHILD = 8;
const real fgamma = real(5) / real(3);

const real ZERO = real(0);
const real ONE = real(1);
const real TWO = real(2);

const real HALF = real(1) / real(2);
const real TWELFTH = real(1) / real(12);

const real cfl = real(2) / real(15);
const real ei_floor = 1.0e-15;
const integer NRK = 2;
const real rk_beta[2] = { ONE, HALF };

const integer NGF = 4;
const integer phi_i = 0;
const integer gx_i = 1;
const integer gy_i = 2;
const integer gz_i = 3;


const std::array<boundary_type, NFACE> boundary_types = {OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW};

#define SYSTEM(command) \
	if( system( (command).c_str()) != 0) { \
		printf( "System command \"%s\" failed in %s on line %i\n", (command).c_str(), __FILE__, __LINE__); \
		abort(); \
	}

#endif /* TYPES_HPP_ */
