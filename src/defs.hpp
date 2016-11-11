/*
 * types.hpp
 *
 *  Created on: May 26, 2015
 *      Author: dmarce1
 */

#include <hpx/config.hpp>

#ifndef TYPES444_HPP_

//#define CWD
#define BIBI
//#define OLD_SCF

//#define WD_EOS


#define EXPERIMENT

#ifdef OCTOTIGER_HAVE_SILO
# define DO_OUTPUT
#endif

#define USE_SIMD

//#define USE_DRIVING
#define DRIVING_RATE 0.01
#define DRIVING_TIME 1.00

#define ZCORRECT

#define USE_PPM
//#define USE_MINMOD

//#include <hpx/hpx.hpp>

//namespace hpx {
//using mutex = hpx::lcos::local::spinlock;
//}

#include "real.hpp"
typedef long long int integer;
#define TYPES444_HPP_

typedef unsigned char byte;

enum gsolve_type {
	RHO, DRHODT
};

#include <array>

//#include <hpx/runtime/serialization/serialize.hpp>
//#include <hpx/runtime/serialization/list.hpp>
//#include <hpx/runtime/serialization/set.hpp>
//#include <hpx/runtime/serialization/array.hpp>
//#include <hpx/runtime/serialization/vector.hpp>
//#include <hpx/runtime/serialization/shared_ptr.hpp>
//#include <mutex>

#define USE_ROTATING_FRAME
//#define OUTPUT_FREQ (100.0)

//#define USE_SPHERICAL
constexpr integer M_POLES = 3;
constexpr integer L_POLES = M_POLES;



//#define GRID_SIZE real(2.0)

const real DEFAULT_OMEGA = 0.0;

//const integer MAX_LEVEL = 5;

enum boundary_type {
	OUTFLOW, REFLECT
};

const integer NDIM = 3;

const integer NSPECIES = 5;

const integer INX = 8;
const integer H_BW = 3;
const integer R_BW = 2;

const integer H_NX = 2 * H_BW + INX;
const integer G_NX = INX;
const integer H_N3 = H_NX * H_NX * H_NX;
const integer F_N3 = ((INX+1)*(INX+1)*(INX+1));
const integer G_N3 = G_NX * G_NX * G_NX;


#define h0index(i,j,k) ((i)*INX*INX+(j)*INX+(k))
#define hindex(i,j,k) ((i)*H_DNX + (j)*H_DNY + (k)*H_DNZ)
#define findex(i,j,k) ((i)*(INX+1)*(INX+1) + (j)*(INX+1) + (k))
#define gindex(i,j,k) ((i)*G_DNX + (j)*G_DNY + (k)*G_DNZ)


const integer F_DNX = (INX+1)*(INX+1);
const integer F_DNZ = 1;
const integer F_DNY = (INX+1);


const integer NF = 15;
const integer NDIR = 27;
const integer H_DNX = H_NX * H_NX;
const integer H_DNY = H_NX;
const integer H_DNZ = 1;
const integer H_DN[NDIM] = { H_NX * H_NX, H_NX, 1 };
const integer F_DN[NDIM] = {(INX+1)*(INX+1),INX+1,1 };
const integer G_DNX = G_NX * G_NX;
const integer G_DNY = G_NX;
const integer G_DNZ = 1;
const integer G_DN[NDIM] = { G_NX * G_NX, G_NX, 1 };

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
const integer spc_i = 10;
const integer spc_ac_i = 10;
const integer spc_ae_i = 11;
const integer spc_dc_i = 12;
const integer spc_de_i = 13;
const integer spc_vac_i = 14;

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

const real ZERO = real(0);
const real ONE = real(1);
const real TWO = real(2);
const real THREE = real(3);
const real FOUR = real(4);

const real HALF = real(real(1) / real(2));
const real TWELFTH = real(real(1) / real(12));

const real cfl = real(real(2) / real(15));
const real ei_floor = 1.0e-15;
const integer NRK = 2;
const real rk_beta[2] = { ONE, HALF };

const integer NGF = 4;
const integer phi_i = 0;
const integer gx_i = 1;
const integer gy_i = 2;
const integer gz_i = 3;

const std::array<boundary_type, NFACE> boundary_types = { OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW };
/*
#define SYSTEM(command) \
	if( system( (command).c_str()) != 0) { \
		printf( "System command \"%s\" failed in %s on line %i\n", (command).c_str(), __FILE__, __LINE__); \
		abort(); \
	}
*/
#endif /* TYPES_HPP_ */
