/*
 * types.hpp
 *
 *  Created on: May 26, 2015
 *      Author: dmarce1
 */

#include <hpx/config.hpp>

#ifdef OCTOTIGER_HAVE_GRAV_PAR
# define USE_GRAV_PAR
#endif

#ifdef OCTOTIGER_HAVE_SILO
# define DO_OUTPUT
#endif

//#define OCTOTIGER_HAVE_RADIATION

#ifdef OCTOTIGER_HAVE_RADIATION
#define RADIATION
#endif

#ifndef TYPES444_HPP_

//#define CWD
#define BIBI
//#define OLD_SCF

//#define WD_EOS


#define EXPERIMENT
#ifdef RADIATION
#define NRF 4
#else
#define NRF 0
#define NRADF 0
#endif

#define LIGHTSPEED 1.0

#define NPF 5


#define abort_error() printf( "Error in %s on line %i\n", __FILE__, __LINE__); abort()



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

constexpr real DEFAULT_OMEGA = 0.0;

//const integer MAX_LEVEL = 5;

enum boundary_type {
	OUTFLOW, REFLECT
};

constexpr integer NDIM = 3;

constexpr integer NSPECIES = 5;

constexpr integer INX = 8;
constexpr integer H_BW = 3;
constexpr integer R_BW = 2;

constexpr integer H_NX = 2 * H_BW + INX;
constexpr integer G_NX = INX;
constexpr integer H_N3 = H_NX * H_NX * H_NX;
constexpr integer F_N3 = ((INX+1)*(INX+1)*(INX+1));
constexpr integer G_N3 = G_NX * G_NX * G_NX;

constexpr integer F_DNX = (INX+1)*(INX+1);
constexpr integer F_DNZ = 1;
constexpr integer F_DNY = (INX+1);


constexpr integer NF = 15;
constexpr integer NDIR = 27;
constexpr integer H_DNX = H_NX * H_NX;
constexpr integer H_DNY = H_NX;
constexpr integer H_DNZ = 1;
constexpr integer H_DN[NDIM] = { H_NX * H_NX, H_NX, 1 };
constexpr integer F_DN[NDIM] = {(INX+1)*(INX+1),INX+1,1 };
constexpr integer G_DNX = G_NX * G_NX;
constexpr integer G_DNY = G_NX;
constexpr integer G_DNZ = 1;
constexpr integer G_DN[NDIM] = { G_NX * G_NX, G_NX, 1 };

constexpr integer rho_i = 0;
constexpr integer egas_i = 1;
constexpr integer sx_i = 2;
constexpr integer sy_i = 3;
constexpr integer sz_i = 4;
constexpr integer tau_i = 5;
constexpr integer pot_i = 6;
constexpr integer zx_i = 7;
constexpr integer zy_i = 8;
constexpr integer zz_i = 9;
constexpr integer spc_i = 10;
constexpr integer spc_ac_i = 10;
constexpr integer spc_ae_i = 11;
constexpr integer spc_dc_i = 12;
constexpr integer spc_de_i = 13;
constexpr integer spc_vac_i = 14;


constexpr integer vx_i = sx_i;
constexpr integer vy_i = sy_i;
constexpr integer vz_i = sz_i;
constexpr integer h_i = egas_i;

constexpr integer XDIM = 0;
constexpr integer YDIM = 1;
constexpr integer ZDIM = 2;

constexpr integer FXM = 0;
constexpr integer FXP = 1;
constexpr integer FYM = 2;
constexpr integer FYP = 3;
constexpr integer FZM = 4;
constexpr integer FZP = 5;

constexpr integer NFACE = 2 * NDIM;
constexpr integer NVERTEX = 8;
constexpr integer NCHILD = 8;

constexpr real ZERO = real(0);
constexpr real ONE = real(1);
constexpr real TWO = real(2);
constexpr real THREE = real(3);
constexpr real FOUR = real(4);

constexpr real HALF = real(real(1) / real(2));
constexpr real SIXTH = real(real(1) / real(6));
constexpr real TWELFTH = real(real(1) / real(12));

constexpr real cfl = real(real(2) / real(15));
constexpr real ei_floor = 1.0e-15;
constexpr integer NRK = 2;
constexpr real rk_beta[2] = { ONE, HALF };

constexpr integer NGF = 4;
constexpr integer phi_i = 0;
constexpr integer gx_i = 1;
constexpr integer gy_i = 2;
constexpr integer gz_i = 3;

constexpr std::array<boundary_type, NFACE> boundary_types = { OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW };

// #define h0index(i,j,k) ((i)*INX*INX+(j)*INX+(k))
constexpr inline integer h0index(integer i, integer j, integer k)
{
    return i * INX * INX + j * INX + k;
}

// #define hindex(i,j,k) ((i)*H_DNX + (j)*H_DNY + (k)*H_DNZ)
constexpr inline integer hindex(integer i, integer j, integer k)
{
    return i * H_DNX + j * H_DNY + k * H_DNZ;
}

// #define findex(i,j,k) ((i)*(INX+1)*(INX+1) + (j)*(INX+1) + (k))
constexpr inline integer findex(integer i, integer j, integer k)
{
    return i * (INX + 1) * (INX + 1) + j * (INX + 1) + k;
}

// #define gindex(i,j,k) ((i)*G_DNX + (j)*G_DNY + (k)*G_DNZ)
constexpr inline integer gindex(integer i, integer j, integer k)
{
    return i * G_DNX + j * G_DNY + k * G_DNZ;
}

template <typename T>
constexpr inline T sqr(T const& val)
{
    return val * val;
}

template <typename T>
constexpr inline T cube(T const& val)
{
    return val * val * val;
}

template <typename T>
constexpr inline T average(T const& s1, T const& s2)
{
    return 0.5 * (s1 + s2);
};

template <typename T>
constexpr inline void inplace_average(T& s1, T& s2)
{
    s1 = s2 = average(s1, s2);
};

/*
#define SYSTEM(command) \
	if( system( (command).c_str()) != 0) { \
		printf( "System command \"%s\" failed in %s on line %i\n", (command).c_str(), __FILE__, __LINE__); \
		abort(); \
	}
*/
#endif /* TYPES_HPP_ */
