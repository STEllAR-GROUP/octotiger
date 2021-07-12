//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(DEFS_HPP_)
#define DEFS_HPP_

#include "octotiger/real.hpp"

#include <array>
//#include <cstddef>
//#include <iostream>

//#define FACES_ONLY
//#define CHECK_FLUXES

constexpr double MARSHAK_OPAC = 1.0e+2;
//#define OCTOTIGER_RESTART_LOAD_SEQ

//#define OCTOTIGER_USE_NODE_CACHE

//#define OCTOTIGER_FLUX_CHECK

constexpr real rho_floor = 1.0e-12;
constexpr int REFINE_BW = 2;

//#define FIND_AXIS_V2

//#define CWD
//#define BIBI
//#define OLD_SCF

//#define WD_EOS

//#define USE_NIECE_BOOL


//#define EXPERIMENT
constexpr int NRF = 7;

#define abort_error() printf( "Error in %s on line %i\n", __FILE__, __LINE__); abort()



//#define USE_SIMD

//#define ZCORRECT

//#define USE_PPM
//#define USE_MINMOD


//#define DISABLE_VERTEX_AVG


#if !defined(OCTOTIGER_FORCEINLINE)
#   if defined(__NVCC__) || defined(__CUDACC__)
#       define OCTOTIGER_FORCEINLINE inline
#   elif defined(_MSC_VER)
#       define OCTOTIGER_FORCEINLINE __forceinline
#   elif defined(__GNUC__)
#       define OCTOTIGER_FORCEINLINE inline __attribute__ ((__always_inline__))
#   else
#       define OCTOTIGER_FORCEINLINE inline
#   endif
#endif

using integer = long long int;

//using byte = unsigned char;

enum gsolve_type {
	RHO, DRHODT
};


//#define USE_ROTATING_FRAME
//#define OUTPUT_FREQ (100.0)

//#define USE_SPHERICAL
//constexpr integer M_POLES = 3;
//constexpr integer L_POLES = M_POLES;



//#define GRID_SIZE real(2.0)

constexpr real DEFAULT_OMEGA = 0.0;

//const integer MAX_LEVEL = 5;

#include "hydro_defs.hpp"

constexpr integer R_BW = 2;

constexpr integer HS_NX = (2 * H_BW + INX/2);
constexpr integer G_NX = INX;
constexpr integer HS_N3 = HS_NX * HS_NX * HS_NX;
constexpr integer F_N3 = ((INX+1)*(INX+1)*(INX+1));
constexpr integer G_N3 = G_NX * G_NX * G_NX;

constexpr integer F_DNX = (INX+1)*(INX+1);
constexpr integer F_DNZ = 1;
constexpr integer F_DNY = (INX+1);


constexpr integer NDIR = 27;
constexpr integer HS_DNX = HS_NX * HS_NX;
constexpr integer HS_DNY = HS_NX;
constexpr integer HS_DNZ = 1;
constexpr integer HS_DN[NDIM] = { HS_NX * HS_NX, HS_NX, 1 };
constexpr integer F_DN[NDIM] = {(INX+1)*(INX+1),INX+1,1 };
constexpr integer G_DNX = G_NX * G_NX;
constexpr integer G_DNY = G_NX;
constexpr integer G_DNZ = 1;
constexpr integer G_DN[NDIM] = { G_NX * G_NX, G_NX, 1 };

// Radiation {{{
constexpr integer RAD_BW = 3;
constexpr integer RAD_NX = INX + 2 * RAD_BW;
constexpr integer RAD_N3 = RAD_NX * RAD_NX * RAD_NX;

constexpr inline integer rindex(integer x, integer y, integer z)
{
    return z + RAD_NX * (y + RAD_NX * x);
}
// }}}


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

constexpr real ei_floor = 1.0e-15;

#define USE_RK3

#ifdef USE_RK3
constexpr integer NRK = 3;
constexpr real rk_beta[3] = { 1.0, 1.0/4.0, 2.0/3.0 };
#else
constexpr integer NRK = 2;
constexpr real rk_beta[2] = { ONE, HALF };
#endif


constexpr integer MAX_LEVEL = 21;

constexpr integer NGF = 4;
constexpr integer phi_i = 0;
constexpr integer gx_i = 1;
constexpr integer gy_i = 2;
constexpr integer gz_i = 3;

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

// #define hindex(i,j,k) ((i)*H_DNX + (j)*H_DNY + (k)*H_DNZ)
constexpr inline integer hSindex(integer i, integer j, integer k)
{
    return i * HS_DNX + j * HS_DNY + k * HS_DNZ;
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
inline void inplace_average(T& s1, T& s2)
{
    s1 = s2 = average(s1, s2);
};

/*
#define SYSTEM(command)                                                        \
    if (system((command).c_str()) != 0)                                        \
    {                                                                          \
        printf("System command \"%s\" failed in %s on line %i\n",              \
            (command).c_str(), __FILE__, __LINE__);                            \
        abort();                                                               \
    }
*/

//template <typename T>
//std::size_t write(std::ostream& strm, T && t)
//{
//    using output_type = typename std::decay<T>::type;
//    strm.write(reinterpret_cast<char const*>(&t), sizeof(output_type));
//    return sizeof(output_type);
//}
//
//template <typename T>
//std::size_t write(std::ostream& strm, T* t, std::size_t size)
//{
//    strm.write(reinterpret_cast<char const*>(t), sizeof(T) * size);
//    return sizeof(T) * size;
//}

//template <typename T>
//std::size_t read(std::istream& strm, T & t)
//{
//    using input_type = typename std::decay<T>::type;
//    strm.read(reinterpret_cast<char*>(&t), sizeof(input_type));
//    return sizeof(input_type);
//}
//
//template <typename T>
//std::size_t read(std::istream& strm, T* t, std::size_t size)
//{
//    strm.read(reinterpret_cast<char*>(t), sizeof(T) * size);
//    return sizeof(T) * size;
//}

#endif
