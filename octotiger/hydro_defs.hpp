/*
 * field_defs.hpp
 *
 *  Created on: Jul 29, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_HYDRO_DEFS_HPP_
#define OCTOTIGER_HYDRO_DEFS_HPP_

constexpr int NDIM = 3;


constexpr int rho_i = 0;
constexpr int egas_i = 1;
constexpr int sx_i = 2;
constexpr int sy_i = 3;
constexpr int sz_i = 4;
constexpr int tau_i = 5;
constexpr int pot_i = 6;
constexpr int zx_i = 7;
constexpr int zy_i = 8;
constexpr int zz_i = 9;
constexpr int spc_i = 10;

constexpr int INX = OCTOTIGER_GRIDDIM;
constexpr int H_BW = 3;
constexpr int H_NX = 2 * H_BW + INX;
constexpr int H_DNX = H_NX * H_NX;
constexpr int H_DNY = H_NX;
constexpr int H_DNZ = 1;
constexpr int H_DN[NDIM] = { H_NX * H_NX, H_NX, 1 };
constexpr int H_N3 = H_NX * H_NX * H_NX;


#endif /* OCTOTIGER_HYDRO_DEFS_HPP_ */
