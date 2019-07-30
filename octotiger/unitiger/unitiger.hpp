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
constexpr int NDIM = 2;
constexpr int H_BW = 3;
constexpr int H_NX = (2 * H_BW + 128);
constexpr int H_DNX = 1;
constexpr int H_DN[3] = { 1, H_NX, H_NX * H_NX };
constexpr int H_DNY = H_NX;
constexpr int H_DNZ = (H_NX * H_NX);
constexpr int H_N3 = std::pow(H_NX, NDIM);
constexpr auto rho_i = 0;
constexpr auto egas_i = 1;
constexpr auto sx_i = 2;
constexpr auto sy_i = 3;
constexpr auto sz_i = 4;
constexpr auto tau_i = 4;
#endif
constexpr int H_DN0 = 0;
constexpr int NDIR = std::pow(3, NDIM);
constexpr int NFACEDIR = std::pow(3, NDIM - 1);
constexpr int ORDER = 3;
constexpr double CFL = (0.4/ORDER/NDIM);

static constexpr char *field_names[] = { "rho", "egas", "sx", "sy", "sz", "tau" };


constexpr int NF = (3 + NDIM);


inline static double bound_width() {
	int bw = 1;
	int next_bw = 1;
	for (int dim = 1; dim < NDIM; dim++) {
		next_bw *= H_NX;
		bw += next_bw;
	}
	return bw;
}


#define FGAMMA (7.0/5.0)

double hydro_flux(std::vector<std::vector<double>> &U, std::vector<std::vector<std::vector<double>>> &F, std::vector<std::array<double,NDIM>>& X, double omega);


static constexpr int directions[3][27] = { {
/**/-H_DNX, +H_DN0, +H_DNX /**/
}, {
/**/-H_DNX - H_DNY, +H_DN0 - H_DNY, +H_DNX - H_DNY,/**/
/**/-H_DNX + H_DN0, +H_DN0 + H_DN0, +H_DNX + H_DN0,/**/
/**/-H_DNX + H_DNY, +H_DN0 + H_DNY, +H_DNX + H_DNY, /**/
}, {
/**/-H_DNX - H_DNY - H_DNZ, +H_DN0 - H_DNY - H_DNZ, +H_DNX - H_DNY - H_DNZ,/**/
/**/-H_DNX + H_DN0 - H_DNZ, +H_DN0 + H_DN0 - H_DNZ, +H_DNX + H_DN0 - H_DNZ,/**/
/**/-H_DNX + H_DNY - H_DNZ, +H_DN0 + H_DNY - H_DNZ, +H_DNX + H_DNY - H_DNZ,/**/
/**/-H_DNX - H_DNY + H_DN0, +H_DN0 - H_DNY + H_DN0, +H_DNX - H_DNY + H_DN0,/**/
/**/-H_DNX + H_DN0 + H_DN0, +H_DN0 + H_DN0 + H_DN0, +H_DNX + H_DN0 + H_DN0,/**/
/**/-H_DNX + H_DNY + H_DN0, +H_DN0 + H_DNY + H_DN0, +H_DNX + H_DNY + H_DN0,/**/
/**/-H_DNX - H_DNY + H_DNZ, +H_DN0 - H_DNY + H_DNZ, +H_DNX - H_DNY + H_DNZ,/**/
/**/-H_DNX + H_DN0 + H_DNZ, +H_DN0 + H_DN0 + H_DNZ, +H_DNX + H_DN0 + H_DNZ,/**/
/**/-H_DNX + H_DNY + H_DNZ, +H_DN0 + H_DNY + H_DNZ, +H_DNX + H_DNY + H_DNZ/**/

} };

#endif /* UNITIGER_HPP_ */
