/*
 * taylor.cpp
 *
 *  Created on: Jun 9, 2015
 *      Author: dmarce1
 */

#include "octotiger/taylor.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/real.hpp"

integer taylor_consts::map2[3][3] = {
    { 0, 1, 2 },
    { 1, 3, 4 },
    { 2, 4, 5 }
};
integer taylor_consts::map3[3][3][3] = {
    { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } },
    { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } },
    { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } }
};
integer taylor_consts::map4[3][3][3][3] = {
    { { { 0,  1,  2 }, { 1,  3,  4 }, { 2,  4,  5 } },
      { { 1,  3,  4 }, { 3,  6,  7 }, { 4,  7,  8 } },
      { { 2,  4,  5 }, { 4,  7,  8 }, { 5,  8,  9 } } },
    { { { 1,  3,  4 }, { 3,  6,  7 }, { 4,  7,  8 } },
      { { 3,  6,  7 }, { 6, 10, 11 }, { 7, 11, 12 } },
      { { 4,  7,  8 }, { 7, 11, 12 }, { 8, 12, 13 } } },
    { { { 2,  4,  5 }, { 4,  7,  8 }, { 5,  8,  9 } },
      { { 4,  7,  8 }, { 7, 11, 12 }, { 8, 12, 13 } },
      { { 5,  8,  9 }, { 8, 12, 13 }, { 9, 13, 14 } } }
};

const real taylor_consts::delta[3][3] = {
    { ONE, ZERO, ZERO },
    { ZERO, ONE, ZERO },
    { ZERO, ZERO, ONE }
};

static /*__attribute__((constructor))*/ void init() {
    // The taylor coefficients for the various dimensions are all stored in an
    // one-dimensional array:
    //     dim  indices
    //      0     0
    //      1     1 ... 3
    //      2     4 ... 9
    //      3    10 ... 19
    //      4    20 ... 34
    for (integer i = 0; i != NDIM; ++i) {
        for (integer j = 0; j != NDIM; ++j) {
            taylor_consts::map2[i][j] += 4;
            for (integer k = 0; k != NDIM; ++k) {
                taylor_consts::map3[i][j][k] += 10;
                for (integer l = 0; l != NDIM; ++l) {
                    taylor_consts::map4[i][j][k][l] += 20;
                }
            }
        }
    }
}

//  integer m = 20;
// 	for (integer i = 0; i != NDIM; ++i) {
// 		for (integer j = i; j != NDIM; ++j) {
// 			for (integer k = j; k != NDIM; ++k) {
// 				for (integer l = k; l != NDIM; ++l) {
// 					taylor_consts::map4[i][j][k][l] = m;
// 					taylor_consts::map4[i][j][l][k] = m;
// 					taylor_consts::map4[i][l][k][j] = m;
// 					taylor_consts::map4[i][k][l][j] = m;
// 					taylor_consts::map4[i][k][j][l] = m;
// 					taylor_consts::map4[i][l][j][k] = m;
// 					taylor_consts::map4[l][j][k][i] = m;
// 					taylor_consts::map4[k][j][l][i] = m;
// 					taylor_consts::map4[j][l][k][i] = m;
// 					taylor_consts::map4[j][k][l][i] = m;
// 					taylor_consts::map4[l][k][j][i] = m;
// 					taylor_consts::map4[k][l][j][i] = m;
// 					taylor_consts::map4[k][j][i][l] = m;
// 					taylor_consts::map4[l][j][i][k] = m;
// 					taylor_consts::map4[k][l][i][j] = m;
// 					taylor_consts::map4[l][k][i][j] = m;
// 					taylor_consts::map4[j][k][i][l] = m;
// 					taylor_consts::map4[j][l][i][k] = m;
// 					taylor_consts::map4[j][i][k][l] = m;
// 					taylor_consts::map4[j][i][l][k] = m;
// 					taylor_consts::map4[l][i][k][j] = m;
// 					taylor_consts::map4[k][i][l][j] = m;
// 					taylor_consts::map4[k][i][j][l] = m;
// 					taylor_consts::map4[l][i][j][k] = m;
// 					++m;
// 				}
// 			}
// 		}
// 	}
// }

struct init_taylor_data
{
    init_taylor_data()
    {
        init();
    }
};
init_taylor_data init_taylor;

