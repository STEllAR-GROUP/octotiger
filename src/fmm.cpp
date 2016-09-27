/*
 * fmm.cpp
 *
 *  Created on: Sep 27, 2016
 *      Author: dmarce1
 */

#include "fmm.hpp"

void fmm::set_particles(const std::vector<real>& rho) {
	for (integer i = 0; i != NMX; ++i) {
		for (integer j = 0; j != NMX; ++j) {
			for (integer k = 0; k != NMX; ++k) {
				const integer iii = mindex(i, j, k);
				for (integer i0 = 0; i0 != NPX; ++i0) {
					for (integer j0 = 0; j0 != NPX; ++j0) {
						for (integer k0 = 0; k0 != NPX; ++k0) {
							const integer iiih = hindex(NPX * i + i0 + H_BW, NPX * j + j0 + H_BW, NPX * k + k0 + H_BW);
							P[iii].set_mass(i0, j0, k0, rho[iiih]);
						}
					}
				}
				M[iii] = P[iii].get_multipole(dx);
			}
		}
	}
}

std::vector<taylor<ORDER, real>> fmm::get_multipoles() const {
	std::vector < taylor < ORDER, real >> data;
	data.reserve(NM3 / 8);
	taylor<ORDER, real> Mc;
	for (integer i = 0; i != NMX; i += 2) {
		for (integer j = 0; j != NMX; j += 2) {
			for (integer k = 0; k != NMX; k += 2) {
				Mc = real(0);
				for (const auto& ci : geo::octant::full_set()) {
					const integer iii = mindex(i + ci.get_side(XDIM), j + ci.get_side(YDIM), k + ci.get_side(ZDIM));
					space_vector x;
					x[XDIM] = (ci.get_side(XDIM) - real(0.5)) * dx;
					x[YDIM] = (ci.get_side(YDIM) - real(0.5)) * dx;
					x[ZDIM] = (ci.get_side(ZDIM) - real(0.5)) * dx;
					Mc += M[iii] >> x;
				}
				data.push_back(Mc);
			}
		}
	}
	return data;
}

void fmm::set_multipoles(const std::vector<taylor<ORDER, real>>& data, const geo::octant& ci) {
	integer index = 0;
	for (integer i = 0; i != NMX / 2; i++) {
		for (integer j = 0; j != NMX / 2; j++) {
			for (integer k = 0; k != NMX / 2; k++) {
				const integer i0 = ci.get_side(XDIM) * NMX / 2 + i;
				const integer j0 = ci.get_side(YDIM) * NMX / 2 + j;
				const integer k0 = ci.get_side(ZDIM) * NMX / 2 + k;
				const integer iii = mindex(i0,j0,k0);
				M[iii] = data[index++];
			}
		}
	}
}

