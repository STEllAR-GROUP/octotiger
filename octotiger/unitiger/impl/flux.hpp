#ifdef NOHPX
#include "../basis.hpp"
#else
#include "octotiger/unitiger/basis.hpp"
#endif

#include "../physics.hpp"

template<int NDIM, int INX, int ORDER>
safe_real hydro_computer<NDIM, INX, ORDER>::flux(const std::vector<std::vector<std::array<safe_real, geo::NDIR>>> &Q,
		std::vector<std::vector<std::vector<safe_real>>> &F, std::vector<std::array<safe_real, NDIM>> &X, safe_real omega) {

	static constexpr auto faces = geo::lower_face_members[NDIM - 1];
	static constexpr auto weights = geo::quad_weights[NDIM - 1];
	static constexpr auto face_loc = geo::face_locs[NDIM - 1];
	static constexpr auto kdelta = geo::kdeltas[NDIM - 1];
	static constexpr auto dx = 1.0 / INX;

	static constexpr auto dir = geo::directions[NDIM - 1];

	int bw = bound_width();

	const auto flip_dim = [](const int d, int flip_dim) {
		std::array<int, NDIM> dims;
		int k = d;
		for (int dim = 0; dim < NDIM; dim++) {
			dims[dim] = k % 3;
			k /= 3;
		}
		k = 0;
//		printf( "%i | %i %i %i | ", flip_dim, d, dims[0], dims[1]);
		dims[flip_dim] = 2 - dims[flip_dim];
		for (int dim = 0; dim < NDIM; dim++) {
			k *= 3;
			k += dims[NDIM - 1 - dim];
		}
		//	printf( "%i %i %i\n", k,  dims[0], dims[1]);
		return k;
	};

	std::array < safe_real, 3 > amax = { 0.0, 0.0, 0.0 };
	for (int dim = 0; dim < NDIM; dim++) {
		std::vector<safe_real> UR(nf), UL(nf), this_flux(nf);
		for (const auto &i : find_indices(2, geo::H_NX - 2)) {
			safe_real a = -1.0;
			for (int fi = 0; fi < geo::NFACEDIR; fi++) {
				const auto d = faces[dim][fi];
				const auto di = dir[d];
				for (int f = 0; f < nf; f++) {
					UR[f] = Q[f][i][d];
					UL[f] = Q[f][i - geo::H_DN[dim]][flip_dim(d, dim)];
				}
				std::array < safe_real, NDIM > vg;
				if constexpr (NDIM > 1) {
					vg[0] = -0.5 * omega * (X[i][1] + X[i - geo::H_DN[dim]][1]);
					vg[1] = +0.5 * omega * (X[i][0] + X[i - geo::H_DN[dim]][0]);
					if constexpr (NDIM == 3) {
						vg[2] = 0.0;
					}
				}
				physics < NDIM > ::flux(UL, UR, this_flux, dim, a, vg, dx);
				for (int f = 0; f < nf; f++) {
					fluxes[dim][f][i][fi] = this_flux[f];
				}
			}
			amax[dim] = std::max(a, amax[dim]);
		}
		for (int f = 0; f < nf; f++) {
			for (const auto &i : find_indices(3, geo::H_NX - 2)) {
				F[dim][f][i] = 0.0;
				for (int fi = 0; fi < geo::NFACEDIR; fi++) {
					F[dim][f][i] += weights[fi] * fluxes[dim][f][i][fi];
				}
			}
		}
		for (int angmom_pair = 0; angmom_pair < angmom_count_; angmom_pair++) {
			const int sx_i = angmom_index_ + angmom_pair * (NDIM + geo::NANGMOM);
			const int zx_i = sx_i + NDIM;
			for (int n = 0; n < geo::NANGMOM; n++) {
				for (int m = 0; m < NDIM; m++) {
					if (dim != m) {
						for (int l = 0; l < NDIM; l++) {
							for (int fi = 0; fi < geo::NFACEDIR; fi++) {
								const auto d = faces[dim][fi];
								for (const auto &i : find_indices(3, geo::H_NX - 2)) {
									F[dim][zx_i + n][i] += weights[fi] * kdelta[n][m][l] * face_loc[d][m] * 0.5 * dx * fluxes[dim][sx_i + l][i][fi];
								}
							}
						}
					}
				}
			}
		}
	}
	for (int d = 1; d < NDIM; d++) {
		amax[0] = std::max(amax[0], amax[d]);
	}
	return amax[0];
}
