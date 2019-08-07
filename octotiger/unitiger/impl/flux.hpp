
#include "../physics.hpp"

template<int NDIM, int INX>
safe_real hydro_computer<NDIM, INX>::flux(const hydro::state_type& U, const hydro::recon_type<NDIM>& Q,
		hydro::flux_type &F, hydro::x_type<NDIM> &X,
		safe_real omega) {

	static const auto indices2 = geo::find_indices(2, geo::H_NX - 2);
	static constexpr auto faces = geo::face_pts();
	static constexpr auto weights = geo::face_weight();
	static constexpr auto xloc = geo::xloc();
	static constexpr auto kdelta = geo::kronecker_delta();
	static constexpr auto dir = geo::direction();

	const auto dx = X[0][geo::H_DNX] - X[0][0];

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

	std::array < safe_real, 3 > amax = {0.0, 0.0, 0.0};
	for (int dim = 0; dim < NDIM; dim++) {
		std::vector<safe_real> UR(nf_), UL(nf_), this_flux(nf_);
		std::vector<safe_real> UR0(nf_), UL0(nf_);
		for (const auto &i : indices2) {
			safe_real ap = -1.0;
			safe_real am = +1.0;
			for (int fi = 0; fi < geo::NFACEDIR; fi++) {
				const auto d = faces[dim][fi];
				const auto di = dir[d];
				for (int f = 0; f < nf_; f++) {
					UR0[f] = U[f][i];
					UL0[f] = U[f][i-geo::H_DN[dim]];
					UR[f] = Q[f][i][d];
					UL[f] = Q[f][i - geo::H_DN[dim]][flip_dim(d, dim)];
				}
				std::array < safe_real, NDIM > vg;
				if constexpr (NDIM > 1) {
					vg[0] = -omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
					vg[1] = +omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
					vg[2] = 0.0;
				} else {
					vg[0] = 0.0;
				}


				physics < NDIM > ::flux(UL, UR, UL0, UR0, this_flux, dim, am, ap, vg, dx);
				for (int f = 0; f < nf_; f++) {
					fluxes[dim][f][i][fi] = this_flux[f];
				}
			}
			amax[dim] = std::max(ap, amax[dim]);
		}
		for (int f = 0; f < nf_; f++) {
			for (const auto &i : geo::find_indices(3, geo::H_NX - 2)) {
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
								for (const auto &i : geo::find_indices(3, geo::H_NX - 2)) {
									F[dim][zx_i + n][i] += weights[fi] * kdelta[n][m][l] * xloc[d][m] * 0.5 * dx
									* fluxes[dim][sx_i + l][i][fi];
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
