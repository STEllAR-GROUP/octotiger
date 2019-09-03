//  Copyright (c) 2019 Dominic C Marcello
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::boundaries(hydro::state_type &U) {

	for (int f = 0; f < nf_; f++) {
		if CONSTEXPR(NDIM == 1) {
			for (int i = 0; i < geo::H_BW; i++) {
				if (bc_[0] == OUTFLOW) {
					U[f][i] = U[f][geo::H_BW];
				} else {
					U[f][i] = U[f][i + INX];
				}

				if (bc_[1] == OUTFLOW) {
					U[f][geo::H_NX - 1 - i] = U[f][geo::H_NX - geo::H_BW - 1];
				} else {
					U[f][geo::H_NX - 1 - i] = U[f][2 * geo::H_BW - 1 - i];
				}

			}

		} else if CONSTEXPR(NDIM == 2) {

			const auto index = [](int i, int j) {
				return j + geo::H_NX * i;
			};

			for (int i = 0; i < geo::H_BW; i++) {
				for (int j = 0; j < geo::H_NX; j++) {
					int j0 = j;
					j0 = std::max(j0, geo::H_BW);
					j0 = std::min(j0, geo::H_NX - 1 - geo::H_BW);

					if (bc_[0] == OUTFLOW) {
						U[f][index(i, j)] = U[f][index(geo::H_BW, j0)];
					} else {
						U[f][index(i, j)] = U[f][index(i + INX, j)];
					}

					if (bc_[2] == OUTFLOW) {
						U[f][index(j, i)] = U[f][index(j0, geo::H_BW)];
					} else {
						U[f][index(j, i)] = U[f][index(j, i + INX)];
					}

					if (bc_[1] == OUTFLOW) {
						U[f][index(geo::H_NX - 1 - i, j)] = U[f][index(geo::H_NX - 1 - geo::H_BW, j0)];
					} else {
						U[f][index(geo::H_NX - 1 - i, j)] = U[f][index(2 * geo::H_BW - 1 - i, j)];
					}

					if (bc_[3] == OUTFLOW) {
						U[f][index(j, geo::H_NX - 1 - i)] = U[f][index(j0, geo::H_NX - 1 - geo::H_BW)];
					} else {
						U[f][index(j, geo::H_NX - 1 - i)] = U[f][index(j, 2 * geo::H_BW - 1 - i)];
					}

				}
			}
		} else {
			const auto index = [](int i, int j, int k) {
				return k + geo::H_NX * j + i * geo::H_NX * geo::H_NX;
			};

			for (int i = 0; i < geo::H_BW; i++) {
				for (int j = 0; j < geo::H_NX; j++) {
					for (int k = 0; k < geo::H_NX; k++) {
						int j0 = j;
						j0 = std::max(j0, geo::H_BW);
						j0 = std::min(j0, geo::H_NX - 1 - geo::H_BW);
						int k0 = k;
						k0 = std::max(k0, geo::H_BW);
						k0 = std::min(k0, geo::H_NX - 1 - geo::H_BW);

						if (bc_[0] == OUTFLOW) {
							U[f][index(i, j, k)] = U[f][index(geo::H_BW, j0, k0)];
						} else {
							U[f][index(i, j, k)] = U[f][index(i + INX, j, k)];
						}

						if (bc_[2] == OUTFLOW) {
							U[f][index(j, i, k)] = U[f][index(j0, geo::H_BW, k0)];
						} else {
							U[f][index(j, i, k)] = U[f][index(j0, i + INX, k0)];
						}

						if (bc_[4] == OUTFLOW) {
							U[f][index(j, k, i)] = U[f][index(j0, k0, geo::H_BW)];
						} else {
							U[f][index(j, k, i)] = U[f][index(j0, k0, i + INX)];
						}

						if (bc_[1] == OUTFLOW) {
							U[f][index(geo::H_NX - 1 - i, j, k)] = U[f][index(geo::H_NX - 1 - geo::H_BW, j0, k0)];
						} else {
							U[f][index(geo::H_NX - 1 - i, j, k)] = U[f][index(2 * geo::H_BW - 1 - i, j, k)];
						}

						if (bc_[3] == OUTFLOW) {
							U[f][index(j, geo::H_NX - 1 - i, k)] = U[f][index(j0, geo::H_NX - 1 - geo::H_BW, k0)];
						} else {
							U[f][index(j, geo::H_NX - 1 - i, k)] = U[f][index(j0, 2 * geo::H_BW - 1 - i, k0)];
						}

						if (bc_[5] == OUTFLOW) {
							U[f][index(j, k, geo::H_NX - 1 - i)] = U[f][index(j0, k0, geo::H_NX - 1 - geo::H_BW)];
						} else {
							U[f][index(j, k, geo::H_NX - 1 - i)] = U[f][index(j0, k0, 2 * geo::H_BW - 1 - i)];
						}
					}
				}
			}
		}
	}
}
