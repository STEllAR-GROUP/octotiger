

template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::boundaries(hydro::state_type &U) {

	for (int f = 0; f < nf_; f++) {
		if (NDIM == 1) {
			for (int i = 0; i < geo::H_BW + 20; i++) {
				U[f][i] = U[f][geo::H_BW];
				U[f][geo::H_NX - 1 - i] = U[f][geo::H_NX - geo::H_BW - 1];
			}
		} else if (NDIM == 2) {

			const auto index = [](int i, int j) {
				return i + geo::H_NX * j;
			};

			for (int i = 0; i < geo::H_BW; i++) {
				for (int j = 0; j < geo::H_NX; j++) {
					int j0 = j;
					j0 = std::max(j0, geo::H_BW);
					j0 = std::min(j0, geo::H_NX - 1 - geo::H_BW);
					U[f][index(i, j)] = U[f][index(geo::H_BW, j0)];
					U[f][index(j, i)] = U[f][index(j0, geo::H_BW)];
					U[f][index(geo::H_NX - 1 - i, j)] = U[f][index(geo::H_NX - 1 - geo::H_BW, j0)];
					U[f][index(j, geo::H_NX - 1 - i)] = U[f][index(j0, geo::H_NX - 1 - geo::H_BW)];
				}
			}
		} else {
			const auto index = [](int i, int j, int k) {
				return i + geo::H_NX * j + k * geo::H_NX * geo::H_NX;
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
						U[f][index(i, j, k)] = U[f][index(geo::H_BW, j0, k0)];
						U[f][index(j, i, k)] = U[f][index(j0, geo::H_BW, k0)];
						U[f][index(j, k, i)] = U[f][index(j0, k0, geo::H_BW)];
						U[f][index(geo::H_NX - 1 - i, j, k)] = U[f][index(geo::H_NX - 1 - geo::H_BW, j0, k0)];
						U[f][index(j, geo::H_NX - 1 - i, k)] = U[f][index(j0, geo::H_NX - 1 - geo::H_BW, k0)];
						U[f][index(j, k, geo::H_NX - 1 - i)] = U[f][index(j0, k0, geo::H_NX - 1 - geo::H_BW)];
					}
				}
			}
		}
	}
}
