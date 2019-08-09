#include "../util.hpp"

template<int NDIM, int INX>
const hydro::recon_type<NDIM> hydro_computer<NDIM, INX>::reconstruct(hydro::state_type &U_, const hydro::x_type<NDIM> &X, safe_real omega) {

	static constexpr auto xloc = geo::xloc();
	static constexpr auto kdelta = geo::kronecker_delta();
	static constexpr auto vw = geo::volume_weight();
	static constexpr auto dir = geo::direction();

	static const auto indices1 = geo::find_indices(1, geo::H_NX - 1);
	static const auto indices2 = geo::find_indices(2, geo::H_NX - 2);

	const auto dx = X[0][geo::H_DNX] - X[0][0];
	auto U = physics < NDIM > ::template pre_recon<INX>(U_, X, omega, angmom_count_ > 0);

	const auto measure_angmom = [dx](const std::array<std::array<safe_real, geo::NDIR>, NDIM> &C) {
		std::array < safe_real, geo::NANGMOM > L;
		for (int n = 0; n < geo::NANGMOM; n++) {
			L[n] = 0.0;
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l < NDIM; l++) {
					for (int d = 0; d < geo::NDIR; d++) {
						if (d != geo::NDIR / 2) {
							L[n] += vw[d] * kdelta[n][m][l] * 0.5 * xloc[d][m] * C[l][d] * dx;
						}
					}
				}
			}
		}
		return L;
	};

	const auto add_angmom = [dx](std::array<std::array<safe_real, geo::NDIR>, NDIM> &C, std::array<safe_real, geo::NANGMOM> &Z) {
		for (int d = 0; d < geo::NDIR; d++) {
			if (d != geo::NDIR / 2) {
				for (int n = 0; n < geo::NANGMOM; n++) {
					for (int m = 0; m < NDIM; m++) {
						for (int l = 0; l < NDIM; l++) {
							const auto tmp = 6.0 * Z[n] / dx;
							C[l][d] += kdelta[n][m][l] * 0.5 * xloc[d][m] * tmp;
						}
					}
				}
			}
		}
	};

	const auto derivs = [this](const std::vector<safe_real> &u, hydro::inner_recon_type<NDIM> &d1) {
		for (const auto &i : indices1) {
			for (int d = 0; d < geo::NDIR / 2; d++) {
				const auto di = dir[d];
				d1[i][d] = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
			}
		}
	};

	const auto recon = [this](hydro::inner_recon_type<NDIM> &q, const std::vector<safe_real> &u, const hydro::inner_recon_type<NDIM> &d1) {
		for (const auto &i : indices1) {
			for (int d = 0; d < geo::NDIR / 2; d++) {
				const auto di = dir[d];
				q[i][d] = 0.5 * (u[i] + u[i + di]);
				q[i][d] += (1.0 / 6.0) * (d1[i][d] - d1[i + di][d]);
				q[i + di][geo::flip(d)] = q[i][d];
			}
		}
	};

	const auto limit = [this](hydro::inner_recon_type<NDIM> &q, const std::vector<safe_real> &u) {
		for (const auto i : indices2) {
			for (int d = 0; d < geo::NDIR / 2; d++) {
				limit_slope(q[i][d], u[i], q[i][geo::flip(d)]);
			}
		}
	};

	if (angmom_count_ == 0) {
		for (int f = 0; f < nf_; f++) {
			derivs(U[f], D1[f]);
			recon(Q[f], U[f], D1[f]);
			if (!smooth_field_[f]) {
				limit(Q[f], U[f]);
			}
		}

	} else {
		for (int f = 0; f < angmom_index_; f++) {
			derivs(U[f], D1[f]);
			recon(Q[f], U[f], D1[f]);
			if (!smooth_field_[f]) {
				limit(Q[f], U[f]);
			}
		}

		int sx_i = angmom_index_;
		int zx_i = sx_i + NDIM;

		for (int angmom_pair = 0; angmom_pair < angmom_count_; angmom_pair++) {
			for (int f = sx_i; f < sx_i + NDIM; f++) {
				derivs(U[f], D1[f]);

			}


			for (int f = sx_i; f < sx_i + NDIM; f++) {
				recon(Q[f], U[f], D1[f]);
				limit(Q[f], U[f]);

			}

			for (const auto &i : indices2) {
				std::array < safe_real, geo::NANGMOM > Z;
				for (int dim = 0; dim < geo::NANGMOM; dim++) {
					Z[dim] = U[zx_i + dim][i];
				}
				std::array<std::array<safe_real, geo::NDIR>, NDIM> S;
				for (int dim = 0; dim < NDIM; dim++) {
					for (int d = 0; d < geo::NDIR; d++) {
						S[dim][d] = Q[sx_i + dim][i][d];
					}
				}
				auto am1 = measure_angmom(S);
				decltype(Z) am2;
				for (int dim = 0; dim < geo::NANGMOM; dim++) {
					am2[dim] = U[zx_i + dim][i] - am1[dim];
				}
				add_angmom(S, am2);
				for (int dim = 0; dim < NDIM; dim++) {
					for (int d = 0; d < geo::NDIR; d++) {
						if (d != geo::NDIR / 2) {
							auto &s = S[dim][d];
							const auto &q = U[sx_i + dim][i + dir[d]];
							const auto &u0 = U[sx_i + dim][i];
							const auto M = std::max(u0, q);
							const auto m = std::min(u0, q);
							s = std::min(s, M);
							s = std::max(s, m);
						}
					}
				}
				for (int f = sx_i; f < sx_i + NDIM; f++) {
					const auto dim = f - sx_i;
					for (int d = 0; d < geo::NDIR / 2; d++) {
						limit_slope(S[dim][d], U[f][i], S[dim][geo::flip(d)]);
					}
				}
				am2 = measure_angmom(S);
				for (int n = 0; n < geo::NANGMOM; n++) {
					U[zx_i + n][i] = Z[n] - am2[n];
				}
				for (int dim = 0; dim < NDIM; dim++) {
					for (int d = 0; d < geo::NDIR; d++) {
						Q[sx_i + dim][i][d] = S[dim][d];
					}
				}
			}
			for (int f = zx_i; f < zx_i + geo::NANGMOM; f++) {
				derivs(U[f], D1[f]);
				recon(Q[f], U[f], D1[f]);
				limit(Q[f], U[f]);
			}
			sx_i += geo::NANGMOM + NDIM;
			zx_i += geo::NANGMOM + NDIM;
		}
		for (int f = zx_i + 1 - geo::NANGMOM - NDIM; f < nf_; f++) {
			derivs(U[f], D1[f]);
			recon(Q[f], U[f], D1[f]);
			if (!smooth_field_[f]) {
				limit(Q[f], U[f]);
			}
		}

	}

	Q = physics < NDIM > ::template post_recon<INX>(Q, X, omega, angmom_count_ > 0);
	return Q;
}

