#include "../util.hpp"

#define NEW_LIMITER

template<int NDIM, int INX>
const hydro::recon_type<NDIM> hydro_computer<NDIM, INX>::reconstruct(hydro::state_type &U_, const hydro::x_type &X,
		safe_real omega) {

	static thread_local auto D1 = std::vector<std::array<safe_real, geo::NDIR / 2>>(geo::H_N3);
	static thread_local auto Q = std::vector < std::vector<std::array<safe_real, geo::NDIR>>
			> (nf_, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));

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

	const auto add_angmom =
			[dx](std::array<std::array<safe_real, geo::NDIR>, NDIM> &C, std::array<safe_real, geo::NANGMOM> &Z) {
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

	static const auto center_value = [](const double u0, const std::array<double,geo::NDIR>& q) {
		double sum = 0.0;
		for( int d = 0; d < geo::NDIR; d++) {
			if( d != geo::NDIR/2 ) {
				sum += vw[d] * q[d];
			}
		}
		return (u0 - sum) / vw[geo::NDIR/2];
	};

	static const auto limit_slope2 = [](double &ql, double qc, double &qr) {

		const auto q0 = (2.0/3.0)*qc + (1.0/6.0)*(ql + qr);

		const double tmp1 = qr - ql;
		const double tmp2 = qr + ql;

		if (bool(qr < q0) != bool(q0 < ql)) {
			qr = ql = q0;
			return;
		}
		const double tmp3 = tmp1 * tmp1 / 6.0;
		const double tmp4 = tmp1 * (q0 - 0.5 * tmp2);
		if (tmp4 > tmp3) {
			ql = 3.0 * q0 - 2.0 * qr;
		} else if (-tmp3 > tmp4) {
			qr = 3.0 * q0 - 2.0 * ql;
		}
	};

	const auto reconstruct =
			[this](std::vector<std::array<safe_real, geo::NDIR>> &q, const std::vector<safe_real> &u, bool smooth) {
				for (const auto &i : indices1) {
					for (int d = 0; d < geo::NDIR / 2; d++) {
						const auto di = dir[d];
						D1[i][d] = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
					}
				}
				for (const auto &i : indices1) {
					for (int d = 0; d < geo::NDIR / 2; d++) {
						const auto di = dir[d];
						q[i][d] = 0.5 * (u[i] + u[i + di]);
						q[i][d] += (1.0 / 6.0) * (D1[i][d] - D1[i + di][d]);
						q[i + di][geo::flip(d)] = q[i][d];
					}
				}
				for (const auto &i : indices1) {
					for (int gi = 0; gi < geo::group_count(); gi++) {
						safe_real sum = 0.0;
						for (int n = 0; n < geo::group_size(gi); n++) {
							const auto pair = geo::group_pair(gi, n);
							sum += q[i + pair.first][pair.second];
						}
						sum /= safe_real(geo::group_size(gi));
						for (int n = 0; n < geo::group_size(gi); n++) {
							const auto pair = geo::group_pair(gi, n);
							q[i + pair.first][pair.second] = sum;
						}
					}
				}
				if (!smooth) {
					for (const auto i : indices2) {
						const auto qc = center_value(u[i], q[i]);
						for (int d = 0; d < geo::NDIR / 2; d++) {
							auto& qp = q[i][geo::flip(d)];
							auto& qm = q[i][d];
							const auto avg = (2./3.)*qc + (1./6.0)*(qp+qm);
#ifdef NEW_LIMITER
							limit_slope2(qm, qc, qp);
#else
							limit_slope(qm, u[i], qp);
#endif
						}
					}
				}
		}	;

	if (angmom_count_ == 0 || NDIM == 1) {
		for (int f = 0; f < nf_; f++) {
			reconstruct(Q[f], U[f], smooth_field_[f]);
		}

	} else {
		for (int f = 0; f < angmom_index_; f++) {
			reconstruct(Q[f], U[f], smooth_field_[f]);
		}

		int sx_i = angmom_index_;
		int zx_i = sx_i + NDIM;

		for (int angmom_pair = 0; angmom_pair < angmom_count_; angmom_pair++) {
			for (int f = sx_i; f < sx_i + NDIM; f++) {
				reconstruct(Q[f], U[f], false);
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
							const auto &q = Q[sx_i + dim][i][dir[d]];
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
					const auto qc = center_value(U[sx_i + dim][i], S[dim]);
					for (int d = 0; d < geo::NDIR / 2; d++) {
#ifdef NEW_LIMITER
						limit_slope2(S[dim][d], qc, S[dim][geo::flip(d)]);
#else
						limit_slope(S[dim][d], U[f][i], S[dim][geo::flip(d)]);
#endif
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
				reconstruct(Q[f], U[f], false);
			}
			sx_i += geo::NANGMOM + NDIM;
			zx_i += geo::NANGMOM + NDIM;
		}
		for (int f = angmom_index_ + angmom_count_ * (geo::NANGMOM + NDIM); f < nf_; f++) {
			reconstruct(Q[f], U[f], smooth_field_[f]);
		}

	}

	Q = physics < NDIM > ::template post_recon<INX>(Q, X, omega, angmom_count_ > 0);
	return Q;
}

