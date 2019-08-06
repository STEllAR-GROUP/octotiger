
template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::advance(const hydro::state_type &U0, hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type<NDIM> &X,
		safe_real dx, safe_real dt, safe_real beta, safe_real omega) {
	static thread_local std::vector<std::vector<safe_real>> dudt(nf_, std::vector < safe_real > (geo::H_N3));
	for (int f = 0; f < nf_; f++) {
		for (const auto &i : geo::find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
			dudt[f][i] = 0.0;
		}
	}
	for (int dim = 0; dim < NDIM; dim++) {
		for (int f = 0; f < nf_; f++) {
			for (const auto &i : geo::find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
				const auto fr = F[dim][f][i + geo::H_DN[dim]];
				const auto fl = F[dim][f][i];
				dudt[f][i] -= (fr - fl) * INVERSE(dx);
			}
		}
	}
	physics < NDIM > ::template source<INX>(dudt, U, F, X, omega, dx);
	for (int f = 0; f < nf_; f++) {
		for (const auto &i : geo::find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
			safe_real u0 = U0[f][i];
			safe_real u1 = U[f][i] + dudt[f][i] * dt;
			U[f][i] = u0 * (1.0 - beta) + u1 * beta;
		}
	}

}

