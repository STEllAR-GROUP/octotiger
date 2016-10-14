#include "grid.hpp"
#include "problem.hpp"
#include "options.hpp"
#include <cmath>
#include <cassert>
#include "profiler.hpp"
#include "taylor.hpp"
#include <boost/thread/tss.hpp>

extern options opts;

real grid::omega = ZERO;
space_vector grid::pivot(ZERO);
real grid::scaling_factor = 1.0;

integer grid::max_level = 0;

struct tls_data_t {
	std::vector<state> v;
	std::vector<std::vector<state>> dvdx;
	std::vector<std::vector<state>> dudx;
	std::vector<std::vector<state>> uf;
	std::vector<std::vector<real>> zz;
};

class tls_t {
private:
	pthread_key_t key;
public:
	static void cleanup(void* ptr) {
		tls_data_t* _ptr = (tls_data_t*) ptr;
		delete _ptr;
	}
	tls_t() {
		pthread_key_create(&key, cleanup);
	}
	tls_data_t* get_ptr() {
		tls_data_t* ptr = (tls_data_t*) pthread_getspecific(key);
		if (ptr == nullptr) {
			ptr = new tls_data_t;
			ptr->v.resize(H_N3);
			ptr->zz.resize(NDIM, std::vector < real > (H_N3));
			ptr->dvdx.resize(NDIM, std::vector < state > (H_N3));
			ptr->dudx.resize(NDIM, std::vector < state > (H_N3));
			ptr->uf.resize(NFACE, std::vector < state > (H_N3));
			pthread_setspecific(key, ptr);
		}
		return ptr;
	}
};

static tls_t tls;

static std::vector<state>& TLS_V() {
	return tls.get_ptr()->v;
}

static std::vector<std::vector<state>>& TLS_dVdx() {
	return tls.get_ptr()->dvdx;
}

static std::vector<std::vector<state>>& TLS_dUdx() {
	return tls.get_ptr()->dudx;
}

static std::vector<std::vector<real>>& TLS_zz() {
	return tls.get_ptr()->zz;
}

static std::vector<std::vector<state>>& TLS_Uf() {
	return tls.get_ptr()->uf;
}

space_vector grid::get_cell_center(integer i, integer j, integer k) {
	const integer iii0 = hindex(H_BW,H_BW,H_BW);
	space_vector c;
	c[XDIM] = X[XDIM][iii0] + (i) * dx;
	c[YDIM] = X[XDIM][iii0] + (j) * dx;
	c[ZDIM] = X[XDIM][iii0] + (k) * dx;
	return c;
}

void grid::set_hydro_boundary(const std::vector<real>& data, const geo::direction& dir, integer width, bool etot_only) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	if (!etot_only) {
		get_boundary_size(lb, ub, dir, OUTER, INX, width);
	} else {
		get_boundary_size(lb, ub, dir, OUTER, INX, width);
	}
	integer iter = 0;

	for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
			for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
				for (integer field = 0; field != NF; ++field) {
					if (!etot_only || (etot_only && field == egas_i)) {
						U[hindex( i, j, k)](field) = data[iter];
						++iter;
					}
				}
			}
		}
	}
	PROF_END;
}

std::vector<real> grid::get_hydro_boundary(const geo::direction& dir, integer width, bool etot_only) {
	PROF_BEGIN;
	std::array<integer, NDIM> lb, ub;
	std::vector < real > data;
	integer size;
	if (!etot_only) {
		size = NF * get_boundary_size(lb, ub, dir, INNER, INX, width);
	} else {
		size = get_boundary_size(lb, ub, dir, INNER, INX, width);
	}
	data.resize(size);
	integer iter = 0;

	for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
			for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
				for (integer field = 0; field != NF; ++field) {
					if (!etot_only || (etot_only && field == egas_i)) {
						data[iter] = U[hindex( i, j, k)](field);
						++iter;
					}
				}
			}
		}
	}
	PROF_END;
	return data;

}

line_of_centers_t grid::line_of_centers(const std::pair<space_vector, space_vector>& line) {
	PROF_BEGIN;
	line_of_centers_t loc;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i-H_BW, j-H_BW, k-H_BW);
				space_vector a = line.first;
				const space_vector& o = line.second;
				space_vector b;
				real aa = 0.0;
				real bb = 0.0;
				real ab = 0.0;
				for (integer d = 0; d != NDIM; ++d) {
					a[d] -= o[d];
					b[d] = X[d][iii] - o[d];
				}
				for (integer d = 0; d != NDIM; ++d) {
					aa += a[d] * a[d];
					bb += b[d] * b[d];
					ab += a[d] * b[d];
				}
				const real d = std::sqrt((aa * bb - ab * ab) / aa);
				real p = ab / std::sqrt(aa);
				std::vector < real > data(NF + NGF);
				if (d < std::sqrt(3.0) * dx / 2.0) {
					for (integer ui = 0; ui != NF; ++ui) {
						data[ui] = U[iii](ui);
					}
					for (integer gi = 0; gi != NGF; ++gi) {
						data[NF + gi] = G[gi][iiig];
					}
					loc.resize(loc.size() + 1);
					loc[loc.size() - 1].first = p;
					loc[loc.size() - 1].second = std::move(data);
				}
			}
		}
	}
	PROF_END;
	return loc;
}

std::pair<std::vector<real>, std::vector<real>> grid::diagnostic_error() const {
	PROF_BEGIN;
	std::pair < std::vector<real>, std::vector < real >> e;
	const real dV = dx * dx * dx;
	if (opts.problem == SOLID_SPHERE) {
		e.first.resize(8, ZERO);
		e.second.resize(8, ZERO);
	}
	for (integer i = 0; i != G_NX; ++i) {
		for (integer j = 0; j != G_NX; ++j) {
			for (integer k = 0; k != G_NX; ++k) {
				const integer iii = gindex(i, j, k);
				const integer bdif = H_BW;
				const integer iiih = hindex(i + bdif, j + bdif, k + bdif);
				const real x = X[XDIM][iiih];
				const real y = X[YDIM][iiih];
				const real z = X[ZDIM][iiih];
				if (opts.problem == SOLID_SPHERE) {
					const auto a = solid_sphere_analytic_phi(x, y, z, 0.25);
					std::vector < real > n(4);
					n[phi_i] = G[phi_i][iii];
					n[gx_i] = G[gx_i][iii];
					n[gy_i] = G[gy_i][iii];
					n[gz_i] = G[gz_i][iii];
					const real rho = U[iiih](rho_i);
					for (integer l = 0; l != 4; ++l) {
						e.first[l] += std::abs(a[l] - n[l]) * dV * rho;
						e.first[4 + l] += std::abs(a[l]) * dV * rho;
						e.second[l] += std::pow((a[l] - n[l]) * rho, 2) * dV;
						e.second[4 + l] += std::pow(a[l] * rho, 2) * dV;
					}
				}
			}
		}
	}
//	printf("%e\n", e[0]);
	PROF_END;
	return e;
}

real grid::get_omega() {
	return omega;
}

void grid::velocity_inc(const space_vector& dv) {

	for (integer iii = 0; iii != H_N3; ++iii) {
		const real rho = U[iii](rho_i);
		if (rho != ZERO) {
			const real rhoinv = ONE / rho;
			real& sx = U[iii](sx_i);
			real& sy = U[iii](sy_i);
			real& sz = U[iii](sz_i);
			real& egas = U[iii](egas_i);
			egas -= HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
			sx += dv[XDIM] * rho;
			sy += dv[YDIM] * rho;
			sz += dv[ZDIM] * rho;
			egas += HALF * (sx * sx + sy * sy + sz * sz) * rhoinv;
		}
	}

}

void grid::set_pivot(const space_vector& p) {
	pivot = p;
}

inline real minmod(real a, real b) {
	return (std::copysign(HALF, a) + std::copysign(HALF, b)) * std::min(std::abs(a), std::abs(b));
}

inline real minmod_theta(real a, real b, real c, real theta) {
	return minmod(theta * minmod(a, b), c);
}

inline real minmod_theta(real a, real b, real theta = 1.0) {
	return minmod(theta * minmod(a, b), HALF * (a + b));
}

std::vector<real> grid::get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::dimension& dim) const {
	PROF_BEGIN;
	std::vector < real > data;
	integer size = 1;
	for (auto& dim : geo::dimension::full_set()) {
		size *= (ub[dim] - lb[dim]);
	}
	size /= (NCHILD / 2);
	size *= NF;
	data.reserve(size);
	const integer stride1 = (dim == XDIM) ? (INX + 1) : (INX + 1) * (INX + 1);
	const integer stride2 = (dim == ZDIM) ? (INX + 1) : 1;
	for (integer field = 0; field != NF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; i += 2) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; j += 2) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; k += 2) {
					const integer i00 = findex(i, j, k);
					const integer i10 = i00 + stride1;
					const integer i01 = i00 + stride2;
					const integer i11 = i00 + stride1 + stride2;
					real value = ZERO;
					value += F[dim][i00](field);
					value += F[dim][i10](field);
					value += F[dim][i01](field);
					value += F[dim][i11](field);
					const real f = dx / TWO;
					if (field == zx_i) {
						if (dim == YDIM) {
							value += F[dim][i00](sy_i) * f;
							value += F[dim][i10](sy_i) * f;
							value -= F[dim][i01](sy_i) * f;
							value -= F[dim][i11](sy_i) * f;
						} else if (dim == ZDIM) {
							value -= F[dim][i00](sz_i) * f;
							value -= F[dim][i10](sz_i) * f;
							value += F[dim][i01](sz_i) * f;
							value += F[dim][i11](sz_i) * f;
						} else if (dim == XDIM) {
							value += F[dim][i00](sy_i) * f;
							value += F[dim][i10](sy_i) * f;
							value -= F[dim][i01](sy_i) * f;
							value -= F[dim][i11](sy_i) * f;
							value -= F[dim][i00](sz_i) * f;
							value += F[dim][i10](sz_i) * f;
							value -= F[dim][i01](sz_i) * f;
							value += F[dim][i11](sz_i) * f;
						}
					} else if (field == zy_i) {
						if (dim == XDIM) {
							value -= F[dim][i00](sx_i) * f;
							value -= F[dim][i10](sx_i) * f;
							value += F[dim][i01](sx_i) * f;
							value += F[dim][i11](sx_i) * f;
						} else if (dim == ZDIM) {
							value += F[dim][i00](sz_i) * f;
							value -= F[dim][i10](sz_i) * f;
							value += F[dim][i01](sz_i) * f;
							value -= F[dim][i11](sz_i) * f;
						} else if (dim == YDIM) {
							value -= F[dim][i00](sx_i) * f;
							value -= F[dim][i10](sx_i) * f;
							value += F[dim][i01](sx_i) * f;
							value += F[dim][i11](sx_i) * f;
							value += F[dim][i00](sz_i) * f;
							value -= F[dim][i10](sz_i) * f;
							value += F[dim][i01](sz_i) * f;
							value -= F[dim][i11](sz_i) * f;
						}
					} else if (field == zz_i) {
						if (dim == XDIM) {
							value += F[dim][i00](sx_i) * f;
							value -= F[dim][i10](sx_i) * f;
							value += F[dim][i01](sx_i) * f;
							value -= F[dim][i11](sx_i) * f;
						} else if (dim == YDIM) {
							value -= F[dim][i00](sy_i) * f;
							value += F[dim][i10](sy_i) * f;
							value -= F[dim][i01](sy_i) * f;
							value += F[dim][i11](sy_i) * f;
						} else if (dim == ZDIM) {
							value -= F[dim][i00](sy_i) * f;
							value += F[dim][i10](sy_i) * f;
							value -= F[dim][i01](sy_i) * f;
							value += F[dim][i11](sy_i) * f;
							value += F[dim][i00](sx_i) * f;
							value += F[dim][i10](sx_i) * f;
							value -= F[dim][i01](sx_i) * f;
							value -= F[dim][i11](sx_i) * f;
						}
					}
					value /= real(4);
					data.push_back(value);
				}
			}
		}
	}
	PROF_END;
	return data;
}

void grid::set_flux_restrict(const std::vector<real>& data, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
	const geo::dimension& dim) {
	PROF_BEGIN;
	integer index = 0;
	for (integer field = 0; field != NF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					const integer iii = findex(i, j, k);
					F[dim][iii](field) = data[index];
					++index;
				}
			}
		}
	}
	PROF_END;
}

void grid::set_outflows(std::vector<real>&& u) {
	U_out = std::move(u);
}

void grid::set_prolong(const std::vector<real>& data, std::vector<real>&& outflows) {
	PROF_BEGIN;
	integer index = 0;
	U_out = std::move(outflows);
	for (integer field = 0; field != NF; ++field) {
		for (integer i = H_BW; i != H_NX - H_BW; ++i) {
			for (integer j = H_BW; j != H_NX - H_BW; ++j) {
				for (integer k = H_BW; k != H_NX - H_BW; ++k) {
					const integer iii = hindex(i, j, k);
					auto& value = U[iii](field);
					value = data[index];
					++index;
				}
			}
		}
	}
	PROF_END;
}

std::vector<real> grid::get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, bool etot_only) {
	PROF_BEGIN;
	auto& dUdx = TLS_dUdx();
	auto& tmpz = TLS_zz();
	std::vector < real > data;

	integer size = NF;
	for (integer dim = 0; dim != NDIM; ++dim) {
		size *= (ub[dim] - lb[dim]);
	}
	data.reserve(size);
	auto lb0 = lb;
	auto ub0 = ub;
	for (integer d = 0; d != NDIM; ++d) {
		lb0[d] /= 2;
		ub0[d] = (ub[d] - 1) / 2 + 1;
	}
	compute_primitives(lb0, ub0, etot_only);
	compute_primitive_slopes(1.0, lb0, ub0, etot_only);
	compute_conserved_slopes(lb0, ub0, etot_only);

	if (!etot_only) {
		for (integer i = lb0[XDIM]; i != ub0[XDIM]; ++i) {
			for (integer j = lb0[YDIM]; j != ub0[YDIM]; ++j) {
#pragma GCC ivdep
				for (integer k = lb0[ZDIM]; k != ub0[ZDIM]; ++k) {
					const integer iii = hindex(i,j,k);
					tmpz[XDIM][iii] = U[iii](zx_i);
					tmpz[YDIM][iii] = U[iii](zy_i);
					tmpz[ZDIM][iii] = U[iii](zz_i);
				}
			}
		}
	}

	for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
		const real xsgn = (i % 2) ? +1 : -1;
		for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
			const real ysgn = (j % 2) ? +1 : -1;
#pragma GCC ivdep
			for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
				const integer iii = hindex(i / 2, j / 2, k / 2);
				for (integer field = 0; field != NF; ++field) {
					if (!etot_only || (etot_only && field == egas_i)) {
						const real zsgn = (k % 2) ? +1 : -1;
						real value = U[iii](field);
						value += xsgn * dUdx[XDIM][iii](field) * 0.25;
						value += ysgn * dUdx[YDIM][iii](field) * 0.25;
						value += zsgn * dUdx[ZDIM][iii](field) * 0.25;
						if (field == sx_i) {
							U[iii](zy_i) -= 0.25 * zsgn * value * dx / 8.0;
							U[iii](zz_i) += 0.25 * ysgn * value * dx / 8.0;
						} else if (field == sy_i) {
							U[iii](zx_i) += 0.25 * zsgn * value * dx / 8.0;
							U[iii](zz_i) -= 0.25 * xsgn * value * dx / 8.0;
						} else if (field == sz_i) {
							U[iii](zx_i) -= 0.25 * ysgn * value * dx / 8.0;
							U[iii](zy_i) += 0.25 * xsgn * value * dx / 8.0;
						}
						data.push_back(value);
					}
				}
			}
		}
	}

	if (!etot_only) {
		for (integer i = lb0[XDIM]; i != ub0[XDIM]; ++i) {
			for (integer j = lb0[YDIM]; j != ub0[YDIM]; ++j) {
#pragma GCC ivdep
				for (integer k = lb0[ZDIM]; k != ub0[ZDIM]; ++k) {
					const integer iii = hindex(i,j,k);
					U[iii](zx_i) = tmpz[XDIM][iii];
					U[iii](zy_i) = tmpz[YDIM][iii];
					U[iii](zz_i) = tmpz[ZDIM][iii];
				}
			}
		}
	}

	PROF_END;
	return data;
}

std::vector<real> grid::get_restrict() const {
	PROF_BEGIN;
	constexpr
	integer Size = NF * INX * INX * INX / NCHILD + NF;
	std::vector < real > data;
	data.reserve(Size);
	for (integer field = 0; field != NF; ++field) {
		for (integer i = H_BW; i < H_NX - H_BW; i += 2) {
			for (integer j = H_BW; j < H_NX - H_BW; j += 2) {
				for (integer k = H_BW; k < H_NX - H_BW; k += 2) {
					const integer iii = hindex(i, j, k);
					real pt = ZERO;
					for (integer x = 0; x != 2; ++x) {
						for (integer y = 0; y != 2; ++y) {
							for (integer z = 0; z != 2; ++z) {
								const integer jjj = iii + x * H_DNX + y * H_DNY + z * H_DNZ;
								pt += U[jjj](field);
								if (field == zx_i) {
									pt += X[YDIM][jjj] * U[jjj](sz_i);
									pt -= X[ZDIM][jjj] * U[jjj](sy_i);
								} else if (field == zy_i) {
									pt -= X[XDIM][jjj] * U[jjj](sz_i);
									pt += X[ZDIM][jjj] * U[jjj](sx_i);
								} else if (field == zz_i) {
									pt += X[XDIM][jjj] * U[jjj](sy_i);
									pt -= X[YDIM][jjj] * U[jjj](sx_i);
								}
							}
						}
					}
					pt /= real(NCHILD);
					data.push_back(pt);
				}
			}
		}
	}
	for (integer field = 0; field != NF; ++field) {
		data.push_back(U_out[field]);
	}
	PROF_END;
	return data;
}

void grid::set_restrict(const std::vector<real>& data, const geo::octant& octant) {
	PROF_BEGIN;
	integer index = 0;
	const integer i0 = octant.get_side(XDIM) * (INX / 2);
	const integer j0 = octant.get_side(YDIM) * (INX / 2);
	const integer k0 = octant.get_side(ZDIM) * (INX / 2);
	for (integer field = 0; field != NF; ++field) {
		for (integer i = H_BW; i != H_NX / 2; ++i) {
			for (integer j = H_BW; j != H_NX / 2; ++j) {
				for (integer k = H_BW; k != H_NX / 2; ++k) {
					const integer iii = (i + i0) * H_DNX + (j + j0) * H_DNY + (k + k0) * H_DNZ;
					auto& v = U[iii](field);
					v = data[index];
					if (field == zx_i) {
						v -= X[YDIM][iii] * U[iii](sz_i);
						v += X[ZDIM][iii] * U[iii](sy_i);
					} else if (field == zy_i) {
						v += X[XDIM][iii] * U[iii](sz_i);
						v -= X[ZDIM][iii] * U[iii](sx_i);
					} else if (field == zz_i) {
						v -= X[XDIM][iii] * U[iii](sy_i);
						v += X[YDIM][iii] * U[iii](sx_i);
					}
					++index;
				}
			}
		}
	}
	PROF_END;
}

std::pair<std::vector<real>, std::vector<real> > grid::field_range() const {
	PROF_BEGIN;
	std::pair < std::vector<real>, std::vector<real> > minmax;
	minmax.first.resize(NF);
	minmax.second.resize(NF);
	for (integer field = 0; field != NF; ++field) {
		minmax.first[field] = +std::numeric_limits < real > ::max();
		minmax.second[field] = -std::numeric_limits < real > ::max();
	}
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				for (integer field = 0; field != NF; ++field) {
					minmax.first[field] = std::min(minmax.first[field], U[iii](field));
					minmax.second[field] = std::max(minmax.second[field], U[iii](field));
				}
			}
		}
	}
	PROF_END;
	return minmax;
}

HPX_PLAIN_ACTION(grid::set_omega, set_omega_action);

void grid::set_omega(real omega) {
	if (hpx::get_locality_id() == 0) {
		std::list<hpx::future<void>> futs;
		auto remotes = hpx::find_remote_localities();
		for (auto& l : remotes) {
			futs.push_back(hpx::async < set_omega_action > (l, omega));
		}
		for (auto& f : futs) {
			f.get();
		}
	}
	grid::omega = omega;
}

real grid::roche_volume(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, real cx, bool donor) const {
	PROF_BEGIN;
	const real dV = dx * dx * dx;
	real V = 0.0;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer D = 0 - H_BW;
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				real x0 = X[XDIM][iii];
				real x = x0 - cx;
				real y = X[YDIM][iii];
				real z = X[ZDIM][iii];
				const real R = std::sqrt(x0 * x0 + y * y);
				real phi_eff = G[phi_i][iiig] - 0.5 * std::pow(omega * R, 2);
				//	real factor = axis.first[0] == l1.first ? 0.5 : 1.0;
				if ((x0 <= l1.first && !donor) || (x0 >= l1.first && donor)) {
					if (phi_eff <= l1.second) {
						const real fx = G[gx_i][iiig] + x0 * std::pow(omega, 2);
						const real fy = G[gy_i][iiig] + y * std::pow(omega, 2);
						const real fz = G[gz_i][iiig];
						real g = x * fx + y * fy + z * fz;
						if (g <= 0.0) {
							V += dV;
						}
					}
				}
			}
		}
	}
	PROF_END;
	return V;
}

std::vector<real> grid::frac_volumes() const {
	PROF_BEGIN;
	std::vector < real > V(NSPECIES, 0.0);
	const real dV = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				for (integer si = 0; si != NSPECIES; ++si) {
					if (U[iii](spc_i + si) > 1.0e-5) {
						V[si] += (U[iii](spc_i + si) / U[iii](rho_i)) * dV;
					}
				}
			}
		}
	}
//	printf( "%e", V[0]);
	PROF_END;
	return V;
}

bool grid::is_in_star(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, integer frac, integer iii) const {
	bool use = false;
	if (frac == 0) {
		use = true;
	} else {
		space_vector a = axis.first;
		const space_vector& o = axis.second;
		space_vector b;
		real aa = 0.0;
		real ab = 0.0;
		for (integer d = 0; d != NDIM; ++d) {
			a[d] -= o[d];
			b[d] = X[d][iii] - o[d];
		}
		for (integer d = 0; d != NDIM; ++d) {
			aa += a[d] * a[d];
			ab += a[d] * b[d];
		}
		real p = ab / std::sqrt(aa);
//		printf( "%e\n", l1.first);
		if (p < l1.first && frac == +1) {
			use = true;
		} else if (p >= l1.first && frac == -1) {
			use = true;
		}
	}
	return use;
}

real grid::z_moments(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, integer frac) const {
	PROF_BEGIN;
	real mom = 0.0;
	const real dV = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				if (is_in_star(axis, l1, frac, iii)) {
					mom += (std::pow(X[XDIM][iii], 2) + dx * dx / 6.0) * U[iii](rho_i) * dV;
					mom += (std::pow(X[YDIM][iii], 2) + dx * dx / 6.0) * U[iii](rho_i) * dV;
				}
			}
		}
	}
	PROF_END;
	return mom;
}

std::vector<real> grid::conserved_sums(space_vector& com, space_vector& com_dot, const std::pair<space_vector, space_vector>& axis,
	const std::pair<real, real>& l1, integer frac) const {
	PROF_BEGIN;
	std::vector < real > sum(NF, ZERO);
	com[0] = com[1] = com[2] = 0.0;
	com_dot[0] = com_dot[1] = com_dot[2] = 0.0;
	const real dV = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				if (is_in_star(axis, l1, frac, iii)) {
					com[0] += X[XDIM][iii] * U[iii](rho_i) * dV;
					com[1] += X[YDIM][iii] * U[iii](rho_i) * dV;
					com[2] += X[ZDIM][iii] * U[iii](rho_i) * dV;
					com_dot[0] += U[iii](sx_i) * dV;
					com_dot[1] += U[iii](sy_i) * dV;
					com_dot[2] += U[iii](sz_i) * dV;
					for (integer field = 0; field != NF; ++field) {
						sum[field] += U[iii](field) * dV;
					}
					sum[egas_i] += U[iii](pot_i) * HALF * dV;
					sum[zx_i] += X[YDIM][iii] * U[iii](sz_i) * dV;
					sum[zx_i] -= X[ZDIM][iii] * U[iii](sy_i) * dV;
					sum[zy_i] -= X[XDIM][iii] * U[iii](sz_i) * dV;
					sum[zy_i] += X[ZDIM][iii] * U[iii](sx_i) * dV;
					sum[zz_i] += X[XDIM][iii] * U[iii](sy_i) * dV;
					sum[zz_i] -= X[YDIM][iii] * U[iii](sx_i) * dV;
				}
			}
		}
	}
	if (sum[rho_i] > 0.0) {
		for (integer d = 0; d != NDIM; ++d) {
			com[d] /= sum[rho_i];
			com_dot[d] /= sum[rho_i];
		}
	}
	PROF_END;
	return sum;
}

std::vector<real> grid::gforce_sum(bool torque) const {
	PROF_BEGIN;
	std::vector < real > sum(NDIM, ZERO);
	const real dV = dx * dx * dx;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const auto D = 0 - H_BW;
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				const real& rho = U[iii](rho_i);
				const real x = X[XDIM][iii];
				const real y = X[YDIM][iii];
				const real z = X[ZDIM][iii];
				const real fx = rho * G[gx_i][iiig] * dV;
				const real fy = rho * G[gy_i][iiig] * dV;
				const real fz = rho * G[gz_i][iiig] * dV;
				if (!torque) {
					sum[XDIM] += fx;
					sum[YDIM] += fy;
					sum[ZDIM] += fz;
				} else {
					sum[XDIM] -= z * fy - y * fz;
					sum[YDIM] += z * fx - x * fz;
					sum[ZDIM] -= y * fx - x * fy;
				}
			}
		}
	}
	PROF_END;
	return sum;
}

std::vector<real> grid::l_sums() const {
	PROF_BEGIN;
	std::vector < real > sum(NDIM);
	const real dV = dx * dx * dx;
	std::fill(sum.begin(), sum.end(), ZERO);
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				sum[XDIM] += X[YDIM][iii] * U[iii](sz_i) * dV;
				sum[XDIM] -= X[ZDIM][iii] * U[iii](sy_i) * dV;

				sum[YDIM] -= X[XDIM][iii] * U[iii](sz_i) * dV;
				sum[YDIM] += X[ZDIM][iii] * U[iii](sx_i) * dV;

				sum[ZDIM] += X[XDIM][iii] * U[iii](sy_i) * dV;
				sum[ZDIM] -= X[YDIM][iii] * U[iii](sx_i) * dV;

			}
		}
	}
	PROF_END;
	return sum;
}

bool grid::refine_me(integer lev) const {
	PROF_BEGIN;
	auto test = get_refine_test();
	if (lev < 2) {
		PROF_END;
		return true;
	}
	bool rc = false;
	for (integer i = H_BW - R_BW; i != H_NX - H_BW + R_BW; ++i) {
		for (integer j = H_BW - R_BW; j != H_NX - H_BW + R_BW; ++j) {
			for (integer k = H_BW - R_BW; k != H_NX - H_BW + R_BW; ++k) {
				int cnt = 0;
				if (i < H_BW || i >= H_NX - H_BW) {
					++cnt;
				}
				if (j < H_BW || j >= H_NX - H_BW) {
					++cnt;
				}
				if (k < H_BW || k >= H_NX - H_BW) {
					++cnt;
				}
				if (cnt > 1) {
					continue;
				}
				const integer iii = hindex(i, j, k);
				std::vector < real > state(NF);
				for (integer i = 0; i != NF; ++i) {
					state[i] = U[iii](i);
				}
				if (test(lev, max_level, X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii], state)) {
					rc = true;
					break;
				}
			}
			if (rc) {
				break;
			}
		}
		if (rc) {
			break;
		}
	}
	PROF_END;
	return rc;
}

grid::~grid() {

}

void grid::rho_mult(real f0, real f1) {
	for (integer i = 0; i != H_NX; ++i) {
		for (integer j = 0; j != H_NX; ++j) {
			for (integer k = 0; k != H_NX; ++k) {
				U[hindex(i,j,k)](spc_ac_i) *= f0;
				U[hindex(i,j,k)](spc_dc_i) *= f1;
				U[hindex(i,j,k)](spc_ae_i) *= f0;
				U[hindex(i,j,k)](spc_de_i) *= f1;
				U[hindex(i,j,k)](rho_i) = 0.0;
				for (integer si = 0; si != NSPECIES; ++si) {
					U[hindex(i,j,k)](rho_i) += U[hindex(i, j, k)](spc_i + si);
				}
			}
		}
	}

}

void grid::rho_move(real x) {
	real w = x / dx;
	const real rho_floor = 1.0e-15;

	w = std::max(-0.5, std::min(0.5, w));
	for (integer i = 1; i != H_NX - 1; ++i) {
		for (integer j = 1; j != H_NX - 1; ++j) {
			for (integer k = 1; k != H_NX - 1; ++k) {
				for (integer si = spc_i; si != NSPECIES + spc_i; ++si) {
					U[hindex(i,j,k)](si) += w * U[hindex(i+1,j,k)](si);
					U[hindex(i,j,k)](si) -= w * U[hindex(i-1,j,k)](si);
					U[hindex(i,j,k)](si) = std::max(U[hindex(i,j,k)](si), 0.0);
				}
				U[hindex(i,j,k)](rho_i) = 0.0;
				for (integer si = 0; si != NSPECIES; ++si) {
					U[hindex(i,j,k)](rho_i) += U[hindex(i, j, k)](spc_i + si);
				}
				U[hindex(i,j,k)](rho_i) = std::max(U[hindex(i,j,k)](rho_i), rho_floor);
			}
		}
	}
}
/*
 space_vector& grid::center_of_mass_value(integer i, integer j, integer k) {
 return com[0][gindex(i, j, k)];
 }

 const space_vector& grid::center_of_mass_value(integer i, integer j, integer k) const {
 return com[0][gindex(i, j, k)];
 }*/

space_vector grid::center_of_mass() const {
	PROF_BEGIN;
	space_vector this_com;
	this_com[0] = this_com[1] = this_com[2] = ZERO;
	real m = ZERO;
	for (integer i = 0; i != INX + 0; ++i) {
		for (integer j = 0; j != INX + 0; ++j) {
			for (integer k = 0; k != INX + 0; ++k) {
				const integer iii = gindex(i, j, k);
				const real this_m = is_leaf ? mon[iii] : M[iii]();
				for (auto& dim : geo::dimension::full_set()) {
					this_com[dim] += this_m * com[0][iii][dim];
				}
				m += this_m;
			}
		}
	}
	if (m != ZERO) {
		for (auto& dim : geo::dimension::full_set()) {
			this_com[dim] /= m;
		}
	}
	PROF_END;
	return this_com;
}

grid::grid(real _dx, std::array<real, NDIM> _xmin) :
	U(NF), U0(NF), dUdt(NF), F(NDIM), X(NDIM), G(NGF), is_root(false), is_leaf(true) {
	dx = _dx;
	xmin = _xmin;
	allocate();
}

void grid::compute_primitives(const std::array<integer, NDIM> lb, const std::array<integer, NDIM> ub, bool etot_only) {
	PROF_BEGIN;
	auto& V = TLS_V();
	if (!etot_only) {
		for (integer i = lb[XDIM] - 1; i != ub[XDIM] + 1; ++i) {
			for (integer j = lb[YDIM] - 1; j != ub[YDIM] + 1; ++j) {
#pragma GCC ivdep
				for (integer k = lb[ZDIM] - 1; k != ub[ZDIM] + 1; ++k) {
					const integer iii = hindex(i, j, k);
					V[iii](rho_i) = U[iii](rho_i);
					V[iii](tau_i) = U[iii](tau_i);
					const real rhoinv = 1.0 / V[iii](rho_i);
					V[iii](egas_i) = U[iii](egas_i) * rhoinv;
					for (integer si = 0; si != NSPECIES; ++si) {
						V[iii](spc_i + si) = U[iii](spc_i + si) * rhoinv;
					}
					V[iii](pot_i) = U[iii](pot_i) * rhoinv;
					for (integer d = 0; d != NDIM; ++d) {
						auto& v = V[iii](sx_i + d);
						v = U[iii](sx_i + d) * rhoinv;
						V[iii](egas_i) -= 0.5 * v * v;
						V[iii](zx_i + d) = U[iii](zx_i + d) * rhoinv;
					}
					V[iii](sx_i) += X[YDIM][iii] * omega;
					V[iii](sy_i) -= X[XDIM][iii] * omega;
					V[iii](zz_i) -= dx * dx * omega / 6.0;
				}
			}
		}
	} else {
		for (integer i = lb[XDIM] - 1; i != ub[XDIM] + 1; ++i) {
			for (integer j = lb[YDIM] - 1; j != ub[YDIM] + 1; ++j) {
#pragma GCC ivdep
				for (integer k = lb[ZDIM] - 1; k != ub[ZDIM] + 1; ++k) {
					const integer iii = hindex(i, j, k);
					V[iii](rho_i) = U[iii](rho_i);
					const real rhoinv = 1.0 / V[iii](rho_i);
					V[iii](egas_i) = U[iii](egas_i) * rhoinv;
					for (integer d = 0; d != NDIM; ++d) {
						auto& v = V[iii](sx_i + d);
						v = U[iii](sx_i + d) * rhoinv;
						V[iii](egas_i) -= 0.5 * v * v;
						V[iii](zx_i + d) = U[iii](zx_i + d) * rhoinv;
					}
					V[iii](sx_i) += X[YDIM][iii] * omega;
					V[iii](sy_i) -= X[XDIM][iii] * omega;
					V[iii](zz_i) -= dx * dx * omega / 6.0;
				}
			}
		}
	}
	PROF_END;
}

void grid::compute_primitive_slopes(real theta, const std::array<integer, NDIM> lb, const std::array<integer, NDIM> ub, bool etot_only) {
	PROF_BEGIN;
	auto& dVdx = TLS_dVdx();
	auto& V = TLS_V();
	for (integer f = 0; f != NF; ++f) {
		if (etot_only && (f == tau_i || f == pot_i || (f >= spc_i && f < spc_i + NSPECIES))) {
			continue;
		}
		const auto& v = V;
		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
				for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
					const integer iii = hindex(i,j,k);
					const auto v0 = v[iii](f);
					dVdx[XDIM][iii](f) = minmod_theta(v[iii + H_DNX](f) - v0, v0 - v[iii - H_DNX](f), theta);
					dVdx[YDIM][iii](f) = minmod_theta(v[iii + H_DNY](f) - v0, v0 - v[iii - H_DNY](f), theta);
					dVdx[ZDIM][iii](f) = minmod_theta(v[iii + H_DNZ](f) - v0, v0 - v[iii - H_DNZ](f), theta);
				}
			}
		}
	}
	for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
			for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
				const integer iii = hindex(i,j,k);
				real dV_sym[3][3];
				real dV_ant[3][3];
				for (integer d0 = 0; d0 != NDIM; ++d0) {
					for (integer d1 = 0; d1 != NDIM; ++d1) {
						dV_sym[d1][d0] = (dVdx[d0][iii](sx_i + d1) + dVdx[d1][iii](sx_i + d0)) / 2.0;
						dV_ant[d1][d0] = 0.0;
					}
				}
				dV_ant[XDIM][YDIM] = +6.0 * V[iii](zz_i) / dx;
				dV_ant[XDIM][ZDIM] = -6.0 * V[iii](zy_i) / dx;
				dV_ant[YDIM][ZDIM] = +6.0 * V[iii](zx_i) / dx;
				dV_ant[YDIM][XDIM] = -dV_ant[XDIM][YDIM];
				dV_ant[ZDIM][XDIM] = -dV_ant[XDIM][ZDIM];
				dV_ant[ZDIM][YDIM] = -dV_ant[YDIM][ZDIM];
				for (integer d0 = 0; d0 != NDIM; ++d0) {
					for (integer d1 = 0; d1 != NDIM; ++d1) {
						const real tmp = dV_sym[d0][d1] + dV_ant[d0][d1];
						dVdx[d0][iii](sx_i + d1) = minmod(tmp, 2.0 / theta * dVdx[d0][iii](sx_i + d1));
					}
				}
			}
		}
	}
	PROF_END;
}

void grid::compute_conserved_slopes(const std::array<integer, NDIM> lb, const std::array<integer, NDIM> ub, bool etot_only) {
	PROF_BEGIN;
	auto& dVdx = TLS_dVdx();
	auto& dUdx = TLS_dUdx();
	auto& V = TLS_V();
	const real theta = 1.0;
	if (!etot_only) {
		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
				for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
					const integer iii = hindex(i,j,k);
					V[iii](sx_i) -= X[YDIM][iii] * omega;
					V[iii](sy_i) += X[XDIM][iii] * omega;
					V[iii](zz_i) += dx * dx * omega / 6.0;
					dVdx[YDIM][iii](sx_i) -= dx * omega;
					dVdx[XDIM][iii](sy_i) += dx * omega;
				}
			}
		}
		for (integer d0 = 0; d0 != NDIM; ++d0) {
			auto& dV = dVdx[d0];
			auto& dU = dUdx[d0];
			for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
				for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
					for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
						const integer iii = hindex(i,j,k);
						dU[iii](rho_i) = dV[iii](rho_i);
						for (integer si = 0; si != NSPECIES; ++si) {
							dU[iii](spc_i + si) = V[iii](spc_i + si) * dV[iii](rho_i) + dV[iii](spc_i + si) * V[iii](rho_i);
						}
						dU[iii](pot_i) = V[iii](pot_i) * dV[iii](rho_i) + dV[iii](pot_i) * V[iii](rho_i);
						dU[iii](egas_i) = V[iii](egas_i) * dV[iii](rho_i) + dV[iii](egas_i) * V[iii](rho_i);
						for (integer d1 = 0; d1 != NDIM; ++d1) {
							dU[iii](sx_i + d1) = V[iii](sx_i + d1) * dV[iii](rho_i) + dV[iii](sx_i + d1) * V[iii](rho_i);
							dU[iii](egas_i) += V[iii](rho_i) * (V[iii](sx_i + d1) * dV[iii](sx_i + d1));
							dU[iii](egas_i) += dV[iii](rho_i) * 0.5 * std::pow(V[iii](sx_i + d1), 2);
							dU[iii](zx_i + d1) = V[iii](zx_i + d1) * dV[iii](rho_i); // + dV[zx_i + d1][iii] * V[iii](rho_i);
						}
						dU[iii](tau_i) = dV[iii](tau_i);
					}
				}
			}
		}
	} else {
		for (integer d0 = 0; d0 != NDIM; ++d0) {
			auto& dV = dVdx[d0];
			auto& dU = dUdx[d0];
			for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
				for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
#pragma GCC ivdep
					for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
						const integer iii = hindex(i,j,k);
						dU[iii](egas_i) = V[iii](egas_i) * dV[iii](rho_i) + dV[iii](egas_i) * V[iii](rho_i);
						for (integer d1 = 0; d1 != NDIM; ++d1) {
							dU[iii](egas_i) += V[iii](rho_i) * (V[iii](sx_i + d1) * dV[iii](sx_i + d1));
							dU[iii](egas_i) += dV[iii](rho_i) * 0.5 * std::pow(V[iii](sx_i + d1), 2);
						}
					}
				}
			}
		}
	}

	PROF_END;
}

void grid::set_root(bool flag) {
	is_root = flag;
}

void grid::set_leaf(bool flag) {
	if (is_leaf != flag) {
		is_leaf = flag;
	}
}

void grid::set_coordinates() {
	PROF_BEGIN;
	for (integer i = 0; i != H_NX; ++i) {
		for (integer j = 0; j != H_NX; ++j) {
			for (integer k = 0; k != H_NX; ++k) {
				const integer iii = hindex(i, j, k);
				X[XDIM][iii] = (real(i - H_BW) + HALF) * dx + xmin[XDIM] - pivot[XDIM];
				X[YDIM][iii] = (real(j - H_BW) + HALF) * dx + xmin[YDIM] - pivot[YDIM];
				X[ZDIM][iii] = (real(k - H_BW) + HALF) * dx + xmin[ZDIM] - pivot[ZDIM];
			}
		}
	}
	PROF_END;
}

void grid::allocate() {
	PROF_BEGIN;
//	static std::once_flag flag;
//	std::call_once(flag, compute_ilist);
	U_out0 = std::vector < real > (NF, ZERO);
	U_out = std::vector < real > (NF, ZERO);
	dphi_dt = std::vector < real > (INX * INX * INX);
	for (integer field = 0; field != NGF; ++field) {
		G[field].resize(G_N3);
	}
	for (integer dim = 0; dim != NDIM; ++dim) {
		X[dim].resize(H_N3);
	}
	U0.resize(INX * INX * INX);
	U.resize(H_N3);
	dUdt.resize(INX * INX * INX);
	for (integer dim = 0; dim != NDIM; ++dim) {
		F[dim].resize(F_N3);
	}
	com.resize(2);
	L.resize(G_N3);
	L_c.resize(G_N3);
	integer nlevel = 0;
	com[0].resize(G_N3);
	com[1].resize(G_N3 / 8);

	set_coordinates();
	PROF_END;
}

grid::grid() :
	U(NF), U0(NF), dUdt(NF), F(NDIM), X(NDIM), G(NGF), is_root(false), is_leaf(true), U_out(NF, ZERO), U_out0(NF, ZERO), dphi_dt(H_N3) {
//	allocate();
}

grid::grid(const init_func_type& init_func, real _dx, std::array<real, NDIM> _xmin) :
	U(NF), U0(NF), dUdt(NF), F(NDIM), X(NDIM), G(NGF), is_root(false), is_leaf(true), U_out(NF, ZERO), U_out0(NF, ZERO), dphi_dt(H_N3) {
	PROF_BEGIN;
	dx = _dx;
	xmin = _xmin;
	allocate();
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				std::vector < real > this_u = init_func(X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii], dx);
				for (integer field = 0; field != NF; ++field) {
					U[iii](field) = this_u[field];
				}
				U[iii](zx_i) = ZERO;
				U[iii](zy_i) = ZERO;
				U[iii](zz_i) = ZERO;
			}
		}
	}
	PROF_END;
}

inline real limit_range(real a, real b, real& c) {
	const real max = std::max(a, b);
	const real min = std::min(a, b);
	c = std::min(max, std::max(min, c));
}
;

inline real limit_range_all(real am, real ap, real& bl, real& br) {
	real avg = (br + bl) / 2.0;
	limit_range(am, ap, avg);
	limit_range(am, avg, bl);
	limit_range(ap, avg, br);
}
;

inline real limit_slope(real& ql, real q0, real& qr) {
	const real tmp1 = qr - ql;
	const real tmp2 = qr + ql;
	if ((qr - q0) * (q0 - ql) <= 0.0) {
		qr = ql = q0;
	} else if (tmp1 * (q0 - 0.5 * tmp2) > (1.0 / 6.0) * tmp1 * tmp1) {
		ql = 3.0 * q0 - 2.0 * qr;
	} else if (-(1.0 / 6.0) * tmp1 * tmp1 > tmp1 * (q0 - 0.5 * tmp2)) {
		qr = 3.0 * q0 - 2.0 * ql;
	}
}
;

void grid::reconstruct() {

	PROF_BEGIN;
	auto& Uf = TLS_Uf();
	auto& dUdx = TLS_dUdx();
	auto& dVdx = TLS_dVdx();
	auto& V = TLS_V();

	auto& slpx = dUdx[XDIM];
	auto& slpy = dUdx[YDIM];
	auto& slpz = dUdx[ZDIM];

	compute_primitives();

	for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
#pragma GCC ivdep
		for (integer field = 0; field != NF; ++field) {
			const real theta_x = (field == sy_i || field == sz_i) ? 1.0 : 2.0;
			const real theta_y = (field == sx_i || field == sz_i) ? 1.0 : 2.0;
			const real theta_z = (field == sx_i || field == sy_i) ? 1.0 : 2.0;
			slpx[iii](field) = minmod_theta(V[iii + H_DNX](field) - V[iii](field), V[iii](field) - V[iii - H_DNX](field), theta_x);
			slpy[iii](field) = minmod_theta(V[iii + H_DNY](field) - V[iii](field), V[iii](field) - V[iii - H_DNY](field), theta_y);
			slpz[iii](field) = minmod_theta(V[iii + H_DNZ](field) - V[iii](field), V[iii](field) - V[iii - H_DNZ](field), theta_z);
		}
	}
	for (integer iii = 0; iii != H_N3 - H_NX * H_NX; ++iii) {
#pragma GCC ivdep
		for (integer field = 0; field != NF; ++field) {
			const real& u0 = V[iii](field);
			Uf[FXP][iii](field) = Uf[FXM][iii + H_DNX](field) = (V[iii + H_DNX](field) + u0) * HALF;
			Uf[FYP][iii](field) = Uf[FYM][iii + H_DNY](field) = (V[iii + H_DNY](field) + u0) * HALF;
			Uf[FZP][iii](field) = Uf[FZM][iii + H_DNZ](field) = (V[iii + H_DNZ](field) + u0) * HALF;
		}
	}
	for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
#pragma GCC ivdep
		for (integer field = 0; field != NF; ++field) {
			const real& u0 = V[iii](field);
			const real& sx = slpx[iii](field);
			const real& sy = slpy[iii](field);
			const real& sz = slpz[iii](field);
			Uf[FXP][iii](field) += (-(slpx[iii + H_DNX](field) - sx) / 3.0) * HALF;
			Uf[FXM][iii](field) += (+(slpx[iii - H_DNX](field) - sx) / 3.0) * HALF;
			Uf[FYP][iii](field) += (-(slpy[iii + H_DNY](field) - sy) / 3.0) * HALF;
			Uf[FYM][iii](field) += (+(slpy[iii - H_DNY](field) - sy) / 3.0) * HALF;
			Uf[FZP][iii](field) += (-(slpz[iii + H_DNZ](field) - sz) / 3.0) * HALF;
			Uf[FZM][iii](field) += (+(slpz[iii - H_DNZ](field) - sz) / 3.0) * HALF;
		}
#pragma GCC ivdep
		for (integer field = 0; field != NF; ++field) {
			const real& u0 = V[iii](field);
			limit_slope(Uf[FXM][iii](field), u0, Uf[FXP][iii](field));
			limit_slope(Uf[FYM][iii](field), u0, Uf[FYP][iii](field));
			limit_slope(Uf[FZM][iii](field), u0, Uf[FZP][iii](field));
		}
	}

#pragma GCC ivdep
	for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {

		slpx[iii](sy_i) = slpy[iii](sx_i) = 0.5 * (slpx[iii](sy_i) + slpy[iii](sx_i));
		slpx[iii](sz_i) = slpz[iii](sx_i) = 0.5 * (slpx[iii](sz_i) + slpz[iii](sx_i));
		slpy[iii](sz_i) = slpz[iii](sy_i) = 0.5 * (slpy[iii](sz_i) + slpz[iii](sy_i));

		slpy[iii](sz_i) += 6.0 * V[iii](zx_i) / dx;
		slpz[iii](sy_i) -= 6.0 * V[iii](zx_i) / dx;
		slpx[iii](sz_i) -= 6.0 * V[iii](zy_i) / dx;
		slpz[iii](sx_i) += 6.0 * V[iii](zy_i) / dx;
		slpx[iii](sy_i) += 6.0 * V[iii](zz_i) / dx;
		slpy[iii](sx_i) -= 6.0 * V[iii](zz_i) / dx;

		slpx[iii](sy_i) = minmod(slpx[iii](sy_i), 2.0 * minmod(V[iii + H_DNX](sy_i) - V[iii](sy_i), V[iii](sy_i) - V[iii - H_DNX](sy_i)));
		slpx[iii](sz_i) = minmod(slpx[iii](sz_i), 2.0 * minmod(V[iii + H_DNX](sz_i) - V[iii](sz_i), V[iii](sz_i) - V[iii - H_DNX](sz_i)));
		slpy[iii](sx_i) = minmod(slpy[iii](sx_i), 2.0 * minmod(V[iii + H_DNY](sx_i) - V[iii](sx_i), V[iii](sx_i) - V[iii - H_DNY](sx_i)));
		slpy[iii](sz_i) = minmod(slpy[iii](sz_i), 2.0 * minmod(V[iii + H_DNY](sz_i) - V[iii](sz_i), V[iii](sz_i) - V[iii - H_DNY](sz_i)));
		slpz[iii](sx_i) = minmod(slpz[iii](sx_i), 2.0 * minmod(V[iii + H_DNZ](sx_i) - V[iii](sx_i), V[iii](sx_i) - V[iii - H_DNZ](sx_i)));
		slpz[iii](sy_i) = minmod(slpz[iii](sy_i), 2.0 * minmod(V[iii + H_DNZ](sy_i) - V[iii](sy_i), V[iii](sy_i) - V[iii - H_DNZ](sy_i)));

		const real zx_lim = +(slpy[iii](sz_i) - slpz[iii](sy_i)) / 12.0;
		const real zy_lim = -(slpx[iii](sz_i) - slpz[iii](sx_i)) / 12.0;
		const real zz_lim = +(slpx[iii](sy_i) - slpy[iii](sx_i)) / 12.0;
		for (int face = 0; face != NFACE; ++face) {
			Uf[face][iii](zx_i) = V[iii](zx_i) - zx_lim * dx;
			Uf[face][iii](zy_i) = V[iii](zy_i) - zy_lim * dx;
			Uf[face][iii](zz_i) = V[iii](zz_i) - zz_lim * dx;
		}
		Uf[FXP][iii](sy_i) = V[iii](sy_i) + 0.5 * slpx[iii](sy_i);
		Uf[FXM][iii](sy_i) = V[iii](sy_i) - 0.5 * slpx[iii](sy_i);
		Uf[FXP][iii](sz_i) = V[iii](sz_i) + 0.5 * slpx[iii](sz_i);
		Uf[FXM][iii](sz_i) = V[iii](sz_i) - 0.5 * slpx[iii](sz_i);
		Uf[FYP][iii](sx_i) = V[iii](sx_i) + 0.5 * slpy[iii](sx_i);
		Uf[FYM][iii](sx_i) = V[iii](sx_i) - 0.5 * slpy[iii](sx_i);
		Uf[FYP][iii](sz_i) = V[iii](sz_i) + 0.5 * slpy[iii](sz_i);
		Uf[FYM][iii](sz_i) = V[iii](sz_i) - 0.5 * slpy[iii](sz_i);
		Uf[FZP][iii](sx_i) = V[iii](sx_i) + 0.5 * slpz[iii](sx_i);
		Uf[FZM][iii](sx_i) = V[iii](sx_i) - 0.5 * slpz[iii](sx_i);
		Uf[FZP][iii](sy_i) = V[iii](sy_i) + 0.5 * slpz[iii](sy_i);
		Uf[FZM][iii](sy_i) = V[iii](sy_i) - 0.5 * slpz[iii](sy_i);
	}

	for (integer iii = 0; iii != H_N3; ++iii) {
#pragma GCC ivdep
		for (integer face = 0; face != NFACE; ++face) {
			real w = 0.0;
			for (integer si = 0; si != NSPECIES; ++si) {
				w += Uf[face][iii](spc_i + si);
			}
			if (w > ZERO) {
				for (integer si = 0; si != NSPECIES; ++si) {
					Uf[face][iii](spc_i + si) /= w;
				}
			}
		}
	}

#pragma GCC ivdep
	for (integer iii = H_NX * H_NX; iii != H_N3 - H_NX * H_NX; ++iii) {
		const real phi_x = HALF * (Uf[FXM][iii](pot_i) + Uf[FXP][iii - H_DNX](pot_i));
		const real phi_y = HALF * (Uf[FYM][iii](pot_i) + Uf[FYP][iii - H_DNY](pot_i));
		const real phi_z = HALF * (Uf[FZM][iii](pot_i) + Uf[FZP][iii - H_DNZ](pot_i));
		Uf[FXM][iii](pot_i) = phi_x;
		Uf[FYM][iii](pot_i) = phi_y;
		Uf[FZM][iii](pot_i) = phi_z;
		Uf[FXP][iii - H_DNX](pot_i) = phi_x;
		Uf[FYP][iii - H_DNY](pot_i) = phi_y;
		Uf[FZP][iii - H_DNZ](pot_i) = phi_z;
	}

	for (integer face = 0; face != NFACE; ++face) {
		for (integer iii = 0; iii != H_N3; ++iii) {
#pragma GCC ivdep
			for (integer field = 0; field != NF; ++field) {
				if (field != rho_i && field != tau_i) {
					Uf[face][iii](field) *= Uf[face][iii](rho_i);
				}
			}
		}
	}

	for (integer i = H_BW - 1; i != H_NX - H_BW + 1; ++i) {
		for (integer j = H_BW - 1; j != H_NX - H_BW + 1; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW - 1; k != H_NX - H_BW + 1; ++k) {
				const integer iii = hindex(i, j, k);
				for (integer face = 0; face != NFACE; ++face) {
					real x0 = ZERO;
					real y0 = ZERO;
					if (face == FXP) {
						x0 = +HALF * dx;
					} else if (face == FXM) {
						x0 = -HALF * dx;
					} else if (face == FYP) {
						y0 = +HALF * dx;
					} else if (face == FYM) {
						y0 = -HALF * dx;
					}
					Uf[face][iii](sx_i) -= omega * (X[YDIM][iii] + y0) * Uf[face][iii](rho_i);
					Uf[face][iii](sy_i) += omega * (X[XDIM][iii] + x0) * Uf[face][iii](rho_i);
					Uf[face][iii](zz_i) += dx * dx * omega * Uf[face][iii](rho_i) / 6.0;
					Uf[face][iii](egas_i) += HALF * Uf[face][iii](sx_i) * Uf[face][iii](sx_i) / Uf[face][iii](rho_i);
					Uf[face][iii](egas_i) += HALF * Uf[face][iii](sy_i) * Uf[face][iii](sy_i) / Uf[face][iii](rho_i);
					Uf[face][iii](egas_i) += HALF * Uf[face][iii](sz_i) * Uf[face][iii](sz_i) / Uf[face][iii](rho_i);
				}
			}
		}
	}
	PROF_END;
}

real grid::compute_fluxes() {
	PROF_BEGIN;
	const auto& Uf = TLS_Uf();
	real max_lambda = ZERO;
	std::array<std::vector<real>, NF> ur, ul, f;
	std::vector<space_vector> x;

	const integer line_sz = H_NX - 2 * H_BW + 1;
	for (integer field = 0; field != NF; ++field) {
		ur[field].resize(line_sz);
		ul[field].resize(line_sz);
		f[field].resize(line_sz);
	}
	x.resize(line_sz);

	for (integer dim = 0; dim != NDIM; ++dim) {

		const integer dx_i = dim;
		const integer dy_i = (dim == XDIM ? YDIM : XDIM);
		const integer dz_i = (dim == ZDIM ? YDIM : ZDIM);
		const integer face_p = 2 * dim + 1;
		const integer face_m = 2 * dim;

		for (integer k = H_BW; k != H_NX - H_BW; ++k) {
			for (integer j = H_BW; j != H_NX - H_BW; ++j) {
				for (integer i = H_BW; i != H_NX - H_BW + 1; ++i) {
					const integer i0 = H_DN[dx_i] * i + H_DN[dy_i] * j + H_DN[dz_i] * k;
					const integer im = i0 - H_DN[dx_i];
					for (integer field = 0; field != NF; ++field) {
						ur[field][i - H_BW] = Uf[face_m][i0](field);
						ul[field][i - H_BW] = Uf[face_p][im](field);
					}
					for (integer d = 0; d != NDIM; ++d) {
						x[i - H_BW][d] = (X[d][i0] + X[d][im]) * HALF;
					}
				}
				const real this_max_lambda = roe_fluxes(f, ul, ur, x, omega, dim, dx);
				max_lambda = std::max(max_lambda, this_max_lambda);
				for (integer field = 0; field != NF; ++field) {
					for (integer i = H_BW; i != H_NX - H_BW + 1; ++i) {
						const integer i0 = F_DN[dx_i] * (i - H_BW) + F_DN[dy_i] * (j - H_BW) + F_DN[dz_i] * (k - H_BW);
						F[dim][i0](field) = f[field][i - H_BW];
					}
				}
			}
		}
	}

	PROF_END;
	return max_lambda;
}

void grid::set_max_level(integer l) {
	max_level = l;
}

void grid::store() {
	for (integer field = 0; field != NF; ++field) {
#pragma GCC ivdep
		for (integer i = 0; i != INX; ++i) {
			for (integer j = 0; j != INX; ++j) {
				for (integer k = 0; k != INX; ++k) {
					U0[h0index(i, j, k)](field) = U[hindex(i+H_BW,j+H_BW,k+H_BW)](field);
				}
			}
		}
	}
	U_out0 = U_out;
}

void grid::boundaries() {
	PROF_BEGIN;
	for (integer face = 0; face != NFACE; ++face) {
		set_physical_boundaries(face);
	}
	PROF_END;
}

void grid::set_physical_boundaries(const geo::face& face) {
	const auto dim = face.get_dimension();
	const auto side = face.get_side();
	const integer dni = dim == XDIM ? H_DNY : H_DNX;
	const integer dnj = dim == ZDIM ? H_DNY : H_DNZ;
	const integer dnk = H_DN[dim];
	const integer klb = side == geo::MINUS ? 0 : H_NX - H_BW;
	const integer kub = side == geo::MINUS ? H_BW : H_NX;
	const integer ilb = 0;
	const integer iub = H_NX;
	const integer jlb = 0;
	const integer jub = H_NX;

	for (integer field = 0; field != NF; ++field) {
		for (integer k = klb; k != kub; ++k) {
			for (integer j = jlb; j != jub; ++j) {
				for (integer i = ilb; i != iub; ++i) {
					integer k0;
					switch (boundary_types[face]) {
					case REFLECT:
						k0 = side == geo::MINUS ? (2 * H_BW - k - 1) : (2 * (H_NX - H_BW) - k - 1);
						break;
					case OUTFLOW:
						k0 = side == geo::MINUS ? H_BW : H_NX - H_BW - 1;
						break;
					default:
						k0 = -1;
						assert(false);
						abort();
					}
					const real value = U[i * dni + j * dnj + k0 * dnk](field);
					const integer iii = i * dni + j * dnj + k * dnk;
					real& ref = U[iii](field);
					if (field == sx_i + dim) {
						real s0;
						if (field == sx_i) {
							s0 = -omega * X[YDIM][iii] * U[iii](rho_i);
						} else if (field == sy_i) {
							s0 = +omega * X[XDIM][iii] * U[iii](rho_i);
						} else {
							s0 = ZERO;
						}
						switch (boundary_types[face]) {
						case REFLECT:
							ref = -value;
							break;
						case OUTFLOW:
							const real before = value;
							if (side == geo::MINUS) {
								ref = s0 + std::min(value - s0, ZERO);
							} else {
								ref = s0 + std::max(value - s0, ZERO);
							}
							const real after = ref;
							assert(rho_i < field);
							assert(egas_i < field);
							real this_rho = U[iii](rho_i);
							if (this_rho != ZERO) {
								U[iii](egas_i) += HALF * (after * after - before * before) / this_rho;
							}
							break;
						}
					} else {
						ref = +value;
					}
				}
			}
		}
	}

}

void grid::compute_sources(real t) {
	PROF_BEGIN;
	auto& src = dUdt;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				const integer iii = hindex(i, j, k);
				const integer iiif = findex(i - H_BW, j - H_BW, k - H_BW);
				const integer iiig = gindex(i - H_BW, j - H_BW, k - H_BW);
				for (integer field = 0; field != NF; ++field) {
					src[iii0](field) = ZERO;
				}
				const real rho = U[iii](rho_i);
				src[iii0](zx_i) = (-(F[YDIM][iiif + F_DNY](sz_i) + F[YDIM][iiif](sz_i)) + (F[ZDIM][iiif + F_DNZ](sy_i) + F[ZDIM][iiif](sy_i))) * HALF;
				src[iii0](zy_i) = (+(F[XDIM][iiif + F_DNX](sz_i) + F[XDIM][iiif](sz_i)) - (F[ZDIM][iiif + F_DNZ](sx_i) + F[ZDIM][iiif](sx_i))) * HALF;
				src[iii0](zz_i) = (-(F[XDIM][iiif + F_DNX](sy_i) + F[XDIM][iiif](sy_i)) + (F[YDIM][iiif + F_DNY](sx_i) + F[YDIM][iiif](sx_i))) * HALF;
				src[iii0](sx_i) += rho * G[gx_i][iiig];
				src[iii0](sy_i) += rho * G[gy_i][iiig];
				src[iii0](sz_i) += rho * G[gz_i][iiig];
				src[iii0](sx_i) += omega * U[iii](sy_i);
				src[iii0](sy_i) -= omega * U[iii](sx_i);
				src[iii0](egas_i) -= omega * X[YDIM][iii] * rho * G[gx_i][iiig];
				src[iii0](egas_i) += omega * X[XDIM][iii] * rho * G[gy_i][iiig];
#ifdef USE_DRIVING
				const real period = (2.0 * M_PI / grid::omega);
				if (t < DRIVING_TIME * period) {
					const real ff = -DRIVING_RATE / period;
					const real rho = U[iii](rho_i);
					const real sx = U[iii](sx_i);
					const real sy = U[iii](sy_i);
					const real zz = U[iii](zz_i);
					const real x = X[XDIM][iii];
					const real y = X[YDIM][iii];
					const real R = std::sqrt(x * x + y * y);
					const real lz = (x * sy - y * sx);
					const real dsx = -y / R / R * lz * ff;
					const real dsy = +x / R / R * lz * ff;
					src[sx_i][iii0] += dsx;
					src[sy_i][iii0] += dsy;
					src[iii0](egas_i) += (sx * dsx + sy * dsy) / rho;
					src[iii0](zz_i) += ff * zz;

				}
#endif

			}
		}
	}
	PROF_END;
}

void grid::compute_dudt() {
	PROF_BEGIN;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer field = 0; field != NF; ++field) {
#pragma GCC ivdep
				for (integer k = H_BW; k != H_NX - H_BW; ++k) {
					const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
					const integer iiif = findex(i - H_BW, j - H_BW, k - H_BW);
					dUdt[iii0](field) -= (F[XDIM][iiif + F_DNX](field) - F[XDIM][iiif](field)) / dx;
					dUdt[iii0](field) -= (F[YDIM][iiif + F_DNY](field) - F[YDIM][iiif](field)) / dx;
					dUdt[iii0](field) -= (F[ZDIM][iiif + F_DNZ](field) - F[ZDIM][iiif](field)) / dx;
				}
			}
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				dUdt[iii0](egas_i) += dUdt[iii0](pot_i);
				dUdt[iii0](pot_i) = ZERO;
			}
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				const integer iiig = gindex(i - H_BW, j - H_BW, k - H_BW);
				dUdt[iii0](egas_i) -= (dUdt[iii0](rho_i) * G[phi_i][iiig]) * HALF;
			}
		}
	}
	PROF_END;
//	solve_gravity(DRHODT);
}

void grid::egas_to_etot() {
	PROF_BEGIN;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				U[iii](egas_i) += U[iii](pot_i) * HALF;
			}
		}
	}
	PROF_END;
}

void grid::etot_to_egas() {
	PROF_BEGIN;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				U[iii](egas_i) -= U[iii](pot_i) * HALF;
			}
		}
	}
	PROF_END;
}

void grid::next_u(integer rk, real t, real dt) {
	PROF_BEGIN;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				const integer iii = hindex(i, j, k);
				dUdt[iii0](egas_i) += (dphi_dt[iii0] * U[iii](rho_i)) * HALF;
				dUdt[iii0](zx_i) -= omega * X[ZDIM][iii] * U[iii](sx_i);
				dUdt[iii0](zy_i) -= omega * X[ZDIM][iii] * U[iii](sy_i);
				dUdt[iii0](zz_i) += omega * (X[XDIM][iii] * U[iii](sx_i) + X[YDIM][iii] * U[iii](sy_i));
			}
		}
	}

	std::vector < real > du_out(NF, ZERO);

	std::vector < real > ds(NDIM, ZERO);
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				const integer iii0 = h0index(i - H_BW, j - H_BW, k - H_BW);
				for (integer field = 0; field != NF; ++field) {
					const real u1 = U[iii](field) + dUdt[iii0](field) * dt;
					const real u0 = U0[h0index(i - H_BW, j - H_BW, k - H_BW)](field);
					U[iii](field) = (ONE - rk_beta[rk]) * u0 + rk_beta[rk] * u1;
				}
			}
		}
	}

	du_out[sx_i] += omega * U_out[sy_i] * dt;
	du_out[sy_i] -= omega * U_out[sx_i] * dt;

	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
#pragma GCC ivdep
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			const real dx2 = dx * dx;
			const integer iii_p0 = findex(INX, i - H_BW, j - H_BW);
			const integer jjj_p0 = findex(j - H_BW, INX, i - H_BW);
			const integer kkk_p0 = findex(i - H_BW, j - H_BW, INX);
			const integer iii_m0 = findex(0, i - H_BW, j - H_BW);
			const integer jjj_m0 = findex(j - H_BW, 0, i - H_BW);
			const integer kkk_m0 = findex(i - H_BW, j - H_BW, 0);
			const integer iii_p = H_DNX * (H_NX - H_BW) + H_DNY * i + H_DNZ * j;
			const integer jjj_p = H_DNY * (H_NX - H_BW) + H_DNZ * i + H_DNX * j;
			const integer kkk_p = H_DNZ * (H_NX - H_BW) + H_DNX * i + H_DNY * j;
			const integer iii_m = H_DNX * (H_BW) + H_DNY * i + H_DNZ * j;
			const integer jjj_m = H_DNY * (H_BW) + H_DNZ * i + H_DNX * j;
			const integer kkk_m = H_DNZ * (H_BW) + H_DNX * i + H_DNY * j;
			std::vector < real > du(NF);
			for (integer field = 0; field != NF; ++field) {
//	if (field < zx_i || field > zz_i) {
				du[field] = ZERO;
				if (X[XDIM][iii_p] + pivot[XDIM] > scaling_factor) {
					du[field] += (F[XDIM][iii_p0](field)) * dx2;
				}
				if (X[YDIM][jjj_p] + pivot[YDIM] > scaling_factor) {
					du[field] += (F[YDIM][jjj_p0](field)) * dx2;
				}
				if (X[ZDIM][kkk_p] + pivot[ZDIM] > scaling_factor) {
					du[field] += (F[ZDIM][kkk_p0](field)) * dx2;
				}
				if (X[XDIM][iii_m] + pivot[XDIM] < -scaling_factor + dx) {
					du[field] += (-F[XDIM][iii_m0](field)) * dx2;
				}
				if (X[YDIM][jjj_m] + pivot[YDIM] < -scaling_factor + dx) {
					du[field] += (-F[YDIM][jjj_m0](field)) * dx2;
				}
				if (X[ZDIM][kkk_m] + pivot[ZDIM] < -scaling_factor + dx) {
					du[field] += (-F[ZDIM][kkk_m0](field)) * dx2;
				}
//			}
			}

			if (X[XDIM][iii_p] + pivot[XDIM] > scaling_factor) {
				const real xp = X[XDIM][iii_p] - HALF * dx;
				du[zx_i] += (X[YDIM][iii_p] * F[XDIM][iii_p0](sz_i)) * dx2;
				du[zx_i] -= (X[ZDIM][iii_p] * F[XDIM][iii_p0](sy_i)) * dx2;
				du[zy_i] -= (xp * F[XDIM][iii_p0](sz_i)) * dx2;
				du[zy_i] += (X[ZDIM][iii_p] * F[XDIM][iii_p0](sx_i)) * dx2;
				du[zz_i] += (xp * F[XDIM][iii_p0](sy_i)) * dx2;
				du[zz_i] -= (X[YDIM][iii_p] * F[XDIM][iii_p0](sx_i)) * dx2;
			}
			if (X[YDIM][jjj_p] + pivot[YDIM] > scaling_factor) {
				const real yp = X[YDIM][jjj_p] - HALF * dx;
				du[zx_i] += (yp * F[YDIM][jjj_p0](sz_i)) * dx2;
				du[zx_i] -= (X[ZDIM][jjj_p] * F[YDIM][jjj_p0](sy_i)) * dx2;
				du[zy_i] -= (X[XDIM][jjj_p] * F[YDIM][jjj_p0](sz_i)) * dx2;
				du[zy_i] += (X[ZDIM][jjj_p] * F[YDIM][jjj_p0](sx_i)) * dx2;
				du[zz_i] += (X[XDIM][jjj_p] * F[YDIM][jjj_p0](sy_i)) * dx2;
				du[zz_i] -= (yp * F[YDIM][jjj_p0](sx_i)) * dx2;
			}
			if (X[ZDIM][kkk_p] + pivot[ZDIM] > scaling_factor) {
				const real zp = X[ZDIM][kkk_p] - HALF * dx;
				du[zx_i] -= (zp * F[ZDIM][kkk_p0](sy_i)) * dx2;
				du[zx_i] += (X[YDIM][kkk_p] * F[ZDIM][kkk_p0](sz_i)) * dx2;
				du[zy_i] += (zp * F[ZDIM][kkk_p0](sx_i)) * dx2;
				du[zy_i] -= (X[XDIM][kkk_p] * F[ZDIM][kkk_p0](sz_i)) * dx2;
				du[zz_i] += (X[XDIM][kkk_p] * F[ZDIM][kkk_p0](sy_i)) * dx2;
				du[zz_i] -= (X[YDIM][kkk_p] * F[ZDIM][kkk_p0](sx_i)) * dx2;
			}

			if (X[XDIM][iii_m] + pivot[XDIM] < -scaling_factor + dx) {
				const real xm = X[XDIM][iii_m] - HALF * dx;
				du[zx_i] += (-X[YDIM][iii_m] * F[XDIM][iii_m0](sz_i)) * dx2;
				du[zx_i] -= (-X[ZDIM][iii_m] * F[XDIM][iii_m0](sy_i)) * dx2;
				du[zy_i] -= (-xm * F[XDIM][iii_m0](sz_i)) * dx2;
				du[zy_i] += (-X[ZDIM][iii_m] * F[XDIM][iii_m0](sx_i)) * dx2;
				du[zz_i] += (-xm * F[XDIM][iii_m0](sy_i)) * dx2;
				du[zz_i] -= (-X[YDIM][iii_m] * F[XDIM][iii_m0](sx_i)) * dx2;
			}
			if (X[YDIM][jjj_m] + pivot[YDIM] < -scaling_factor + dx) {
				const real ym = X[YDIM][jjj_m] - HALF * dx;
				du[zx_i] -= (-X[ZDIM][jjj_m] * F[YDIM][jjj_m0](sy_i)) * dx2;
				du[zx_i] += (-ym * F[YDIM][jjj_m0](sz_i)) * dx2;
				du[zy_i] -= (-X[XDIM][jjj_m] * F[YDIM][jjj_m0](sz_i)) * dx2;
				du[zy_i] += (-X[ZDIM][jjj_m] * F[YDIM][jjj_m0](sx_i)) * dx2;
				du[zz_i] += (-X[XDIM][jjj_m] * F[YDIM][jjj_m0](sy_i)) * dx2;
				du[zz_i] -= (-ym * F[YDIM][jjj_m0](sx_i)) * dx2;
			}
			if (X[ZDIM][kkk_m] + pivot[ZDIM] < -scaling_factor + dx) {
				const real zm = X[ZDIM][kkk_m] - HALF * dx;
				du[zx_i] -= (-zm * F[ZDIM][kkk_m0](sy_i)) * dx2;
				du[zx_i] += (-X[YDIM][kkk_m] * F[ZDIM][kkk_m0](sz_i)) * dx2;
				du[zy_i] += (-zm * F[ZDIM][kkk_m0](sx_i)) * dx2;
				du[zy_i] -= (-X[XDIM][kkk_m] * F[ZDIM][kkk_m0](sz_i)) * dx2;
				du[zz_i] += (-X[XDIM][kkk_m] * F[ZDIM][kkk_m0](sy_i)) * dx2;
				du[zz_i] -= (-X[YDIM][kkk_m] * F[ZDIM][kkk_m0](sx_i)) * dx2;
			}
			for (integer field = 0; field != NF; ++field) {
				du_out[field] += du[field] * dt;
			}
		}
	}
#pragma GCC ivdep
	for (integer field = 0; field != NF; ++field) {
		const real out1 = U_out[field] + du_out[field];
		const real out0 = U_out0[field];
		U_out[field] = (ONE - rk_beta[rk]) * out0 + rk_beta[rk] * out1;
	}
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				U[iii](rho_i) = ZERO;
				for (integer si = 0; si != NSPECIES; ++si) {
					U[iii](rho_i) += U[iii](spc_i + si);
				}
				if (U[iii](tau_i) < ZERO) {
					printf("Tau is negative- %e\n", double(U[iii](tau_i)));
					abort();
				} else if (U[iii](rho_i) <= ZERO) {
					printf("Rho is non-positive - %e\n", double(U[iii](rho_i)));
					abort();
				}

			}
		}
	}
	PROF_END;
}

void grid::dual_energy_update() {
	PROF_BEGIN;
	bool in_bnd;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
#pragma GCC ivdep
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				real ek = ZERO;
				ek += HALF * pow(U[iii](sx_i), 2) / U[iii](rho_i);
				ek += HALF * pow(U[iii](sy_i), 2) / U[iii](rho_i);
				ek += HALF * pow(U[iii](sz_i), 2) / U[iii](rho_i);
				real ei = U[iii](egas_i) - ek;
				real et = U[iii](egas_i);
				et = std::max(et, U[iii + H_DNX](egas_i));
				et = std::max(et, U[iii - H_DNX](egas_i));
				et = std::max(et, U[iii + H_DNY](egas_i));
				et = std::max(et, U[iii - H_DNY](egas_i));
				et = std::max(et, U[iii + H_DNZ](egas_i));
				et = std::max(et, U[iii - H_DNZ](egas_i));
				if (ei > de_switch1 * et) {
					U[iii](tau_i) = std::pow(ei, ONE / fgamma);
				}
			}
		}
	}
	PROF_END;
}

std::vector<real> grid::conserved_outflows() const {
	auto Uret = U_out;
	Uret[egas_i] += Uret[pot_i];
	return Uret;
}
