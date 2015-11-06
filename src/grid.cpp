/*
 * grid.cpp
 *
 *  Created on: May 26, 2015
 *      Author: dmarce1
 */

#include "grid.hpp"
#include <cmath>
#include <cassert>

real grid::omega = ZERO;
space_vector grid::pivot(ZERO);
void grid::set_omega(real o) {
	omega = o;
}

void grid::set_pivot(const space_vector& p) {
	pivot = p;
}

std::pair<real, real> grid::omega_part(const space_vector& pivot) const {
	real I = ZERO;
	real L = ZERO;
	const real dV = dx * dx * dx;
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = DNX * i + DNY * j + DNZ * k;
				const real x = X[XDIM][iii] - pivot[XDIM];
				const real y = X[YDIM][iii] - pivot[YDIM];
				const real sx = U[sx_i][iii];
				const real sy = U[sy_i][iii];
				const real zz = U[zz_i][iii];
				const real rho = U[rho_i][iii];
				L += (x * sy - y * sx + zz) * dV;
				I += (x * x + y * y) * rho * dV;
			}
		}
	}
//	printf( "%e\n", L);
	return std::make_pair(L, I);
}

std::vector<real> grid::get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
		const geo::dimension& dim) const {
	std::vector<real> data;
	integer size = 1;
	for (auto& dim : geo::dimension::full_set()) {
		size *= (ub[dim] - lb[dim]);
	}
	size /= (NCHILD / 2);
	size *= NF;
	data.reserve(size);
	const integer stride1 = (dim == XDIM) ? DNY : DNX;
	const integer stride2 = (dim == ZDIM) ? DNY : DNZ;
	for (integer field = 0; field != NF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; i += 2) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; j += 2) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; k += 2) {
					const integer i00 = DNX * i + DNY * j + DNZ * k;
					const integer i10 = i00 + stride1;
					const integer i01 = i00 + stride2;
					const integer i11 = i00 + stride1 + stride2;
					real value = ZERO;
					value += F[dim][field][i00];
					value += F[dim][field][i10];
					value += F[dim][field][i01];
					value += F[dim][field][i11];
					const real f = dx / 2.0;
					if (field == zx_i) {
						if (dim == YDIM) {
							value += F[dim][sy_i][i00] * f;
							value += F[dim][sy_i][i10] * f;
							value -= F[dim][sy_i][i01] * f;
							value -= F[dim][sy_i][i11] * f;
						} else if (dim == ZDIM) {
							value -= F[dim][sz_i][i00] * f;
							value -= F[dim][sz_i][i10] * f;
							value += F[dim][sz_i][i01] * f;
							value += F[dim][sz_i][i11] * f;
						} else if (dim == XDIM) {
							value += F[dim][sy_i][i00] * f;
							value += F[dim][sy_i][i10] * f;
							value -= F[dim][sy_i][i01] * f;
							value -= F[dim][sy_i][i11] * f;
							value -= F[dim][sz_i][i00] * f;
							value += F[dim][sz_i][i10] * f;
							value -= F[dim][sz_i][i01] * f;
							value += F[dim][sz_i][i11] * f;
						}
					} else if (field == zy_i) {
						if (dim == XDIM) {
							value -= F[dim][sx_i][i00] * f;
							value -= F[dim][sx_i][i10] * f;
							value += F[dim][sx_i][i01] * f;
							value += F[dim][sx_i][i11] * f;
						} else if (dim == ZDIM) {
							value += F[dim][sz_i][i00] * f;
							value -= F[dim][sz_i][i10] * f;
							value += F[dim][sz_i][i01] * f;
							value -= F[dim][sz_i][i11] * f;
						} else if (dim == YDIM) {
							value -= F[dim][sx_i][i00] * f;
							value -= F[dim][sx_i][i10] * f;
							value += F[dim][sx_i][i01] * f;
							value += F[dim][sx_i][i11] * f;
							value += F[dim][sz_i][i00] * f;
							value -= F[dim][sz_i][i10] * f;
							value += F[dim][sz_i][i01] * f;
							value -= F[dim][sz_i][i11] * f;
						}
					} else if (field == zz_i) {
						if (dim == XDIM) {
							value += F[dim][sx_i][i00] * f;
							value -= F[dim][sx_i][i10] * f;
							value += F[dim][sx_i][i01] * f;
							value -= F[dim][sx_i][i11] * f;
						} else if (dim == YDIM) {
							value -= F[dim][sy_i][i00] * f;
							value += F[dim][sy_i][i10] * f;
							value -= F[dim][sy_i][i01] * f;
							value += F[dim][sy_i][i11] * f;
						} else if (dim == ZDIM) {
							value -= F[dim][sy_i][i00] * f;
							value += F[dim][sy_i][i10] * f;
							value -= F[dim][sy_i][i01] * f;
							value += F[dim][sy_i][i11] * f;
							value += F[dim][sx_i][i00] * f;
							value += F[dim][sx_i][i10] * f;
							value -= F[dim][sx_i][i01] * f;
							value -= F[dim][sx_i][i11] * f;
						}
					}
					value /= real(4);
					data.push_back(value);
				}
			}
		}
	}
//				src[zz_i][iii] = (-(F[XDIM][sy_i][iii + DNX] + F[XDIM][sy_i][iii])
//						+ (F[YDIM][sx_i][iii + DNY] + F[YDIM][sx_i][iii])) * HALF;

	return data;
}

void grid::set_flux_restrict(const std::vector<real>& data, const std::array<integer, NDIM>& lb,
		const std::array<integer, NDIM>& ub, const geo::dimension& dim) {
	integer index = 0;
	for (integer field = 0; field != NF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					const integer iii = i * DNX + j * DNY + k * DNZ;
					F[dim][field][iii] = data[index];
					++index;
				}
			}
		}
	}
}

void grid::set_outflows(std::vector<real>&& u) {
	U_out = std::move(u);
}

void grid::set_prolong(const std::vector<real>& data, std::vector<real>&& outflows) {
	integer index = 0;
	U_out = std::move(outflows);
	for (integer field = 0; field != NF; ++field) {
		for (integer i = HBW; i != HNX - HBW; ++i) {
			for (integer j = HBW; j != HNX - HBW; ++j) {
				for (integer k = HBW; k != HNX - HBW; ++k) {
					const integer iii = i * DNX + j * DNY + k * DNZ;
					auto& value = U[field][iii];
					value = data[index];
					/*	if (field == zx_i) {
					 value -= X[YDIM][iii] * U[sz_i][iii];
					 value += X[ZDIM][iii] * U[sy_i][iii];
					 } else if (field == zy_i) {
					 value += X[XDIM][iii] * U[sz_i][iii];
					 value -= X[ZDIM][iii] * U[sx_i][iii];
					 } else if (field == zz_i) {
					 value -= X[XDIM][iii] * U[sy_i][iii];
					 value += X[YDIM][iii] * U[sx_i][iii];
					 }*/
					++index;
				}
			}
		}
	}
}

std::vector<real> grid::get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub) const {
	std::vector<real> data;
	integer size = 1;
	for (integer dim = 0; dim != NDIM; ++dim) {
		size *= (ub[dim] - lb[dim]);
	}
	data.reserve(size);
	for (integer field = 0; field != NF; ++field) {
		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
			for (integer i1 = 0; i1 != 2; ++i1) {
				for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
					for (integer j1 = 0; j1 != 2; ++j1) {
						for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
							for (integer k1 = 0; k1 != 2; ++k1) {
								const integer iii = i * DNX + j * DNY + k * DNZ;
								real value = U[field][iii];
								/*	if (field == zx_i) {
								 value += X[YDIM][iii] * U[sz_i][iii];
								 value -= X[ZDIM][iii] * U[sy_i][iii];
								 } else if (field == zy_i) {
								 value -= X[XDIM][iii] * U[sz_i][iii];
								 value += X[ZDIM][iii] * U[sx_i][iii];
								 } else if (field == zz_i) {
								 value += X[XDIM][iii] * U[sy_i][iii];
								 value -= X[YDIM][iii] * U[sx_i][iii];
								 }*/
								data.push_back(value);
							}
						}
					}
				}
			}
		}
	}
	return data;
}

std::vector<real> grid::get_restrict() const {
	constexpr
	integer Size = NF * INX * INX * INX / NCHILD + NF;
	std::vector<real> data;
	data.reserve(Size);
	for (integer field = 0; field != NF; ++field) {
		for (integer i = HBW; i < HNX - HBW; i += 2) {
			for (integer j = HBW; j < HNX - HBW; j += 2) {
				for (integer k = HBW; k < HNX - HBW; k += 2) {
					const integer iii = i * DNX + j * DNY + k * DNZ;
					real pt = ZERO;
					for (integer x = 0; x != 2; ++x) {
						for (integer y = 0; y != 2; ++y) {
							for (integer z = 0; z != 2; ++z) {
								const integer jjj = iii + x * DNX + y * DNY + z * DNZ;
								pt += U[field][jjj];
								if (field == zx_i) {
									pt += X[YDIM][jjj] * U[sz_i][jjj];
									pt -= X[ZDIM][jjj] * U[sy_i][jjj];
								} else if (field == zy_i) {
									pt -= X[XDIM][jjj] * U[sz_i][jjj];
									pt += X[ZDIM][jjj] * U[sx_i][jjj];
								} else if (field == zz_i) {
									pt += X[XDIM][jjj] * U[sy_i][jjj];
									pt -= X[YDIM][jjj] * U[sx_i][jjj];
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

	return data;
}

void grid::set_restrict(const std::vector<real>& data, const geo::octant& octant) {
	integer index = 0;
	const integer i0 = octant.get_side(XDIM) * (INX / 2);
	const integer j0 = octant.get_side(YDIM) * (INX / 2);
	const integer k0 = octant.get_side(ZDIM) * (INX / 2);
	for (integer field = 0; field != NF; ++field) {
		for (integer i = HBW; i != HNX / 2; ++i) {
			for (integer j = HBW; j != HNX / 2; ++j) {
				for (integer k = HBW; k != HNX / 2; ++k) {
					const integer iii = (i + i0) * DNX + (j + j0) * DNY + (k + k0) * DNZ;
					auto& v = U[field][iii];
					v = data[index];
					if (field == zx_i) {
						v -= X[YDIM][iii] * U[sz_i][iii];
						v += X[ZDIM][iii] * U[sy_i][iii];
					} else if (field == zy_i) {
						v += X[XDIM][iii] * U[sz_i][iii];
						v -= X[ZDIM][iii] * U[sx_i][iii];
					} else if (field == zz_i) {
						v -= X[XDIM][iii] * U[sy_i][iii];
						v += X[YDIM][iii] * U[sx_i][iii];
					}
					++index;
				}
			}
		}
	}
}

std::pair<std::vector<real>, std::vector<real> > grid::field_range() const {
	std::pair<std::vector<real>, std::vector<real> > minmax;
	minmax.first.resize(NF);
	minmax.second.resize(NF);
	for (integer field = 0; field != NF; ++field) {
		minmax.first[field] = +std::numeric_limits<real>::max();
		minmax.second[field] = -std::numeric_limits<real>::max();
	}
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = i * DNX + j * DNY + k * DNZ;
				for (integer field = 0; field != NF; ++field) {
					minmax.first[field] = std::min(minmax.first[field], U[field][iii]);
					minmax.second[field] = std::max(minmax.second[field], U[field][iii]);
				}
			}
		}
	}
	return minmax;
}

std::vector<real> grid::conserved_sums() const {
	std::vector<real> sum(NF, ZERO);
	const real dV = dx * dx * dx;
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = i * DNX + j * DNY + k * DNZ;
				for (integer field = 0; field != NF; ++field) {
					sum[field] += U[field][iii] * dV;
				}
				sum[egas_i] += U[pot_i][iii] * HALF * dV;
				sum[zx_i] += X[YDIM][iii] * U[sz_i][iii] * dV;
				sum[zx_i] -= X[ZDIM][iii] * U[sy_i][iii] * dV;
				sum[zy_i] -= X[XDIM][iii] * U[sz_i][iii] * dV;
				sum[zy_i] += X[ZDIM][iii] * U[sx_i][iii] * dV;
				sum[zz_i] += X[XDIM][iii] * U[sy_i][iii] * dV;
				sum[zz_i] -= X[YDIM][iii] * U[sx_i][iii] * dV;
			}
		}
	}
	return sum;
}

std::vector<real> grid::l_sums() const {
	std::vector<real> sum(NDIM);
	const real dV = dx * dx * dx;
	std::fill(sum.begin(), sum.end(), ZERO);
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = i * DNX + j * DNY + k * DNZ;
				sum[XDIM] += X[YDIM][iii] * U[sz_i][iii] * dV;
				sum[XDIM] -= X[ZDIM][iii] * U[sy_i][iii] * dV;

				sum[YDIM] -= X[XDIM][iii] * U[sz_i][iii] * dV;
				sum[YDIM] += X[ZDIM][iii] * U[sx_i][iii] * dV;

				sum[ZDIM] += X[XDIM][iii] * U[sy_i][iii] * dV;
				sum[ZDIM] -= X[YDIM][iii] * U[sx_i][iii] * dV;

			}
		}
	}
	return sum;
}

bool grid::refine_me(integer lev) const {
	if (lev == 0) {
		return true;
	}
	bool rc;
	for (integer iii = 0; iii != HN3; ++iii) {
		real rho_ref;
		if (U[acc_i][iii] > U[don_i][iii]) {
			rho_ref = 7.5 * 1.0e-6;
		} else {
			rho_ref = 0.28 * 1.0e-6;
		}
		if (U[rho_i][iii] > rho_ref) {
			rc = (lev < MAX_LEVEL);
		} else if (U[rho_i][iii] > rho_ref / 8.0) {
			rc = (lev < MAX_LEVEL - 1);
		} else if (U[rho_i][iii] > rho_ref / 64.0) {
			rc = (lev < MAX_LEVEL - 2);
		} else {
			rc = false;
		}
		if (rc) {
			return true;
		}
	}
	return false;
}

integer grid::level_count() const {
	return nlevel;
}

grid::~grid() {

}

real& grid::hydro_value(integer f, integer i, integer j, integer k) {
	return U[f][i * DNX + j * DNY + k * DNZ];
}

space_vector& grid::center_of_mass_value(integer i, integer j, integer k) {
	return com[0][i * HNX * HNX + j * HNX + k];
}

const space_vector& grid::center_of_mass_value(integer i, integer j, integer k) const {
	return com[0][i * HNX * HNX + j * HNX + k];
}

space_vector grid::center_of_mass() const {
	space_vector this_com;
	this_com[0] = this_com[1] = this_com[2] = ZERO;
	real m = ZERO;
	for (integer i = HBW; i != INX + HBW; ++i) {
		for (integer j = HBW; j != INX + HBW; ++j) {
			for (integer k = HBW; k != INX + HBW; ++k) {
				const integer iii = i * DNX + j * DNY + k * DNZ;
				const real& this_m = M[0][iii]();
				for (auto& dim : geo::dimension::full_set()) {
					this_com[dim] += this_m * com[dim][iii][dim];
				}
				m += this_m;
			}
		}
	}
	for (auto& dim : geo::dimension::full_set()) {
		this_com[dim] /= m;
	}
	return this_com;
}

multipole& grid::multipole_value(integer lev, integer i, integer j, integer k) {
	const integer bw = HBW;
	const integer inx = INX >> lev;
	const integer nx = 2 * bw + inx;
	return M[lev][i * nx * nx + j * nx + k];
}

const multipole& grid::multipole_value(integer lev, integer i, integer j, integer k) const {
	const integer bw = HBW;
	const integer inx = INX >> lev;
	const integer nx = 2 * bw + inx;
	return M[lev][i * nx * nx + j * nx + k];
}

real grid::hydro_value(integer f, integer i, integer j, integer k) const {
	return U[f][i * DNX + j * DNY + k * DNZ];
}

inline real minmod(real a, real b) {
	return (std::copysign(HALF, a) + std::copysign(HALF, b)) * std::min(std::abs(a), std::abs(b));
}

inline real minmod_theta(real a, real b, real c, real theta) {
	return minmod(theta * minmod(a, b), c);
}

inline real minmod_theta(real a, real b, real theta) {
	return minmod(theta * minmod(a, b), HALF * (a + b));
}

grid::grid(real _dx, std::array<real, NDIM> _xmin) :
		U(NF), U0(NF), dUdt(NF), Uf(NFACE), F(NDIM), X(NDIM), G(NGF), G0(NGF), src(NF), ilist_d_bnd(
				geo::direction::count()), ilist_n_bnd(geo::direction::count()), is_root(false), is_leaf(true) {
	dx = _dx;
	xmin = _xmin;
	allocate();
}

void grid::set_root(bool flag) {
	if (is_root != flag) {
		is_root = flag;
		integer this_nlevel = 0;
		for (integer inx = INX; inx > 1; inx /= 2) {
			const integer this_nx = inx + 2 * HBW;
			const integer sz = this_nx * this_nx * this_nx;
			com[this_nlevel].resize(sz);
			if (is_root || (inx >= INX / 2)) {
				M[this_nlevel].resize(sz);
			}
			if (is_root || (inx >= INX)) {
				L[this_nlevel].resize(sz);
				L_c[this_nlevel].resize(sz);
			}
			++this_nlevel;
		}
		compute_ilist();
	}
}

void grid::set_leaf(bool flag) {
	if (is_leaf != flag) {
		is_leaf = flag;
		compute_ilist();
	}
}

void grid::set_coordinates() {
	for (integer i = 0; i != HNX; ++i) {
		for (integer j = 0; j != HNX; ++j) {
			for (integer k = 0; k != HNX; ++k) {
				const integer iii = i * DNX + j * DNY + k * DNZ;
				X[XDIM][iii] = (real(i - HBW) + HALF) * dx + xmin[XDIM] - pivot[XDIM];
				X[YDIM][iii] = (real(j - HBW) + HALF) * dx + xmin[YDIM] - pivot[YDIM];
				X[ZDIM][iii] = (real(k - HBW) + HALF) * dx + xmin[ZDIM] - pivot[ZDIM];
			}
		}
	}
}

void grid::allocate() {
	U_out0 = std::vector<real>(NF, ZERO);
	U_out = std::vector<real>(NF, ZERO);
	dphi_dt = std::vector<real>(HN3);
	for (integer field = 0; field != NGF; ++field) {
		G[field].resize(HN3);
		G0[field].resize(HN3);
	}
	for (integer dim = 0; dim != NDIM; ++dim) {
		X[dim].resize(HN3);
	}
	for (integer field = 0; field != NF; ++field) {
		src[field].resize(HN3);
		U0[field].resize(HN3);
		U[field].resize(HN3);
		dUdt[field].resize(HN3);
		for (integer dim = 0; dim != NDIM; ++dim) {
			F[dim][field].resize(HN3);
		}
		for (integer face = 0; face != NFACE; ++face) {
			Uf[face][field].resize(HN3);
		}
	}
	nlevel = 0;
	for (integer inx = INX; inx > 1; inx /= 2) {
		++nlevel;
	}
	com.resize(nlevel);
	M.resize(nlevel);
	L.resize(nlevel);
	L_c.resize(nlevel);
	nlevel = 0;
	for (integer inx = INX; inx > 1; inx /= 2) {
		const integer this_nx = inx + 2 * HBW;
		const integer sz = this_nx * this_nx * this_nx;
		com[nlevel].resize(sz);
		if (is_root || (inx >= INX / 2)) {
			M[nlevel].resize(sz);
		}
		if (is_root || (inx >= INX)) {
			L[nlevel].resize(sz);
			L_c[nlevel].resize(sz);
		}
		++nlevel;
	}
	set_coordinates();
	compute_ilist();

}

grid::grid() :
		U(NF), U0(NF), dUdt(NF), Uf(NFACE), F(NDIM), X(NDIM), G(NGF), G0(NGF), src(NF), ilist_d_bnd(
				geo::direction::count()), ilist_n_bnd(geo::direction::count()) {
	;
}

grid::grid(const std::function<std::vector<real>(real, real, real)>& init_func, real _dx, std::array<real, NDIM> _xmin) :
		U(NF), U0(NF), dUdt(NF), Uf(NFACE), F(NDIM), X(NDIM), G(NGF), G0(NGF), src(NF), ilist_d_bnd(
				geo::direction::count()), ilist_n_bnd(geo::direction::count()), is_root(false), is_leaf(true), U_out(NF,
				ZERO), U_out0(NF, ZERO), dphi_dt(HN3) {
	dx = _dx;
	xmin = _xmin;
	allocate();
	for (integer i = 0; i != HNX; ++i) {
		for (integer j = 0; j != HNX; ++j) {
			for (integer k = 0; k != HNX; ++k) {
				const integer iii = i * DNX + j * DNY + k * DNZ;
				std::vector<real> this_u = init_func(X[XDIM][iii], X[YDIM][iii], X[ZDIM][iii]);
				for (integer field = 0; field != NF; ++field) {
					U[field][iii] = this_u[field];
				}
				U[zx_i][iii] = ZERO;
				U[zy_i][iii] = ZERO;
				U[zz_i][iii] = ZERO;
			}
		}
	}
}


void grid::reconstruct() {

//#pragma GCC ivdep
	for (integer iii = 0; iii != HN3; ++iii) {
		const real x = X[XDIM][iii];
		const real y = X[YDIM][iii];
		const real& rho = U[rho_i][iii];
		U[egas_i][iii] -= HALF * U[sx_i][iii] * U[sx_i][iii] / rho;
		U[egas_i][iii] -= HALF * U[sy_i][iii] * U[sy_i][iii] / rho;
		U[sx_i][iii] += omega * y * rho;
		U[sy_i][iii] -= omega * x * rho;
		U[zz_i][iii] -= std::pow(dx * omega, 2) * rho / 6.0;
		U[egas_i][iii] += HALF * U[sx_i][iii] * U[sx_i][iii] / rho;
		U[egas_i][iii] += HALF * U[sy_i][iii] * U[sy_i][iii] / rho;
	}

	std::array<std::vector<real>, NF> slpx, slpy, slpz;
	for (integer field = 0; field != NF; ++field) {
		slpx[field].resize(HN3);
		slpy[field].resize(HN3);
		slpz[field].resize(HN3);
		if (field != rho_i && field != tau_i) {
#pragma GCC ivdep
			for (integer iii = 0; iii != HN3; ++iii) {
				U[field][iii] /= U[rho_i][iii];
			}
		}
#pragma GCC ivdep
		for (integer iii = HNX * HNX; iii != HN3 - HNX * HNX; ++iii) {
			const real u0 = U[field][iii];
			if (field < zx_i || field > zz_i) {
				const real theta = 1.3;
				slpx[field][iii] = minmod_theta(U[field][iii + DNX] - u0, u0 - U[field][iii - DNX], theta);
				slpy[field][iii] = minmod_theta(U[field][iii + DNY] - u0, u0 - U[field][iii - DNY], theta);
				slpz[field][iii] = minmod_theta(U[field][iii + DNZ] - u0, u0 - U[field][iii - DNZ], theta);
			} else {
				slpx[field][iii] = slpy[field][iii] = slpz[field][iii] = ZERO;
			}
		}

	}
	const real TWELVE = 12;
	auto slpx0 = slpx;
	auto slpy0 = slpy;
	auto slpz0 = slpz;
#pragma GCC ivdep
	for (integer iii = HNX * HNX; iii != HN3 - HNX * HNX; ++iii) {
		slpx0[sy_i][iii] = HALF * (U[sy_i][iii + DNX] - U[sy_i][iii - DNX]);
		slpx0[sz_i][iii] = HALF * (U[sz_i][iii + DNX] - U[sz_i][iii - DNX]);
		slpy0[sx_i][iii] = HALF * (U[sx_i][iii + DNY] - U[sx_i][iii - DNY]);
		slpy0[sz_i][iii] = HALF * (U[sz_i][iii + DNY] - U[sz_i][iii - DNY]);
		slpz0[sx_i][iii] = HALF * (U[sx_i][iii + DNZ] - U[sx_i][iii - DNZ]);
		slpz0[sy_i][iii] = HALF * (U[sy_i][iii + DNZ] - U[sy_i][iii - DNZ]);
		const real lx_sum = slpy0[sz_i][iii] + slpz0[sy_i][iii];
		const real ly_sum = slpx0[sz_i][iii] + slpz0[sx_i][iii];
		const real lz_sum = slpx0[sy_i][iii] + slpy0[sx_i][iii];
		const real lx_dif = (+U[zx_i][iii] / dx) * TWELVE;
		const real ly_dif = (-U[zy_i][iii] / dx) * TWELVE;
		const real lz_dif = (+U[zz_i][iii] / dx) * TWELVE;
		slpx0[sy_i][iii] = (lz_sum + lz_dif) * HALF;
		slpy0[sx_i][iii] = (lz_sum - lz_dif) * HALF;
		slpx0[sz_i][iii] = (ly_sum + ly_dif) * HALF;
		slpz0[sx_i][iii] = (ly_sum - ly_dif) * HALF;
		slpy0[sz_i][iii] = (lx_sum + lx_dif) * HALF;
		slpz0[sy_i][iii] = (lx_sum - lx_dif) * HALF;
		const real theta = 1.99;
		slpx[sy_i][iii] = minmod_theta(U[sy_i][iii + DNX] - U[sy_i][iii], U[sy_i][iii] - U[sy_i][iii - DNX],
				slpx0[sy_i][iii], theta);
		slpx[sz_i][iii] = minmod_theta(U[sz_i][iii + DNX] - U[sz_i][iii], U[sz_i][iii] - U[sz_i][iii - DNX],
				slpx0[sz_i][iii], theta);
		slpy[sx_i][iii] = minmod_theta(U[sx_i][iii + DNY] - U[sx_i][iii], U[sx_i][iii] - U[sx_i][iii - DNY],
				slpy0[sx_i][iii], theta);
		slpy[sz_i][iii] = minmod_theta(U[sz_i][iii + DNY] - U[sz_i][iii], U[sz_i][iii] - U[sz_i][iii - DNY],
				slpy0[sz_i][iii], theta);
		slpz[sx_i][iii] = minmod_theta(U[sx_i][iii + DNZ] - U[sx_i][iii], U[sx_i][iii] - U[sx_i][iii - DNZ],
				slpz0[sx_i][iii], theta);
		slpz[sy_i][iii] = minmod_theta(U[sy_i][iii + DNZ] - U[sy_i][iii], U[sy_i][iii] - U[sy_i][iii - DNZ],
				slpz0[sy_i][iii], theta);
	}
	for (integer field = 0; field != NF; ++field) {
#pragma GCC ivdep
		for (integer iii = HNX * HNX; iii != HN3 - HNX * HNX; ++iii) {
			const real u0 = U[field][iii];
			Uf[FXP][field][iii] = u0 + HALF * slpx[field][iii];
			Uf[FXM][field][iii] = u0 - HALF * slpx[field][iii];
			Uf[FYP][field][iii] = u0 + HALF * slpy[field][iii];
			Uf[FYM][field][iii] = u0 - HALF * slpy[field][iii];
			Uf[FZP][field][iii] = u0 + HALF * slpz[field][iii];
			Uf[FZM][field][iii] = u0 - HALF * slpz[field][iii];

		}
		if (field == pot_i) {
#pragma GCC ivdep
			for (integer iii = HNX * HNX; iii != HN3 - HNX * HNX; ++iii) {
				const real phi_x = HALF * (Uf[FXM][field][iii] + Uf[FXP][field][iii - DNX]);
				const real phi_y = HALF * (Uf[FYM][field][iii] + Uf[FYP][field][iii - DNY]);
				const real phi_z = HALF * (Uf[FZM][field][iii] + Uf[FZP][field][iii - DNZ]);
				Uf[FXM][field][iii] = phi_x;
				Uf[FYM][field][iii] = phi_y;
				Uf[FZM][field][iii] = phi_z;
				Uf[FXP][field][iii - DNX] = phi_x;
				Uf[FYP][field][iii - DNY] = phi_y;
				Uf[FZP][field][iii - DNZ] = phi_z;
			}
		}
		if (field != rho_i && field != tau_i) {
#pragma GCC ivdep
			for (integer iii = 0; iii != HN3; ++iii) {
				U[field][iii] *= U[rho_i][iii];
				for (integer face = 0; face != NFACE; ++face) {
					Uf[face][field][iii] *= Uf[face][rho_i][iii];
				}
			}
		}
	}
	for (integer iii = 0; iii != HN3; ++iii) {
		const real& rho = U[rho_i][iii];
		U[egas_i][iii] -= HALF * U[sx_i][iii] * U[sx_i][iii] / rho;
		U[egas_i][iii] -= HALF * U[sy_i][iii] * U[sy_i][iii] / rho;
		U[sx_i][iii] -= omega * X[YDIM][iii] * rho;
		U[sy_i][iii] += omega * X[XDIM][iii] * rho;
		U[zz_i][iii] += std::pow(dx * omega, 2) * rho / 6.0;
		U[egas_i][iii] += HALF * U[sx_i][iii] * U[sx_i][iii] / rho;
		U[egas_i][iii] += HALF * U[sy_i][iii] * U[sy_i][iii] / rho;
	}
	for (integer i = HBW - 1; i != HNX - HBW + 1; ++i) {
		for (integer j = HBW - 1; j != HNX - HBW + 1; ++j) {
			for (integer k = HBW - 1; k != HNX - HBW + 1; ++k) {
				const integer iii = i * DNX + j * DNY + k * DNZ;
				for (integer face = 0; face != NFACE; ++face) {
					Uf[face][egas_i][iii] -= HALF * Uf[face][sx_i][iii] * Uf[face][sx_i][iii] / Uf[face][rho_i][iii];
					Uf[face][egas_i][iii] -= HALF * Uf[face][sy_i][iii] * Uf[face][sy_i][iii] / Uf[face][rho_i][iii];
					real x0 = ZERO;
					real y0 = ZERO;
					if (face == FXP) {
						x0 = +HALF * dx;
					}
					if (face == FXM) {
						x0 = -HALF * dx;
					}
					if (face == FYP) {
						y0 = +HALF * dx;
					}
					if (face == FYM) {
						y0 = -HALF * dx;
					}
					Uf[face][sx_i][iii] -= omega * (X[YDIM][iii] + y0) * Uf[face][rho_i][iii];
					Uf[face][sy_i][iii] += omega * (X[XDIM][iii] + x0) * Uf[face][rho_i][iii];
					Uf[face][zz_i][iii] += std::pow(dx * omega, 2) * Uf[face][rho_i][iii] / 6.0;
					Uf[face][egas_i][iii] += HALF * Uf[face][sx_i][iii] * Uf[face][sx_i][iii] / Uf[face][rho_i][iii];
					Uf[face][egas_i][iii] += HALF * Uf[face][sy_i][iii] * Uf[face][sy_i][iii] / Uf[face][rho_i][iii];
				}
			}
		}
	}
}

real grid::compute_fluxes() {
	real max_lambda = ZERO;
	std::array<std::vector<real>, NF> ur, ul, f;
	std::vector<space_vector> x;

	const integer line_sz = HNX - 2 * HBW + 1;
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

		for (integer k = HBW; k != HNX - HBW; ++k) {
			for (integer j = HBW; j != HNX - HBW; ++j) {
				for (integer i = HBW; i != HNX - HBW + 1; ++i) {
					const integer i0 = DN[dx_i] * i + DN[dy_i] * j + DN[dz_i] * k;
					const integer im = i0 - DN[dx_i];
					for (integer field = 0; field != NF; ++field) {
						ur[field][i - HBW] = Uf[face_m][field][i0];
						ul[field][i - HBW] = Uf[face_p][field][im];
					}
					for (integer d = 0; d != NDIM; ++d) {
						x[i - HBW][d] = (X[d][i0] + X[d][im]) * HALF;
					}
				}
				const real this_max_lambda = roe_fluxes(f, ul, ur, x, omega, dim);
				max_lambda = std::max(max_lambda, this_max_lambda);
				for (integer field = 0; field != NF; ++field) {
					for (integer i = HBW; i != HNX - HBW + 1; ++i) {
						const integer i0 = DN[dx_i] * i + DN[dy_i] * j + DN[dz_i] * k;
						F[dim][field][i0] = f[field][i - HBW];
					}
				}
			}
		}
	}

	return max_lambda;
}

void grid::store() {
	for (integer field = 0; field != NF; ++field) {
#pragma GCC ivdep
		for (integer iii = 0; iii != HN3; ++iii) {
			U0[field][iii] = U[field][iii];
		}
	}
	for (integer field = 0; field != NGF; ++field) {
#pragma GCC ivdep
		for (integer iii = 0; iii != HN3; ++iii) {
			G0[field][iii] = G[field][iii];
		}
	}
	U_out0 = U_out;
}

void grid::restore() {
	for (integer field = 0; field != NF; ++field) {
#pragma GCC ivdep
		for (integer iii = 0; iii != HN3; ++iii) {
			U[field][iii] = U0[field][iii];
		}
	}
	for (integer field = 0; field != NGF; ++field) {
#pragma GCC ivdep
		for (integer iii = 0; iii != HN3; ++iii) {
			G[field][iii] = G0[field][iii];
		}
	}
	U_out = U_out0;
}

void grid::boundaries() {
	for (integer face = 0; face != NFACE; ++face) {
		set_physical_boundaries(face);
	}
}

void grid::set_physical_boundaries(const geo::face& face) {
	const auto dim = face.get_dimension();
	const auto side = face.get_side();
	const integer dni = dim == XDIM ? DNY : DNX;
	const integer dnj = dim == ZDIM ? DNY : DNZ;
	const integer dnk = DN[dim];
	const integer klb = side == geo::MINUS ? 0 : HNX - HBW;
	const integer kub = side == geo::MINUS ? HBW : HNX;
	const integer ilb = 0;
	const integer iub = HNX;
	const integer jlb = 0;
	const integer jub = HNX;

	for (integer field = 0; field != NF; ++field) {
		for (integer k = klb; k != kub; ++k) {
			for (integer j = jlb; j != jub; ++j) {
				for (integer i = ilb; i != iub; ++i) {
					integer k0;
					switch (boundary_types[face]) {
					case REFLECT:
						k0 = side == geo::MINUS ? (2 * HBW - k - 1) : (2 * (HNX - HBW) - k - 1);
						break;
					case OUTFLOW:
						k0 = side == geo::MINUS ? HBW : HNX - HBW - 1;
						break;
					default:
						k0 = -1;
						assert(false);
						abort();
					}
					const real value = U[field][i * dni + j * dnj + k0 * dnk];
					const integer iii = i * dni + j * dnj + k * dnk;
					real& ref = U[field][iii];
					if (field == sx_i + dim) {
						real s0;
						if (field == sx_i) {
							s0 = -omega * X[YDIM][iii] * U[rho_i][iii];
						} else if (field == sy_i) {
							s0 = +omega * X[XDIM][iii] * U[rho_i][iii];
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
							real this_rho = U[rho_i][iii];
							if (this_rho != ZERO) {
								U[egas_i][iii] += HALF * (after * after - before * before) / this_rho;
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

void grid::compute_sources() {
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
#pragma GCC ivdep
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = DNX * i + DNY * j + DNZ * k;
				for (integer field = 0; field != NF; ++field) {
					src[field][iii] = ZERO;
				}
				const real rho = U[rho_i][iii];
				src[zx_i][iii] = (-(F[YDIM][sz_i][iii + DNY] + F[YDIM][sz_i][iii])
						+ (F[ZDIM][sy_i][iii + DNZ] + F[ZDIM][sy_i][iii])) * HALF;
				src[zy_i][iii] = (+(F[XDIM][sz_i][iii + DNX] + F[XDIM][sz_i][iii])
						- (F[ZDIM][sx_i][iii + DNZ] + F[ZDIM][sx_i][iii])) * HALF;
				src[zz_i][iii] = (-(F[XDIM][sy_i][iii + DNX] + F[XDIM][sy_i][iii])
						+ (F[YDIM][sx_i][iii + DNY] + F[YDIM][sx_i][iii])) * HALF;
				src[sx_i][iii] += rho * G[gx_i][iii];
				src[sy_i][iii] += rho * G[gy_i][iii];
				src[sz_i][iii] += rho * G[gz_i][iii];
				src[sx_i][iii] += omega * U[sy_i][iii];
				src[sy_i][iii] -= omega * U[sx_i][iii];
				src[egas_i][iii] -= omega * X[YDIM][iii] * rho * G[gx_i][iii];
				src[egas_i][iii] += omega * X[XDIM][iii] * rho * G[gy_i][iii];
			}
		}
	}
}

void grid::compute_dudt() {
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
			for (integer field = 0; field != NF; ++field) {
#pragma GCC ivdep
				for (integer k = HBW; k != HNX - HBW; ++k) {
					const integer iii = DNX * i + DNY * j + DNZ * k;
					dUdt[field][iii] = ZERO;
					dUdt[field][iii] -= (F[XDIM][field][iii + DNX] - F[XDIM][field][iii]) / dx;
					dUdt[field][iii] -= (F[YDIM][field][iii + DNY] - F[YDIM][field][iii]) / dx;
					dUdt[field][iii] -= (F[ZDIM][field][iii + DNZ] - F[ZDIM][field][iii]) / dx;
					dUdt[field][iii] += src[field][iii];
				}
			}
#pragma GCC ivdep
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = DNX * i + DNY * j + DNZ * k;
				dUdt[egas_i][iii] += dUdt[pot_i][iii];
				dUdt[pot_i][iii] = ZERO;
			}
#pragma GCC ivdep
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = DNX * i + DNY * j + DNZ * k;
				dUdt[egas_i][iii] -= (dUdt[rho_i][iii] * G[phi_i][iii]) * HALF;
			}
		}
	}
//	solve_gravity(DRHODT);
}

void grid::egas_to_etot() {
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
#pragma GCC ivdep
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = i * DNX + j * DNY + k * DNZ;
				U[egas_i][iii] += U[pot_i][iii] * HALF;
			}
		}
	}
}

void grid::etot_to_egas() {
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
#pragma GCC ivdep
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = i * DNX + j * DNY + k * DNZ;
				U[egas_i][iii] -= U[pot_i][iii] * HALF;
			}
		}
	}
}

void grid::next_u(integer rk, real dt) {

	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
#pragma GCC ivdep
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = DNX * i + DNY * j + DNZ * k;
				dUdt[egas_i][iii] += (dphi_dt[iii] * U[rho_i][iii]) * HALF;
				dUdt[zx_i][iii] -= omega * X[ZDIM][iii] * U[sx_i][iii];
				dUdt[zy_i][iii] -= omega * X[ZDIM][iii] * U[sy_i][iii];
				dUdt[zz_i][iii] += omega * (X[XDIM][iii] * U[sx_i][iii] + X[YDIM][iii] * U[sy_i][iii]);
			}
		}
	}

	std::vector<real> du_out(NF, ZERO);

	std::vector<real> ds(NDIM, ZERO);
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = DNX * i + DNY * j + DNZ * k;
				for (integer field = 0; field != NF; ++field) {
					const real u1 = U[field][iii] + dUdt[field][iii] * dt;
					const real u0 = U0[field][iii];
					U[field][iii] = (ONE - rk_beta[rk]) * u0 + rk_beta[rk] * u1;
				}
			}
		}
	}

	du_out[sx_i] += omega * U_out[sy_i] * dt;
	du_out[sy_i] -= omega * U_out[sx_i] * dt;

	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
			const real dx2 = dx * dx;
			const integer iii_p = DNX * (HNX - HBW) + DNY * i + DNZ * j;
			const integer jjj_p = DNY * (HNX - HBW) + DNZ * i + DNX * j;
			const integer kkk_p = DNZ * (HNX - HBW) + DNX * i + DNY * j;
			const integer iii_m = DNX * (HBW) + DNY * i + DNZ * j;
			const integer jjj_m = DNY * (HBW) + DNZ * i + DNX * j;
			const integer kkk_m = DNZ * (HBW) + DNX * i + DNY * j;
			std::vector<real> du(NF);
			for (integer field = 0; field != NF; ++field) {
				if (field < zx_i || field > zz_i) {
					du[field] = ZERO;
					if (X[XDIM][iii_p] + pivot[XDIM] > ONE) {
						du[field] += (F[XDIM][field][iii_p]) * dx2;
					}
					if (X[YDIM][jjj_p] + pivot[YDIM] > ONE) {
						du[field] += (F[YDIM][field][jjj_p]) * dx2;
					}
					if (X[ZDIM][kkk_p] + pivot[ZDIM] > ONE) {
						du[field] += (F[ZDIM][field][kkk_p]) * dx2;
					}
					if (X[XDIM][iii_m] + pivot[XDIM] < -ONE + dx) {
						du[field] += (-F[XDIM][field][iii_m]) * dx2;
					}
					if (X[YDIM][jjj_m] + pivot[YDIM] < -ONE + dx) {
						du[field] += (-F[YDIM][field][jjj_m]) * dx2;
					}
					if (X[ZDIM][kkk_m] + pivot[ZDIM] < -ONE + dx) {
						du[field] += (-F[ZDIM][field][kkk_m]) * dx2;
					}
				}
			}

			if (X[XDIM][iii_p] + pivot[XDIM] > ONE) {
				const real xp = X[XDIM][iii_p] - HALF * dx;
				du[zx_i] += (X[YDIM][iii_p] * F[XDIM][sz_i][iii_p]) * dx2;
				du[zx_i] -= (X[ZDIM][iii_p] * F[XDIM][sy_i][iii_p]) * dx2;
				du[zy_i] -= (xp * F[XDIM][sz_i][iii_p]) * dx2;
				du[zy_i] += (X[ZDIM][iii_p] * F[XDIM][sx_i][iii_p]) * dx2;
				du[zz_i] += (xp * F[XDIM][sy_i][iii_p]) * dx2;
				du[zz_i] -= (X[YDIM][iii_p] * F[XDIM][sx_i][iii_p]) * dx2;
			}
			if (X[YDIM][jjj_p] + pivot[YDIM] > ONE) {
				const real yp = X[YDIM][jjj_p] - HALF * dx;
				du[zx_i] += (yp * F[YDIM][sz_i][jjj_p]) * dx2;
				du[zx_i] -= (X[ZDIM][jjj_p] * F[YDIM][sy_i][jjj_p]) * dx2;
				du[zy_i] -= (X[XDIM][jjj_p] * F[YDIM][sz_i][jjj_p]) * dx2;
				du[zy_i] += (X[ZDIM][jjj_p] * F[YDIM][sx_i][jjj_p]) * dx2;
				du[zz_i] += (X[XDIM][jjj_p] * F[YDIM][sy_i][jjj_p]) * dx2;
				du[zz_i] -= (yp * F[YDIM][sx_i][jjj_p]) * dx2;
			}
			if (X[ZDIM][kkk_p] + pivot[ZDIM] > ONE) {
				const real zp = X[ZDIM][kkk_p] - HALF * dx;
				du[zx_i] -= (zp * F[ZDIM][sy_i][kkk_p]) * dx2;
				du[zx_i] += (X[YDIM][kkk_p] * F[ZDIM][sz_i][kkk_p]) * dx2;
				du[zy_i] += (zp * F[ZDIM][sx_i][kkk_p]) * dx2;
				du[zy_i] -= (X[XDIM][kkk_p] * F[ZDIM][sz_i][kkk_p]) * dx2;
				du[zz_i] += (X[XDIM][kkk_p] * F[ZDIM][sy_i][kkk_p]) * dx2;
				du[zz_i] -= (X[YDIM][kkk_p] * F[ZDIM][sx_i][kkk_p]) * dx2;
			}

			if (X[XDIM][iii_m] + pivot[XDIM] < -ONE + dx) {
				const real xm = X[XDIM][iii_m] - HALF * dx;
				du[zx_i] += (-X[YDIM][iii_m] * F[XDIM][sz_i][iii_m]) * dx2;
				du[zx_i] -= (-X[ZDIM][iii_m] * F[XDIM][sy_i][iii_m]) * dx2;
				du[zy_i] -= (-xm * F[XDIM][sz_i][iii_m]) * dx2;
				du[zy_i] += (-X[ZDIM][iii_m] * F[XDIM][sx_i][iii_m]) * dx2;
				du[zz_i] += (-xm * F[XDIM][sy_i][iii_m]) * dx2;
				du[zz_i] -= (-X[YDIM][iii_m] * F[XDIM][sx_i][iii_m]) * dx2;
			}
			if (X[YDIM][jjj_m] + pivot[YDIM] < -ONE + dx) {
				const real ym = X[YDIM][jjj_m] - HALF * dx;
				du[zx_i] -= (-X[ZDIM][jjj_m] * F[YDIM][sy_i][jjj_m]) * dx2;
				du[zx_i] += (-ym * F[YDIM][sz_i][jjj_m]) * dx2;
				du[zy_i] -= (-X[XDIM][jjj_m] * F[YDIM][sz_i][jjj_m]) * dx2;
				du[zy_i] += (-X[ZDIM][jjj_m] * F[YDIM][sx_i][jjj_m]) * dx2;
				du[zz_i] += (-X[XDIM][jjj_m] * F[YDIM][sy_i][jjj_m]) * dx2;
				du[zz_i] -= (-ym * F[YDIM][sx_i][jjj_m]) * dx2;
			}
			if (X[ZDIM][kkk_m] + pivot[ZDIM] < -ONE + dx) {
				const real zm = X[ZDIM][kkk_m] - HALF * dx;
				du[zx_i] -= (-zm * F[ZDIM][sy_i][kkk_m]) * dx2;
				du[zx_i] += (-X[YDIM][kkk_m] * F[ZDIM][sz_i][kkk_m]) * dx2;
				du[zy_i] += (-zm * F[ZDIM][sx_i][kkk_m]) * dx2;
				du[zy_i] -= (-X[XDIM][kkk_m] * F[ZDIM][sz_i][kkk_m]) * dx2;
				du[zz_i] += (-X[XDIM][kkk_m] * F[ZDIM][sy_i][kkk_m]) * dx2;
				du[zz_i] -= (-X[YDIM][kkk_m] * F[ZDIM][sx_i][kkk_m]) * dx2;
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
	for (integer i = HBW; i != HNX - HBW; ++i) {
		for (integer j = HBW; j != HNX - HBW; ++j) {
			for (integer k = HBW; k != HNX - HBW; ++k) {
				const integer iii = DNX * i + DNY * j + DNZ * k;
				U[rho_i][iii] = U[acc_i][iii] + U[don_i][iii];
				real ek = ZERO;
				ek += HALF * std::pow(U[sx_i][iii], 2) / U[rho_i][iii];
				ek += HALF * std::pow(U[sy_i][iii], 2) / U[rho_i][iii];
				ek += HALF * std::pow(U[sz_i][iii], 2) / U[rho_i][iii];
				real ei = U[egas_i][iii] - ek;
				if (ei > de_switch1 * U[egas_i][iii]) {
					U[tau_i][iii] = std::pow(ei, ONE / fgamma);
				}
				if (U[tau_i][iii] < ZERO) {
					printf("Tau is negative- %e\n", double(U[tau_i][iii]));
					abort();
				} else if (U[rho_i][iii] <= ZERO) {
					printf("Rho is non-positive - %e\n", double(U[rho_i][iii]));
					abort();
				}
			}
		}
	}
}

std::vector<real> grid::conserved_outflows() const {
	auto Uret = U_out;
	Uret[egas_i] += Uret[pot_i];
	return Uret;
}
