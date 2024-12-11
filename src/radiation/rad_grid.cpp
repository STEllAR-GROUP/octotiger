//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/matrix.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/radiation/implicit.hpp"
#include "octotiger/radiation/kernel_interface.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/real.hpp"
#include "octotiger/roe.hpp"
#include "octotiger/space_vector.hpp"

#include <hpx/include/future.hpp>

#include <fenv.h>
#include <array>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "octotiger/unitiger/radiation/radiation_physics_impl.hpp"
#include "octotiger/matrix.hpp"

#if !defined(HPX_COMPUTE_DEVICE_CODE)

using real = double;

#define CHECK_FLUX( er, fx, fy, fz) if( ((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 ) {printf( "flux exceded %s %i %e fx %e fy %e fz %e er %e\n", __FILE__, __LINE__, sqrt(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c)), fx, fy, fz, er*physcon().c); abort();}
#define CHECK_FLUX1( er, fx, fy, fz, rho) if( ((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 ) {printf( "flux exceded %s %i %e fx %e fy %e fz %e er %e rho %e\n", __FILE__, __LINE__, sqrt(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c)), fx, fy, fz, er*physcon().c, rho); abort();}
#define BAD_FLUX( er, fx, fy, fz, rho) bool(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 )

std::unordered_map<std::string, int> rad_grid::str_to_index;
std::unordered_map<int, std::string> rad_grid::index_to_str;

void node_server::compute_radiation(real dt, real omega) {
	const bool root = bool(my_location.level() == 0);
	auto rgrid = rad_grid_ptr;
	rgrid->set_dx(grid_ptr->get_dx());
	rgrid->set_X(grid_ptr->get_X());
	rgrid->compute_mmw(grid_ptr->U);
	const real min_dx = TWO * grid::get_scaling_factor() / real(INX << opts().max_level);
	const real c = physcon().c;
	const real max_dt = min_dx / c / 3.0;
	const integer Nsubstep = std::max(int(std::ceil(dt * INVERSE(max_dt))), 1);
	const real sub_dt = dt * INVERSE(real(Nsubstep));
	const auto &rho = grid_ptr->get_field(rho_i);
	auto &egas = grid_ptr->get_field(egas_i);
	auto &tau = grid_ptr->get_field(tau_i);
	auto &sx = grid_ptr->get_field(sx_i);
	auto &sy = grid_ptr->get_field(sy_i);
	auto &sz = grid_ptr->get_field(sz_i);
	rgrid->implicit_source(egas, tau, sx, sy, sz, rho, dt);
	if (root) {
		printf("Radiation Transport\n");
		fflush(stdout);
	}
	for (integer i = 0; i != Nsubstep; ++i) {
		if (root) {
			printf("radiation sub-step %i of %i\r", int(i + 1), int(Nsubstep));
			fflush(stdout);
		}
		all_rad_bounds();
		rgrid->store();
		rgrid->compute_flux(omega);
		GET(exchange_rad_flux_corrections());
		rgrid->advance(sub_dt, 1.0);
		all_rad_bounds();
		if (root) {
			printf("                                          \r");
			printf("begin - implicit radiation\n");
			fflush(stdout);
		}
		rgrid->implicit_source(egas, tau, sx, sy, sz, rho, sub_dt);
		if (root) {
			printf("end - implicit radiation\n");
			fflush(stdout);
		}
		rgrid->compute_flux(omega);
		GET(exchange_rad_flux_corrections());
		rgrid->advance(sub_dt, 0.5);
		all_rad_bounds();
	}
	if (my_location.level() == 0) {
		printf("\rradiation done\n");
		fflush(stdout);
	}
}

const auto computeEddingtonTensor = [](real const &rEnergy, std::array<real, NDIM> rFlux) {
	constexpr real tiny = std::numeric_limits<real>::min();
	std::array<std::array<real, NDIM>, NDIM> eddingtonTensor;
	std::array<real, NDIM> fluxUnitVector;
	real const lightSpeed = physcon().c;
	real const invLightSpeed = real(1) / lightSpeed;
	real const invEnergy = real(1.0) / rEnergy;
	assert(rEnergy > real(0));
	real fluxNormSquared = real(0);
	for (int d = 0; d < NDIM; d++) {
		rFlux[d] *= invLightSpeed * invEnergy;
	}
	for (int d = 0; d < NDIM; d++) {
		fluxNormSquared += sqr(rFlux[d]);
	}
	assert(real(0) <= fluxNormSquared);
	assert(fluxNormSquared <= real(1));
	real const invFluxNorm = std::sqrt(fluxNormSquared) / (fluxNormSquared + tiny);
	for (int d = 0; d < NDIM; d++) {
		fluxUnitVector[d] = rFlux[d] * invFluxNorm;
	}
	real const eddingtonFactorNum = real(3) + real(4) * fluxNormSquared;
	real const eddingtonFactorDen = real(5) + std::sqrt(real(16) - real(12) * fluxNormSquared);
	real const eddingtonFactor = eddingtonFactorNum / eddingtonFactorDen;
	real const diffusionCoefficient = real(0.5) - real(0.5) * eddingtonFactorDen;
	real const streamingCoefficient = real(1.5) * eddingtonFactorDen - real(0.5);
	for (int l = 0; l < NDIM; l++) {
		eddingtonTensor[l][l] = diffusionCoefficient + streamingCoefficient * sqr(fluxUnitVector[l]);
		for (int m = 0; m < l; m++) {
			eddingtonTensor[m][l] = eddingtonTensor[l][m] = fluxUnitVector[l] * fluxUnitVector[m] * streamingCoefficient;
		}
	}
	return eddingtonTensor;
};

void rad_grid::implicit_source(std::vector<real> &egas_, std::vector<real> &tau_, std::vector<real> &sx_,
		std::vector<real> &sy_, std::vector<real> &sz_, const std::vector<real> &rho_, real dt) {
	PROFILE()
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);
	const real sigma = physcon().sigma;
	const real mh = physcon().mh;
	const real kb = physcon().kb;
	const real gam = grid::get_fgamma();
	const real c = physcon().c;
	const real ar = 4.0 * sigma * INVERSE(c);
	const real inv_c = INVERSE(physcon().c);
	const real inv_gam = INVERSE(gam);
	const real c_hat = c * opts().clight_reduced;
	const real inv_cc_hat = INVERSE(c * c_hat);
	const real inv_c2 = INVERSE(c * c);
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer D = H_BW - RAD_BW;
				const integer ir = rindex(xi, yi, zi);
				const integer ih = hindex(xi + D, yi + D, zi + D);
				real &tau = tau_[ih];
				real &egas = egas_[ih];
				real &erad = U[er_i][ir];
				real &fx = U[fx_i][ir];
				real &fy = U[fy_i][ir];
				real &fz = U[fz_i][ir];
				fx *= inv_c;
				fy *= inv_c;
				fz *= inv_c;
				real &sx = sx_[ih];
				real &sy = sy_[ih];
				real &sz = sz_[ih];
				const real erad0 = erad;
				const real egas0 = egas;
				const real fx0 = fx;
				const real fy0 = fy;
				const real fz0 = fz;
				const real sx0 = sx;
				const real sy0 = sy;
				const real sz0 = sz;
				const real mu = mmw[ih];
				const real rho = rho_[ih];
				CHECK_FLUX1(erad, fx, fy, fz, rho);
				const real inv_rho = INVERSE(rho);
				const real vx = sx * inv_rho;
				const real vy = sy * inv_rho;
				const real vz = sz * inv_rho;
				const real beta_x = vx * inv_c;
				const real beta_y = vy * inv_c;
				const real beta_z = vz * inv_c;
				real eint = std::max(egas - 0.5 * rho * (vx * vx + vy * vy + vz * vz), 0.0);
				const real kap = kappa_R(rho, eint, mmw[ih], X_spc[ih], Z_spc[ih]);
				const real chi = kap;
				const real fmag = SQRT(fx * fx + fy * fy + fz * fz);
				const real nx = fx / fmag;
				const real ny = fy / fmag;
				const real nz = fz / fmag;
				const real f = fmag / erad;
				const real eta = (3 + 4 * f * f) / (5 + 2 * SQRT(4 - 3 * f * f));
				const real f1 = (1 - eta) / 2;
				const real f2 = (3 * eta - 1) / 2;
				const real Dxx = f2 * nx * nx + f1;
				const real Dxy = f2 * nx * ny;
				const real Dxz = f2 * nx * nz;
				const real Dyx = Dxy;
				const real Dyy = f2 * ny * ny + f1;
				const real Dyz = f2 * ny * nz;
				const real Dzx = Dxz;
				const real Dzy = Dyz;
				const real Dzz = f2 * nz * nz + f1;
				auto F = [Dxx, Dxy, Dxz, Dyx, Dyy, Dyz, Dzx, Dzy, Dzz, dt, mu, rho, kap, chi, mh, kb, gam, c, ar, erad0,
						fx0, fy0, fz0, egas0, sx0, sy0, sz0](std::array<real, 4> X) {
					const real erad = X[0];
					const real fx = X[1];
					const real fy = X[2];
					const real fz = X[3];
					const real egas = erad0 - erad + egas0;
					const real sx = sx0 + (fx0 - fx) / c;
					const real sy = sy0 + (fy0 - fy) / c;
					const real sz = sz0 + (fz0 - fz) / c;
					const real beta_x = sx / (rho * c);
					const real beta_y = sy / (rho * c);
					const real beta_z = sz / (rho * c);
					const real eint = std::max(
							egas - rho * c * c * (beta_x * beta_x + beta_y * beta_y + beta_z * beta_z) / 2, 0.0);
					const real T = ((gam - 1) * mu * mh / (rho * kb)) * eint;
					const real bPx = erad * (beta_x * Dxx + beta_y * Dyx + beta_z * Dzx);
					const real bPy = erad * (beta_x * Dxy + beta_y * Dyy + beta_z * Dzy);
					const real bPz = erad * (beta_x * Dxz + beta_y * Dyz + beta_z * Dzz);
					const real g0 = kap * (erad - ar * T * T * T * T - 2.0 * (beta_x * fx + beta_y * fy + beta_z * fz));
					const real gx = chi * (fx - erad * beta_x - bPx);
					const real gy = chi * (fy - erad * beta_y - bPy);
					const real gz = chi * (fz - erad * beta_z - bPz);
					const real G0 = g0 + beta_x * gx + beta_y * gy + beta_z * gz;
					const real Gx = beta_x * g0 + gx;
					const real Gy = beta_y * g0 + gy;
					const real Gz = beta_z * g0 + gz;
					X[0] = erad - erad0 + dt * G0;
					X[1] = fx - fx0 + dt * Gx;
					X[2] = fy - fy0 + dt * Gy;
					X[3] = fz - fz0 + dt * Gz;
					return X;
				};
				std::array<real, 4> x = { erad, fx, fy, fz };
				x = solve<real, 4>(F, x);
				erad = x[0];
				fx = x[1];
				fy = x[2];
				fz = x[3];
				egas += erad0 - erad;
				sx += (fx0 - fx) * inv_c;
				sy += (fy0 - fy) * inv_c;
				sz += (fz0 - fz) * inv_c;
				eint = std::max(egas - (sx * sx + sy * sy + sz * sz) / (2 * rho), 0.0);
				tau = POWER(std::max(eint, 0.0), inv_gam);
				fx *= c;
				fy *= c;
				fz *= c;
				CHECK_FLUX1(erad, fx, fy, fz, rho);
			}
		}
	}
}


template<typename T>
int sign(T const &x) {
	if (x < T(0)) {
		return -1;
	} else if (x > T(0)) {
		return +1;
	} else {
		return 0;
	}
}

struct index_iterator {
	index_iterator(int dim, int lb = R_BW, int ub = RAD_NX - R_BW) {
		int xdim = XDIM, ydim = YDIM, zdim = ZDIM;
		if (dim == XDIM) {
			std::swap(xdim, zdim);
		} else if (dim == YDIM) {
			std::swap(ydim, zdim);
		}
		length[XDIM] = length[YDIM] = length[ZDIM] = RAD_NX - 2 * RAD_BW;
		length[dim] = ub - lb;
		offset = stride[dim] * (lb - R_BW);
		for (int d = 0; d < NDIM; d++) {
			length[d] *= stride[d];
			offset += stride[d] * R_BW;
		}
		stride[xdim] = sqr(RAD_NX);
		stride[ydim] = RAD_NX;
		stride[zdim] = 1;
		position = offset;
	}
	bool operator++(int) {
		position -= offset;
		for (int d = NDIM - 1; d >= 0; d--) {
			position += stride[d];
			real const new_position = position % length[d];
			if (new_position == position) {
				position += offset;
				return true;
			}
			position = new_position;
		}
		position += offset;
		return false;
	}
	operator int() const {
		return position;
	}
private:
	int position;
	int offset;
	std::array<int, NDIM> length;
	std::array<int, NDIM> stride;
};

void rad_grid::compute_flux(real omega) {
	PROFILE()
	real const c = physcon().c;
	real const c2 = sqr(c);
	real const cInv = real(1.0) / c;
	int DN = -1;
	std::array<std::vector<real>, NRF> dUdx;
	for (auto &du : dUdx) {
		du.resize(RAD_N3);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		index_iterator I(dim, RAD_INTERIOR_BEGIN - 1, RAD_INTERIOR_END + 1);
		do {
			int const i0 = int(I);
			real const um = U[er_i][i0 - DN];
			real const u0 = U[er_i][i0];
			real const up = U[er_i][i0 + DN];
			real const pSlope = up - u0;
			real const mSlope = u0 - um;
			real const sign = std::copysign(real(0.25), pSlope) + std::copysign(real(0.5), mSlope);
			dUdx[er_i][i0] = sign * std::min(std::abs(pSlope), std::abs(mSlope));
			for (int field = fx_i; field <= fz_i; field++) {
				real const rEnergyM = U[field][i0 - DN];
				real const rEnergy0 = U[field][i0];
				real const rEnergyP = U[field][i0 + DN];
				real const um = U[field][i0 - DN] / rEnergyM;
				real const u0 = U[field][i0] / rEnergy0;
				real const up = U[field][i0 + DN] / rEnergyP;
				assert(real(1) >= up && up >= -real(1));
				assert(real(1) >= u0 && u0 >= -real(1));
				assert(real(1) >= um && um >= -real(1));
				real const pSlope = up - u0;
				real const mSlope = u0 - um;
				real const sign = std::copysign(real(0.25), pSlope) + std::copysign(real(0.5), mSlope);
				dUdx[field][i0] = sign * std::min(std::abs(pSlope), std::abs(mSlope));
			}
		} while (I++);
		I = index_iterator(dim, RAD_INTERIOR_BEGIN, RAD_INTERIOR_END + 1);
		do {
			int const rI = int(I);
			int const lI = rI + DN;
			std::array<real, NDIM> rFluxFlow;
			std::array<real, NDIM> lFluxFlow;
			real const rEnergySlope = dUdx[er_i][rI];
			real const lEnergySlope = dUdx[er_i][lI];
			real const rEnergy = U[er_i][rI] - rEnergySlope;
			real const lEnergy = U[er_i][lI] + lEnergySlope;
			std::array<real, NDIM> const rFluxSlope = { dUdx[fx_i][rI], dUdx[fy_i][rI], dUdx[fz_i][rI] };
			std::array<real, NDIM> const lFluxSlope = { dUdx[fx_i][lI], dUdx[fy_i][lI], dUdx[fz_i][lI] };
			std::array<real, NDIM> rFlux = { U[fx_i][rI], U[fy_i][rI], U[fz_i][rI] };
			std::array<real, NDIM> lFlux = { U[fx_i][lI], U[fy_i][lI], U[fz_i][lI] };
			for (int d = 0; d < NDIM; d++) {
				rFlux[d] += (rFlux[d] - rFluxSlope[d]) * rEnergy;
				lFlux[d] += (lFlux[d] + lFluxSlope[d]) * lEnergy;
			}
			auto const rEddingtonTensor = computeEddingtonTensor(rEnergy, rFlux);
			auto const lEddingtonTensor = computeEddingtonTensor(lEnergy, lFlux);
			real const rEnergyFlow = rFlux[dim] - c * rEnergy;
			real const lEnergyFlow = lFlux[dim] + c * lEnergy;
			for (int d = 0; d < NDIM; d++) {
				rFluxFlow[d] = c2 * rEnergy * rEddingtonTensor[dim][d] - c * rFlux[d];
				lFluxFlow[d] = c2 * lEnergy * lEddingtonTensor[dim][d] + c * lFlux[d];
			}
			flux[dim][er_i][rI] = real(0.5) * (rEnergyFlow + lEnergyFlow);
			for (int d = 0; d < NDIM; d++) {
				flux[dim][fx_i + d][rI] = real(0.5) * (rFluxFlow[d] + lFluxFlow[d]);
			}
		} while (I++);
		DN *= RAD_INTERIOR;
	}
}

void rad_grid::static_init() {
	str_to_index["er"] = er_i;
	str_to_index["fx"] = fx_i;
	str_to_index["fy"] = fy_i;
	str_to_index["fz"] = fz_i;
	for (const auto &s : str_to_index) {
		index_to_str[s.second] = s.first;
	}
}

std::vector<std::string> rad_grid::get_field_names() {
	std::vector<std::string> rc;
	for (auto i : str_to_index) {
		rc.push_back(i.first);
	}
	return rc;
}

void rad_grid::set(const std::string name, real *data) {
	auto iter = str_to_index.find(name);
	real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
	if (iter != str_to_index.end()) {
		int f = iter->second;
		int jjj = 0;
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
					data[jjj] /= f == er_i ? eunit : funit;
					U[f][iii] = data[jjj];
					jjj++;
				}
			}
		}
	}

}

std::vector<silo_var_t> rad_grid::var_data() const {
	std::vector<silo_var_t> s;
	real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
	for (auto l : str_to_index) {
		const int f = l.second;
		std::string this_name = l.first;
		int jjj = 0;
		silo_var_t this_s(this_name);
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
					this_s(jjj) = U[f][iii];
					this_s(jjj) *= f == er_i ? eunit : funit;
					this_s.set_range(this_s(jjj));
					jjj++;
				}
			}
		}
		s.push_back(std::move(this_s));
	}
	return std::move(s);
}

constexpr auto _0 = real(0);
constexpr auto _1 = real(1);
constexpr auto _2 = real(2);
constexpr auto _3 = real(3);
constexpr auto _4 = real(4);
constexpr auto _5 = real(5);

using set_rad_grid_action_type = node_server::set_rad_grid_action;
HPX_REGISTER_ACTION (set_rad_grid_action_type);

hpx::future<void> node_client::set_rad_grid(std::vector<real> &&g, std::vector<real> &&o) const {
	return hpx::async<typename node_server::set_rad_grid_action>(get_unmanaged_gid(), g, o);
}

void node_server::set_rad_grid(const std::vector<real> &data, const std::vector<real> &outflows) {
	rad_grid_ptr->set_prolong(data, outflows);
}

using send_rad_boundary_action_type = node_server::send_rad_boundary_action;
HPX_REGISTER_ACTION (send_rad_boundary_action_type);

using send_rad_flux_correct_action_type = node_server::send_rad_flux_correct_action;
HPX_REGISTER_ACTION (send_rad_flux_correct_action_type);

void node_client::send_rad_flux_correct(std::vector<real> &&data, const geo::face &face, const geo::octant &ci) const {
	hpx::apply<typename node_server::send_rad_flux_correct_action>(get_unmanaged_gid(), std::move(data), face, ci);
}

void node_server::recv_rad_flux_correct(std::vector<real> &&data, const geo::face &face, const geo::octant &ci) {
	const geo::quadrant index(ci, face.get_dimension());
	niece_rad_channels[face][index].set_value(std::move(data));
}

void node_client::send_rad_boundary(std::vector<real> &&data, const geo::direction &dir, std::size_t cycle) const {
	hpx::apply<typename node_server::send_rad_boundary_action>(get_gid(), std::move(data), dir, cycle);
}

void node_server::recv_rad_boundary(std::vector<real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_rad_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

using send_rad_children_action_type = node_server::send_rad_children_action;
HPX_REGISTER_ACTION (send_rad_children_action_type);

void node_server::recv_rad_children(std::vector<real> &&data, const geo::octant &ci, std::size_t cycle) {
	child_rad_channels[ci].set_value(std::move(data), cycle);
}

#include <fenv.h>

void node_client::send_rad_children(std::vector<real> &&data, const geo::octant &ci, std::size_t cycle) const {
	hpx::apply<typename node_server::send_rad_children_action>(get_unmanaged_gid(), std::move(data), ci, cycle);
}

#define ASSERT(a) if (!a) { \
   printf( "Error in %s on line %i\n", __FILE__, __LINE__); \
   abort(); \
   }

//Er_np1 + c * dt * kap * Er_np1 - 4.0 * dt * kap * sigma * ((gam - 1)*(mh*mmw * (Ei_n - Er_np1 + Er_n) / (rho * kb))^4;

#include <iostream>
#include <cmath>
#include <stdexcept>

void rad_grid::set_dx(real _dx) {
	dx = _dx;
}

void rad_grid::set_X(const std::vector<std::vector<real>> &x) {
	X.resize(NDIM);
	for (integer d = 0; d != NDIM; ++d) {
		X[d].resize(RAD_N3);
		for (integer xi = 0; xi != RAD_NX; ++xi) {
			for (integer yi = 0; yi != RAD_NX; ++yi) {
				for (integer zi = 0; zi != RAD_NX; ++zi) {
					const auto D = H_BW - RAD_BW;
					const integer ir = rindex(xi, yi, zi);
					const integer ih = hindex(xi + D, yi + D, zi + D);
					X[d][ir] = x[d][ih];
				}
			}
		}
	}
}

real rad_grid::hydro_signal_speed(const std::vector<real> &egas_, const std::vector<real> &tau_,
		const std::vector<real> &sx_, const std::vector<real> &sy_, const std::vector<real> &sz_,
		const std::vector<real> &rho_) {
	real a = 0.0;
	const real fgamma = grid::get_fgamma();
	const integer D = H_BW - RAD_BW;
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer ir = rindex(xi, yi, zi);
				const integer ih = hindex(xi + D, yi + D, zi + D);
				const real sx = sx_[ih];
				const real sy = sy_[ih];
				const real sz = sz_[ih];
				const real egas = egas_[ih];
				const real Erad = U[er_i][ir];
				const real Fx = U[fx_i][ir];
				const real Fy = U[fy_i][ir];
				const real Fz = U[fz_i][ir];
				const real rho = rho_[ih];
				CHECK_FLUX1(Erad, Fx, Fy, Fz, rho);
				const real inv_rho = INVERSE(rho);
				const real vx = sx * inv_rho;
				const real vy = sy * inv_rho;
				const real vz = sz * inv_rho;
				const real eint = std::max(egas - 0.5 * rho * (vx * vx + vy * vy + vz * vz), 0.0);
				const real kap = kappa_R(rho, eint, mmw[ih], X_spc[ih], Z_spc[ih]);
				real cs2 = (4.0 / 9.0) * Erad * inv_rho;
				const real cons = kap * dx;
				if (cons < 32.0) {
					cs2 *= 1.0 - std::exp(-std::min(32.0, cons));
				}
				a = std::max(cs2, a);
			}
		}
	}
	return SQRT(a);
}

void rad_grid::compute_mmw(const std::vector<std::vector<safe_real>> &U) {
	mmw.resize(RAD_N3);
	X_spc.resize(RAD_N3);
	Z_spc.resize(RAD_N3);
	for (integer i = 0; i != RAD_NX; ++i) {
		for (integer j = 0; j != RAD_NX; ++j) {
			for (integer k = 0; k != RAD_NX; ++k) {
				const integer d = H_BW - RAD_BW;
				const integer iiir = rindex(i, j, k);
				const integer iiih = hindex(i + d, j + d, k + d);
				specie_state_t<real> spc;
				for (integer si = 0; si != opts().n_species; ++si) {
					spc[si] = U[spc_i + si][iiih];
				}
				mean_ion_weight(spc, mmw[iiir], X_spc[iiir], Z_spc[iiir]);
			}
		}
	}

}

template<class T>
T minmod(T a, T b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

void rad_grid::allocate() {
	rad_grid::dx = dx;
	U.resize(NRF);
	Ushad.resize(NRF);
	flux.resize(NDIM);
	for (int d = 0; d < NDIM; d++) {
		flux[d].resize(NRF);
	}
	for (integer f = 0; f != NRF; ++f) {
		U0[f].resize(RAD_N3);
		U[f].resize(RAD_N3);
		Ushad[f].resize(RAD_N3);
		for (integer d = 0; d != NDIM; ++d) {
			flux[d][f].resize(RAD_N3);
		}
	}
}

void rad_grid::store() {
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = 0; i != RAD_N3; ++i) {
			U0[f][i] = U[f][i];
		}
	}
}

void rad_grid::restore() {
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = 0; i != RAD_N3; ++i) {
			U[f][i] = U0[f][i];
		}
	}
}

void rad_grid::sanity_check() {
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer iiir = rindex(xi, yi, zi);
				if (U[er_i][iiir] <= 0.0) {
					printf("INSANE\n");
					//		printf("%e %i %i %i\n", U[er_i][iiir], xi, yi, zi);
					abort();
				}
			}
		}
	}
}

void rad_grid::change_units(real m, real l, real t, real k) {
	const real l2 = l * l;
	const real t2 = t * t;
	const real t2inv = 1.0 * INVERSE(t2);
	const real tinv = 1.0 * INVERSE(t);
	const real l3 = l2 * l;
	const real l3inv = 1.0 * INVERSE(l3);
	for (integer i = 0; i != RAD_N3; ++i) {
		U[er_i][i] *= (m * l2 * t2inv) * l3inv;
		U[fx_i][i] *= tinv * (m * t2inv);
		U[fy_i][i] *= tinv * (m * t2inv);
		U[fz_i][i] *= tinv * (m * t2inv);
	}
}

void rad_grid::advance(real dt, real beta) {
	const real l = dt * INVERSE(dx);
	const integer D[3] = { DX, DY, DZ };
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer iii = rindex(xi, yi, zi);
				CHECK_FLUX(U[er_i][iii], U[fx_i][iii], U[fy_i][iii], U[fz_i][iii]);
				for (integer f = 0; f != NRF; ++f) {
					const real &u0 = U0[f][iii];
					real u1 = U[f][iii];
					for (integer d = 0; d != NDIM; ++d) {
						u1 -= l * (flux[d][f][iii + D[d]] - flux[d][f][iii]);
					}
					U[f][iii] = u0 * (1.0 - beta) + beta * u1;
				}
				CHECK_FLUX(U[er_i][iii], U[fx_i][iii], U[fy_i][iii], U[fz_i][iii]);
			}
		}
	}
}

void rad_grid::set_physical_boundaries(geo::face face, real t) {
	for (integer i = 0; i != RAD_NX; ++i) {
		for (integer j = 0; j != RAD_NX; ++j) {
			for (integer k = 0; k != RAD_BW; ++k) {
				integer iii1, iii0;
				switch (face) {
				case 0:
					iii1 = rindex(k, i, j);
					iii0 = rindex(RAD_BW, i, j);
					break;
				case 1:
					iii1 = rindex(RAD_NX - 1 - k, i, j);
					iii0 = rindex(RAD_NX - 1 - RAD_BW, i, j);
					break;
				case 2:
					iii1 = rindex(i, k, j);
					iii0 = rindex(i, RAD_BW, j);
					break;
				case 3:
					iii1 = rindex(i, RAD_NX - 1 - k, j);
					iii0 = rindex(i, RAD_NX - 1 - RAD_BW, j);
					break;
				case 4:
					iii1 = rindex(i, j, k);
					iii0 = rindex(i, j, RAD_BW);
					break;
				case 5:
				default:
					iii1 = rindex(i, j, RAD_NX - 1 - k);
					iii0 = rindex(i, j, RAD_NX - 1 - RAD_BW);
				}
				for (integer f = 0; f != NRF; ++f) {
					U[f][iii1] = U[f][iii0];
				}
				switch (face) {
				case 0:
					if (opts().problem == MARSHAK) {
						if (t > 0) {
							auto u = marshak_wave_analytic(-opts().xscale, 0, 0, t);
							U[fx_i][iii1] = u[opts().n_fields + fx_i];
							U[er_i][iii1] = std::max(u[opts().n_fields + er_i], 1.0e-10);
						} else {
							U[fx_i][iii1] = 0.0;
							U[er_i][iii1] = 1.0e-10;
						}
					} else {
						U[fx_i][iii1] = std::min(U[fx_i][iii1], 0.0);
					}
					break;
				case 1:
					U[fx_i][iii1] = std::max(U[fx_i][iii1], 0.0);
					break;
				case 2:
					U[fy_i][iii1] = std::min(U[fy_i][iii1], 0.0);
					break;
				case 3:
					U[fy_i][iii1] = std::max(U[fy_i][iii1], 0.0);
					break;
				case 4:
					U[fz_i][iii1] = std::min(U[fz_i][iii1], 0.0);
					break;
				case 5:
					U[fz_i][iii1] = std::max(U[fz_i][iii1], 0.0);
					break;
				}
			}
		}
	}
}

hpx::future<void> node_server::exchange_rad_flux_corrections() {
	const geo::octant ci = my_location.get_child_index();
	constexpr auto full_set = geo::face::full_set();
	for (auto &f : full_set) {
		const auto face_dim = f.get_dimension();
		auto const &this_aunt = aunts[f];
		if (!this_aunt.empty()) {
			std::array<integer, NDIM> lb, ub;
			lb[XDIM] = lb[YDIM] = lb[ZDIM] = RAD_BW;
			ub[XDIM] = ub[YDIM] = ub[ZDIM] = INX + RAD_BW;
			if (f.get_side() == geo::MINUS) {
				lb[face_dim] = RAD_BW;
			} else {
				lb[face_dim] = INX + RAD_BW;
			}
			ub[face_dim] = lb[face_dim] + 1;
			auto data = rad_grid_ptr->get_flux_restrict(lb, ub, face_dim);
			this_aunt.send_rad_flux_correct(std::move(data), f.flip(), ci);
		}
	}

	constexpr integer size = geo::face::count() * geo::quadrant::count();
	std::array<future<void>, size> futs;
	for (auto &f : futs) {
		f = hpx::make_ready_future();
	}
	integer index = 0;
	for (auto const &f : geo::face::full_set()) {
		if (this->nieces[f] == +1) {
			for (auto const &quadrant : geo::quadrant::full_set()) {
				futs[index++] = niece_rad_channels[f][quadrant].get_future().then(
						[this, f, quadrant](hpx::future<std::vector<real> > &&fdata) -> void {
							const auto face_dim = f.get_dimension();
							std::array<integer, NDIM> lb, ub;
							switch (face_dim) {
							case XDIM:
								lb[XDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
								lb[YDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
								lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
								ub[XDIM] = lb[XDIM] + 1;
								ub[YDIM] = lb[YDIM] + (INX / 2);
								ub[ZDIM] = lb[ZDIM] + (INX / 2);
								break;
							case YDIM:
								lb[XDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
								lb[YDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
								lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
								ub[XDIM] = lb[XDIM] + (INX / 2);
								ub[YDIM] = lb[YDIM] + 1;
								ub[ZDIM] = lb[ZDIM] + (INX / 2);
								break;
							case ZDIM:
							default:
								lb[XDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
								lb[YDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
								lb[ZDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
								ub[XDIM] = lb[XDIM] + (INX / 2);
								ub[YDIM] = lb[YDIM] + (INX / 2);
								ub[ZDIM] = lb[ZDIM] + 1;
								break;
							}
							rad_grid_ptr->set_flux_restrict(GET(fdata), lb, ub, face_dim);
						});
			}
		}
	}
	return hpx::when_all(std::move(futs)).then([](future<decltype(futs)> fout) {
		auto fin = GET(fout);
		for (auto &f : fin) {
			GET(f);
		}
	});
}

void rad_grid::set_flux_restrict(const std::vector<real> &data, const std::array<integer, NDIM> &lb,
		const std::array<integer, NDIM> &ub, const geo::dimension &dim) {

	integer index = 0;
	for (integer field = 0; field != NRF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					const integer iii = rindex(i, j, k);
					flux[dim][field][iii] = data[index];
					++index;
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_flux_restrict(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub,
		const geo::dimension &dim) const {

	std::vector<real> data;
	integer size = 1;
	for (auto &dim : geo::dimension::full_set()) {
		size *= (ub[dim] - lb[dim]);
	}
	size /= (NCHILD / 2);
	size *= NRF;
	data.reserve(size);
	const integer stride1 = (dim == XDIM) ? (RAD_NX) : (RAD_NX) * (RAD_NX);
	const integer stride2 = (dim == ZDIM) ? (RAD_NX) : 1;
	for (integer field = 0; field != NRF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; i += 2) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; j += 2) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; k += 2) {
					const integer i00 = rindex(i, j, k);
					const integer i10 = i00 + stride1;
					const integer i01 = i00 + stride2;
					const integer i11 = i00 + stride1 + stride2;
					real value = ZERO;
					value += flux[dim][field][i00];
					value += flux[dim][field][i10];
					value += flux[dim][field][i01];
					value += flux[dim][field][i11];
					value /= real(4);
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

void node_server::all_rad_bounds() {
//	if( my_location.level() == 0 ) printf( "\nbounds 1\n");
	GET(exchange_interlevel_rad_data());
//	if( my_location.level() == 0 ) printf( "\nbounds 2\n");
	collect_radiation_bounds();
//	if( my_location.level() == 0 ) printf( "\nbounds 3\n");
	send_rad_amr_bounds();
//	if( my_location.level() == 0 ) printf( "\nbounds 4\n");
	rcycle++;
}

hpx::future<void> node_server::exchange_interlevel_rad_data() {

	hpx::future<void> f = hpx::make_ready_future();
	integer ci = my_location.get_child_index();

	if (is_refined) {
		for (auto const &ci : geo::octant::full_set()) {
			auto data = GET(child_rad_channels[ci].get_future(rcycle));
			rad_grid_ptr->set_restrict(data, ci);
		}
	}
	if (my_location.level() > 0) {
		auto data = rad_grid_ptr->get_restrict();
		parent.send_rad_children(std::move(data), ci, rcycle);
	}
	return hpx::make_ready_future();
}

void node_server::collect_radiation_bounds() {

	rad_grid_ptr->clear_amr();
	for (auto const &dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			const integer width = H_BW;
			auto bdata = rad_grid_ptr->get_boundary(dir);
			neighbors[dir].send_rad_boundary(std::move(bdata), dir.flip(), rcycle);
		}
	}

	std::array<future<void>, geo::direction::count()> results;
	integer index = 0;
	for (auto const &dir : geo::direction::full_set()) {
		if (!(neighbors[dir].empty() && my_location.level() == 0)) {
			results[index++] = sibling_rad_channels[dir].get_future(rcycle).then(
			/*hpx::util::annotated_function(*/[this, dir](future<sibling_rad_type> &&f) -> void {
				auto &&tmp = GET(f);
				if (!neighbors[dir].empty()) {
					rad_grid_ptr->set_boundary(tmp.data, tmp.direction);
				} else {
					rad_grid_ptr->set_rad_amr_boundary(tmp.data, tmp.direction);

				}
			}/*, "node_server::collect_rad_boundaries::set_rad_boundary")*/);
		}
	}
	while (index < geo::direction::count()) {
		results[index++] = hpx::make_ready_future();
	}
//	wait_all_and_propagate_exceptions(std::move(results));
	for (auto &f : results) {
		GET(f);
	}
	rad_grid_ptr->complete_rad_amr_boundary();
	for (auto &face : geo::face::full_set()) {
		if (my_location.is_physical_boundary(face)) {
			rad_grid_ptr->set_physical_boundaries(face, current_time);
		}
	}

}

void rad_grid::initialize_erad(const std::vector<safe_real> rho, const std::vector<safe_real> tau) {
	const real fgamma = grid::get_fgamma();
	for (integer xi = 0; xi != RAD_NX; ++xi) {
		for (integer yi = 0; yi != RAD_NX; ++yi) {
			for (integer zi = 0; zi != RAD_NX; ++zi) {
				const auto D = H_BW - RAD_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				const real ei = POWER(tau[iiih], fgamma);
				if (opts().problem == STAR || opts().problem == ROTATING_STAR || opts().problem == RADIATION_COUPLING) {
					U[er_i][iiir] = 10.0 * B_p((double) rho[iiih], (double) ei, (double) mmw[iiir])
							* (4.0 * M_PI / physcon().c);
					U[fx_i][iiir] = U[fy_i][iiir] = U[fz_i][iiir] = 0.0;
				}
			}
		}
	}
}

rad_grid::rad_grid(real _dx) :
		dx(_dx), is_coarse(RAD_N3), has_coarse(RAD_N3) {
	allocate();
}

rad_grid::rad_grid() :
		is_coarse(RAD_N3), has_coarse(RAD_N3), U_out(NRF, ZERO), U_out0(NRF, ZERO) {
	allocate();
}

void rad_grid::set_boundary(const std::vector<real> &data, const geo::direction &dir) {

	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, dir, OUTER, INX, RAD_BW);
	integer iter = 0;

	for (integer field = 0; field != NRF; ++field) {
		auto &Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					Ufield[rindex(i, j, k)] = data[iter];
					++iter;
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_boundary(const geo::direction &dir) {

	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size = NRF * get_boundary_size(lb, ub, dir, INNER, INX, RAD_BW);
	data.resize(size);
	integer iter = 0;

	for (integer field = 0; field != NRF; ++field) {
		auto &Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					data[iter] = Ufield[rindex(i, j, k)];
					++iter;
				}
			}
		}
	}

	return data;
}

void rad_grid::set_field(real v, integer f, integer i, integer j, integer k) {
	U[f][rindex(i, j, k)] = v;
}

real rad_grid::get_field(integer f, integer i, integer j, integer k) const {
	return U[f][rindex(i, j, k)];
}

void rad_grid::set_prolong(const std::vector<real> &data, const std::vector<real> &out) {
	integer index = 0;
	U_out = std::move(out);
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = RAD_BW; i != RAD_NX - RAD_BW; ++i) {
			for (integer j = RAD_BW; j != RAD_NX - RAD_BW; ++j) {
				for (integer k = RAD_BW; k != RAD_NX - RAD_BW; ++k) {
					const integer iii = rindex(i, j, k);
					U[f][iii] = data[index];
					++index;
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_prolong(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub) {
	std::vector<real> data;
	integer size = NRF;
	for (integer dim = 0; dim != NDIM; ++dim) {
		size *= (ub[dim] - lb[dim]);
	}
	auto lb0 = lb;
	auto ub0 = ub;
	for (integer d = 0; d != NDIM; ++d) {
		lb0[d] /= 2;
		ub0[d] /= 2;
	}

	for (integer f = 0; f != NRF; ++f) {
		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
					const integer iii = rindex(i / 2, j / 2, k / 2);
					real value = U[f][iii];
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

std::vector<real> rad_grid::get_restrict() const {
	std::vector<real> data;
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = RAD_BW; i < RAD_NX - RAD_BW; i += 2) {
			for (integer j = RAD_BW; j < RAD_NX - RAD_BW; j += 2) {
				for (integer k = RAD_BW; k < RAD_NX - RAD_BW; k += 2) {
					const integer iii = rindex(i, j, k);
					real v = ZERO;
					for (integer x = 0; x != 2; ++x) {
						for (integer y = 0; y != 2; ++y) {
							for (integer z = 0; z != 2; ++z) {
								const integer jjj = iii + x * RAD_NX * RAD_NX + y * RAD_NX + z;
								v += U[f][jjj];
							}
						}
					}
					v /= real(NCHILD);
					data.push_back(v);
				}
			}
		}
	}
	return data;
}

void rad_grid::set_restrict(const std::vector<real> &data, const geo::octant &octant) {
	integer index = 0;
	const integer i0 = octant.get_side(XDIM) * (INX / 2);
	const integer j0 = octant.get_side(YDIM) * (INX / 2);
	const integer k0 = octant.get_side(ZDIM) * (INX / 2);
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = RAD_BW; i != RAD_NX / 2; ++i) {
			for (integer j = RAD_BW; j != RAD_NX / 2; ++j) {
				for (integer k = RAD_BW; k != RAD_NX / 2; ++k) {
					const integer iii = rindex(i + i0, j + j0, k + k0);
					U[f][iii] = data[index];
					++index;
					if (index > int(data.size())) {
						printf("rad_grid::set_restrict error %i %i\n", int(index), int(data.size()));
					}
				}
			}
		}
	}

}
;

void node_server::send_rad_amr_bounds() {
	if (is_refined) {
		constexpr auto full_set = geo::octant::full_set();
		for (auto &ci : full_set) {
			const auto &flags = amr_flags[ci];
			for (auto &dir : geo::direction::full_set()) {
				if (flags[dir]) {
					std::array<integer, NDIM> lb, ub;
					std::vector<real> data;
					get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
					for (integer dim = 0; dim != NDIM; ++dim) {
						lb[dim] = std::max(lb[dim] - 1, integer(0));
						ub[dim] = std::min(ub[dim] + 1, integer(HS_NX));
						lb[dim] = lb[dim] + ci.get_side(dim) * (INX / 2);
						ub[dim] = ub[dim] + ci.get_side(dim) * (INX / 2);
					}
					data = rad_grid_ptr->get_subset(lb, ub);
					children[ci].send_rad_amr_boundary(std::move(data), dir, rcycle);
				}
			}
		}
	}
}

using erad_init_action_type = node_server::erad_init_action;
HPX_REGISTER_ACTION (erad_init_action_type);

hpx::future<void> node_client::erad_init() const {
	return hpx::async<typename node_server::erad_init_action>(get_unmanaged_gid());
}

void node_server::erad_init() {
	std::array<hpx::future<void>, NCHILD> futs;
	int index = 0;
	if (is_refined) {
		for (auto &child : children) {
			futs[index++] = child.erad_init();
		}
	}
	grid_ptr->rad_init();
	if (is_refined) {
		hpx::wait_all(futs);
	}
}

void rad_grid::clear_amr() {
	std::fill(is_coarse.begin(), is_coarse.end(), 0);
	std::fill(has_coarse.begin(), has_coarse.end(), 0);
}

void rad_grid::set_rad_amr_boundary(const std::vector<real> &data, const geo::direction &dir) {
	PROFILE();

	std::array<integer, NDIM> lb, ub;
	int l = 0;
	get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
	for (int i = lb[0]; i < ub[0]; i++) {
		for (int j = lb[1]; j < ub[1]; j++) {
			for (int k = lb[2]; k < ub[2]; k++) {
				is_coarse[hSindex(i, j, k)]++;
				assert(i < H_BW || i >= HS_NX - H_BW || j < H_BW || j >= HS_NX - H_BW || k < H_BW || k >= HS_NX - H_BW);
			}
		}
	}

	for (int dim = 0; dim < NDIM; dim++) {
		lb[dim] = std::max(lb[dim] - 1, integer(0));
		ub[dim] = std::min(ub[dim] + 1, integer(HS_NX));
	}

	for (int f = 0; f < NRF; f++) {
		for (int i = lb[0]; i < ub[0]; i++) {
			for (int j = lb[1]; j < ub[1]; j++) {
				for (int k = lb[2]; k < ub[2]; k++) {
					has_coarse[hSindex(i, j, k)]++;
					Ushad[f][hSindex(i, j, k)] = data[l++];
				}
			}
		}
	}
	assert(l == data.size());
}

std::vector<std::pair<std::string, real>> rad_grid::get_outflows() const {
	std::vector<std::pair<std::string, real>> rc;
	rc.reserve(NRF);
	rc.push_back(std::make_pair("er", U_out[0]));
	rc.push_back(std::make_pair("fx", U_out[1]));
	rc.push_back(std::make_pair("fy", U_out[2]));
	rc.push_back(std::make_pair("fz", U_out[3]));
	return rc;
}

void rad_grid::set_outflows(std::vector<std::pair<std::string, real>> &&u) {
	for (const auto &p : u) {
		set_outflow(p);
	}
}

void rad_grid::set_outflow(const std::pair<std::string, real> &p) {
	if (p.first == "er") {
		U_out[0] = p.second;
	} else if (p.first == "fx") {
		U_out[1] = p.second;
	} else if (p.first == "fy") {
		U_out[2] = p.second;
	} else if (p.first == "fz") {
		U_out[3] = p.second;
	} else {
		assert(false);
	}
}

void rad_grid::complete_rad_amr_boundary() {
	PROFILE();

	using oct_array = std::array<std::array<std::array<double, 2>, 2>, 2>;
	static thread_local std::vector<std::vector<oct_array>> Uf(NRF, std::vector<oct_array>(HS_N3));

	const auto limiter = [](double a, double b) {
		return minmod_theta(a, b, 64. / 37.);
	};

	for (int f = 0; f < NRF; f++) {

		for (int i0 = 1; i0 < HS_NX - 1; i0++) {
			for (int j0 = 1; j0 < HS_NX - 1; j0++) {
				for (int k0 = 1; k0 < HS_NX - 1; k0++) {
					const int iii0 = hSindex(i0, j0, k0);
					if (is_coarse[iii0]) {
						for (int ir = 0; ir < 2; ir++) {
							for (int jr = 0; jr < 2; jr++) {
								for (int kr = 0; kr < 2; kr++) {
									const auto is = ir % 2 ? +1 : -1;
									const auto js = jr % 2 ? +1 : -1;
									const auto ks = kr % 2 ? +1 : -1;
									const auto &u0 = Ushad[f][iii0];
									const auto &uc = Ushad[f];
									const auto s_x = limiter(uc[iii0 + is * HS_DNX] - u0, u0 - uc[iii0 - is * HS_DNX]);
									const auto s_y = limiter(uc[iii0 + js * HS_DNY] - u0, u0 - uc[iii0 - js * HS_DNY]);
									const auto s_z = limiter(uc[iii0 + ks * HS_DNZ] - u0, u0 - uc[iii0 - ks * HS_DNZ]);
									const auto s_xy = limiter(uc[iii0 + is * HS_DNX + js * HS_DNY] - u0,
											u0 - uc[iii0 - is * HS_DNX - js * HS_DNY]);
									const auto s_xz = limiter(uc[iii0 + is * HS_DNX + ks * HS_DNZ] - u0,
											u0 - uc[iii0 - is * HS_DNX - ks * HS_DNZ]);
									const auto s_yz = limiter(uc[iii0 + js * HS_DNY + ks * HS_DNZ] - u0,
											u0 - uc[iii0 - js * HS_DNY - ks * HS_DNZ]);
									const auto s_xyz = limiter(uc[iii0 + is * HS_DNX + js * HS_DNY + ks * HS_DNZ] - u0,
											u0 - uc[iii0 - is * HS_DNX - js * HS_DNY - ks * HS_DNZ]);
									auto &uf = Uf[f][iii0][ir][jr][kr];
									uf = u0;
									uf += (9.0 / 64.0) * (s_x + s_y + s_z);
									uf += (3.0 / 64.0) * (s_xy + s_yz + s_xz);
									uf += (1.0 / 64.0) * s_xyz;
								}
							}
						}
					}
				}
			}
		}
	}

	for (int f = 0; f < NRF; f++) {
		for (int i = 0; i < H_NX; i++) {
			for (int j = 0; j < H_NX; j++) {
				for (int k = 0; k < H_NX; k++) {
					const int i0 = (i + H_BW) / 2;
					const int j0 = (j + H_BW) / 2;
					const int k0 = (k + H_BW) / 2;
					const int iii0 = hSindex(i0, j0, k0);
					const int iiir = hindex(i, j, k);
					if (is_coarse[iii0]) {
						int ir, jr, kr;
						if constexpr (H_BW % 2 == 0) {
							ir = i % 2;
							jr = j % 2;
							kr = k % 2;
						} else {
							ir = 1 - (i % 2);
							jr = 1 - (j % 2);
							kr = 1 - (k % 2);
						}
						U[f][iiir] = Uf[f][iii0][ir][jr][kr];
					}
				}
			}
		}
	}

}

std::vector<real> rad_grid::get_subset(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub) {
	PROFILE();
	std::vector<real> data;
	for (int f = 0; f < NRF; f++) {
		for (int i = lb[0]; i < ub[0]; i++) {
			for (int j = lb[1]; j < ub[1]; j++) {
				for (int k = lb[2]; k < ub[2]; k++) {
					data.push_back(U[f][hindex(i, j, k)]);
				}
			}
		}
	}
	return std::move(data);

}

using send_rad_amr_boundary_action_type = node_server:: send_rad_amr_boundary_action;
HPX_REGISTER_ACTION (send_rad_amr_boundary_action_type);

void node_server::recv_rad_amr_boundary(std::vector<real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_rad_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

void node_client::send_rad_amr_boundary(std::vector<real> &&data, const geo::direction &dir, std::size_t cycle) const {
	hpx::apply<typename node_server::send_rad_amr_boundary_action>(get_unmanaged_gid(), std::move(data), dir, cycle);
}

#endif
