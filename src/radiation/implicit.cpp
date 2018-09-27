#include <string>
#include <array>
#include <numeric>
#include <cmath>
#include <assert.h>
#include <functional>
#include "../space_vector.hpp"
#include "opacities.hpp"

#include "../physcon.hpp"

#include <functional>
#define _3DIM 3

using quad = long double;
using function_type = std::function<quad(quad)>;

namespace std {
__float128 abs(__float128 a) {
	return fabsq(a);
}
}

class quad_space_vector: public std::array<quad, NDIM> {
public:
	quad_space_vector() = default;
	quad_space_vector(const quad_space_vector& other) = default;
	quad_space_vector& operator=(const quad_space_vector& other) = default;
	quad_space_vector(const space_vector& other) {
		for (int d = 0; d < NDIM; d++) {
			(*this)[d] = other[d];
		}
	}
	quad_space_vector& operator=(const space_vector& other) {
		for (int d = 0; d < NDIM; d++) {
			(*this)[d] = other[d];
		}
		return *this;
	}
	quad_space_vector& operator+=(const quad_space_vector& other) {
		for (int d = 0; d < NDIM; ++d) {
			(*this)[d] += other[d];
		}
		return *this;
	}
	quad_space_vector& operator-=(const quad_space_vector& other) {
		for (int d = 0; d < NDIM; ++d) {
			(*this)[d] -= other[d];
		}
		return *this;
	}
	quad_space_vector operator+(const quad_space_vector& other) const {
		quad_space_vector rc;
		for (int d = 0; d < NDIM; ++d) {
			rc[d] = (*this)[d] + other[d];
		}
		return rc;
	}
	quad_space_vector operator-(const quad_space_vector& other) const {
		quad_space_vector rc;
		for (int d = 0; d < NDIM; ++d) {
			rc[d] = (*this)[d] - other[d];
		}
		return rc;
	}
	quad_space_vector operator*(const quad& other) const {
		quad_space_vector rc;
		for (int d = 0; d < NDIM; ++d) {
			rc[d] = (*this)[d] * other;
		}
		return rc;
	}
	quad_space_vector operator/(const quad& other) const {
		quad_space_vector rc;
		for (int d = 0; d < NDIM; ++d) {
			rc[d] = (*this)[d] * INVERSE(other);
		}
		return rc;
	}
};

inline quad_space_vector operator*(const quad& a, const quad_space_vector& b) {
	return b * a;
}

using mat_function_type = std::function<quad(quad, quad, quad)>;

inline quad dot(const quad_space_vector& a, quad_space_vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void implicit_radiation_step(quad& E0, quad& e0, quad_space_vector& F0, quad_space_vector& u0, quad rho, quad mmw, quad kp,
		quad kr, quad dt) {

	const quad c = physcon.c;
	const quad cinv = quad(1.0) / c;
	const quad c2 = c * c;

	quad_space_vector F1, u1;
	quad e1;
	quad E1;

	const auto f =
			[&](quad this_E) {
				F1 = (quad(3)*c*F0*rho + quad(4)*this_E*dt*kr*(F0 + c2*rho*u0))*INVERSE(quad(4)*this_E*dt*kr + quad(3)*c*(quad(1) + c*dt*kr)*rho);
				u1 = u0 + (F0 - F1)*INVERSE(rho*c2);
				e1 = std::max(e0 - ((this_E - E0) + (quad(1.0)/quad(2.0)) * rho * (dot(u1,u1) - dot(u0,u0))),quad(0.0));
				const quad B = quad(4)*quad(M_PI)*B_p(rho,e1,mmw)*cinv;
				quad f1 = this_E - E0;
				quad f2 = dt * (c * kp * (this_E - B));
				quad f3 = dt * ( - (quad(2) * kp - kr) * dot(u1,F1) * cinv);
				E1 = this_E;
				if( std::abs(f1) > std::abs(f2) ) {
					std::swap(f1,f2);
				}
				if( std::abs(f2) > std::abs(f3) ) {
					std::swap(f2,f3);
				}
				if( std::abs(f1) > std::abs(f2) ) {
					std::swap(f1,f2);
				}
				return (f1 + f2) + f3;
			};

	const quad ke = quad(0.5) * rho * dot(u0, u0);
	quad minE = quad(0.0);
	quad maxE = E0 + (e0 + ke);
	quad error;
	int iter = 0;
	do {
		quad fmax, fmid;
		quad midE = quad(0.5) * (maxE + minE);
		fmax = f(maxE);
		fmid = f(midE);
		const quad score = fmax * fmid;
		if (score != quad(0.0)) {
			error = (maxE - minE) * INVERSE(midE);
		} else {
			error = quad(0.0);
		}

		if (++iter > 100) {
			printf("v1 %e %e %e %e | %e %e %e %e | %e %e \n", double(E0), (double) (F0[0] / (E0 * c)),
					(double) (F0[1] / (E0 * c)), (double) (F0[2] / (E0 * c)), double(e0), (double) (u0[0] / (c)),
					(double) (u0[1] / c), (double) (u0[2] / c), (double) kr, (double) kp);
			printf("binary search for Er failed\n");
			abort();
		}
		if (score < quad(0.0)) {
			minE = midE;
		} else if (score > quad(0.0)) {
			maxE = midE;
		}
	} while (error > quad(1.0e-14));

	e0 = e1;
	E0 = quad(0.5) * (maxE + minE);
	F0 = F1;
	u0 = u1;
}

void implicit_radiation_step_2nd_order(quad& E0, quad& e0, quad_space_vector& F0, quad_space_vector& u0, quad rho, quad mmw,
		quad kp, quad kr, quad dt) {

	const quad c2 = quad(physcon.c) * quad(physcon.c);
	const auto step = [rho, mmw, kp, kr, c2](quad E, quad e, quad_space_vector F, quad_space_vector u, quad dt ) {
		quad E1 = E;
		quad e1 = e;
		quad_space_vector F1 = F;
		quad_space_vector u1 = u;
		implicit_radiation_step(E1, e1, F1, u1, rho, mmw, kp, kr, dt);
		quad dE;
		quad_space_vector dF;
		if( E1 > e1 ) {
			dE = -e1 + e + 0.5 * rho * (dot(u,u) - dot(u1,u1));
			dF = rho * c2 * (u - u1);
		} else {
			dE = E1 - E;
			dF = F1 - F;
		}
		return std::make_pair<quad,quad_space_vector>(dE/dt, dF/dt);

	};

	constexpr int N = 2;
	const quad gam = quad(1) + std::sqrt(quad(2)) / quad(2);
	const quad A[N][N] = { { gam, quad(0) }, { quad(1) - gam, gam } };
	const quad B[N] = { quad(1) - gam, gam };
	quad kdE[N];
	quad_space_vector kdF[N];
	quad E, e;
	quad_space_vector F, u;
	for (int i = 0; i < N; i++) {
		E = E0;
		F = F0;
		u = u0;
		e = e0;
		for (int j = 0; j < i; j++) {
			const quad dE = (A[i][j] * kdE[j]) * dt;
			const quad_space_vector dF = (A[i][j] * kdF[j]) * dt;
			E += dE;
			F += dF;
			auto u0 = u;
			u -= dF * INVERSE(rho * c2);
			e -= dE + rho * (dot(u, u) - dot(u0, u0)) / quad(2);
		}
		const auto tmp = step(E, e, F, u, A[i][i] * dt);
		kdE[i] = tmp.first;
		kdF[i] = tmp.second;
	}
	E = E0;
	F = F0;
	u = u0;
	for (int i = 0; i < N; i++) {
		E0 += B[i] * kdE[i] * dt;
		F0 += B[i] * kdF[i] * dt;
	}
	u0 -= (F0 - F) * INVERSE(rho * c2);
	e0 -= (E0 - E) + quad(0.5) * rho * (dot(u0, u0) - dot(u, u));
}

space_vector quad_space_vector_to_space_vector(const quad_space_vector& q) {
	space_vector r;
	for (int d = 0; d < NDIM; d++) {
		r[d] = q[d];
	}
	return r;
}

std::pair<real, space_vector> implicit_radiation_step_2nd_order(real E0_, real& e0_, const space_vector& F0_,
		const space_vector& u0_, real rho_, real mmw_, real dt_) {

	quad rho = rho_;
	quad mmw = mmw_;
	quad dt = dt_;
	quad E0 = E0_;
	quad e0 = e0_;
	quad_space_vector F0 = F0_;
	quad_space_vector u0 = u0_;

	const quad dtinv = INVERSE(dt);


// Take a single backward Euler step to get opacities at end of timestep to average with start of timestep
	quad E1 = E0;
	quad e1 = e0;
	quad_space_vector F1 = F0;
	quad_space_vector u1 = u0;
	quad kp = kappa_p(rho, e0, mmw);
	quad kr = kappa_R(rho, e0, mmw);
	implicit_radiation_step(E1, e1, F1, u1, rho, mmw, kp, kr, dt);
	kp += 0.5 * (kappa_p(rho, e1, mmw) - kp);
	kr += 0.5 * (kappa_R(rho, e1, mmw) - kr);

	//Now take 2nd order step

	implicit_radiation_step(E1, e1, F1, u1, rho, mmw, kp, kr, dt);
	e0_ = e1;
//	implicit_radiation_step_2nd_order(E0, e0, F0, u0, rho, mmw, kp, kr, dt);
	return std::make_pair(real((E1 - E0) * dtinv), quad_space_vector_to_space_vector((F1 - F0) * dtinv));
}

