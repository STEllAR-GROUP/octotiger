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

inline quad dot(const quad_space_vector& a, const quad_space_vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void implicit_radiation_step(quad& E0, quad& e0, quad_space_vector& F0, quad_space_vector& u0, quad rho, quad mmw, quad kp,
		quad kr, quad dt) {
//	F0[0] = F0[1] = F0[2] = 0.0;
//	u0[0] = u0[1] = u0[2] = 0.0;

	const quad c = physcon.c;
	const quad cinv = quad(1.0) / c;
	const quad c2 = c * c;

	quad e1 = e0;
	quad dE1 = 0;

	quad_space_vector base1, base2;

	if (dot(u0, u0) == quad(0)) {
		if (dot(F0, F0) == quad(0)) {
			base1[1] = base1[2] = base2[0] = base2[2] = quad(0);
			base1[0] = base2[1] = quad(1);
		} else {
			base1[0] = base1[1] = base1[2] = quad(0);
			base2 = F0 * INVERSE(SQRT(dot(F0,F0)));
		}
	} else {

		base1 = u0 * INVERSE(SQRT(dot(u0,u0)));
		base2 = F0 - base1 * (dot(base1, F0));
		base2 = base2 * INVERSE(SQRT(dot(base2,base2)));
	}

	quad Fa0 = dot(F0, base1);
	quad Fb0 = dot(F0, base2);
	quad ua0 = dot(u0, base1);
	quad ub0 = dot(u0, base2);

	quad dFa1 = 0;
	quad dFb1 = 0;

	quad Fa1 = Fa0;
	quad Fb1 = Fb0;
	quad ua1 = ua0;
	quad ub1 = ub0;

	quad error = 1e+99;
	const quad cdt = c * dt;
	quad fe, fa, fb;
	int iter = 0;
	do {

		quad B = quad(4) * quad(M_PI) * B_p(rho, e1, mmw) * cinv;
		quad dBde = quad(4) * quad(M_PI) * dB_p_de(rho, e1, mmw) * cinv;

		fe = dE1 + cdt * kp * (E0 + dE1 - B) + dt * (kr - quad(2) * kp) * (ua1 * (Fa0 + dFa1) + ub1 * (Fb0 + dFb1)) * cinv;
		fa = dFa1 + cdt * kr * ((Fa0 + dFa1) - quad(4) / quad(3) * ua1 * (E0 + dE1));
		fb = dFb1 + cdt * kr * ((Fb0 + dFb1) - quad(4) / quad(3) * ub1 * (E0 + dE1));

		if( iter > 50 ) {
		printf("%3i %16.9e | %16.9e %16.9e %16.9e | %16.9e %16.9e %16.9e | %16.9e %16.9e %16.9e | %16.9e\n", iter, double(dE1),
				double(E0 + dE1), double(Fa1 / (c * (E0 + dE1))), double(Fb1 / (c * (E0 + dE1))), double(e1),
				double(ua1 * cinv), double(ub1 * cinv), double(fe), double(fa), double(fb), double(error));
		}

		const quad A00 = quad(1) + cdt * kp * (quad(1) + dBde);
		const quad A01 = -kp * dt * ua1 * cinv * dBde / quad(2)
				+ dt * (kr - quad(2) * kp) * (ua1 - (Fa0 + dFa1) * INVERSE(rho * c2)) * cinv;
		const quad A02 = -kp * dt * ub1 * cinv * dBde / quad(2)
				+ dt * (kr - quad(2) * kp) * (ub1 - (Fb0 + dFb1) * INVERSE(rho * c2)) * cinv;
		const quad A10 = -quad(4) / quad(3) * cdt * kr * ua1;
		const quad A20 = -quad(4) / quad(3) * cdt * kr * ub1;
		const quad A11 = (quad(1) + cdt * kr * (quad(1) + quad(4) / quad(3) * INVERSE(rho * c2) * (E0 + dE1)));

		const quad detAinv = INVERSE(A00 * A11 * A11 - A01 * A10 * A11 - A02 * A11 * A20);
		const quad B00 = +A11 * A11 * detAinv;
		const quad B01 = -A01 * A11 * detAinv;
		const quad B02 = -A02 * A11 * detAinv;
		const quad B10 = -A10 * A11 * detAinv;
		const quad B11 = -(A02 * A20 - A00 * A11) * detAinv;
		const quad B12 = +A02 * A10 * detAinv;
		const quad B20 = -A11 * A20 * detAinv;
		const quad B21 = +A01 * A20 * detAinv;
		const quad B22 = (-A01 * A10 + A00 * A11) * detAinv;

		quad dE = -(B00 * fe + B01 * fa + B02 * fb);
		const quad da = -(B10 * fe + B11 * fa + B12 * fb);
		const quad db = -(B20 * fe + B21 * fa + B22 * fb);

		dE = std::max(dE, -(E0 + dE1) / quad(2));
		dE1 += dE;
		dFa1 += da;
		dFb1 += db;
		const quad dua = -da * INVERSE(rho * c2);
		const quad dub = -db * INVERSE(rho * c2);
		const quad dk = quad(0.5) * rho * (quad(2) * dua * ua1 + dua * dua + quad(2) * dub * ub1 + dub * dub);
		ua1 += dua;
		ub1 += dub;
		e1 = std::max(e1 - dE - dk, e1 / quad(2));

		error = std::sqrt((fe * fe + fa * fa * cinv * cinv + fb * fb * cinv * cinv)) * INVERSE(E0 + dE1);

		if (iter > 100) {
			printf("Implicit radiation failed to converge\n");
			abort();
			break;
		}
		iter++;
	} while (std::abs(error) > 5.0e-7);
	if( iter > 50 ) {
		printf("%3i %16.9e | %16.9e %16.9e %16.9e | %16.9e %16.9e %16.9e | %16.9e %16.9e %16.9e | %16.9e\n", iter, double(dE1),
			double(E0 + dE1), double(Fa1 / (c * (E0 + dE1))), double(Fb1 / (c * (E0 + dE1))), double(e1),
			double(ua1 * cinv), double(ub1 * cinv), double(fe), double(fa), double(fb), double(error));
	}
	E0 = E0 + dE1;
	e0 = e1;
	for (int i = 0; i < NDIM; i++) {
		F0[i] = Fa1 * base1[i] + Fb1 * base2[i];
		u0[i] = ua1 * base1[i] + ub1 * base2[i];
	}

}

/*
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

 //				F1 = (quad(3)*c*F0*rho + quad(4)*dt*kr*this_E*(F0 + c2*u0*rho))*INVERSE(quad(4)*dt*kr*this_E + quad(3)*c*(quad(1) + c*dt*kr)*rho);

 u1 = u0 + (F0 - F1)*INVERSE(rho*c2);
 e1 = std::max(e0 - ((this_E - E0) + (quad(1.0)/quad(2.0)) * rho * (dot(u1,u1) - dot(u0,u0))),quad(0.0));
 const quad B = quad(4)*quad(M_PI)*B_p(rho,e1,mmw)*cinv;
 quad f1 = this_E - E0;
 quad f2 = dt * (c * kp * (this_E - B));
 quad f3 = dt * ((-quad(2) * kp + kr) * dot(u1,F1) * cinv - kr * quad(4)/quad(3)*dot(u1,u1) * this_E * cinv);
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

 printf("%i %e %e %e %e | %e %e %e %e | %e %e | %.20e \n", iter, double(E1), (double) (F1[0] / (E1 * c)),
 (double) (F1[1] / (E1 * c)), (double) (F1[2] / (E1 * c)), double(e1), (double) (u1[0] / (c)),
 (double) (u1[1] / c), (double) (u1[2] / c), (double) fmax, (double) fmid, double(error));
 if (++iter > 75) {
 if (iter > 100) {
 printf("binary search for Er failed\n");
 printf("%i %e %e %e %e | %e %e %e %e | %e %e | %e \n", iter, double(E0), (double) (F0[0] / (E0 * c)),
 (double) (F0[1] / (E0 * c)), (double) (F0[2] / (E0 * c)), double(e0), (double) (u0[0] / (c)),
 (double) (u0[1] / c), (double) (u0[2] / c), (double) fmax, (double) fmid, error);
 abort();
 }
 }
 if (score < quad(0.0)) {
 minE = midE;
 } else if (score > quad(0.0)) {
 maxE = midE;
 }
 } while (std::abs(error) > quad(1.0e-14));

 e0 = e1;
 E0 = quad(0.5) * (maxE + minE);
 F0 = F1;
 u0 = u1;
 }

 */

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
//	kp += 0.5 * (kappa_p(rho, e1, mmw) - kp);
//	kr += 0.5 * (kappa_R(rho, e1, mmw) - kr);

//Now take 2nd order step

//	implicit_radiation_step_2nd_order(E0, e0, F0, u0, rho, mmw, kp, kr, dt);
	e0_ = e1;
	return std::make_pair(real((E1 - E0) * dtinv), quad_space_vector_to_space_vector((F1 - F0) * dtinv));
}

