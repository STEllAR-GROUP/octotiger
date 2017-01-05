/*
 * lane_emden.cpp
 *
 *  Created on: Feb 6, 2015
 *      Author: dmarce1
 */

#include "lane_emden.hpp"
#include <cmath>

static inline real pow_1_5(real y) {
    return y * std::sqrt(y);
}

static inline real fy(real y, real z, real r) {
	return z;
}

static inline real fz(real y, real z, real r) {
	if (r != 0.0) {
		return -(pow_1_5(y) + 2.0 * z / r);
	} else {
		return -3.0;
	}
}

static inline real fm(real theta, real dummy, real r) {
    constexpr static real four_pi = real(4) * M_PI;
	return four_pi * pow_1_5(theta) * r * r;
}

real lane_emden(real r0, real dr, real* m_enc) {
	int N;
	real dy1, dz1, y, z, r, dy2, dz2, dy3, dz3, dy4, dz4, y0, z0;
	real dm1, m, dm2, dm3, dm4, m0;
	int done = 0;
	y = 1.0;
	z = 0.0;
	m = 0.0;
	N = (int) (r0 / dr + 0.5);
	if (N < 1) {
		N = 1;
	}
	r = 0.0;
	do {
		if (r + dr > r0) {
			dr = r0 - r;
			done = 1;
		}
		y0 = y;
		z0 = z;
		m0 = m;
		dy1 = fy(y, z, r) * dr;
		dz1 = fz(y, z, r) * dr;
		dm1 = fm(y, z, r) * dr;
		y += 0.5 * dy1;
		z += 0.5 * dz1;
		m += 0.5 * dm1;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
        real rdr2 = r + 0.5 * dr;
		dy2 = fy(y, z, rdr2) * dr;
		dz2 = fz(y, z, rdr2) * dr;
		dm2 = fm(y, z, rdr2) * dr;
		y = y0 + 0.5 * dy2;
		z = z0 + 0.5 * dz2;
		m = m0 + 0.5 * dm2;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
		dy3 = fy(y, z, rdr2) * dr;
		dz3 = fz(y, z, rdr2) * dr;
		dm3 = fm(y, z, rdr2) * dr;
		y = y0 + dy3;
		z = z0 + dz3;
		m = m0 + dm3;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
        real rdr = r + dr;
		dy4 = fy(y, z, rdr) * dr;
		dz4 = fz(y, z, rdr) * dr;
		dm4 = fm(y, z, rdr) * dr;
		y = y0 + (dy1 + dy4 + 2.0 * (dy3 + dy2)) / 6.0;
		z = z0 + (dz1 + dz4 + 2.0 * (dz3 + dz2)) / 6.0;
		m = m0 + (dm1 + dm4 + 2.0 * (dm3 + dm2)) / 6.0;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
		r += dr;
	} while (done == 0);
	if (m_enc != nullptr) {
		*m_enc = m;
	}
	if (y < 0.0) {
		return 0.0;
	} else {
		return y;
	}
}

real wd_radius(real mass, real* rho0) {
	real rho_min, rho_max, rho_mid;
	real test_mass;
	rho_min = 1.0e-3;
	rho_max = 1.0e+3;
	real r;
	do {
		rho_mid = sqrt(rho_min * rho_max);
		r = lane_emden(rho_mid, 0.001, &test_mass);
		if (test_mass > mass) {
			rho_max = rho_mid;
		} else {
			rho_min = rho_mid;
		}
//		printf("%e %e %e %e %e\n", rho_min, rho_mid, rho_max, test_mass, mass);
	} while (log(rho_max / rho_min) > 1.0e-9);
	*rho0 = rho_mid;
	return r;
}

double find_l1(double q) {

	double y, f, df, f1, f2, fr, df1, df2, dfr;
	const double qp1 = 1.0 + q;
	const double qp1inv = 1.0 / qp1;

	y = 0.0;
	do {
		f1 = -pow(y + q * qp1inv, -2);
		f2 = +q * pow(y - qp1inv, -2);
		fr = qp1 * y;
		f = f1 + f2 + fr;
		df1 = +2.0 * pow(y + q * qp1inv, -3);
		df2 = -2.0 * q * pow(y - qp1inv, -3);
		dfr = qp1;
		df = df1 + df2 + dfr;
		y -= f / df;
		//	printf( "%e %e %e\n", y, df, f );
	} while (fabs(f) > 1.0e-10);
	return y;

}

real find_V(real q) {

	const real qp1 = 1.0 + q;
	const real qp1inv = 1.0 / qp1;
	real x, y, z;
	real fx, fy, fz, r1inv, r2inv, phi, phi_l1;
	real h = 5.0e-2;
	int in;
	real r1inv3, r2inv3;
	const real l1_x = find_l1(q);
	r1inv = 1.0 / sqrt(pow(l1_x + q * qp1inv, 2));
	r2inv = 1.0 / sqrt(pow(l1_x - qp1inv, 2));
	phi_l1 = -1.0 * r1inv - q * r2inv - 0.5 * qp1 * (l1_x * l1_x);
	in = 0;
	real dx;

	for (x = l1_x; x < 1.0 + l1_x; x += h) {
		for (y = h / 2.0; y < 0.5; y += h) {
			for (z = h / 2.0; z < 0.5; z += h) {
				r1inv = 1.0 / sqrt(pow(x + q * qp1inv, 2) + y * y + z * z);
				r2inv = 1.0 / sqrt(pow(x - qp1inv, 2) + y * y + z * z);
				phi = -r1inv - q * r2inv - 0.5 * qp1 * (x * x + y * y);
				if (phi < phi_l1) {
					r1inv3 = r1inv * r1inv * r1inv;
					r2inv3 = r2inv * r2inv * r2inv;
					dx = x - qp1inv;
					fx = -(x + q * qp1inv) * r1inv3 - q * dx * r2inv3 + qp1 * x;
					fy = -y * r1inv3 - y * q * r2inv3 + qp1 * y;
					fz = -z * r1inv3 - z * q * r2inv3;
					if (fx * dx + fy * y + fz * z <= 0.0) {
						in++;
					}
				}
			}
		}
	}
//	printf( "!\n");
	return 4.0 * in * h * h * h;
}

real binary_separation(real accretor_mass, real donor_mass, real donor_radius, real fill_factor) {
    constexpr static real pi_4_3 = 4.0 / 3.0 * M_PI;
	real q = donor_mass / accretor_mass;
	real normalized_roche_volume = find_V(q) * fill_factor;
	real roche_radius = std::pow(normalized_roche_volume / pi_4_3, 1.0 / 3.0);
	real separation = donor_radius / roche_radius;
	return separation;
}
