//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "octotiger/lane_emden.hpp"
#include "octotiger/math/Real.hpp"

#include <cmath>

static inline Real pow_n(Real y, Real n) {
    return std::pow(y, n);
}

static inline Real fy(Real y, Real z, Real r) {
	return z;
}

static inline Real fz(Real y, Real z, Real r, Real n) {
	if (r != 0.0) {
		return -(pow_n(y, n) + 2.0 * z / r);
	}
	return -3.0;
}

static inline Real fm(Real theta, Real dummy, Real r, Real n) {
    constexpr static Real four_pi = Real(4) * M_PI;
	return four_pi * pow_n(theta, n) * r * r;
}

Real lane_emden(Real r0, Real dr, Real n, Real* m_enc) {
    Real dy1, dz1, y, z, r, dy2, dz2, dy3, dz3, dy4, dz4, y0, z0;
	Real dm1, m, dm2, dm3, dm4, m0;
	int done = 0;
	y = 1.0;
	z = 0.0;
	m = 0.0;
	int N = static_cast<int>(r0 / dr + 0.5);
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
		dz1 = fz(y, z, r, n) * dr;
		dm1 = fm(y, z, r, n) * dr;
		y += 0.5 * dy1;
		z += 0.5 * dz1;
		m += 0.5 * dm1;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
        Real rdr2 = r + 0.5 * dr;
		dy2 = fy(y, z, rdr2) * dr;
		dz2 = fz(y, z, rdr2, n) * dr;
		dm2 = fm(y, z, rdr2, n) * dr;
		y = y0 + 0.5 * dy2;
		z = z0 + 0.5 * dz2;
		m = m0 + 0.5 * dm2;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
		dy3 = fy(y, z, rdr2) * dr;
		dz3 = fz(y, z, rdr2, n) * dr;
		dm3 = fm(y, z, rdr2, n) * dr;
		y = y0 + dy3;
		z = z0 + dz3;
		m = m0 + dm3;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
        Real rdr = r + dr;
		dy4 = fy(y, z, rdr) * dr;
		dz4 = fz(y, z, rdr, n) * dr;
		dm4 = fm(y, z, rdr, n) * dr;
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
	}
	return y;
}

Real wd_radius(Real mass, Real* rho0) {
	Real rho_min, rho_max, rho_mid;
	Real test_mass;
	rho_min = 1.0e-3;
	rho_max = 1.0e+3;
	Real r;
	do {
		rho_mid = sqrt(rho_min * rho_max);
		r = lane_emden(rho_mid, 0.001, 1.5, &test_mass);
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

Real find_V(Real q) {

	const Real qp1 = 1.0 + q;
	const Real qp1inv = 1.0 / qp1;
	Real x, y, z;
	Real fx, fy, fz, r1inv, r2inv, phi, phi_l1;
	Real h = 5.0e-2;
    Real r1inv3, r2inv3;
	const Real l1_x = find_l1(q);
	r1inv = 1.0 / sqrt(pow(l1_x + q * qp1inv, 2));
	r2inv = 1.0 / sqrt(pow(l1_x - qp1inv, 2));
	phi_l1 = -1.0 * r1inv - q * r2inv - 0.5 * qp1 * (l1_x * l1_x);
	int in = 0;

    for (x = l1_x; x < 1.0 + l1_x; x += h) {
		for (y = h / 2.0; y < 0.5; y += h) {
			for (z = h / 2.0; z < 0.5; z += h) {
				r1inv = 1.0 / sqrt(pow(x + q * qp1inv, 2) + y * y + z * z);
				r2inv = 1.0 / sqrt(pow(x - qp1inv, 2) + y * y + z * z);
				phi = -r1inv - q * r2inv - 0.5 * qp1 * (x * x + y * y);
				if (phi < phi_l1) {
					r1inv3 = r1inv * r1inv * r1inv;
					r2inv3 = r2inv * r2inv * r2inv;
					Real dx = x - qp1inv;
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

Real binary_separation(Real accretor_mass, Real donor_mass, Real donor_radius, Real fill_factor) {
    constexpr static Real pi_4_3 = 4.0 / 3.0 * M_PI;
	Real q = donor_mass / accretor_mass;
	Real normalized_roche_volume = find_V(q) * fill_factor;
	Real roche_radius = std::pow(normalized_roche_volume / pi_4_3, 1.0 / 3.0);
	Real separation = donor_radius / roche_radius;
	return separation;
}
