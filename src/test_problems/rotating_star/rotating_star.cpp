//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "octotiger/test_problems/rotating_star.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

class rotating_star_analytic {
private:
	oct::vector<oct::vector<double>> rho_;
	oct::vector<oct::vector<double>> ene_;
	double dr_, dz_, omega_;
	int nr_, nz_;
public:
	double interpolate(const oct::vector<oct::vector<double>>& f, double R, double z) const {
                // bicubic interpolation by the prescreption: f(x,y) = (x^3 x^2 x 1)A(y^3 y^2 y 1)^T, and F = BAB^T
                static constexpr real mycoeff[4][4] = {{1., 0., 0., 0.}, {-1.83333333, 3., -1.5, 0.333333333}, {1., -2.5,
  2., -0.5}, {-0.166666667, 0.5, -0.5, 0.166666667}}; // B^-1
                real coeff2[4][4] = { }; // initialize A, the coefficients per interval, with zeros
                R = std::abs(R);
                z = std::abs(z);
                int i = int(R / dr_ - 0.5) + nr_ / 2; // dens(R_i)=dens((i+nr/2+0.5)*dr)
                int k = int(z / dz_ - 0.5) + nz_ / 2; // z_i = (k+nz/2+0.5)dz
                real rc = 0.0;
                if (i >= 0 && i < nr_ - 3 && k >= 0 && k < nz_ - 3) {
                        for (int i0 = 0; i0 < 4; i0++) {
                                for (int k0 = 0; k0 < 4; k0++) {
                                        for (int i1 = 0; i1 < 4; i1++) {
                                                for (int k1 = 0; k1 < 4; k1++) {
                                                        coeff2[i0][k0] += mycoeff[i0][i1] * f[i+i1][k+k1] * mycoeff[k0][k1]; // A = B^-1F(B^-1)^T
                                                }
                                        }
                                }
                        }
                        double x = R / dr_ - 0.5 - int(R / dr_ - 0.5); // x = (R - R_i) / dr
                        double y = z / dz_ - 0.5 - int(z / dz_ - 0.5); // y = (z - z_i) / dz
                        for (int i0 = 0; i0 < 4; i0++) {
                                for (int k0 = 0; k0 < 4; k0++) {
                                        rc += coeff2[i0][k0] * std::pow(x, i0) * std::pow(y, k0); // f(x,y) = \sigma a_ij x^i y^j = (x^3 x^2 x 1)A(y^3 y^2 y 1)^T
                                }
                        }
                        return std::max(rc, 0.0);
                } else {
                        return 0.0;
                }
        }
	rotating_star_analytic() {
		std::ifstream fp("rotating_star.bin", std::ios::in | std::ios::binary);
		if (fp.fail()) {
			std::cout << "Could not open rotating_star.bin, aborting\n";
			throw;
		}
		std::cout << "Reading rotating_star.bin\n";

		fp.read(reinterpret_cast<char*>(&nr_), sizeof(decltype(nr_)));
		fp.read(reinterpret_cast<char*>(&nz_), sizeof(decltype(nz_)));
		dr_ = 1.0 / nr_;
		dz_ = 1.0 / nz_;
		nr_ *= 2;
		nz_ *= 2;
		fp.read(reinterpret_cast<char*>(&omega_), sizeof(double));
		rho_.resize(nr_, oct::vector<double>(nz_));
		ene_.resize(nr_, oct::vector<double>(nz_));
		for (int i = 0; i < nr_; i++) {
			for (int k = 0; k < nz_; k++) {
				fp.read(reinterpret_cast<char*>(&(rho_[i][k])), sizeof(double));
				fp.read(reinterpret_cast<char*>(&(ene_[i][k])), sizeof(double));
			}
		}
		std::cout << "Done reading rotating_star.bin\n";
		print( "Omega = %e\n", omega_);
	}
	void state_at(double& rho, double& ene, double& sx, double& sy, double x, double y, double z) const {
		const double R = std::sqrt(x * x + y * y);
		rho = interpolate(rho_, R, z);
		ene = interpolate(ene_, R, z);
		sx = -y * rho * omega_;
		sy = +x * rho * omega_;
	}
	double get_omega() {
		return omega_;
	}

};

oct::vector<real> rotating_star(real x, real y, real z, real dx) {
	oct::vector<real> u(opts().n_fields, real(0));

	x -= opts().rotating_star_x;

	static rotating_star_analytic rs;
	const real fgamma = 5.0 / 3.0;
	rs.state_at(u[rho_i], u[egas_i], u[sx_i], u[sy_i], x, y, z);
//	u[egas_i] = (1.681244e-01) * std::pow(u[rho_i],fgamma) / (fgamma-1.0);
	u[rho_i] = std::max(u[rho_i], 1.0e-10);
	u[egas_i] = std::max(u[egas_i], 1.0e-10);
	u[tau_i] = std::pow(u[egas_i], 1.0 / fgamma);
	u[egas_i] += 0.5 * (std::pow(u[sx_i], 2) + std::pow(u[sy_i], 2)) / u[rho_i];
	u[spc_i] = u[rho_i];
	if (u[rho_i] > 1.0e-10) {
		u[spc_i + 1] = 0.0;
        } else {
                u[spc_i] = 0.0;
                u[spc_i + 1] = u[rho_i];
        }
//	if( u[rho_i] < 0.5 ) {
//		u[spc_i + 1] = u[rho_i];
//		u[spc_i + 0] = 0;
//	} else {
//		u[spc_i + 0] = u[rho_i];
//		u[spc_i + 1] = 0;
//	}
//	u[zz_i] = rs.get_omega() * dx * dx / 6.0 * u[rho_i];
	for (int s = 2; s < opts().n_species; s++) {
		u[spc_i + s] = 0.0;
	}
	return u;
}

oct::vector<real> rotating_star_a(real x, real y, real z, real) {
	return rotating_star(x, y, z, 0);
}
