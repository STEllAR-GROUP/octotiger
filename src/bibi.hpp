/*
 * bibi.hpp
 *
 *  Created on: Jan 27, 2016
 *      Author: dmarce1
 */

#ifndef BIBI_HPP_
#define BIBI_HPP_

#include <math.h>
#include <functional>
#include "defs.hpp"

struct bibi_polytrope {
	real rho0;
	void set_P0( real p ) {
		if( p < 0.0) {
			printf( "!!!!!1\n");
		}
		P0 = p;
	}
	real get_P0() const {
		return P0;
	}
private:
	real P0;
	real  rho_C, rho_E;
	real theta_E, theta_C;
	real n_E, n_C;
	real mass, radius;
	void solve_for_mass(real target_mass);
	void solve_for_radius(real target_radius);
	void solve_for_mass_and_radius(real tmass, real tradius);
	void solve();
	void print();

	double dy_dx(real x, real y, real z) {
		return z;
	}
	real K_C(), K_E();

	double y_to_rho(real y) {
		if (y < theta_E) {
			return rho0 * pow(std::max(y, 0.0), n_E);
		} else {
			return rho0 * pow(std::max(y + theta_C - theta_E, 0.0), n_C);
		}
	}

	double dz_dx(real x, real y, real z) {
		const real c1 = pow(theta_E, 1.0 + n_E) / (1.0 + n_E);
		const real c2 = pow(theta_C, 1.0 + n_C) / (1.0 + n_C);
		real rc;
		if (y < theta_E) {
			rc = c1 * pow(std::max(y, 0.0), n_E);
		} else {
			rc = c2 * pow(std::max(y + theta_C - theta_E, 0.0), n_C);
		}
		if (x != 0.0) {
			if (y < theta_E) {
				rc += 2.0 * z / x;
			} else {
				rc += 2.0 * z / x;
			}
		}
		return -rc;
	}

	real dm_dx(real x, real y, real z) {
		real alpha = sqrt(P0 / 4.0 / M_PI / rho0 / rho0);
		return 4.0 * M_PI * alpha * alpha * alpha * x * x * y_to_rho(y);
	}

public:
	real H_C(){
		return K_C() * (1.0 + n_C) * pow(rho_C, 1.0 / n_C);
	}
	real H_E(){
		return K_E() * (1.0 + n_E) * pow(rho_E, 1.0 / n_E);
	}
	bibi_polytrope(real m, real r, real nc, real ne, real ac, real ae);
	bool solve_at(real r, real& den, real& pre, real& menc);

	bool in_core(double rho) {
		return rho > rho_E;
	}

	double rho2ent(double rho) {
		if (rho <= rho_E) {
			return K_E() * (1.0 + n_E) * pow(rho, 1.0 / n_E);
		} else if (rho < rho_C) {
			return H_E();
		} else {
			return K_C() * (1.0 + n_C) * pow(rho, 1.0 / n_C) - H_C() + H_E();
		}
	}

	double rho2ene(double rho) {
		double p;
		if (rho <= rho_E) {
			p = K_E() * pow(rho, 1.0 + 1.0 / n_E);
		} else if (rho < rho_C) {
			p = P0;
		} else {
			p = K_C() * pow(rho, 1.0 + 1.0 / n_C);
		}
		return p / (fgamma - 1.0);
	}

	double ent2rho(double H) {
		H = std::max(0.0, H);
		if (H <= H_E()) {
			return pow(H / (K_E() * (1.0 + n_E)), n_E);
		} else {
			return pow((H + H_C() - H_E()) / (K_C() * (1.0 + n_C)), n_C);
		}
	}

};

#endif /* BIBI_HPP_ */
