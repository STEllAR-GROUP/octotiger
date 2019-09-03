//  Copyright (c) 2019
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "1d_polytrope.hpp"
#include "cubic_table.hpp"
#include <memory>
#include <cmath>
#include <vector>

spherical_polytrope make_1d_spherical_polytrope(
		const std::function<double(double)>& d_of_h_norm,
		const std::function<double(double)>& p_of_d_norm, double M, double R) {

	double h, hdot, r, dr;

	double d0 = 1.0;

	double h0 = d0;
	double err;
	std::vector<double> r_data;
	std::vector<double> p_data;
	std::vector<double> d_data;
	do {
		h = h0;
		hdot = 0.0;
		r = 0.0;
		dr = 1.0;
		double dh, dhdot, d;
		double m = 0.0;
		double dm1, dm2, dm3, dm;
		r_data.resize(0);
		d_data.resize(0);
		p_data.resize(0);
		do {
			auto d_of_h = [&d_of_h_norm,h0,d0]( double h) {
				if( h > 0.0 ) {
					return d0 * d_of_h_norm(h/h0);
				} else {
					return 0.0;
				}
			};
			const auto dhdot_dr = [&](double h, double hdot, double r) {
				if( r == 0.0) {
					return 0.0;
				} else {
					return -2.0 * hdot / r - 4.0 * M_PI *  d_of_h(std::max(h,0.0));
				}
			};

			const auto dh_dr = [&](double hdot) {
				return hdot;
			};

			d = d_of_h(h);
			r_data.push_back(r);
			d_data.push_back(d);
			p_data.push_back(p_of_d_norm(d)*h0*d0);
			while ([&]() {
				double dh1, dh2, dh3;;
				double dhdot1, dhdot2, dhdot3;

				d = d_of_h(h);
				dhdot1 = dhdot_dr(h, hdot, r) * dr;
				dh1 = dh_dr(hdot) * dr;
				dm1 = 4.0 * M_PI * r * r * d * dr;

				d = d_of_h(h + dh1);
				dhdot2 = dhdot_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
				dh2 = dh_dr(hdot + dhdot1) * dr;
				dm2 = 4.0 * M_PI * r * r * d_of_h(h) * dr;

				d = h + (dh1 + dh2)/4.0;
				dhdot3 = dhdot_dr(h + (dh1 + dh2)/4.0, hdot + (dhdot1 + dhdot2)/4.0, r + 0.5 * dr) * dr;
				dh3 = dh_dr(hdot + (dhdot1 + dhdot2)/4.0) * dr;
				dm3 = 4.0 * M_PI * r * r * d_of_h(h + (dh1 + dh2)/4.0) * dr;

				dh = (dh1 + 4.0 * dh3 + dh2) / 6.0;
				dhdot = (dhdot1 + 4.0 * dhdot3 + dhdot2) / 6.0;
				dm = (dm1 + 4.0 * dm3 + dm2)/6.0;

				double dlog_h = std::abs(dh/std::max(h, h0 * 1.0e-6));
				if( dlog_h > 1.0e-3) {
					dr /= std::pow(10,0.25);
					return -1;
				} else if( dlog_h < 1.0e-4 ) {
					dr *= std::pow(10,0.25);
					return +1;
				} else {
					return 0;
				}
			}() != 0) {
			}
			h += dh;
			hdot += dhdot;
			r += dr;
			h = std::max(h, 0.0);
			m += dm;
		} while (h > 0.0);
		double mf = m / M;
		double rf = r / R;
		d0 /= mf / (rf * rf * rf);
		h0 /= mf / rf;
		err = std::max(std::abs(r / R - 1.0), std::abs(m / M - 1.0));
	} while (err > 1.0e-6);
	const double rmax = r;
	spherical_polytrope rc;

	const auto d_of_r_table = std::make_shared < cubic_table > (d_data, r_data);
	const auto p_of_r_table = std::make_shared < cubic_table > (p_data, r_data);

	rc.rho_of_r = [rmax,d_of_r_table](double r) {
		if( r <= rmax) {
			return (*d_of_r_table)(r);
		} else {
			return 0.0;
		}
	};
	rc.p_of_r = [rmax,p_of_r_table](double r) {
		if( r <= rmax) {
			return (*p_of_r_table)(r);
		} else {
			return 0.0;
		}
	};
	rc.p_scale = h0 * d0;
	rc.rho_scale = d0;
	return rc;

}
