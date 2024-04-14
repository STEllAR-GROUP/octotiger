//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MESA_STAR_HPP_
#define MESA_STAR_HPP_

#include "octotiger/lane_emden.hpp"
#include "octotiger/grid.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <array>
#include <vector>
#include <memory>
#include "octotiger/config/export_definitions.hpp"
#include "octotiger/real.hpp"
#include "octotiger/defs.hpp"
#include <string>
#include <functional>

#define BUFFER_SIZE (1024 * 16)
#define HEADER_LINES 13
#define NCOEF 4

std::pair< std::function<double(double)>, std::function<double(double)> > build_rho_of_h_from_mesa(
		const std::string& filename, real&, real&, real&, real&, std::vector<double>& P, std::vector<double>& rho, std::vector<double>& h);

std::pair< std::function<double(double)>, std::function<double(double)> > build_rho_of_h_from_relation(std::vector<double> const P, std::vector<double> const rho, std::vector<double> const h);

OCTOTIGER_EXPORT std::vector<real> mesa_star(real, real, real, real);

OCTOTIGER_EXPORT bool mesa_refine_test(integer level, integer maxl, real, real, real,
    std::vector<real> const& U,
    std::array<std::vector<real>, NDIM> const& dudx);

class mesa_profiles {
private:
        std::vector<double> rho_;
        std::vector<double> P_;
        std::vector<double> r_;
        std::vector<double> omega_;
        int lines_;

        double interpolate_mono(std::vector<double> y, double r) const {
        // interpolating by a monotonic cubic interpolation scheme that conserves monotonicity
                auto i = find_i(r);
                if (i == 0) {
                        printf("could not find r = %, at the mesa profile. Aborting ...\n", r);
                        abort();
                        return 0;
                }
                double m0, m1;
                if (i == lines_ - 1) {
                        m1 = (y[i] - y[i-1]) / (r_[i] - r_[i-1]); // adjusting the slopes at the edges to conserve monotonicity
                } else {
                        m1 = 0.5 * (((y[i] - y[i-1]) / (r_[i] - r_[i-1])) + ((y[i+1] - y[i]) / (r_[i+1] - r_[i])));
                }
                i--;
                if (i == 0) {
                        m0 = (y[i+1] - y[i]) / (r_[i+1] - r_[i]);
                } else {
                        m0 = 0.5 * (((y[i] - y[i-1]) / (r_[i] - r_[i-1])) + ((y[i+1] - y[i]) / (r_[i+1] - r_[i])));
                }
                return cubic_hermit_interp(r, r_[i], r_[i+1], y[i], y[i+1], m0, m1);
        }
        int find_i(double r) const {
                for (int i = 1; i < lines_; i++) {
                        if (r_[i] < r) {
                                return i;
                        }
                }
                return 0;
        }
        double cubic_hermit_interp(double x, double x0, double x1, double p0, double p1, double m0, double m1) const {
                // interpolating by an Hermite cubic spline scheme
                auto const delta = x1 - x0;
                auto const t = (x - x0) / delta;
                auto h00 = [] (double t) { return  2 * t * t * t - 3 * t * t + 1; }; // Hermite basis functions
                auto h10 = [] (double t) { return  t * t * t - 2 * t * t + t; };
                auto h01 = [] (double t) { return  -2 * t * t * t + 3 * t * t; };
                auto h11 = [] (double t) { return t * t * t - t * t; };
                return p0 * h00(t) + delta * m0 * h10(t) + p1 * h01(t) + delta * m1 * h11(t);
        }
public:
        mesa_profiles(const std::string& filename) {
                char line[BUFFER_SIZE];
                char dummy[BUFFER_SIZE];
                char log10_P[BUFFER_SIZE];
                char log10_R[BUFFER_SIZE];
                char log10_rho[BUFFER_SIZE];
                char vrot_kms[BUFFER_SIZE];
                double file_to_g = 1.0;
                double file_to_cm = 1.0;
                double file_to_s = 1.0;
                FILE* fp = fopen(filename.c_str(), "rt");
                if (fp == NULL) {
                        printf("%s not found!\n", filename.c_str());
                        abort();
                }
                int linenum = 1;
                while (fgets(line, sizeof line, fp) != NULL) {
                        if (linenum > HEADER_LINES) {
                                std::sscanf(line,
                                                "%s %s %s %s %s %s %[^\]",
                                        dummy, dummy, log10_rho, log10_P, log10_R,
                                        vrot_kms, dummy);
                                P_.push_back(std::pow(10, std::atof(log10_P)) * std::pow(opts().code_to_s/file_to_g, 2) * (opts().code_to_cm / file_to_cm) / (opts().code_to_g / file_to_g));
                                double tmp = std::pow(10, std::atof(log10_R)) / (opts().code_to_cm / file_to_g);
                                if (hpx::get_locality_id() == 0) {
                                        printf("rho(%e) = %e\n", tmp, std::pow(10, std::atof(log10_rho)));
                                }
                                r_.push_back(tmp);
                                rho_.push_back(std::pow(10, std::atof(log10_rho)) * std::pow(opts().code_to_cm / file_to_cm, 3) / (opts().code_to_g / file_to_g));
                                omega_.push_back(std::atof(vrot_kms) * 100 * 1000 * (opts().code_to_s / file_to_s) / (opts().code_to_cm / file_to_cm) / tmp);
                        }
                        linenum++;
                }
                lines_ = linenum - HEADER_LINES;
		r_.resize(lines_);
                fclose(fp);
		if (hpx::get_locality_id() == 0) {
			printf("closing file and exiting read, r_ size: %i, read %i lines", r_.size(), linenum);
		}
        }

        mesa_profiles() = default;
        ~mesa_profiles() = default;

        double get_R0() {
                return r_[0];
	}
        void state_at(double& rho, double& p, double& omega, double r) const {
                rho = interpolate_mono(rho_, r);
                p = interpolate_mono(P_, r);
                omega = interpolate_mono(omega_, r);
        }
};

#endif /* MESA_STAR_HPP_ */
