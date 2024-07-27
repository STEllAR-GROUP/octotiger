//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/mesa/mesa.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"
//#include "octotiger/lane_emden.hpp"
//#include "octotiger/grid.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <array>
#include <vector>
#include <memory>

//#define BUFFER_SIZE (1024 * 16)
//#define HEADER_LINES 14
//#define NCOEF 4

class cubic_table {
private:
	const int N;
	const std::vector<double> x;
	std::vector<std::array<double, NCOEF>> A;

	std::array<std::array<double, NCOEF>, NCOEF> coef_matrix(double a,
			double b) {
		return { {
				{	-(1 + 2*b)/(8.*(a + std::pow(a,2))*(1 + a + b)),(1 + 2*a + 2*b + 4*a*b)/(8*a + 8*a*b),(1 + 2*a + 2*b + 4*a*b)/(8*b + 8*a*b),-(1 + 2*a)/(8.*(1 + a + b)*(b + std::pow(b,2)))},
				{	1/(4.*(a + std::pow(a,2))*(1 + a + b)),-1 - 1/(4*a + 4*a*b),1 + 1/(4*b + 4*a*b),-1/(4.*(1 + a + b)*(b + std::pow(b,2)))},
				{	(1 + 2*b)/(2.*(a + std::pow(a,2))*(1 + a + b)),-((1 - 2*a + 2*b)/(2*a + 2*a*b)),-((1 + 2*a - 2*b)/(2*b + 2*a*b)),(1 + 2*a)/(2.*(1 + a + b)*(b + std::pow(b,2)))},
				{	-(1/((a + std::pow(a,2))*(1 + a + b))),1/(a + a*b),-(1/(b + a*b)),1/((1 + a + b)*(b + std::pow(b,2)))}
			}};
	}

	std::array<double, NCOEF> translate(const std::array<double, NCOEF>& in,
			double dx) {
		std::array<double, NCOEF> out;
		out[0] = out[1] = out[2] = out[3] = 0.0;

		out[3] += 1.0 * in[3];
		out[2] += 3.0 * in[3] * dx;
		out[1] += 3.0 * in[3] * dx * dx;
		out[0] += 1.0 * in[3] * dx * dx * dx;

		out[2] += 1.0 * in[2];
		out[1] += 2.0 * in[2] * dx;
		out[0] += 1.0 * in[2] * dx * dx;

		out[1] += 1.0 * in[1];
		out[0] += 1.0 * in[1] * dx;

		out[0] += 1.0 * in[0];

		return out;
	}
public:
	cubic_table(const std::vector<double>& y, const std::vector<double>& _x) :
			N(y.size()), x(_x), A(y.size()) {
		for (int i = 1; i < N - 2; i++) {
			const double dx0 = (x[i] - x[i - 1]);
			const double dx1 = (x[i + 1] - x[i]);
			const double dx2 = (x[i + 2] - x[i + 1]);
			const auto C = coef_matrix(dx0 / dx1, dx2 / dx1);

			for (int n = 0; n < NCOEF; n++) {
				A[i][n] = 0.0;
				for (int m = 0; m < NCOEF; m++) {
					A[i][n] += C[n][m] * y[i + m - 1];
				}
			}

		}
		const double dx0 = (x[1] - x[0]) / (x[2] - x[1]);
		const double dxN = (x[N - 1] - x[N - 2]) / (x[N - 2] - x[N - 3]);
		A[0] = translate(A[1], -dx0);
		A[N - 1] = translate(A[N - 3], +2.0 * dxN);
		A[N - 2] = translate(A[N - 3], +dxN);
	}
	double operator()(double x0) const {
		for (int i = 0; i < N - 1; i++) {
			if (x0 < x[i + 1]) {
				//i = std::max(i, 1);
				i = std::min(i, N - 3);
				x0 = (x0 - 0.5 * (x[i + 1] + x[i])) / (x[i + 1] - x[i]);
				double y0 = 0.0;
				for (int n = 0; n < NCOEF; n++) {
					y0 += A[i][n] * std::pow(x0, n);
				}
				return y0;
			}
		}
		printf("x0 not found\n");
		abort();
		return 1.0 / 0.0;
	}

	double derivative(double x0) const {
		constexpr double b[4] = { 0.0, 1.0, 2.0, 3.0 };
		for (int i = 0; i < N - 1; i++) {
			if (x0 <= x[i + 1]) {
				i = std::max(i, 1);
				i = std::min(i, N - 3);
				x0 = (x0 - 0.5 * (x[i + 1] + x[i])) / (x[i + 1] - x[i]);
				double y0 = 0.0;
				for (int n = 1; n < NCOEF; n++) {
					y0 += A[i][n] * std::pow(x0, n - 1) * b[n] / (x[i + 1] - x[i]);
				}
				return y0;
			}
		}
		printf("x0 not found\n");
		abort();
		return 1.0 / 0.0;
	}
};

std::pair< std::function<double(double)>, std::function<double(double)> > build_rho_of_h_from_mesa(
		const std::string& filename, real &HE_prof, real& dE_prof, real& PE_prof, real& rE_prof, real& ME_prof, std::vector<double>& P, std::vector<double>& rho, std::vector<double>& h) {
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
	P.clear();
	rho.clear();
	h.clear();
	std::vector<double> r;
	int linenum = 1;
	while (fgets(line, sizeof line, fp) != NULL) {
		if (linenum > HEADER_LINES) {
			std::sscanf(line,
                                                "%s %s %s %s %s %s %[^\]",
                                        dummy, dummy, log10_rho, log10_P, log10_R,
                                        vrot_kms, dummy);
				real const cur_rho = std::pow(10, std::atof(log10_rho)) * std::pow(opts().code_to_cm / file_to_cm, 3) / (opts().code_to_g / file_to_g);
				if (cur_rho <= dE_prof) {
	                                P.push_back(std::pow(10, std::atof(log10_P)) * std::pow(opts().code_to_s/file_to_g, 2) * (opts().code_to_cm / file_to_cm) / (opts().code_to_g / file_to_g));
        	                        rho.push_back(cur_rho);
					r.push_back(std::pow(10, std::atof(log10_R)) / (opts().code_to_cm / file_to_g));
				}
		}
		linenum++;
	}
	fclose(fp);
	h.resize(rho.size());
	double rho_max = rho[rho.size() - 1];
	double P_max = P[rho.size() - 1];
	double r_min = r[rho.size() - 1];
	cubic_table p_of_rho(P, rho);
	h[0] = 0.0;
	ME_prof = 0.0;
	for (std::size_t i = 0; i < rho.size() - 1; i++) {
		const double drho = rho[i + 1] - rho[i];
		const double rho0 = rho[i];
		const double rho1 = 0.5 * (rho[i] + rho[i + 1]);
		const double rho2 = rho[i + 1];
		if (drho < 0.0) {
			printf("negative drho: %e, at %i, %e, %e\n", drho, i, rho2, rho0);
			printf("please make sure the given profile is a monotonic function of rho\n");
			abort();
		}
		const double dp_drho0 = p_of_rho.derivative(rho0);
		const double dp_drho1 = p_of_rho.derivative(rho1);
		const double dp_drho2 = p_of_rho.derivative(rho2);
		h[i + 1] = h[i];
		if (rho0 != 0.0) {
			h[i + 1] += (1.0 / 6.0) * dp_drho0 / rho0 * drho;
		}
		h[i + 1] += (4.0 / 6.0) * dp_drho1 / rho1 * drho;
		h[i + 1] += (1.0 / 6.0) * dp_drho2 / rho2 * drho;
                if (hpx::get_locality_id() == 0) {
			printf("rho[%i] = %e, P[%i] = %e, h[%i] = %e\n",i, rho0, i, P[i], i, h[i]);
		}
		ME_prof += real(4) * real(M_PI) * rho0 * r[i] * r[i] * (r[i] - r[i+1]);
	}
	const double h_max = h[rho.size() - 1];
	HE_prof = h_max;
	dE_prof = rho_max;
	PE_prof = P_max;
	rE_prof = r_min;
	
	const auto rho_of_h_table = std::make_shared<cubic_table>(rho, h);	

	const auto P_of_rho_table = std::make_shared<cubic_table>(P, rho);

	auto f1 = [rho_of_h_table](double h) {
		return (*rho_of_h_table)(h);
	};

	auto f2 = [P_of_rho_table](double rho) {
                return (*P_of_rho_table)(rho);
        };

        if (hpx::get_locality_id() == 0) {
                printf("TESTING!!\n==================\n");
		for (auto cur_h = h[1] / 10.0; cur_h < h[1]; cur_h += h[1] / 10.0) {
			printf("rho_from_h[h=%e] = %e\n", cur_h, f1(cur_h));
		}
                for (std::size_t i = 0; i < rho.size() - 1; i++) {
                        printf("h[%i] = %e, rho_from_h[%i] = %e, rho[%i] = %e\n", i, h[i], i, f1(h[i]), i, rho[i]);
                }
        }

	std::pair< std::function<double(double)>, std::function<double(double)> > fs;
	fs.first = f1;
	fs.second = f2;
	return fs;
}

std::pair< std::function<double(double)>, std::function<double(double)> > build_rho_of_h_from_relation(std::vector<double> const P, std::vector<double> const rho, std::vector<double> const h) {
        const auto rho_of_h_table = std::make_shared<cubic_table>(rho, h);

        const auto P_of_rho_table = std::make_shared<cubic_table>(P, rho);

        auto f1 = [rho_of_h_table](double h) {
                return (*rho_of_h_table)(h);
        };
        auto f2 = [P_of_rho_table](double rho) {
                return (*P_of_rho_table)(rho);
        };

        std::pair< std::function<double(double)>, std::function<double(double)> > fs;
        fs.first = f1;
        fs.second = f2;
        return fs;
}

std::vector<real> mesa_star(real x, real y, real z, real dx) {
        std::vector<real> u(opts().n_fields, real(0));
        const real fgamma = grid::get_fgamma();
        mesa_profiles mesa_p(opts().mesa_filename);
	x = x - opts().star_xcenter;
	y = y - opts().star_ycenter;
	z = z - opts().star_zcenter;
        const real rcut = opts().star_rmax;
        const real alpha = opts().star_alpha;
        const real rho_avg = opts().star_rho_out;
        const real dr = opts().star_dr;
        const real rho_c = opts().star_rho_center;
        const real rho_f = rho_avg / rho_c;
        const real n = opts().star_n;
        const real R0 = mesa_p.get_R0();
        real rho = 0.0;
	real p = 0.0;
        const int M = opts().interp_points;
        int nsamp = 0;
        for (double x0 = x - dx / 2.0 + dx / 2.0 / M; x0 < x + dx / 2.0; x0 += dx / M) {
                for (double y0 = y - dx / 2.0 + dx / 2.0 / M; y0 < y + dx / 2.0; y0 += dx / M) {
                        for (double z0 = z - dx / 2.0 + dx / 2.0 / M; z0 < z + dx / 2.0; z0 += dx / M) {
                                auto r = SQRT(x0 * x0 + y0 * y0 + z0 * z0);
                                ++nsamp;
				if (r <= rcut) {
					real theta = lane_emden(r/alpha, dr/alpha, n, rho_f, opts().p_smooth_l/alpha);
					real rho_theta = rho_c * std::pow(theta, n);
					const auto c0 = real(4) * real(M_PI) * std::pow(alpha, 2) * std::pow(rho_c, (n - real(1))/n) / (n + real(1));
					rho += rho_theta;
					p += c0 * std::pow(rho_theta, (n + real(1))/n);
                                } else if (r <= R0) {
					real rho_t, p_t, omega_t;
					mesa_p.state_at(rho_t, p_t, omega_t, r);
                                        rho += rho_t;
                                        p += p_t;
					auto sx_t = -y0 * rho_t * omega_t;
					auto sy_t = x0 * rho_t * omega_t;
                                }
                        }
                }
        }
        rho = std::max(rho / nsamp, opts().rho_floor);
        auto ene = std::max(p / (fgamma - 1.0) / nsamp, std::pow(opts().tau_floor, fgamma));
        u[rho_i] = rho;
        u[spc_i] = rho;
        u[egas_i] = ene;
	u[tau_i] = std::pow(ene, 1.0 / fgamma);
        return u;
}

bool mesa_refine_test(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U,
                std::array<std::vector<real>, NDIM> const& dudx) {
        bool rc = false;
        real dx = (opts().xscale / INX) / real(1 << level);
        if (opts().min_level + level - 1 < max_level / 2) {
                return std::sqrt(x * x + y * y + z * z) < 10.0 * dx;
        }
        int test_level = max_level;
        bool majority_accretor = U[spc_i] + U[spc_i + 1] > 0.5 * U[rho_i];
        bool majority_donor = U[spc_i + 2] + U[spc_i + 3] > 0.5 * U[rho_i];
        real den_floor = opts().refinement_floor;
        if (!majority_donor) {
                test_level -= opts().donor_refine;
        } else if (opts().refinement_floor_donor > 0.0) {
                den_floor = opts().refinement_floor_donor;
        }
        if (!majority_accretor) {
                test_level -= opts().accretor_refine;
        }
        for (integer this_test_level = test_level; this_test_level >= 1; --this_test_level) {
                if (U[rho_i] > den_floor) {
                        rc = rc || (level < this_test_level);
                }
                if (rc) {
                        break;
                }
                den_floor /= 8.0;
        }
        return rc;
}

