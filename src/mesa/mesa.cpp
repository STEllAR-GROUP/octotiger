//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/mesa/mesa.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"
#include "octotiger/grid.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <array>
#include <vector>
#include <memory>

#define BUFFER_SIZE (1024 * 16)
#define HEADER_LINES 14
#define NCOEF 4

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
			if (x0 <= x[i + 1]) {
				i = std::max(i, 1);
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
					y0 += A[i][n] * std::pow(x0, n - 1) * b[n];
				}
				return y0;
			}
		}
		printf("x0 not found\n");
		abort();
		return 1.0 / 0.0;
	}
};

std::function<double(double)> build_rho_of_h_from_mesa(
		const std::string& filename) {
	char line[BUFFER_SIZE];
	char dummy[BUFFER_SIZE];
	char log10_P[BUFFER_SIZE];
	char log10_R[BUFFER_SIZE];
	char log10_rho[BUFFER_SIZE];
	char vrot_kms[BUFFER_SIZE];
	FILE* fp = fopen(filename.c_str(), "rt");
	if (fp == NULL) {
		printf("%s not found!\n", filename.c_str());
		abort();
	}
	std::vector<double> P, rho, h;
	int linenum = 1;
	while (fgets(line, sizeof line, fp) != NULL) {
		if (linenum > HEADER_LINES) {
			std::sscanf(line,
					"%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
					dummy, dummy, log10_R, dummy, log10_rho, log10_P, dummy,
					dummy, dummy, dummy, dummy, dummy, vrot_kms, dummy, dummy,
					dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
			P.push_back(std::pow(10, std::atof(log10_P)));
//			const double tmp = std::pow(10, std::atof(log10_R)) * 6.957e+10;
//			r.push_back(tmp);
			rho.push_back(std::pow(10, std::atof(log10_rho)));
//			omega.push_back(std::atof(vrot_kms) * 100 * 1000 / tmp);
		}
		linenum++;
	}
	fclose(fp);

	h.resize(rho.size());
	double rho_max = rho[rho.size() - 1];
	for (auto& x : rho) {
		x /= rho_max;
	};
	cubic_table p_of_rho(P, rho);
	h[0] = 0.0;
	for (std::size_t i = 0; i < rho.size() - 1; i++) {
		const double drho = rho[i + 1] - rho[i];
		const double rho0 = rho[i];
		const double rho1 = 0.5 * (rho[i] + rho[i + 1]);
		const double rho2 = rho[i + 1];
		const double dp_drho0 = p_of_rho.derivative(rho0);
		const double dp_drho1 = p_of_rho.derivative(rho1);
		const double dp_drho2 = p_of_rho.derivative(rho2);
		h[i + 1] = h[i];
		if (rho0 != 0.0) {
			h[i + 1] += (1.0 / 6.0) * dp_drho0 / rho0 * drho;
		}
		h[i + 1] += (4.0 / 6.0) * dp_drho1 / rho1 * drho;
		h[i + 1] += (1.0 / 6.0) * dp_drho2 / rho2 * drho;
	}
	double h_max = h[rho.size() - 1];
	for (auto& x : h) {
		x /= h_max;
	};
	const auto rho_of_h_table = std::make_shared<cubic_table>(rho, h);

	return [rho_of_h_table](double h) {
		return (*rho_of_h_table)(h);
	};
}

class mesa_profiles {
private:
        std::vector<double> rho_;
        std::vector<double> P_;
	std::vector<double> r_;
	std::vector<double> omega_;
        int lines_;
public:
	mesa_profiles(
                const std::string& filename) { //, std::vector<double>& P, std::vector<double>& rho, std::vector<double>& r, std::vector<double>& omega) {
//		) {
	// reads the profiles from the mesa file at initialization
		printf("in profiles\n");
		char line[BUFFER_SIZE];
		char dummy[BUFFER_SIZE];
		char log10_P[BUFFER_SIZE];
		char log10_R[BUFFER_SIZE];
		char log10_rho[BUFFER_SIZE];
		char vrot_kms[BUFFER_SIZE];
		FILE* fp = fopen(filename.c_str(), "rt");
//		std::string filename = "mesa_star.data"; 
//		FILE* fp = fopen("mesa_star.data", "rt");
		if (fp == NULL) {
			printf("%s not found!\n", filename.c_str());
			abort();
		}
		//std::vector<double> P, rho, h, r, omega;
		int linenum = 1;
		while (fgets(line, sizeof line, fp) != NULL) {
//			printf("%i: %s, %i\n\n\n\n", linenum, line, sizeof(line));  
			//std::string substr = line.substr(0,100);
			//printf("%i: %s\n\n\n\n", linenum, substr);
			if (linenum > HEADER_LINES) {
				std::sscanf(line,
//						"%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %.1000s \n",
//                                        dummy, dummy, dummy, log10_rho, dummy, 
//					dummy, log10_P, log10_R, dummy, dummy, 
//					vrot_kms, dummy, dummy, dummy, dummy, 
                                                "%s %s %s %s %s %s %[^\]",
                                        dummy, dummy, log10_rho, log10_P, log10_R,
                                   //     dummy, dummy, dummy, dummy, dummy,
                                   //     dummy, dummy, dummy, dummy, dummy,
				//	dummy, dummy, dummy, 
					vrot_kms, dummy);
//				printf("after read: %s, %s, %s\n\n\n", log10_rho, log10_P, log10_R);
				P_.push_back(std::pow(10, std::atof(log10_P)) * std::pow(opts().code_to_s, 2) * opts().code_to_cm / opts().code_to_g);
				double tmp = std::pow(10, std::atof(log10_R)) * 6.957e+10 / opts().code_to_cm;
				printf("rho(%e) = %e\n", tmp, std::pow(10, std::atof(log10_rho)));
				r_.push_back(tmp);
				rho_.push_back(std::pow(10, std::atof(log10_rho)) * std::pow(opts().code_to_cm, 3) / opts().code_to_g);
				omega_.push_back(std::atof(vrot_kms) * 100 * 1000 * opts().code_to_s / opts().code_to_cm / tmp);
			}
			linenum++;
		}
		lines_ = linenum - HEADER_LINES;
		printf("exiting read %i\n", linenum);
		fclose(fp);
	}
	double get_R0() {
//		printf("lines: %d, size: %d, r0: %e, r(lines-1): %e\n", lines_, sizeof(r_), r_[20], r_[lines_-20]); 
		return r_[0];
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
        void state_at(double& rho, double& p, double& omega, double r) const {
                rho = interpolate_mono(rho_, r);
                p = interpolate_mono(P_, r);
		omega = interpolate_mono(omega_, r);
        }
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
};

std::vector<real> mesa_star(real x, real y, real z, real dx) {
        std::vector<real> u(opts().n_fields, real(0));
        const real fgamma = grid::get_fgamma();
        static mesa_profiles mesa_p(opts().mesa_filename);
        real R0 = mesa_p.get_R0();
	//printf(" R0: %e, f: %e\n", mesa_p.get_R0(), opts().code_to_cm);
        real rho = 0.0;
	real p = 0.0;
        const int M = 1;
        int nsamp = 0;
        for (double x0 = x - dx / 2.0 + dx / 2.0 / M; x0 < x + dx / 2.0; x0 += dx / M) {
                for (double y0 = y - dx / 2.0 + dx / 2.0 / M; y0 < y + dx / 2.0; y0 += dx / M) {
                        for (double z0 = z - dx / 2.0 + dx / 2.0 / M; z0 < z + dx / 2.0; z0 += dx / M) {
                                auto r = SQRT(x0 * x0 + y0 * y0 + z0 * z0);
                                ++nsamp;
                                if (r <= R0) {
					real rho_t, p_t, omega_t;
					mesa_p.state_at(rho_t, p_t, omega_t, r);
                       //                 printf("rho(%e) = %e, p = %e, omega = %e, (x,y,z) = (%e,%e,%e)\n", r, rho_t, p_t, omega_t, x, y, z);
                                        rho += rho_t;
                                        p += p_t;
					auto sx_t = -y0 * rho_t * omega_t;
					auto sy_t = x0 * rho_t * omega_t;
                                }
                        }
                }
        }
//      grid::set_AB(this_struct_eos->A, this_struct_eos->B());
//	printf("rho floor: %e, egas floor: %e\n", opts().rho_floor, std::pow(opts().tau_floor, fgamma));
        rho = std::max(rho / nsamp, opts().rho_floor);
        auto ene = std::max(p / (fgamma - 1.0) / nsamp, std::pow(opts().tau_floor, fgamma));
        u[rho_i] = rho;
        u[spc_i] = rho; //u[spc_i + 1] = u[spc_i + 2] = u[spc_i + 3] = u[spc_i + 4] = rho / 5.0;
        u[egas_i] = ene;
	u[tau_i] = std::pow(ene, 1.0 / fgamma);
        return u;
}

bool mesa_refine_test(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U,
                std::array<std::vector<real>, NDIM> const& dudx) {
        bool rc = false;
        real dx = (opts().xscale / INX) / real(1 << level);
        if (level < max_level / 2) {
                return std::sqrt(x * x + y * y + z * z) < 10.0 * dx;
        }
        int test_level = max_level;
        //bool enuf_core = U[spc_ac_i] + U[spc_dc_i] > 0.25 * U[rho_i];
        //bool majority_accretor = U[spc_ae_i] + U[spc_ac_i] > 0.5 * U[rho_i];
        //bool majority_donor = U[spc_de_i] + U[spc_dc_i] > 0.5 * U[rho_i];
        //if (opts().core_refine) {
        //        if (!enuf_core) {
        //                test_level -= 1;
        //        }
        //}
        //if (!majority_donor) {
        //        test_level -= opts().donor_refine;
        //}
        //if (!majority_accretor) {
        //        test_level -= opts().accretor_refine;
        //}
        real den_floor = opts().refinement_floor;
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

//int main() {
//
//}
