//  Copyright (c) 2019
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mesa.hpp"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <array>
#include <vector>
#include <memory>

#define BUFFER_SIZE (1024*1024)
#define HEADER_LINES 6
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

int main() {

}
