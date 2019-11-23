/*
 * recon.cpp
 *
 *  Created on: Jul 25, 2019
 *      Author: dmarce1
 */
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <map>
#include <vector>
#include <math.h>
#include <iostream>
#include <boost/program_options.hpp>

auto band_filter(double t, double Ps, double Pc) {
	const auto x = 2.0 * M_PI * t / Pc;
	const auto y = 2.0 * M_PI * (t / Ps + 0.5);
	const auto sinc = x == 0 ? 1.0 : sin(x) / x;
	constexpr auto a0 = 7938.0 / 18608.0;
	constexpr auto a1 = 9240.0 / 18608.0;
	constexpr auto a2 = 1430.0 / 18608.0;
	const auto blackman = (a0 - a1 * cos(y) + a2 * cos(2 * y));
	if (blackman > 0.0) {
		return sinc * blackman;
	} else {
		return 0.0;
	}
}

auto derivative(const std::vector<std::vector<double>> &f, double omega) {
	std::vector<std::vector<double>> g;
	std::vector<double> u(f[0].size());
	const auto P = 2.0 * M_PI / omega;
	for (int n = 1; n < f.size(); n++) {
		const auto dt = (f[n][0] - f[n - 1][0]);
		for (int i = 0; i < u.size(); i++) {
			u[i] = P * (f[n][i] - f[n - 1][i]) / dt / f[0][i];
		}
		u[0] = (f[n][0] + f[n - 1][0]) / 2.0;
		g.push_back(u);
	}

	return g;

}

auto filter(const std::vector<std::vector<double>> &f, double omega, double Pc, double Ps) {
	std::vector<std::vector<double>> g;
	std::vector<double> u(f[0].size());

	const auto tmin = f[0][0];
	const auto tmax = f[f.size() - 1][0];
	double rt = 0.0;
	const auto P = 2.0 * M_PI / omega;
	Pc *= P;
	Ps *= P;
	for (int n = 1; n < f.size() - 1; n++) {
		double t0 = f[n][0];
		double dt;
		double weight = 0.0;
		if (Ps / 2.0 + tmin < t0 && t0 < tmax - Ps / 2.0) {
			std::fill(u.begin(), u.end(), 0.0);
			for (int m = 1; m < f.size() - 1; m++) {
				double t = f[m][0];
				if (t0 - Ps / 2.0 < t && t < t0 + Ps / 2.0) {
					dt = (f[m + 1][0] - f[m - 1][0]) / 2.0;
					double y = band_filter(t - t0, Ps, Pc);
					for (int i = 0; i < u.size(); i++) {
						u[i] += f[m][i] * y * dt;
					}
					weight += y * dt;
				}
			}
			for (int i = 0; i < u.size(); i++) {
				u[i] /= weight;
			}
			u[0] = rt;
			g.push_back(u);
		}
		dt = (f[n + 1][0] - f[n - 1][0]) / 2.0;
		rt += dt / P;
	}
	return g;
}

struct options {
	double pmin;
	double pmax;
	std::string input;

	int read_options(int argc, char *argv[]) {
		namespace po = boost::program_options;

		po::options_description command_opts("options");

		command_opts.add_options() //
		("input", po::value<std::string>(&input)->default_value("binary.dat"), "input filename")           //
		("pmin", po::value<double>(&pmin)->default_value(1.25), "minimum period to allow through filter")           //
		("pmax", po::value<double>(&pmax)->default_value(4.75), "period where filter allows 100%")           //
				;
		boost::program_options::variables_map vm;
		po::store(po::parse_command_line(argc, argv, command_opts), vm);
		po::notify(vm);

		FILE *fp = fopen(input.c_str(), "rb");
		if (input == "") {
			std::cout << command_opts << "\n";
			return -1;
		} else if (fp == NULL) {
			printf("Unable to open %s\n", input.c_str());
			return -1;
		} else {
			fclose(fp);
		}

		return 0;
	}
};
int main(int argc, char *argv[]) {
	options opts;
	if (opts.read_options(argc, argv) == 0) {

		FILE *fp = fopen(opts.input.c_str(), "rt");
		static char buffer[100000];
		std::map<double, std::vector<double>> values;

		while (!feof(fp)) {
			fgets(buffer, 100000, fp);
			bool done = false;
			char *ptr = buffer;
			double t;
			int col = 0;
			std::vector<double> these_values;
			do {
				while (isspace(*ptr)) {
					if (*ptr == '\n') {
						done = true;
						break;
					}
					ptr++;
				}
				if (!done) {
					double number = atof(ptr);
					if (col == 0) {
						t = number;
					}
					these_values.push_back(number);
					col++;
				}
				while (!isspace(*ptr)) {
					ptr++;
				}
			} while (!done);
			if (values.find(t) == values.end()) {
				values.insert(std::make_pair(t, std::move(these_values)));
			}
		}

		std::vector<std::vector<double>> v1;
		for (auto &v : values) {
			v1.push_back(std::move(v.second));
		}

		auto Pc = 2.0 * opts.pmax * opts.pmin / (opts.pmax + opts.pmin);
		auto Ps = 4.0 * opts.pmax * opts.pmin / (opts.pmax - opts.pmin);

		printf("Window is +/- %e orbits\n", Ps / 2.0);

		const auto v2 = filter(v1, v1[0][2], Pc, Ps);
		if (v2.size() == 0) {
			printf("Not enough data to produce output\n");
		} else {
			const auto v4 = derivative(v1, v1[0][2]);
			const auto v3 = filter(v4, v1[0][2], Pc, Ps);
			FILE *fp1 = fopen("avg.dat", "wt");
			FILE *fp2 = fopen("drv.dat", "wt");
			for (const auto &i : v2) {
				for (const auto &j : i) {
					fprintf(fp1, " %e ", j);
				}
				fprintf(fp1, "\n");
			}
			for (const auto &i : v3) {
				for (const auto &j : i) {
					fprintf(fp2, " %e ", j);
				}
				fprintf(fp2, "\n");
			}
			fclose(fp1);
			fclose(fp2);
			fclose(fp);
		}
	}
	return 0;
}

