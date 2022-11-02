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
#include <complex>
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

void output_spectrogram(double Ps, double Pc) {
	double pmin = Pc / 100.0;
	double pmax = 2.0 * Ps;
	double dp = (pmax - pmin) / 1000.0;
	const auto dt = Ps / 100000.0;
	FILE *fp = fopen("filter.dat", "wt");
	for (auto p = pmin; p <= pmax; p += dp) {
		double w = 0.0;
		const auto omega = 2.0 * M_PI / p;
		double sum = 0.0;
		constexpr std::complex<double> I(0, 1);
		for (double t = dt / 2.0; t < Ps / 2.0; t += dt) {
			const auto bf = band_filter(t, Ps, Pc);
			w += 2.0 * bf * dt;
			sum += 2.0 * bf * cos(omega * t) * dt;
		}
		sum /= w;
		fprintf(fp, "%.12e %.12e\n", p, sum);
	}
	fclose(fp);

}

auto derivative(const std::vector<std::vector<double>> &f, double omega) {
	std::vector<std::vector<double>> g;
	std::vector<double> u(f[0].size());
	const auto P = 2.0 * M_PI / omega;
	for (int n = 1; n < f.size() - 1; n++) {
		const auto nm = n - 1;
		const auto np = n + 1;
		const auto h1 = (f[n][0] - f[nm][0]);
		const auto h2 = (f[np][0] - f[n][0]);
		for (int i = 0; i < u.size(); i++) {
			u[i] = P * (f[np][i] * h1 * h1 + f[n][i] * (h2 * h2 - h1 * h1) - f[nm][i] * h2 * h2) / (f[0][i] * h1 * h2 * (h1 + h2));
		}
		u[0] = f[n][0];
		g.push_back(u);
	}

	return g;

}

auto filter(const std::vector<std::vector<double>> &f, double omega, double Pc, double Ps) {
	std::vector<std::vector<double>> g;
	std::vector<double> u(f[0].size());

	const auto tmin = f[0][0];
	const auto tmax = f[f.size() - 1][0];
	const auto P = 2.0 * M_PI / omega;
	double rt = tmin / P;
	Pc *= P;
	Ps *= P;
	for (int n = 1; n < f.size() - 1; n++) {
		double t0 = f[n][0];
		double dt;
		double weight = 0.0;

		if (Pc / 2.0 + tmin < t0 && t0 < tmax - Pc / 2.0) {
			std::fill(u.begin(), u.end(), 0.0);
			for (int m = 1; m < f.size() - 1; m++) {
				double t = f[m][0];
				if (t0 - Pc / 2.0 < t && t < t0 + Pc / 2.0) {
					dt = (f[m + 1][0] - f[m - 1][0]) / 2.0;
					const auto h1 = f[m][0] - f[m - 1][0];
					const auto h2 = f[m + 1][0] - f[m][0];
					const auto w1 = (h1 + h2) * (2 * h1 - h2) / h1 / 6.0;
					const auto w2 = std::pow(h1 + h2, 3) / h2 / h1 / 6.0;
					const auto w3 = (h1 + h2) * (2 * h2 - h1) / h2 / 6.0;
					double y1 = band_filter(f[m - 1][0] - t0, Ps, Pc);
					double y2 = band_filter(f[m][0] - t0, Ps, Pc);
					double y3 = band_filter(f[m + 1][0] - t0, Ps, Pc);
					for (int i = 0; i < u.size(); i++) {
						u[i] += w1 * f[m - 1][i] * y1 * dt;
						u[i] += w2 * f[m][i] * y2 * dt;
						u[i] += w3 * f[m + 1][i] * y3 * dt;
					}
					weight += w1 * y1 * dt;
					weight += w2 * y2 * dt;
					weight += w3 * y3 * dt;
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
	bool help;
	bool normalize;
	std::string input;

	int read_options(int argc, char *argv[]) {
		namespace po = boost::program_options;


		po::options_description command_opts("options");

		command_opts.add_options() //
		("input", po::value<std::string>(&input)->default_value("binary.dat"), "input filename")           //
		("pmin", po::value<double>(&pmin)->default_value(1.25), "minimum period to allow through filter")           //
		("pmax", po::value<double>(&pmax)->default_value(5), "period where filter allows 100%")           //
		("normalize", po::value<bool>(&normalize)->default_value(true), "normalize averages to t=0 value")           //
		("help", po::value<bool>(&help)->default_value(false), "show the help page")           //
				;
		boost::program_options::variables_map vm;
		po::store(po::parse_command_line(argc, argv, command_opts), vm);
		po::notify(vm);

		FILE *fp = fopen(input.c_str(), "rb");
		if (help) {
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
			const char* b = fgets(buffer, 100000, fp);
			if( b == 0) {
				printf( "UNable to read file\n");
				abort();
			}
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
		printf("sinc period      = %e\n", Pc);
		printf("Blackmann period = %e\n", Ps);

		printf("Window is +/- %.12e orbits\n", Pc / 2.0);

		auto v2 = filter(v1, v1[0][2], Pc, Ps);
		if (v2.size() == 0) {
			printf("Not enough data to produce output\n");
		} else {
			if (opts.normalize) {
				for (auto &line : v2) {
					for (int i = 1; i < line.size(); i++) {
						line[i] /= v1[0][i];
					}
				}
			}
			const auto v4 = derivative(v1, v1[0][2]);
			const auto v3 = filter(v4, v1[0][2], Pc, Ps);
			FILE *fp1 = fopen("avg.dat", "wt");
			FILE *fp2 = fopen("drv.dat", "wt");
			for (const auto &i : v2) {
				for (const auto &j : i) {
					fprintf(fp1, " %.12e ", j);
				}
				fprintf(fp1, "\n");
			}
			for (const auto &i : v3) {
				for (const auto &j : i) {
					fprintf(fp2, " %.12e ", j);
				}
				fprintf(fp2, "\n");
			}
			fclose(fp1);
			fclose(fp2);
			fclose(fp);
		}

		output_spectrogram(Ps, Pc);
	}
	return 0;
}

