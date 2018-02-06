/*
 * silo_read.cpp
 *
 *  Created on: Sep 25, 2017
 *      Author: dmarce1
 */

#include <silo.h>
#include <cmath>
#include <array>
#include <stdlib.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <valarray>

char const* field_names[] = { "rho", "egas", "sx", "sy", "sz", "tau", "pot", "zx", "zy", "zz", "primary_core",
		"primary_envelope", "secondary_core", "secondary_envelope", "vacuum", "phi", "gx", "gy", "gz", "vx", "vy", "vz", "eint",
		"zzs", "roche" };

constexpr auto rho_i = 0;
constexpr auto egas_i = 1;
constexpr auto sx_i = 2;
constexpr auto sy_i = 3;
constexpr auto sz_i = 4;
constexpr auto tau_i = 5;
constexpr auto pot_i = 6;
constexpr auto zx_i = 7;
constexpr auto zy_i = 8;
constexpr auto zz_i = 9;
constexpr auto primary_core_i = 10;
constexpr auto primary_envelope_i = 11;
constexpr auto secondary_core_i = 12;
constexpr auto secondary_envelope_i = 13;
constexpr auto vacuum_i = 14;
constexpr auto phi_i = 15;
constexpr auto gx_i = 16;
constexpr auto gy_i = 17;
constexpr auto gz_i = 18;
constexpr auto vx_i = 19;
constexpr auto vy_i = 20;
constexpr auto vz_i = 21;
constexpr auto eint_i = 22;
constexpr auto zzs_i = 23;
constexpr auto roche_i = 24;

constexpr int NF = 25;
constexpr int NVERTICES = 8;

struct statistics {
	double relL1, relL2, relLinf;
	double absL1, absL2, absLinf;
	statistics() = default;
	statistics(statistics& other) = default;
	statistics(const std::vector<double>& d1, const std::vector<double>& d2) {
		relL1 = 0.0;
		relL2 = 0.0;
		relLinf = 0.0;
		absL1 = 0.0;
		absL2 = 0.0;
		absLinf = 0.0;
		for (int i = 0; i < d1.size(); i++) {
			const double abs_dif = std::abs(d1[i] - d2[i]);
			const double avg = (0.5 * (std::abs(d1[i]) + std::abs(d2[i])));
			const double rel_dif = abs_dif / avg;
			absL1 += abs_dif;
			absL2 += abs_dif * abs_dif;
			absLinf = std::max(absLinf, abs_dif);
			if (avg != 0.0) {
				relL1 += rel_dif;
				relL2 += rel_dif * rel_dif;
				relLinf = std::max(relLinf, rel_dif);
			}
		}
		relL1 /= d1.size();
		relL2 /= d2.size();
		absL1 /= d1.size();
		absL2 /= d2.size();
		relL2 = std::sqrt(relL2);
		absL2 = std::sqrt(absL2);
	}
	void print(FILE* fp) {
		fprintf(fp, "%13e %13e %13e %13e %13e %13e\n", relL1, relL2, relLinf, absL1, absL2, absLinf);
	}
};

struct amr_data {
	std::vector<double> x, y, z, dx;
	std::vector<double> vars[NF];
	void compare(const amr_data& other) {
		const auto count1 = x.size();
		const auto count2 = other.x.size();
		if (count1 != count2) {
			printf("Files are of unequal size\n");
			printf("%i and %i\n", int(count1), int(count2));
			exit(1);
		}
		statistics x_stat(x, other.x);
		statistics y_stat(y, other.y);
		statistics z_stat(z, other.z);
		statistics dx_stat(dx, other.dx);
		statistics var_stat[NF];
		for (int f = 0; f != NF; ++f) {
			var_stat[f] = statistics(vars[f], other.vars[f]);
		}
		printf("File differences: (relative average, rms, and max, absolute average, rms, and max errors\n");
		printf("X coordinates:     ");
		x_stat.print(stdout);
		printf("Y coordinates:     ");
		y_stat.print(stdout);
		printf("Z coordinates:     ");
		z_stat.print(stdout);
		printf("dx:                ");
		dx_stat.print(stdout);
		for (int f = 0; f != NF; ++f) {
			printf("%18s ", field_names[f]);
			var_stat[f].print(stdout);
		}

	}

	void read_mesh(const char* filename) {
		auto* db = DBOpen(filename, DB_PDB, DB_READ);
		auto* mesh = DBGetUcdmesh(db, "mesh");
		const auto* zonelist = mesh->zones;
		auto* x_vals = reinterpret_cast<double*>(mesh->coords[0]);
		auto* y_vals = reinterpret_cast<double*>(mesh->coords[1]);
		auto* z_vals = reinterpret_cast<double*>(mesh->coords[2]);
		const auto count = zonelist->nzones;
		x.resize(count);
		y.resize(count);
		z.resize(count);
		dx.resize(count);
//		printf("Reading mesh\n");
		for (int i = 0; i < count; ++i) {
			x[i] = 0.0;
			y[i] = 0.0;
			z[i] = 0.0;
			double minx = +std::numeric_limits<double>::max();
			double maxx = -std::numeric_limits<double>::max();
			for (int j = 0; j != 8; ++j) {
				auto k = zonelist->nodelist[NVERTICES * i + j];
				minx = std::min(minx, x_vals[k]);
				maxx = std::max(maxx, x_vals[k]);
				x[i] += x_vals[k];
				y[i] += y_vals[k];
				z[i] += z_vals[k];
			}
			dx[i] = maxx - minx;
			x[i] /= double(NVERTICES);
			y[i] /= double(NVERTICES);
			z[i] /= double(NVERTICES);
		}
//		printf("Reading data\n");
		for (int field = 0; field != NF; ++field) {
//			printf("\rField #%i name: %s\n", field, field_names[field]);
			auto* ucd = DBGetUcdvar(db, field_names[field]);
			vars[field].resize(count);
			const double* array = reinterpret_cast<double*>(ucd->vals[0]);
			for (int i = 0; i < count; ++i) {
				vars[field][i] = array[i];
			}
			DBFreeUcdvar(ucd);
		}
		DBClose(db);
		DBFreeUcdmesh(mesh);
	}
};

int main(int argc, char* argv[]) {

	if (argc != 3) {
		printf("Usage:\n");
		printf("silo_cmp <filename1> <filename2> \n");
		abort();
	}

	const char* filename1 = argv[1];
	const char* filename2 = argv[2];
	amr_data mesh1, mesh2;
	FILE* fp1 = fopen(filename1, "rb");
	FILE* fp2 = fopen(filename2, "rb");
	if (fp1 == NULL) {
		printf("Could not open %s for reading.\n", filename1);
	}
	if (fp2 == NULL) {
		printf("Could not open %s for reading.\n", filename2);
	}
	mesh1.read_mesh(filename1);
	mesh2.read_mesh(filename2);
	mesh1.compare(mesh2);
	return 0;
}
