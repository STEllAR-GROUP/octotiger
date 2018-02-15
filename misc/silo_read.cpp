/*
 * silo_read.cpp
 *
 *  Created on: Sep 25, 2017
 *      Author: dmarce1
 */


#include <silo.h>
#include <cmath>
#include <stdlib.h>
#include <limits>
#include <vector>

char const* field_names[] = {"rho", "egas", "sx", "sy", "sz", "tau", "pot", "zx", "zy", "zz", "primary_core", "primary_envelope", "secondary_core",
	"secondary_envelope", "vacuum", "phi", "gx", "gy", "gz", "vx", "vy", "vz", "eint", "zzs"};


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


constexpr int NF = 24;
constexpr int NVERTICES = 8;

void process_data(const std::vector<double> vars[NF],
		const std::vector<double>& x, const std::vector<double>& y,
		const std::vector<double>& z, const std::vector<double>& dx, int count) {

	double m = 0.0;
	for( int i = 0.0; i != count; ++i) {
		const auto dv = std::pow(dx[i],3);
		const auto rho = vars[rho_i][i];
		m += rho * dv;
	}

	printf( "Mass = %e\n", m);
}

int main(int argc, char* argv[]) {
	std::vector<double> x, y, z, dx;
	std::vector<double> vars[NF];

	if( argc != 2 ) {
		printf( "Usage:\n");
		printf( "read_silo <filename>\n");
		abort();
	}

	const char* filename = argv[1];

	FILE* fp = fopen(filename, "rb");
	if( fp == NULL ) {
		printf( "%s does not exist.\n", filename );
		abort();
	} else {
		fclose(fp);
	}

	auto* db = DBOpen(filename,DB_PDB, DB_READ);
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

	for (int field = 0; field != NF; ++field) {
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

	process_data(vars, x, y, z, dx, count);

	return 0;
}
