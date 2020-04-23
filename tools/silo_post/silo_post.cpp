/*
 * silo_convert.cpp
 *
 *  Created on: Oct 10, 2019
 *      Author: dmarce1
 */

#include <boost/program_options.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <silo.h>
#include <vector>
#include <map>

#define SILO_DRIVER DB_HDF5

auto split_silo_id(const std::string str) {
	std::pair<std::string, std::string> split;
	bool before = true;
	for (int i = 0; i < str.size(); i++) {
		if (str[i] == ':') {
			before = false;
		} else if (before) {
			split.first.push_back(str[i]);
		} else {
			split.second.push_back(str[i]);
		}
	}
	return split;
}

struct options {
	std::string input;

	int read_options(int argc, char *argv[]) {
		namespace po = boost::program_options;

		po::options_description command_opts("options");

		command_opts.add_options() //
		("input", po::value<std::string>(&input)->default_value(""), "input filename")           //
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
		po::notify(vm);
		return 0;
	}
};

std::string strip_nonnumeric(std::string &&s) {
	s.erase(std::remove_if(s.begin(), s.end(), [](char c) {
		return c < '0' || c > '9';
	}), s.end());
	return std::move(s);
}

std::string mesh_to_varname(std::string mesh_name, std::string var_name) {
	std::string token = "quadmesh";
	size_t pos = mesh_name.find(token);
	mesh_name.erase(pos, token.length());
	return mesh_name + var_name;
}

std::string mesh_to_dirname(std::string mesh_name) {
	std::string token = "quadmesh";
	size_t pos = mesh_name.find(token);
	mesh_name.erase(pos, token.length());
	return mesh_name;
}

int main(int argc, char *argv[]) {

	std::set<std::string> var_names;
	std::set<std::string> mesh_names;

	options opts;
	if (opts.read_options(argc, argv) != 0) {
		return 0;
	}

	printf("Opening %s\n", opts.input.c_str());
	printf("Reading table of contents\n");
	auto db = DBOpenReal(opts.input.c_str(), SILO_DRIVER, DB_READ);
	auto toc = DBGetToc(db);

	printf("Variable names:\n");
	for (int i = 0; i < toc->nmultivar; ++i) {
		auto name = std::string(toc->multivar_names[i]);
		var_names.insert(name);
		printf("	%s\n", name.c_str());
	}
	auto mesh = DBGetMultimesh(db, "quadmesh");
	for (int i = 0; i < mesh->nblocks; ++i) {
		auto name = std::string(mesh->meshnames[i]);
		mesh_names.insert(name);
	}

	DBFreeMultimesh(mesh);

	int counter = 0;
	long long int n_species, cycle;
	double omega;
	DBReadVar(db, "n_species", &n_species);
	DBReadVar(db, "omega", &omega);

	std::vector<double> atomic_number(n_species);
	std::vector<double> atomic_mass(n_species);
	DBReadVar(db, "atomic_number", atomic_number.data());
	DBReadVar(db, "atomic_mass", atomic_mass.data());
	printf("cycle = %lli\n", cycle);
	printf("n_species = %lli\n", n_species);
	printf("omega     = %e\n", omega);
	printf("atomic number | atomic mass \n");
	for (int s = 0; s < n_species; s++) {
		printf("%e | %e\n", atomic_number[s], atomic_mass[s]);
	}
	printf("Reading %li meshes\n", mesh_names.size());
	double sum1 = 0.0;
	double sum2 = 0.0;
	double t;
	double dxmin = 1.0d+99;
	for (const auto &mesh_name : mesh_names) {
		auto split_name = split_silo_id(mesh_name);
		const auto &filename = split_name.first;
		const auto &name = split_name.second;
		auto db = DBOpenReal(filename.c_str(), SILO_DRIVER, DB_READ);
		auto mesh = DBGetQuadmesh(db, name.c_str());
		printf("\r%s                              ", mesh_name.c_str());
		const auto dir = mesh_to_dirname(name);
		t = mesh->time;
		const auto dx = (((double*) mesh->coords[0])[1] - ((double*) mesh->coords[0])[0]);
		dxmin = std::min(dxmin,dx);
		DBFreeQuadmesh(mesh);
		DBClose(db);
	}

	double rmax = 1.5;
	int NBIN = rmax / dxmin;
	double dR = (rmax) / NBIN;
	std::vector<double> Ibin(NBIN);
	std::vector<double> Lbin(NBIN);



	for (const auto &mesh_name : mesh_names) {
		auto split_name = split_silo_id(mesh_name);
		const auto &filename = split_name.first;
		const auto &name = split_name.second;
		auto db = DBOpenReal(filename.c_str(), SILO_DRIVER, DB_READ);
		auto mesh = DBGetQuadmesh(db, name.c_str());
		printf("\r%s                              ", mesh_name.c_str());
		const auto dir = mesh_to_dirname(name);
		t = mesh->time;
		const auto dx = (((double*) mesh->coords[0])[1] - ((double*) mesh->coords[0])[0]);
		const auto dV = dx * dx * dx;
		auto var_name = mesh_to_varname(name, "rho_1");
		auto var = DBGetQuadvar(db, var_name.c_str());

		for (int i = 0; i < var->nels; i++) {
			const double rho1 = ((double*) (var->vals[0]))[i];
			if (rho1 < pow(10.0, 5.0)) {
				sum1 += dV * rho1;
			}
			if (rho1 < pow(10.0, 5.2)) {
				sum2 += dV * rho1;
			}

		}

		DBFreeQuadvar(var);

		var_name = mesh_to_varname(name, "rho_1");
		auto rho1 = DBGetQuadvar(db, var_name.c_str());
		var_name = mesh_to_varname(name, "rho_2");
		auto rho2 = DBGetQuadvar(db, var_name.c_str());
		var_name = mesh_to_varname(name, "rho_3");
		auto rho3 = DBGetQuadvar(db, var_name.c_str());
		var_name = mesh_to_varname(name, "rho_4");
		auto rho4 = DBGetQuadvar(db, var_name.c_str());
		var_name = mesh_to_varname(name, "rho_5");
		auto rho5 = DBGetQuadvar(db, var_name.c_str());
		var_name = mesh_to_varname(name, "sx");
		auto sx = DBGetQuadvar(db, var_name.c_str());
		var_name = mesh_to_varname(name, "sy");
		auto sy = DBGetQuadvar(db, var_name.c_str());
		const int NX = mesh->dims[0] - 1;
		for (int i = 0; i < NX; i++) {
			for (int j = 0; j < NX; j++) {
				for (int k = 0; k < NX; k++) {
					const auto iii = k * NX * NX + j * NX + i;
					const auto rho = ((double*) rho1->vals[0])[iii] + ((double*) rho2->vals[0])[iii] + ((double*) rho3->vals[0])[iii]
							+ ((double*) rho4->vals[0])[iii] + ((double*) rho5->vals[0])[iii];
					const auto vx = ((double*) sx->vals[0])[iii] / rho;
					const auto vy = ((double*) sy->vals[0])[iii] / rho;
					const auto x = (((double*) mesh->coords[0])[i] + ((double*) mesh->coords[0])[i + 1]) / 2.0;
					const auto y = (((double*) mesh->coords[1])[j] + ((double*) mesh->coords[1])[j + 1]) / 2.0;
					const auto R = std::sqrt(x * x + y * y);
					const auto omega = (-y * vx + x * vy) / (R * R);
					int I = R / dR;
					if (I < NBIN) {
						Lbin[I] += rho * R * R * omega * dx * dx * dx;
						Ibin[I] += rho * R * R * dx * dx * dx;
					}
				}
			}
		}
		DBFreeQuadvar(rho1);
		DBFreeQuadvar(rho2);
		DBFreeQuadvar(rho3);
		DBFreeQuadvar(rho4);
		DBFreeQuadvar(rho5);
		DBFreeQuadvar(sx);
		DBFreeQuadvar(sy);

		DBFreeQuadmesh(mesh);
		DBClose(db);
		counter++;
	}
	FILE *fp = fopen("omega.dat", "wt");
	for (int i = 0; i < NBIN; i++) {
		fprintf(fp, "%e %e\n", (i + 0.5) * dR, Lbin[i] / Ibin[i]);
	}
	fclose(fp);

	const auto Msol = 1.989e+33;
	fp = fopen("dredge.dat", "at");
	fprintf(fp, "%e %e %e\n", t, sum1 / Msol, sum2 / Msol);
	fclose(fp);
	printf("\rDone!                                                          \n");

	DBClose(db);

	return 0;
}
