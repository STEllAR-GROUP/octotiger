/*
 * silo_convert.cpp
 *
 *  Created on: Oct 10, 2019
 *      Author: dmarce1
 */

#include <boost/program_options.hpp>
#include <stdio.h>
#include <iostream>
#include <silo.h>

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

class plain_silo {
private:
	DBfile *db;
public:

	plain_silo(const std::string filename) {
		db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-Tiger", SILO_DRIVER);
	}

	void add_mesh(std::string dir, DBquadmesh *mesh) {
		DBMkDir(db, dir.c_str());
		DBSetDir(db, dir.c_str());
		DBPutQuadmesh(db, mesh->name, mesh->labels, mesh->coords, mesh->dims, mesh->ndims, mesh->datatype, mesh->coordtype, NULL);
		DBSetDir(db, "/");
	}

	void add_var(std::string dir, std::string name, DBquadvar *var) {
		DBSetDir(db, dir.c_str());
		DBPutQuadvar1(db, var->name, var->meshname, var->vals, var->dims, var->ndims, var->mixvals, var->mixlen, var->datatype, var->centering, NULL);
		DBSetDir(db, "/");
	}

	~plain_silo() {
		DBClose(db);
	}
};

struct options {
	std::string input;
	std::string output;

	int read_options(int argc, char *argv[]) {
		namespace po = boost::program_options;

		po::options_description command_opts("options");

		command_opts.add_options() //
		("output", po::value<std::string>(&output)->default_value(""), "output filename")           //
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
		if (output == "") {
			printf("Must specify file output name\n");
			return -1;
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
	printf("Converting %i meshes\n", mesh_names.size());
	for (const auto &mesh_name : mesh_names) {
		auto split_name = split_silo_id(mesh_name);
		const auto &filename = split_name.first;
		const auto &name = split_name.second;
		auto db = DBOpenReal(filename.c_str(), SILO_DRIVER, DB_READ);
		auto mesh = DBGetQuadmesh(db, name.c_str());
//		printf("\r%s                              ", mesh_name.c_str());
		const auto dir = mesh_to_dirname(name);
		printf( "%s\n", dir.c_str());
		for (const auto &base_name : var_names) {
			const auto var_name = mesh_to_varname(name, base_name);
			auto var = DBGetQuadvar(db, var_name.c_str());

			DBFreeQuadvar(var);
		}

		DBFreeQuadmesh(mesh);
		DBClose(db);
		counter++;
	}
	printf("\r                                                          ");

	DBClose(db);

	return 0;
}
