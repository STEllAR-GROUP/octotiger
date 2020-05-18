/*
 * silo_convert.cpp
 *
 *  Created on: Oct 10, 2019
 *      Author: dmarce1
 */

#include <boost/program_options.hpp>
#include <stdio.h>
#include <iostream>
#include <map>

#include "./silo_convert.hpp"

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
	std::string output;
	int num_groups;

	int read_options(int argc, char *argv[]) {
		namespace po = boost::program_options;

		po::options_description command_opts("options");

		command_opts.add_options() //
		("output", po::value<std::string>(&output)->default_value(""), "output filename")      //
		("input", po::value<std::string>(&input)->default_value(""), "input filename")         //
		("num_groups", po::value<int>(&num_groups)->default_value(0), "number of silo groups") //
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
	silo_vars_t vars;
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

	silo_output *output;
	if (opts.num_groups == 0) {
		output = dynamic_cast<silo_output*>(new plain_silo(opts.output));
	} else {
		output = dynamic_cast<silo_output*>(new split_silo(opts.output, opts.num_groups));
	}

	int counter = 0;

	DBReadVar(db, "n_species", &vars.n_species);
	DBReadVar(db, "node_count", &vars.node_count);
	DBReadVar(db, "leaf_count", &vars.leaf_count);
	vars.atomic_number.resize(vars.n_species);
	vars.atomic_mass.resize(vars.n_species);
	vars.X.resize(vars.n_species);
	vars.Z.resize(vars.n_species);
	vars.node_list.resize(vars.node_count);
	vars.node_positions.resize(vars.node_count);
	DBReadVar(db, "node_list", vars.node_list.data());
	DBReadVar(db, "node_positions", vars.node_positions.data());
	DBReadVar(db, "omega", &vars.omega);
	DBReadVar(db, "atomic_number", vars.atomic_number.data());
	DBReadVar(db, "atomic_mass", vars.atomic_mass.data());
	DBReadVar(db, "X", vars.X.data());
	DBReadVar(db, "Z", vars.Z.data());
	DBReadVar(db, "version", &vars.version);
	DBReadVar(db, "code_to_g", &vars.code_to_g);
	DBReadVar(db, "code_to_s", &vars.code_to_s);
	DBReadVar(db, "code_to_cm", &vars.code_to_cm);
	DBReadVar(db, "eos", &vars.eos);
	DBReadVar(db, "gravity", &vars.gravity);
	DBReadVar(db, "hydro", &vars.hydro);
	DBReadVar(db, "radiation", &vars.radiation);
	DBReadVar(db, "output_frequency", &vars.output_frequency);
	DBReadVar(db, "problem", &vars.problem);
	DBReadVar(db, "refinement_floor", &vars.refinement_floor);
	DBReadVar(db, "cgs_time", &vars.cgs_time);
	DBReadVar(db, "rotational_time", &vars.rotational_time);
	DBReadVar(db, "xscale", &vars.xscale);
	DBReadVar(db, "cycle", &vars.cycle);
	DBReadVar(db, "hostname", vars.hostname);
	DBReadVar(db, "timestamp", &vars.timestamp);
	DBReadVar(db, "epoch", &vars.epoch);
	DBReadVar(db, "locality_count", &vars.locality_count);
	DBReadVar(db, "thread_count", &vars.thread_count);
	DBReadVar(db, "step_count", &vars.step_count);
	DBReadVar(db, "time_elapsed", &vars.time_elapsed);
	DBReadVar(db, "steps_elapsed", &vars.steps_elapsed);
	printf("atomic number| atomic mass | X             | Z\n");
	for (int s = 0; s < vars.n_species; s++) {
		printf("%e | %e | %e | %e\n", vars.atomic_number[s], vars.atomic_mass[s], vars.X[s], vars.Z[s]);
	}
	printf("n_species        = %lli\n", vars.n_species);
	printf("node_count       = %lli\n", vars.node_count);
	printf("leaf_count       = %lli\n", vars.leaf_count);
	printf("omega            = %e\n", vars.omega);
	printf("SILO version     = %lli\n", vars.version);
	printf("code_to_g        = %e\n", vars.code_to_g);
	printf("code_to_s        = %e\n", vars.code_to_s);
	printf("code_to_cm       = %e\n", vars.code_to_cm);
	printf("eos              = %lli\n", vars.eos);
	printf("gravity          = %lli\n", vars.gravity);
	printf("hydro            = %lli\n", vars.hydro);
	printf("radiation        = %lli\n", vars.radiation);
	printf("output frequency = %e\n", vars.output_frequency);
	printf("problem          = %lli\n", vars.problem);
	printf("refinement_floor = %e\n", vars.refinement_floor);
	printf("cgs_time         = %e\n", vars.cgs_time);
	printf("rotational_time  = %e\n", vars.rotational_time);
	printf("xscale           = %e\n", vars.xscale);
	printf("cycle            = %lli\n", vars.cycle);
	printf("hostname         = %s\n", vars.hostname);
	printf("timestamp        = %lli\n", vars.timestamp);
	printf("locality_count   = %lli\n", vars.locality_count);
	printf("thread_count     = %lli\n", vars.thread_count);
	printf("step_count       = %lli\n", vars.step_count);
	printf("time_elapsed     = %lli\n", vars.time_elapsed);
	printf("steps_elapsed    = %lli\n", vars.steps_elapsed);
	printf("epoch            = %lli\n", vars.epoch);

//	for (int i = 0; i < node_count; i++) {
//		printf("%16llx %lli\n", node_list[i], node_positions[i]);
//	}

	printf("Converting %li meshes\n", mesh_names.size());
	output->set_mesh_count(mesh_names.size());
	for (const auto &mesh_name : mesh_names) {
		auto split_name = split_silo_id(mesh_name);
		const auto &filename = split_name.first;
		const auto &name = split_name.second;
		auto db = DBOpenReal(filename.c_str(), SILO_DRIVER, DB_READ);
		auto mesh = DBGetQuadmesh(db, name.c_str());
		printf("\r%s                              ", mesh_name.c_str());
		const auto dir = mesh_to_dirname(name);

		output->add_mesh(dir, mesh);

		for (const auto &base_name : var_names) {
			const auto var_name = mesh_to_varname(name, base_name);
			double outflow;
			const auto outflow_name = var_name + "_outflow";
			if (base_name != "gx" && base_name != "gy" && base_name != "gz" && base_name != "locality" && base_name != "idle_rate") {
				DBReadVar(db, outflow_name.c_str(), &outflow);
				output->add_var_outflow(dir, outflow_name, outflow);
			}
			auto var = DBGetQuadvar(db, var_name.c_str());
			output->add_var(dir, var);
			DBFreeQuadvar(var);
		}

		DBFreeQuadmesh(mesh);
		DBClose(db);
		counter++;
	}
	output->set_vars(vars);

	delete output;

	printf("\rDone!                                                          \n");

	DBClose(db);

	return 0;
}

