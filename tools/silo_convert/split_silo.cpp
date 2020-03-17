/*
 * split_silo.cpp
 *
 *  Created on: Mar 16, 2020
 *      Author: dmarce1
 */

#include "./silo_convert.hpp"
#include <cstring>
#include <sys/stat.h>

split_silo::split_silo(const std::string filename, int _num_groups) {
	mesh_num = 0;
	num_groups = _num_groups;
	db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-Tiger", SILO_DRIVER);
	base_filename = filename;
	std::string dir = base_filename + ".data/";
	mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	for (int i = 0; i < num_groups; i++) {
		std::string filename = base_filename + ".data/" + std::to_string(i) + ".silo";
		DBfile *this_db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-Tiger", SILO_DRIVER);
		db_ptrs.push_back(this_db);
	}
}

void split_silo::add_mesh(std::string dir, DBquadmesh *mesh) {
	const auto group_num = mesh_num * num_groups / mesh_count;
	dir_to_group[dir] = group_num;
	DBfile *this_db = db_ptrs[group_num];
	mesh_num++;
	dtime = mesh->dtime;
	time = mesh->time;
	cycle = mesh->cycle;
	DBMkDir(this_db, dir.c_str());
	DBSetDir(this_db, dir.c_str());
	DBPutQuadmesh(this_db, mesh->name, mesh->labels, mesh->coords, mesh->dims, mesh->ndims, mesh->datatype, mesh->coordtype, NULL);
	DBSetDir(this_db, "/");
	auto tmp = dir;
	dir = base_filename + ".data/" + std::to_string(group_num) + ".silo:" + tmp;
	char *new_name = new char[dir.size() + strlen(mesh->name) + 1];
	strcpy(new_name, (dir + mesh->name).c_str());
	mesh_names.push_back(new_name);
}

void split_silo::add_var(std::string dir, DBquadvar *var) {
	const auto group_num = dir_to_group[dir];
	DBfile *this_db = db_ptrs[group_num];
	DBSetDir(this_db, "/");
	DBSetDir(this_db, dir.c_str());
	auto tmp = dir;
	dir = base_filename + ".data/" + std::to_string(group_num) + ".silo:" + tmp;
	char *mesh_name = new char[dir.size() + strlen(var->meshname) + 1];
	strcpy(mesh_name, (dir + var->meshname).c_str());
	DBPutQuadvar1(this_db, var->name, mesh_name, var->vals[0], var->dims, var->ndims, var->mixvals, var->mixlen, var->datatype, var->centering, NULL);
	char *new_name = new char[dir.size() + strlen(var->name) + 1];
	strcpy(new_name, (dir + var->name).c_str());
	var_names[std::string(var->name)].push_back(new_name);

}

void split_silo::add_var_outflow(std::string dir, std::string var_name, double outflow) {
	const int one = 1;
	const auto group_num = dir_to_group[dir];
	DBfile *this_db = db_ptrs[group_num];
	DBSetDir(this_db, "/");
	DBSetDir(this_db, dir.c_str());
	DBWrite(this_db, var_name.c_str(), &outflow, &one, 1, DB_DOUBLE);

}

split_silo::~split_silo() {
	int mesh_type = DB_QUADMESH;
	auto optlist = DBMakeOptlist(4);
	DBAddOption(optlist, DBOPT_MB_BLOCK_TYPE, &mesh_type);
	DBAddOption(optlist, DBOPT_CYCLE, &cycle);
	DBAddOption(optlist, DBOPT_TIME, &time);
	DBAddOption(optlist, DBOPT_DTIME, &dtime);
	DBPutMultimesh(db, "quadmesh", mesh_names.size(), mesh_names.data(), NULL, optlist);
	DBFreeOptlist(optlist);
	for (auto *ptr : mesh_names) {
		delete[] ptr;
	}
	optlist = DBMakeOptlist(1);
	char mmesh[] = "quadmesh";
	DBAddOption(optlist, DBOPT_MMESH_NAME, mmesh);
	for (auto &these_names : var_names) {
		const auto sz = these_names.second.size();
		DBPutMultivar(db, these_names.first.c_str(), sz, these_names.second.data(), std::vector<int>(sz, DB_QUADVAR).data(), optlist);
		for (auto *ptr : these_names.second) {
			delete[] ptr;
		}
	}
	DBFreeOptlist(optlist);
	DBClose(db);
	for (int i = 0; i < num_groups; i++) {
		DBClose(db_ptrs[i]);
	}
}
