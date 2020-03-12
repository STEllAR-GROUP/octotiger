#include "./silo_convert.hpp"
#include <string.h>

plain_silo::plain_silo(const std::string filename) {
	db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-Tiger", SILO_DRIVER);
}

void plain_silo::add_mesh(std::string dir, DBquadmesh *mesh) {
	dtime = mesh->dtime;
	time = mesh->time;
	cycle = mesh->cycle;
	DBMkDir(db, dir.c_str());
	DBSetDir(db, dir.c_str());
	DBPutQuadmesh(db, mesh->name, mesh->labels, mesh->coords, mesh->dims, mesh->ndims, mesh->datatype, mesh->coordtype, NULL);
	DBSetDir(db, "/");
	char *new_name = new char[dir.size() + strlen(mesh->name) + 1];
	strcpy(new_name, (dir + mesh->name).c_str());
	mesh_names.push_back(new_name);
}

void plain_silo::add_var(std::string dir, DBquadvar *var) {
	DBSetDir(db, dir.c_str());
	DBPutQuadvar1(db, var->name, var->meshname, var->vals[0], var->dims, var->ndims, var->mixvals, var->mixlen, var->datatype, var->centering, NULL);
	DBSetDir(db, "/");
	char *new_name = new char[dir.size() + strlen(var->name) + 1];
	strcpy(new_name, (dir + var->name).c_str());
	var_names[std::string(var->name)].push_back(new_name);
}

plain_silo::~plain_silo() {
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
}
