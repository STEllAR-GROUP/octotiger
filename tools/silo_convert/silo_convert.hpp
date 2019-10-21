#pragma once
#include <string>
#include <silo.h>
#include <vector>
#include <map>

#define SILO_DRIVER DB_HDF5


class silo_output {
public:
	virtual void add_mesh(std::string dir, DBquadmesh *mesh) = 0;
	virtual void add_var(std::string dir, DBquadvar *var) = 0;
	virtual void set_vars( double omega, int n_species, const std::vector<double>& atomic_mass, const std::vector<double>& atomic_number) = 0;
	virtual ~silo_output() {
	}
};


class plain_silo: public silo_output {
private:
	DBfile *db;
	std::vector<char*> mesh_names;
	std::map<std::string, std::vector<char*>> var_names;
	double dtime;
	float time;
	int cycle;
public:

	plain_silo(const std::string filename);
	virtual void add_mesh(std::string dir, DBquadmesh *mesh);
	virtual void add_var(std::string dir, DBquadvar *var);
	virtual void set_vars( double omega, int n_species, const std::vector<double>& atomic_mass, const std::vector<double>& atomic_number);

	virtual ~plain_silo();
};
