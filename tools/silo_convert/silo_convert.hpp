#pragma once
#include <string>
#include <silo.h>
#include <vector>
#include <map>
#include <unordered_map>

#define SILO_DRIVER DB_HDF5
static const int HOST_NAME_LEN = 100;

struct silo_vars_t {
	long long int n_species, cycle;
	double omega, code_to_g, code_to_cm, code_to_s, output_frequency, refinement_floor, cgs_time, rotational_time, xscale;
	long long int version, eos, gravity, hydro, problem, radiation, node_count, leaf_count;
	char hostname[HOST_NAME_LEN];
	long long timestamp, epoch, locality_count, thread_count, step_count, time_elapsed, steps_elapsed;
	std::vector<double> atomic_number;
	std::vector<double> atomic_mass;
	std::vector<double> X;
	std::vector<double> Z;
	std::vector<long long> node_list;
	std::vector<long long> node_positions;

};

class silo_output {
protected:
	DBfile *db;
	int mesh_count;
public:
	virtual void add_mesh(std::string dir, DBquadmesh *mesh) = 0;
	virtual void add_var(std::string dir, DBquadvar *var) = 0;
	virtual void add_var_outflow(std::string dir, std::string var_name, double outflow) = 0;
	void set_vars(silo_vars_t vars);
	void set_mesh_count(int);
	virtual ~silo_output() {
	}
};

class plain_silo: public silo_output {
private:
	std::vector<char*> mesh_names;
	std::map<std::string, std::vector<char*>> var_names;
	double dtime;
	float time;
	int cycle;
public:

	plain_silo(const std::string filename);
	virtual void add_mesh(std::string dir, DBquadmesh *mesh);
	virtual void add_var(std::string dir, DBquadvar *var);
	virtual void add_var_outflow(std::string dir, std::string var_name, double outflow);

	virtual ~plain_silo();
};


class split_silo: public silo_output {
	int num_groups;
	std::string base_filename;
	std::vector<char*> mesh_names;
	std::map<std::string, std::vector<char*>> var_names;
	double dtime;
	float time;
	int cycle;
	int mesh_num;
	std::unordered_map<std::string,int> dir_to_group;
public:
	split_silo(const std::string filename, int);
	virtual void add_mesh(std::string dir, DBquadmesh* mesh);
	virtual void add_var(std::string dir, DBquadvar* var);
	virtual void add_var_outflow(std::string dir, std::string var_name, double outflow);
	virtual ~split_silo();
};

