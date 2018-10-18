/*
 * silo.cpp
 *
 *  Created on: Oct 16, 2018
 *      Author: dmarce1
 */

#include "node_registry.hpp"
#include "silo.hpp"
#include "options.hpp"

std::vector<node_location::node_id> output_stage1(std::string fname, int cycle);
void output_stage2(std::string fname, int cycle);

HPX_PLAIN_ACTION(output_stage1, output1_action);
HPX_PLAIN_ACTION(output_stage2, output2_action);

#include <hpx/include/threads.hpp>
#include <hpx/include/run_as.hpp>

template<class T>
struct db_type {
	static constexpr int d = DB_UNKNOWN;
};

template<>
struct db_type<integer> {
	static constexpr int d = DB_LONG_LONG;
};

template<>
struct db_type<real> {
	static constexpr int d = DB_DOUBLE;
};

template<>
struct db_type<char> {
	static constexpr int d = DB_CHAR;
};

template<class T>
struct write_silo_var {
	void operator()(DBfile* db, const char* name, T var) {
		int one = 1;
		DBWrite(db, name, &var, &one, 1, db_type<T>::d);
	}
};

template<>
struct write_silo_var<std::string> {
	void operator()(DBfile* db, const char* name, const std::string var) {
		int sz = var.size();
		DBWrite(db, name, var.c_str(), &sz, 1, DB_CHAR);
	}
};

struct mesh_vars_t {
	std::vector<silo_var_t> vars;
	std::vector<std::string> var_names;
	std::string mesh_name;
	std::vector<std::array<real, INX + 1>> X;
};

static std::vector<hpx::future<mesh_vars_t>> futs_;
static std::vector<node_location::node_id> loc_ids;
static hpx::lcos::local::mutex mtx_;

std::vector<node_location::node_id> output_stage1(std::string fname, int cycle) {
	std::vector<node_location::node_id> ids;
	futs_.clear();
	for (auto i = node_registry::begin(); i != node_registry::end(); i++) {
		if (!i->second->refined()) {
			futs_.push_back(hpx::async([&ids](node_location loc, node_server* this_ptr) {
				std::vector<std::array<real, INX+1>> X(NDIM);
				const real dx = TWO / real(1 << loc.level()) / real(INX);
				for( int d = 0; d < NDIM; d++) {
					const real x0 = loc.x_location(d);
					for( int i = 0; i <= INX; i++) {
						X[d][i] = x0 + real(i) * dx;
					}
				}
				mesh_vars_t rc;
				const std::string suffix = std::to_string(loc.to_id());
				rc.mesh_name = /*std::string( "mesh_") +*/suffix;
				rc.vars = std::move(this_ptr->get_hydro_grid().var_data(suffix));
				rc.X = std::move(X);
				return std::move(rc);
			}, i->first, i->second));
			ids.push_back(i->first.to_id());
		}
	}
	return ids;
}

void output_stage2(std::string fname, int cycle) {
	std::string this_fname = fname + std::string(".silo");
	DBfile *db;
	if (hpx::get_locality_id() == 0) {
		db = DBCreateReal(this_fname.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-tiger", DB_PDB);
	} else {
		db = DBOpen(this_fname.c_str(), DB_PDB, DB_APPEND);
	}
	int nfields = 0;
	if (opts.hydro) {
		nfields += 14;
	}
	if (opts.gravity) {
		nfields += 4;
	}
	if (node_registry::begin() != node_registry::end()) {
		double dtime = node_registry::begin()->second->get_time();
		float ftime = dtime;
		auto optlist = DBMakeOptlist(5);
		int one = 1;
		int opt1 = DB_CARTESIAN;
		DBAddOption(optlist, DBOPT_HIDE_FROM_GUI, &one);
		DBAddOption(optlist, DBOPT_COORDSYS, &opt1);
		DBAddOption(optlist, DBOPT_CYCLE, &cycle);
		DBAddOption(optlist, DBOPT_DTIME, &dtime);
		DBAddOption(optlist, DBOPT_TIME, &ftime);
		const char* coord_names[] = { "x", "y", "z" };
		constexpr int dims[] = { INX + 1, INX + 1, INX + 1 };
		constexpr int dims2[] = { INX, INX, INX };
		constexpr int data_type = DB_DOUBLE;
		constexpr int ndim = NDIM;
		constexpr int coord_type = DB_COLLINEAR;

		for (auto& this_fut : futs_) {
			auto mesh_vars = this_fut.get();
			const auto& X = mesh_vars.X;
			const real* coords[] = { X[0].data(), X[1].data(), X[2].data() };
			DBPutQuadmesh(db, mesh_vars.mesh_name.c_str(), coord_names, coords, dims, ndim, data_type, coord_type, optlist);
			for (const auto& o : mesh_vars.vars) {
				DBPutQuadvar1(db, o.name_, mesh_vars.mesh_name.c_str(), o.data_, dims2, ndim, (const void*) NULL, 0, DB_DOUBLE,
				DB_ZONECENT, optlist);
			}
		}
		DBFreeOptlist(optlist);
	}

	static const auto ids = hpx::find_all_localities();
	static const auto this_id = hpx::get_locality_id();
	if (this_id < ids.size() - 1) {
		output2_action func;
		func(ids[this_id + 1], fname, cycle);
	}

	if (this_id == 0) {
		double dtime = node_registry::begin()->second->get_time();
		float ftime = dtime;
		double rtime = node_registry::begin()->second->get_rotation_count();
		std::vector<node_location> node_locs;
		std::vector<char*> mesh_names;
		std::vector<std::vector<char*>> field_names(nfields);
		for (auto& i : loc_ids) {
			node_location nloc;
			nloc.from_id(i);
			node_locs.push_back(nloc);
		}
		const auto top_field_names = grid::get_field_names();
		for (int i = 0; i < node_locs.size(); i++) {
			const auto suffix = std::to_string(node_locs[i].to_id());
			const auto str = /*std::string("mesh_") + */suffix;
			char* ptr = new char[str.size() + 1];
			std::strcpy(ptr, str.c_str());
			mesh_names.push_back(ptr);
			for (int f = 0; f < nfields; f++) {
				const auto str = top_field_names[f] + std::string("_") + suffix;
				char* ptr = new char[str.size() + 1];
				strcpy(ptr, str.c_str());
				field_names[f].push_back(ptr);
			}
		}

		const int n_total_domains = mesh_names.size();
		static const std::vector<int> meshtypes(n_total_domains, DB_QUAD_RECT);
		static const std::vector<int> datatypes(n_total_domains, DB_QUADVAR);

		auto optlist = DBMakeOptlist(4);
		int opt1 = DB_CARTESIAN;
		DBAddOption(optlist, DBOPT_COORDSYS, &opt1);
		DBAddOption(optlist, DBOPT_CYCLE, &cycle);
		DBAddOption(optlist, DBOPT_DTIME, &dtime);
		DBAddOption(optlist, DBOPT_TIME, &ftime);
		DBPutMultimesh(db, "mesh", n_total_domains, mesh_names.data(), meshtypes.data(), optlist);
		for (int f = 0; f < nfields; f++) {
			DBPutMultivar(db, top_field_names[f].c_str(), n_total_domains, field_names[f].data(), datatypes.data(), optlist);

		}
		DBFreeOptlist(optlist);
		for (auto ptr : mesh_names) {
			delete[] ptr;
		}
		for (int f = 0; f < nfields; f++) {
			for (auto& s : field_names[f]) {
				delete[] s;
			}
		}

		write_silo_var<integer> fi;
		write_silo_var<real> fr;

		fi(db, "eos", integer(opts.eos));
		fi(db, "gravity", integer(opts.gravity));
		fi(db, "hydro", integer(opts.hydro));
		fr(db, "omega", grid::get_omega());
		fr(db, "output_frequency", opts.output_dt);
		fi(db, "problem", integer(opts.problem));
		fi(db, "radiation", integer(opts.radiation));
		fr(db, "refinement_floor", opts.refinement_floor);
		fr(db, "rotational_time", rtime);
		fr(db, "xscale", opts.xscale);

		DBClose(db);
	}
}

void output_all(std::string fname, int cycle) {
	static auto ids = hpx::find_all_localities();
	std::vector<hpx::future<std::vector<node_location::node_id>>> id_futs;
	for (auto& id : ids) {
		id_futs.push_back(hpx::async<output1_action>(id, fname, cycle));
	}
	loc_ids.clear();
	for (auto& f : id_futs) {
		std::vector<node_location::node_id> these_ids = f.get();
		for (auto& i : these_ids) {
			loc_ids.push_back(i);
		}
	}
	output2_action func;
	func(ids[0], fname, cycle);
}

void create_leaf_nodes(const std::vector<node_location::node_id>& node_ids);

HPX_PLAIN_ACTION (create_leaf_nodes);

void create_leaf_nodes(const std::vector<node_location::node_id>& node_ids) {
	std::vector<hpx::future<void>> futs;
	const auto me = hpx::find_here();
	for (const auto& i : node_ids) {
		node_location l;
		l.from_id(i);
		futs.push_back(hpx::new_ < node_server > (me, l).then([&me,l](hpx::future<hpx::id_type>&& f) {
			const auto id = f.get();
			load_registry::put(l.to_id(), id);
			const auto pid = load_registry::get(l.get_parent().to_id());
			node_client(me).set_parent(pid).get();
			return node_client(pid).notify_parent(l,id);
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
}

void load_from_silo(std::string fname) {
	static auto localities = hpx::find_all_localities();
	static int sz = localities.size();
	DBfile* db = DBOpen(fname.c_str(), DB_PDB, DB_READ);
	DBmultimesh* master_mesh = DBGetMultimesh(db, "mesh");
	std::vector<node_location::node_id> work;
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < master_mesh->nblocks; i++) {
		work.push_back(std::stoi(master_mesh->meshnames[i]));
		const int this_id = i / sz;
		const int next_id = (i + 1) / sz;
		if (this_id != next_id) {
			futs.push_back(hpx::async<create_leaf_nodes_action>(localities[this_id], work));
			work.clear();
		}
	}
	DBFreeMultimesh(master_mesh);
	DBClose(db);
	for (auto& f : futs) {
		f.get();
	}
}

silo_var_t::silo_var_t(const std::string& name) {
	name_ = (char*) malloc(sizeof(char) * (name.size() + 1));
	data_ = (double*) malloc(sizeof(double) * INX * INX * INX);
	strcpy(name_, name.c_str());
}

double& silo_var_t::operator()(int i) {
	return (static_cast<double*>(data_))[i];

}

double silo_var_t::operator()(int i) const {
	return (static_cast<double*>(data_))[i];

}

silo_var_t::~silo_var_t() {
	if (name_ != NULL) {
		free(name_);
		free(data_);
	}
}

silo_var_t::silo_var_t(silo_var_t && other) {
	name_ = other.name_;
	data_ = other.data_;
	other.name_ = NULL;
	other.data_ = NULL;
}
