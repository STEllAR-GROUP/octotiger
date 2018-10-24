/*
 * silo.cpp
 *
 *  Created on: Oct 16, 2018
 *      Author: dmarce1
 */

#include "node_registry.hpp"
#include "silo.hpp"
#include "options.hpp"
#include <hpx/lcos/broadcast.hpp>
#include <future>
#include "util.hpp"

static const auto& localities = options::all_localities;

static std::mutex silo_mtx_;

template<class R, class ...Args1, class ...Args2>
R call_silo(R (*f)(Args1...), int, Args2&& ...args) {
	return std::async(std::launch::async, [&]() {
		std::lock_guard<std::mutex> lock(silo_mtx_);
		return f(args...);
	}).get();
}

template<class ...Args1, class ...Args2>
int call_silo(int (*f)(Args1...), int line, Args2&& ...args) {
	return std::async(std::launch::async, [&]() {
		std::lock_guard<std::mutex> lock(silo_mtx_);
		int rc = f(args...);
		if( rc != 0 ) {
			std::cout << "silo error at " << std::to_string(line) << "\n";
		}
		return rc;
	}).get();
}

#define CALL_SILO( func, args... ) call_silo(func, __LINE__, args)

std::vector<node_location::node_id> output_stage1(std::string fname, int cycle);
void output_stage2(std::string fname, int cycle);

HPX_PLAIN_ACTION(output_stage1, output1_action);
HPX_PLAIN_ACTION(output_stage2, output_stage2_action);

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

constexpr int db_type<integer>::d;
constexpr int db_type<char>::d;
constexpr int db_type<real>::d;

template<class T>
struct write_silo_var {
	void operator()(DBfile* db, const char* name, T var) {
		int one = 1;
		CALL_SILO(DBWrite, db, name, &var, &one, 1, db_type<T>::d);
	}
};

template<class T>
struct read_silo_var {
	T operator()(DBfile* db, const char* name) {
		int one = 1;
		T var;
		CALL_SILO(DBReadVar, db, name, &var);
		return var;
	}
};

struct mesh_vars_t {
	std::vector<silo_var_t> vars;
	std::vector<std::string> var_names;
	std::vector<std::pair<std::string, real>> outflow;
	std::string mesh_name;
	std::vector<std::array<real, INX + 1>> X;
};

static std::vector<hpx::future<mesh_vars_t>> futs_;
static std::vector<node_location::node_id> loc_ids;

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
				rc.outflow = std::move(this_ptr->get_hydro_grid().get_outflows());
				return std::move(rc);
			}, i->first, i->second));
			ids.push_back(i->first.to_id());
		}
	}

	return ids;
}

static const int HOST_NAME_LEN = 100;
static int epoch = 0;
static time_t start_time = time(NULL);
static integer start_step = 0;
static int timestamp;
static int nsteps;
static int time_elapsed;
static int steps_elapsed;

void output_stage2(std::string fname, int cycle) {
	std::string this_fname = fname + std::string(".silo");
	DBfile *db;
	if (hpx::get_locality_id() == 0) {
		db = CALL_SILO(DBCreateReal, this_fname.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-tiger", DB_PDB);
	} else {
		db = CALL_SILO(DBOpenReal, this_fname.c_str(), DB_PDB, DB_APPEND);
	}
	const int nfields = grid::get_field_names().size();
	if (node_registry::begin() != node_registry::end()) {
		double dtime = node_registry::begin()->second->get_time();
		float ftime = dtime;
		auto optlist = CALL_SILO(DBMakeOptlist, 5);
		int one = 1;
		int opt1 = DB_CARTESIAN;
		CALL_SILO(DBAddOption, optlist, DBOPT_HIDE_FROM_GUI, &one);
		CALL_SILO(DBAddOption, optlist, DBOPT_COORDSYS, &opt1);
		CALL_SILO(DBAddOption, optlist, DBOPT_CYCLE, &cycle);
		CALL_SILO(DBAddOption, optlist, DBOPT_DTIME, &dtime);
		CALL_SILO(DBAddOption, optlist, DBOPT_TIME, &ftime);
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
			CALL_SILO(DBPutQuadmesh, db, mesh_vars.mesh_name.c_str(), coord_names, coords, dims, ndim, data_type, coord_type,
					optlist);
			for (const auto& o : mesh_vars.vars) {
				CALL_SILO(DBPutQuadvar1, db, o.name(), mesh_vars.mesh_name.c_str(), o.data(), dims2, ndim, (const void*) NULL,
						0, DB_DOUBLE, DB_ZONECENT, optlist);
			}
			for (const auto& p : mesh_vars.outflow) {
				const std::string name = p.first + std::string("_outflow_") + mesh_vars.mesh_name;
				write_silo_var<real>()(db, name.c_str(), p.second);
			}
		}
		CALL_SILO(DBFreeOptlist, optlist);
		CALL_SILO(DBClose, db);
	}
	static const integer this_id = hpx::get_locality_id();
	if (this_id < integer(localities.size()) - 1) {
		output_stage2_action func;
		func(localities[this_id + 1], fname, cycle);
	}

	if (this_id == 0) {
		db = CALL_SILO(DBOpenReal, this_fname.c_str(), DB_PDB, DB_APPEND);
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

		auto optlist = CALL_SILO(DBMakeOptlist, 4);
		int opt1 = DB_CARTESIAN;
		CALL_SILO(DBAddOption, optlist, DBOPT_COORDSYS, &opt1);
		CALL_SILO(DBAddOption, optlist, DBOPT_CYCLE, &cycle);
		CALL_SILO(DBAddOption, optlist, DBOPT_DTIME, &dtime);
		CALL_SILO(DBAddOption, optlist, DBOPT_TIME, &ftime);
		CALL_SILO(DBPutMultimesh, db, "mesh", n_total_domains, mesh_names.data(), meshtypes.data(), optlist);
		for (int f = 0; f < nfields; f++) {

			CALL_SILO(DBPutMultivar, db, top_field_names[f].c_str(), n_total_domains, field_names[f].data(), datatypes.data(),
					optlist);

		}
		CALL_SILO(DBFreeOptlist, optlist);
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
		fr(db, "time", dtime);
		fr(db, "rotational_time", rtime);
		fr(db, "xscale", opts.xscale);
		char hostname[HOST_NAME_LEN];
		gethostname(hostname, HOST_NAME_LEN);
		CALL_SILO(DBWrite, db, "hostname", hostname, &HOST_NAME_LEN, 1, DB_CHAR);
		write_silo_var<integer>()(db, "timestamp", timestamp);
		write_silo_var<integer>()(db, "epoch", epoch);
		write_silo_var<integer>()(db, "N_localities", localities.size());
		write_silo_var<integer>()(db, "step_count", nsteps);
		write_silo_var<integer>()(db, "time_elapsed", time_elapsed);
		write_silo_var<integer>()(db, "steps_elapsed", steps_elapsed);
		CALL_SILO(DBClose, db);

	}
}

void output_all(std::string fname, int cycle) {
	static hpx::future<void> barrier(hpx::make_ready_future<void>());
	barrier.get();
	nsteps = node_registry::begin()->second->get_step_num();
	timestamp = time(NULL);
	steps_elapsed = nsteps - start_step;
	time_elapsed = time(NULL) - start_time;
	start_time = timestamp;
	start_step = nsteps;
	std::vector<hpx::future<std::vector<node_location::node_id>>> id_futs;
	for (auto& id : localities) {
		id_futs.push_back(hpx::async<output1_action>(id, fname, cycle));
	}
	loc_ids.clear();
	for (auto& f : id_futs) {
		std::vector<node_location::node_id> these_ids = f.get();
		for (auto& i : these_ids) {
			loc_ids.push_back(i);
		}
	}
	barrier = hpx::async<output_stage2_action>(localities[0], fname, cycle);
}

void local_load(const std::string&, std::vector<node_location::node_id> node_ids);
void all_boundaries();

HPX_PLAIN_ACTION(local_load, local_load_action);
HPX_PLAIN_ACTION(all_boundaries, all_boundaries_action);

void all_boundaries() {
	std::vector<hpx::future<void>> futs;
	for (auto i = node_registry::begin(); i != node_registry::end(); i++) {
		futs.push_back(hpx::async([i]() {
			i->second->all_hydro_bounds();
		}));
	}
	hpx::when_all(std::move(futs)).get();
}

void local_load(const std::string& fname, std::vector<node_location::node_id> node_ids) {

	load_options_from_silo(fname);

	read_silo_var<real> rr;

	DBfile* db = CALL_SILO(DBOpenReal, fname.c_str(), DB_PDB, DB_READ);
	const real dtime = rr(db, "time");
	const real rtime = rr(db, "rotational_time");
	CALL_SILO(DBClose, db);

	std::vector<hpx::future<void>> futs;
	const auto me = hpx::find_here();
	static const auto names = grid::get_field_names();
	for (const auto& i : node_ids) {
		node_location l;
		l.from_id(i);
		futs.push_back(hpx::new_<node_server>(me, l).then([&me,l,&fname](hpx::future<hpx::id_type>&& f) {
			const auto id = f.get();
			auto client = node_client(id);
			load_registry::put(l.to_id(), id);
			const auto pid = load_registry::get(l.get_parent().to_id());
			node_server* node_ptr = node_registry::get(l);
			grid& g = node_ptr->get_hydro_grid();
			const auto suffix = std::to_string(l.to_id());
			DBfile* db = CALL_SILO(DBOpenReal, fname.c_str(), DB_PDB, DB_READ);
			assert(db);
			std::vector<std::pair<std::string,real>> outflow;
			for( auto n : names ) {
				const auto name = n + std::string( "_") + suffix;
				const auto quadvar = CALL_SILO(DBGetQuadvar,db,name.c_str());
				g.set(quadvar->name, static_cast<real*>(*(quadvar->vals)));
				CALL_SILO(DBFreeQuadvar,quadvar);
				std::string name_outflow = n + std::string("_outflow_") + suffix;
				const real o = read_silo_var<real>()(db, name_outflow.c_str());
				outflow.push_back(std::make_pair(std::move(name_outflow), o));
			}
			g.set_outflows(std::move(outflow));
			CALL_SILO(DBClose, db);
			node_client p(pid);
			p.notify_parent(l,id).get();
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	for (auto i = node_registry::begin(); i != node_registry::end(); i++) {
		i->second->set_time(dtime, rtime);
	}
}

void load_options_from_silo(std::string fname) {
	DBfile* db = CALL_SILO(DBOpenReal, fname.c_str(), DB_PDB, DB_READ);
	assert(db);
	read_silo_var<integer> ri;
	read_silo_var<real> rr;

	opts.eos = eos_type(ri(db, "eos"));
	opts.gravity = ri(db, "gravity");
	opts.hydro = ri(db, "hydro");
	opts.omega = rr(db, "omega");
	opts.output_dt = rr(db, "output_frequency");
	opts.problem = problem_type(ri(db, "problem"));
	opts.radiation = ri(db, "radiation");
	opts.refinement_floor = rr(db, "refinement_floor");
	opts.xscale = rr(db, "xscale");
	if (db != NULL) {
		CALL_SILO(DBClose, db);
	} else {
		std::cout << "Could not load " << fname;
		throw;
	}
	if (hpx::get_locality_id() == 0) {
		grid::set_omega(opts.omega);
	}

}

hpx::id_type load_data_from_silo(std::string fname, node_server* root_ptr, hpx::id_type root) {
	load_registry::put(1, root);
	static int sz = localities.size();
	DBfile* db = CALL_SILO(DBOpenReal, fname.c_str(), DB_PDB, DB_READ);
	epoch = read_silo_var<integer>()(db, "epoch");
	epoch++;
	if (db != NULL) {
		DBmultimesh* master_mesh = CALL_SILO(DBGetMultimesh, db, "mesh");
		CALL_SILO(DBClose, db);
		std::vector<node_location::node_id> work;
		std::vector<hpx::future<void>> futs;
		const int chunk_size = master_mesh->nblocks / sz;
		for (int i = 0; i < master_mesh->nblocks; i++) {
			work.push_back(std::stoi(master_mesh->meshnames[i]));
			const int this_id = i / chunk_size;
			const int next_id = (i + 1) / chunk_size;
			assert(this_id < localities.size());
			if (this_id != next_id) {
				futs.push_back(hpx::async<local_load_action>(localities[this_id], fname, work));
				work.clear();
			}
		}
		CALL_SILO(DBFreeMultimesh, master_mesh);
		for (auto& f : futs) {
			f.get();
		}
	} else {
		std::cout << "Could not load " << fname;
		throw;
	}
	hpx::id_type rc = load_registry::get(1);
	load_registry::destroy();
	root_ptr->form_tree(root);
	hpx::lcos::broadcast<all_boundaries_action>(localities).get();
	return std::move(rc);
}

silo_var_t::silo_var_t(const std::string& name) :
		name_(name), data_(INX * INX * INX) {
}

double& silo_var_t::operator()(int i) {
	return data_[i];
}

double silo_var_t::operator()(int i) const {
	return data_[i];
}
