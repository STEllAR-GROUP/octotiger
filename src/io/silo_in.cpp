/*
 * silo.cpp
 *
 *  Created on: Oct 16, 2018
 *      Author: dmarce1
 */

//101 - fixed units bug in momentum
#include "octotiger/io/silo.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/util.hpp"

#include "octotiger/node_registry.hpp"

#include <hpx/collectives/broadcast.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <future>
#include <mutex>
#include <map>
#include <vector>

static int version_;

static const auto &localities = options::all_localities;

static std::mutex silo_mtx_;

#include <hpx/include/threads.hpp>
#include <hpx/include/run_as.hpp>

template<class T>
struct read_silo_var {
	T operator()(DBfile *db, const char *name) const {
		T var;
		if (DBReadVar(db, name, &var) != 0) {
			std::cout << "Unable to read " << name << " \n";
		}
		return var;
	}
};

struct node_entry_t {
	bool load;
	integer position;
	integer locality_id;
	std::string filename;
	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & load;
		arc & position;
		arc & locality_id;
		arc & filename;
	}
};

using dir_map_type = std::unordered_map<node_location::node_id, node_entry_t>;

static time_t start_time = time(nullptr);
static DBfile *db_;
static dir_map_type node_dir_;

#define SILO_TEST(i) \
	if( i != 0 ) printf( "SILO call failed at %i\n", __LINE__ );

void load_options_from_silo(std::string fname, DBfile *db) {
	const auto func = [&fname, &db]() {
		bool leaveopen;
		if (db == nullptr) {
			db = DBOpenReal(fname.c_str(), DB_UNKNOWN, DB_READ);
			leaveopen = false;
		} else {
			leaveopen = true;
		}
		if (db != nullptr) {

			read_silo_var<integer> ri;
			read_silo_var<real> rr;
			integer version = ri(db, "version");
			if (version > SILO_VERSION) {
				printf("WARNING: Attempting to load a version %i SILO file, maximum version allowed for this Octo-tiger is %i\n", int(version), SILO_VERSION);
			}
			if (version == 100) {
				printf("Reading version 100 SILO - correcting momentum units\n");
			}
			version_ = version;
			opts().code_to_g = rr(db, "code_to_g");
			opts().code_to_s = rr(db, "code_to_s");
			opts().code_to_cm = rr(db, "code_to_cm");
			opts().n_species = ri(db, "n_species");
	//		opts().eos = eos_type(ri(db, "eos"));
			opts().gravity = ri(db, "gravity");
			opts().hydro = ri(db, "hydro");
			opts().omega = rr(db, "omega") * opts().code_to_s;
			opts().output_dt = rr(db, "output_frequency");
			opts().problem = problem_type(ri(db, "problem"));
			opts().radiation = ri(db, "radiation");
			opts().refinement_floor = rr(db, "refinement_floor");
			opts().xscale = rr(db, "xscale");
			opts().atomic_number.resize(opts().n_species);
			opts().atomic_mass.resize(opts().n_species);
			opts().X.resize(opts().n_species);
			opts().Z.resize(opts().n_species);
			SILO_TEST(DBReadVar(db, "atomic_number", opts().atomic_number.data()));
			SILO_TEST(DBReadVar(db, "atomic_mass", opts().atomic_mass.data()));
			SILO_TEST(DBReadVar(db, "X", opts().X.data()));
			SILO_TEST(DBReadVar(db, "Z", opts().Z.data()));
			if (!leaveopen) {
				DBClose(db);
			}
		} else {
			std::cout << "Could not load " << fname;
			throw;
		}
	};
	if (db == nullptr) {
		GET(hpx::threads::run_as_os_thread(func));
	} else {
		func();
	}
	grid::set_omega(opts().omega, false);
	set_units(1. / opts().code_to_g, 1. / opts().code_to_cm, 1. / opts().code_to_s, 1); /**/

}

void load_open(std::string fname, dir_map_type map) {
//	printf("LOAD OPENED on proc %i\n", hpx::get_locality_id());
	load_options_from_silo(fname, db_); /**/
	hpx::threads::run_as_os_thread([&]() {
		db_ = DBOpenReal(fname.c_str(), DB_UNKNOWN, DB_READ);
		read_silo_var<real> rr;
		silo_output_time() = rr(db_, "cgs_time"); /**/
		silo_output_rotation_time() = 2 * M_PI * rr(db_, "rotational_time"); /**/
//		printf("rotational_time = %e\n", silo_output_rotation_time());
		silo_output_time() /= opts().code_to_s;
		node_dir_ = std::move(map);
	//	printf("%e\n", silo_output_time());
//		sleep(100);
	}).get();
}

void load_close() {
	grid::set_units();
	DBClose(db_);
}

HPX_PLAIN_ACTION(load_close, load_close_action);
HPX_PLAIN_ACTION(load_open, load_open_action);

node_server::node_server(const node_location &loc) :
		my_location(loc) {
	const auto &localities = opts().all_localities;
	initialize(0.0, 0.0);
	step_num = gcycle = hcycle = rcycle = 0;

	auto iter = node_dir_.find(loc.to_id());
	assert(iter != node_dir_.end());

	if (!iter->second.load) {
//		printf("Creating %s on %i\n", loc.to_str().c_str(), int(hpx::get_locality_id()));
		std::atomic<int> nc(0);
		std::vector<hpx::future<void>> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs.push_back(hpx::async([this, ci, &nc, &loc, &localities]() {
				auto cloc = loc.get_child(ci);
				auto iter = node_dir_.find(cloc.to_id());
				if (iter != node_dir_.end()) {
					is_refined = true;
					children[ci] = hpx::new_<node_server>(localities[iter->second.locality_id], cloc);
					nc++;
				}
			}));
		}
		GET(hpx::when_all(futs));
		assert(nc == 0 || nc == NCHILD);
	} else {
	//	printf("Loading %s on %i\n", loc.to_str().c_str(), int(hpx::get_locality_id()));
		silo_load_t load;
		static const auto hydro_names = grid::get_hydro_field_names();
		load.vars.resize(hydro_names.size());
		load.outflows.resize(hydro_names.size());
		hpx::threads::run_as_os_thread([&]() {
			static std::mutex mtx;
			std::lock_guard<std::mutex> lock(mtx);
			const auto this_file = iter->second.filename;
			DBfile *db = DBOpenReal(this_file.c_str(), DB_UNKNOWN, DB_READ);
			if (db == NULL) {
				printf("Unable to open SILO file %s\n", this_file.c_str());
				abort();
			}
			const std::string suffix = oct_to_str(loc.to_id());
			for (int f = 0; f != hydro_names.size(); f++) {
				const auto this_name = suffix + std::string("/") + hydro_names[f]; /**/
				auto var = DBGetQuadvar(db, this_name.c_str());
				load.nx = var->dims[0];
				const int nvar = load.nx * load.nx * load.nx;
				load.outflows[f].first = load.vars[f].first = hydro_names[f];
				load.vars[f].second.resize(nvar);
				read_silo_var<real> rd;
				load.outflows[f].second = rd(db, outflow_name(this_name).c_str());
				std::memcpy(load.vars[f].second.data(), var->vals[0], sizeof(real) * nvar);
				DBFreeQuadvar(var);
			}
			DBClose(db);
		}).get();
		is_refined = false;
		for (integer f = 0; f < hydro_names.size(); f++) {
			grid_ptr->set(load.vars[f].first, load.vars[f].second.data(), version_);
			grid_ptr->set_outflow(std::move(load.outflows[f]));
		}
		grid_ptr->rho_from_species();
	}
	current_time = silo_output_time();
	rotational_time = silo_output_rotation_time();
}

node_server::node_server(const node_location &loc, silo_load_t load) :
		my_location(loc) {
//	printf("Distributing %s on %i\n", loc.to_str().c_str(), int(hpx::get_locality_id()));
	initialize(0.0, 0.0);
	step_num = gcycle = hcycle = rcycle = 0;
	static const auto hydro_names = grid::get_hydro_field_names();
	is_refined = false;
	for (integer f = 0; f < hydro_names.size(); f++) {
		grid_ptr->set(load.vars[f].first, load.vars[f].second.data(), version_);
		grid_ptr->set_outflow(std::move(load.outflows[f]));
	}
	grid_ptr->rho_from_species();
	current_time = silo_output_time();
	rotational_time = silo_output_rotation_time();
	assert(nc == 0 || nc == NCHILD);
}

auto split_mesh_id(const std::string id) {
	std::pair<node_location::node_id, std::string> rc;
	int i = 0;
	for (i = 0; id[i] != ':'; i++) {
		rc.second.push_back(id[i]);
	}
	i += 2;
	std::string tmp;
	for (; i != id.size(); i++) {
		tmp.push_back(id[i]);
	}
	rc.first = std::strtoll(tmp.c_str(), nullptr, 8);
//	printf("%li %s\n", rc.first, rc.second.c_str());
	return rc;
}

void load_data_from_silo(std::string fname, node_server *root_ptr, hpx::id_type root) {
	timings::scope ts(root_ptr->timings_, timings::time_io);
	printf( "Reading %s\n", fname.c_str());
	const auto tstart = time(NULL);

	const integer nprocs = opts().all_localities.size();
	DBfile *db = GET(hpx::threads::run_as_os_thread(DBOpenReal, fname.c_str(), DB_UNKNOWN, DB_READ));
	silo_epoch() = GET(hpx::threads::run_as_os_thread(read_silo_var<integer>(), db, "epoch"));
	silo_epoch()++;std
	::vector<node_location::node_id> node_list;
	std::vector<integer> positions;
	std::vector<hpx::future<void>> futs;
	int node_count;
	if (db != nullptr) {
		DBmultimesh *master_mesh = GET(hpx::threads::run_as_os_thread([&]() {
			return DBGetMultimesh(db, "quadmesh");
		}));
		hpx::threads::run_as_os_thread([&]() {
			const read_silo_var<integer> ri;
			node_count = ri(db, "node_count");
			node_list.resize(node_count);
			positions.resize(node_count);
			DBReadVar(db, "node_list", node_list.data());
			DBReadVar(db, "node_positions", positions.data());
		}).get();
		GET(hpx::threads::run_as_os_thread(DBClose, db));
		std::map<node_location::node_id, std::string> load_locs;
		for (int i = 0; i < master_mesh->nblocks; i++) {
			load_locs.insert(split_mesh_id(master_mesh->meshnames[i]));
		}
		for (int i = 0; i < node_list.size(); i++) {
			node_entry_t entry;
			entry.position = positions[i];
			const auto tmp = load_locs.find(node_list[i]);
			entry.load = bool(tmp != load_locs.end());
			entry.locality_id = positions[i] * nprocs / positions.size();
			if (entry.load) {
				entry.filename = tmp->second;
			} else {
				entry.filename = "";
			}
			node_dir_[node_list[i]] = entry;
		}
		auto this_dir = std::move(node_dir_);
		for (int i = 0; i < nprocs; i++) {
	//		printf("Sending LOAD OPEN to %i\n", i);
			futs.push_back(hpx::async < load_open_action > (opts().all_localities[i], fname, this_dir));
		}
		GET(hpx::threads::run_as_os_thread(DBFreeMultimesh, master_mesh));
		for (auto &f : futs) {
			GET(f);
		}
	} else {
		std::cout << "Could not load " << fname;
		throw;
	}
	root_ptr->reconstruct_tree();
	node_registry::clear();
	futs.clear();
	for (int i = 0; i < nprocs; i++) {
		futs.push_back(hpx::async < load_close_action > (opts().all_localities[i]));
	}
	for (auto &f : futs) {
		GET(f);
	}

	const auto tstop = time(NULL);
	printf( "Read took %li seconds\n", tstop - tstart);
}

void node_server::reconstruct_tree() {
	std::vector<hpx::future<void>> futs;
	is_refined = true;
	for (integer ci = 0; ci < NCHILD; ci++) {
		futs.push_back(hpx::async([this, ci]() {
			auto cloc = my_location.get_child(ci);
			auto iter = node_dir_.find(cloc.to_id());
			children[ci] = hpx::new_<node_server>(localities[iter->second.locality_id], cloc);
		})
		);
	}
	hpx::when_all(futs).get();
	current_time = silo_output_time();
	rotational_time = silo_output_rotation_time();
}

silo_var_t::silo_var_t(const std::string &name, std::size_t nx) :
		name_(name), data_(nx * nx * nx) {
	range_.first = +std::numeric_limits<real>::max();
	range_.second = -std::numeric_limits<real>::max();
}

double&
silo_var_t::operator()(int i) {
	return data_[i];
}

double silo_var_t::operator()(int i) const {
	return data_[i];
}
