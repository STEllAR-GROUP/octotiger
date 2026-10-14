/*
 * silo.cpp
 *
 *  Created on: Oct 16, 2018
 *      Author: dmarce1
 */

#include <hpx/config/compiler_specific.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)

//101 - fixed units bug in momentum
#include "octotiger/io/silo.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/util.hpp"

#include "octotiger/node_registry.hpp"

#include <hpx/collectives/broadcast.hpp>
#include <hpx/io_service/io_service_pool.hpp>

#include <cstddef>
#include <stddef.h>
#include <cmath>
#include <future>
#include <stdexcept>
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
		int one = 1;
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
static integer start_step = 0;
static int timestamp;
static int steps_elapsed;
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
			opts().eos = eos_type(ri(db, "eos"));
			opts().gravity = ri(db, "gravity");
			opts().hydro = ri(db, "hydro");
			opts().omega = rr(db, "omega") * opts().code_to_s;
			opts().output_dt = rr(db, "output_frequency");
			opts().problem = problem_type(ri(db, "problem"));
			opts().radiation = DBInqVarExists(db, "radiation") && ri(db, "radiation") != 0;
			// Checkpoints predating the opt-in reduced-speed model used full c.
			// Saved physics takes precedence over a conflicting restart command line.
			opts().radiation_reduced_c_factor = 1;
			// Missing model metadata denotes the original M1 Euler scheme.
			opts().radiation_model = "m1";
			opts().radiation_integrator = "euler";
			if (DBInqVarExists(db, "radiation_integrator")) {
				const integer method = ri(db, "radiation_integrator");
				if (method != 0 && method != 3) {
					if (!leaveopen) {
						DBClose(db);
					}
					throw std::runtime_error("Unsupported radiation restart integrator identifier");
				}
				opts().radiation_integrator = method == 0 ? "euler" : "backward_euler";
			}
			if (DBInqVarExists(db, "periodic")) {
				opts().periodic = ri(db, "periodic") != 0;
			}
			if (opts().radiation) {
				if (DBInqVarExists(db, "radiation_options_version")) {
					const integer radiation_version = ri(db, "radiation_options_version");
					if (radiation_version < 1 || radiation_version > 3) {
						if (!leaveopen) {
							DBClose(db);
						}
						throw std::runtime_error("Unsupported radiation restart metadata version");
					}
					if (radiation_version >= 2) {
						if (!DBInqVarExists(db, "radiation_reduced_c_factor")) {
							if (!leaveopen) {
								DBClose(db);
							}
							throw std::runtime_error("Missing radiation restart option radiation_reduced_c_factor");
						}
						const real factor = rr(db, "radiation_reduced_c_factor");
						if (!std::isfinite(factor) || !(factor > 0 && factor <= 1)) {
							if (!leaveopen) {
								DBClose(db);
							}
							throw std::runtime_error("Invalid radiation restart option radiation_reduced_c_factor: expected finite 0 < factor <= 1");
						}
						opts().radiation_reduced_c_factor = factor;
					}
					if (radiation_version == 3) {
						try {
							for (const char *name : {"radiation_model", "radiation_integrator",
									"radiation_algorithm_version", "radiation_limiter", "radiation_flux_semantics"}) {
								if (!DBInqVarExists(db, name)) {
									throw std::runtime_error(std::string("Missing radiation restart option ") + name);
								}
							}
							const integer model = ri(db, "radiation_model");
							const integer algorithm = ri(db, "radiation_algorithm_version");
							if ((model != 0 && model != 1) || algorithm != (model == 1 && opts().hydro ? 2 : 1) ||
								ri(db, "radiation_limiter") != model ||
								ri(db, "radiation_flux_semantics") != (model == 1 && algorithm == 2 ? 2 : model) ||
								ri(db, "radiation_integrator") != (model == 0 ? 0 : 3)) {
								throw std::runtime_error("Unsupported or inconsistent radiation restart model metadata");
							}
							opts().radiation_model = model == 0 ? "m1" : "fld";
							opts().radiation_limiter = "levermore_pomraning";
							if (model == 1) {
								if (algorithm == 2) {
									for (const char* name : {"fld_hydro_unigrid", "fld_hydro_reflect_bc", "fld_hydro_cfl"}) {
										if (!DBInqVarExists(db, name)) {
											throw std::runtime_error(std::string("Missing moving FLD restart option ") + name);
										}
									}
									opts().unigrid = ri(db, "fld_hydro_unigrid") != 0;
									opts().reflect_bc = ri(db, "fld_hydro_reflect_bc") != 0;
									opts().cfl = rr(db, "fld_hydro_cfl");
								}
								for (const char *name : {"fld_boundary", "fld_nonlinear_tolerance", "fld_linear_tolerance",
										"fld_max_iterations", "fld_max_linear_iterations"}) {
									if (!DBInqVarExists(db, name)) {
										throw std::runtime_error(std::string("Missing FLD restart option ") + name);
									}
								}
								const integer boundary = ri(db, "fld_boundary");
								if (boundary != 0 && boundary != 1) {
									throw std::runtime_error("Unsupported FLD restart boundary");
								}
								opts().fld_boundary = boundary == 0 ? "reflecting" : "vacuum";
								opts().fld_nonlinear_tolerance = rr(db, "fld_nonlinear_tolerance");
								opts().fld_linear_tolerance = rr(db, "fld_linear_tolerance");
								opts().fld_max_iterations = ri(db, "fld_max_iterations");
								opts().fld_max_linear_iterations = ri(db, "fld_max_linear_iterations");
							}
						} catch (...) {
							if (!leaveopen) {
								DBClose(db);
							}
							throw;
						}
					}
					// A partial new-format checkpoint must not silently alter the model.
					for (const char *name : {"periodic", "rad_implicit", "rad_test_chi",
							"rad_test_width", "rad_test_background", "rad_test_amplitude",
							"rad_test_luminosity", "rad_reference"}) {
						if (!DBInqVarExists(db, name)) {
							if (!leaveopen) {
								DBClose(db);
							}
							throw std::runtime_error(std::string("Missing radiation restart option ") + name);
						}
					}
					opts().rad_implicit = ri(db, "rad_implicit") != 0;
					opts().radTestChi = rr(db, "rad_test_chi");
					opts().radTestWidth = rr(db, "rad_test_width");
					opts().radTestBackground = rr(db, "rad_test_background");
					opts().radTestAmplitude = rr(db, "rad_test_amplitude");
					opts().radTestLuminosity = rr(db, "rad_test_luminosity");
					const int reference_length = DBGetVarLength(db, "rad_reference");
					if (reference_length <= 0) {
						if (!leaveopen) {
							DBClose(db);
						}
						throw std::runtime_error("Invalid radiation restart reference filename");
					}
					std::vector<char> reference(reference_length);
					if (DBReadVar(db, "rad_reference", reference.data()) != 0 || reference.back() != '\0') {
						if (!leaveopen) {
							DBClose(db);
						}
						throw std::runtime_error("Cannot read radiation restart reference filename");
					}
					opts().radReference.assign(reference.data(), reference.size() - 1);
				} else {
					std::cerr << "WARNING: Legacy radiation restart lacks transport/source options; "
						"using periodic, rad_implicit, and radiation parameters from the command line/configuration.\n";
				}
			}
			opts().refinement_floor = rr(db, "refinement_floor");
			if (DBInqVarExists(db, "gradient_amr_options_version")) {
				try {
					if (ri(db, "gradient_amr_options_version") != 1) {
						throw std::runtime_error("Unsupported gradient AMR restart metadata version");
					}
					for (const char* name : {"gradient_refine_field", "gradient_refine_mode", "gradient_refine_threshold",
							"gradient_refine_floor", "gradient_refine_coarsen_factor", "gradient_refine_buffer",
							"gradient_min_level", "gradient_max_level"}) {
						if (!DBInqVarExists(db, name)) {
							throw std::runtime_error(std::string("Missing gradient AMR restart option ") + name);
						}
					}
					const integer field = ri(db, "gradient_refine_field"), mode = ri(db, "gradient_refine_mode");
					if ((field != 1 && field != 2) || (mode != 0 && mode != 1)) {
						throw std::runtime_error("Invalid gradient AMR restart field or mode");
					}
					opts().gradient_refine_field = field == 1 ? "density" : "radiation_energy";
					opts().gradient_refine_mode = mode == 0 ? "absolute" : "relative";
					opts().gradient_refine_threshold = rr(db, "gradient_refine_threshold");
					opts().gradient_refine_floor = rr(db, "gradient_refine_floor");
					opts().gradient_refine_coarsen_factor = rr(db, "gradient_refine_coarsen_factor");
					opts().gradient_refine_buffer = ri(db, "gradient_refine_buffer");
					opts().min_level = ri(db, "gradient_min_level");
					opts().max_level = ri(db, "gradient_max_level");
					opts().unigrid = false;
					opts().validateGradientRefinement();
				} catch (...) {
					if (!leaveopen) {
						DBClose(db);
					}
					throw;
				}
			}
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
	DBClose(db_);
}

HPX_PLAIN_ACTION(load_close, load_close_action);
HPX_PLAIN_ACTION(load_open, load_open_action);

// The root can itself be a saved leaf. Use exactly the same field/unit and
// outflow restoration for it as for leaves created while rebuilding a tree.
static void load_silo_leaf(grid& target, const node_location& loc, const std::string& filename) {
	silo_load_t load;
	const auto hydro_names = grid::get_hydro_field_names();
	auto field_names = hydro_names;
	if (opts().radiation) {
		const auto radiation_names = rad_grid::get_field_names();
		field_names.insert(field_names.end(), radiation_names.begin(), radiation_names.end());
	}
	load.vars.resize(field_names.size());
	load.outflows.resize(hydro_names.size());
	hpx::threads::run_as_os_thread([&]() {
		static std::mutex mtx;
		std::lock_guard<std::mutex> lock(mtx);
		const auto& this_file = filename;
		DBfile *db = DBOpenReal(this_file.c_str(), DB_UNKNOWN, DB_READ);
		if (db == NULL) {
			printf("Unable to open SILO file %s\n", this_file.c_str());
			abort();
		}
		const std::string suffix = oct_to_str(loc.to_id());
		for (size_t f = 0; f != field_names.size(); f++) {
			const auto this_name = suffix + std::string("/") + field_names[f];
			auto var = DBGetQuadvar(db, this_name.c_str());
			if (var == nullptr) {
				DBClose(db);
				throw std::runtime_error("Missing restart field " + this_name);
			}
			load.nx = var->dims[0];
			const int nvar = load.nx * load.nx * load.nx;
			load.vars[f].first = field_names[f];
			load.vars[f].second.resize(nvar);
			if (f < hydro_names.size()) {
				read_silo_var<real> rd;
				load.outflows[f].first = field_names[f];
				load.outflows[f].second = rd(db, outflow_name(this_name).c_str());
			}
			std::memcpy(load.vars[f].second.data(), var->vals[0], sizeof(real) * nvar);
			DBFreeQuadvar(var);
		}
		DBClose(db);
	}).get();
	for (auto &field : load.vars) {
		target.set(field.first, field.second.data(), version_);
	}
	for (auto &outflow : load.outflows) {
		target.set_outflow(std::move(outflow));
	}
	target.rho_from_species();
}

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
		load_silo_leaf(*grid_ptr, loc, iter->second.filename);
		is_refined = false;
	}
	current_time = silo_output_time();
	rotational_time = silo_output_rotation_time();
}

node_server::node_server(const node_location &loc, silo_load_t load) :
		my_location(loc) {
//	printf("Distributing %s on %i\n", loc.to_str().c_str(), int(hpx::get_locality_id()));
	const auto &localities = opts().all_localities;
	initialize(0.0, 0.0);
	step_num = gcycle = hcycle = rcycle = 0;
	int nc = 0;
	is_refined = false;
	for (auto &field : load.vars) {
		grid_ptr->set(field.first, field.second.data(), version_);
	}
	for (auto &outflow : load.outflows) {
		grid_ptr->set_outflow(std::move(outflow));
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

void load_data_from_silo(std::string fname, std::shared_ptr<node_server> root_ptr, hpx::id_type root) {
	timings::scope ts(root_ptr->timings_, timings::time_io);
	printf( "Reading %s\n", fname.c_str());
	const auto tstart = time(NULL);

	const integer nprocs = opts().all_localities.size();
	static int sz = localities.size();
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
		const int chunk_size = std::ceil(real(master_mesh->nblocks) / real(sz));
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
    const auto root_entry = node_dir_.find(my_location.to_id());
    if (root_entry == node_dir_.end()) {
        throw std::runtime_error("Restart tree does not contain its root");
    }
    if (root_entry->second.load) {
        load_silo_leaf(*grid_ptr, my_location, root_entry->second.filename);
        is_refined = false;
    } else {
        std::vector<hpx::future<void>> futs;
        is_refined = true;
        for (integer ci = 0; ci < NCHILD; ci++) {
            futs.push_back(hpx::async([this, ci]() {
                auto cloc = my_location.get_child(ci);
                auto iter = node_dir_.find(cloc.to_id());
                children[ci] = hpx::new_<node_server>(localities[iter->second.locality_id], cloc);
            }));
        }
        hpx::when_all(futs).get();
    }
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
#endif
