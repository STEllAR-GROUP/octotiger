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
#include <memory>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <map>
#include <vector>
#include <algorithm>
#include <set>

static int version_;

static const auto &localities = options::all_localities;

static std::mutex silo_mtx_;

#include <hpx/include/threads.hpp>
#include <hpx/include/run_as.hpp>

template<class T>
struct read_silo_var {
	T operator()(DBfile *db, const char *name) const {
        if (silo_restart::present(db)) {
            return silo_restart::read_array<T>(db, name, 1)[0];
        }
		T var{};
		if (DBReadVar(db, name, &var) != 0) {
            throw std::runtime_error(std::string("Unable to read Silo variable ") + name);
		}
		return var;
	}
};

struct node_entry_t {
	bool load;
    bool exact = false;
	integer position;
	integer locality_id;
	std::string filename;
	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & load;
        arc & exact;
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
static bool exact_restart_ = false;

static std::vector<char> read_node_checkpoint(const node_location& location,
        const node_entry_t& entry) {
    return hpx::threads::run_as_os_thread([&]() {
        std::lock_guard<std::mutex> lock(silo_mtx_);
        DBfile* db = DBOpenReal(entry.filename.c_str(), DB_UNKNOWN, DB_READ);
        if (db == nullptr) {
            throw std::runtime_error("Unable to open checkpoint data file " + entry.filename);
        }
        std::vector<char> state;
        try {
            state = silo_restart::read_state(db, location.to_id());
        } catch (...) {
            DBClose(db);
            throw;
        }
        silo_restart::check(DBClose(db), "unable to close checkpoint data file");
        return state;
    }).get();
}

#define SILO_TEST(i) \
	if( i != 0 ) printf( "SILO call failed at %i\n", __LINE__ );

void load_options_from_silo(std::string fname, DBfile *db, bool restore_integrator) {
	const auto func = [&fname, &db, restore_integrator]() {
        const bool leaveopen = db != nullptr;
        std::unique_ptr<DBfile, decltype(&DBClose)> owned(nullptr, &DBClose);
        if (!leaveopen) {
            db = DBOpenReal(fname.c_str(), DB_UNKNOWN, DB_READ);
            owned.reset(db);
        }
		if (db != nullptr) {

            const bool exact = silo_restart::present(db);
            read_silo_var<integer> ri;
            read_silo_var<real> rr;
            opts().radiation = DBInqVarExists(db, "radiation") > 0 && ri(db, "radiation") != 0;
            if (exact && opts().radiation) {
                throw std::runtime_error("SSP_MULTISTEP exact checkpoints do not support radiation");
            }
            std::vector<integer> exact_layout;
            if (exact) {
                const auto version = silo_restart::read_array<integer>(db, "restart_version", 1)[0];
                exact_layout = silo_restart::read_array<integer>(db, "restart_layout", 7);
                if (restore_integrator && (version == 1 || version == silo_restart::format_version)) {
                    opts().time_integrator = SSP_MULTISTEP;
                }
                if ((version != 1 && version != silo_restart::format_version) || opts().time_integrator != SSP_MULTISTEP) {
                    throw std::runtime_error("Exact checkpoint requires supported format 1 or 2 and --time_integrator=SSP_MULTISTEP");
                }
                if (exact_layout[0] != NDIM || exact_layout[1] != INX || exact_layout[2] != H_NX ||
                        exact_layout[4] <= 0 || exact_layout[4] > 1024 ||
                        exact_layout[3] != exact_layout[4] + 10 ||
                        exact_layout[5] != integer(sizeof(real)) || exact_layout[6] != SSP_MULTISTEP) {
                    throw std::runtime_error("Exact checkpoint has an incompatible field layout");
                }
            }

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
            if (exact && opts().n_species != exact_layout[4]) {
                throw std::runtime_error("Exact checkpoint species metadata is inconsistent");
            }
            opts().n_fields = opts().n_species + 10;
			opts().eos = eos_type(ri(db, "eos"));
			opts().gravity = ri(db, "gravity");
			opts().hydro = ri(db, "hydro");
			opts().omega = rr(db, "omega") * opts().code_to_s;
			opts().output_dt = rr(db, "output_frequency");
            opts().problem = problem_type_from_checkpoint(ri(db, "problem"), opts().radiation);
			// Checkpoints predating the opt-in reduced-speed model used full c.
			// Saved physics takes precedence over a conflicting restart command line.
			opts().radiation_reduced_c_factor = 1;
			// Missing model metadata denotes the original M1 Euler scheme.
			opts().radiation_model = "m1";
			opts().radiation_integrator = "euler";
			if (DBInqVarExists(db, "radiation_integrator")) {
				const integer method = ri(db, "radiation_integrator");
				if (method != 0 && method != 3) {
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
						throw std::runtime_error("Unsupported radiation restart metadata version");
					}
					if (radiation_version >= 2) {
						if (!DBInqVarExists(db, "radiation_reduced_c_factor")) {
							throw std::runtime_error("Missing radiation restart option radiation_reduced_c_factor");
						}
						const real factor = rr(db, "radiation_reduced_c_factor");
						if (!std::isfinite(factor) || !(factor > 0 && factor <= 1)) {
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
							throw;
						}
					}
					// A partial new-format checkpoint must not silently alter the model.
					for (const char *name : {"periodic", "rad_implicit", "rad_test_chi",
							"rad_test_width", "rad_test_background", "rad_test_amplitude",
							"rad_test_luminosity", "rad_reference"}) {
						if (!DBInqVarExists(db, name)) {
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
						throw std::runtime_error("Invalid radiation restart reference filename");
					}
					std::vector<char> reference(reference_length);
					if (DBReadVar(db, "rad_reference", reference.data()) != 0 || reference.back() != '\0') {
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
					throw;
				}
			}
			opts().xscale = rr(db, "xscale");
			opts().atomic_number.resize(opts().n_species);
			opts().atomic_mass.resize(opts().n_species);
			opts().X.resize(opts().n_species);
			opts().Z.resize(opts().n_species);
            if (exact) {
                opts().atomic_number = silo_restart::read_array<real>(db, "atomic_number", opts().n_species);
                opts().atomic_mass = silo_restart::read_array<real>(db, "atomic_mass", opts().n_species);
                opts().X = silo_restart::read_array<real>(db, "X", opts().n_species);
                opts().Z = silo_restart::read_array<real>(db, "Z", opts().n_species);
            } else {
                SILO_TEST(DBReadVar(db, "atomic_number", opts().atomic_number.data()));
                SILO_TEST(DBReadVar(db, "atomic_mass", opts().atomic_mass.data()));
                SILO_TEST(DBReadVar(db, "X", opts().X.data()));
                SILO_TEST(DBReadVar(db, "Z", opts().Z.data()));
            }
            if (!leaveopen) {
                silo_restart::check(DBClose(owned.release()), "unable to close Silo option file");
            }
        } else {
            throw std::runtime_error("Could not load " + fname);
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
	load_options_from_silo(fname); /**/
	hpx::threads::run_as_os_thread([&]() {
		db_ = DBOpenReal(fname.c_str(), DB_UNKNOWN, DB_READ);
        if (db_ == nullptr) {
            throw std::runtime_error("Unable to open checkpoint master file " + fname);
        }
        exact_restart_ = silo_restart::present(db_);
        if (exact_restart_) {
            silo_restart::restore_metadata(db_);
            node_dir_ = std::move(map);
            return;
        }
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
    silo_restart::check(DBClose(db_), "unable to close checkpoint master file");
    db_ = nullptr;
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
    if (iter == node_dir_.end()) {
        throw std::runtime_error("Checkpoint is missing node " + oct_to_str(loc.to_id()));
    }
    if (iter->second.exact) {
        load_checkpoint(read_node_checkpoint(loc, iter->second));
        if (bool(is_refined) == iter->second.load) {
            throw std::runtime_error("Checkpoint node refinement flag disagrees with its mesh topology");
        }
        if (is_refined) {
            std::vector<hpx::future<void>> children_ready;
            for (integer ci = 0; ci < NCHILD; ++ci) {
                children_ready.push_back(hpx::async([this, ci, loc]() {
                    const auto child_location = loc.get_child(ci);
                    const auto child_entry = node_dir_.find(child_location.to_id());
                    if (child_entry == node_dir_.end()) {
                        throw std::runtime_error("Checkpoint refined node has a missing child");
                    }
                    children[ci] = hpx::new_<node_server>(
                        opts().all_localities[child_entry->second.locality_id], child_location);
                }));
            }
            for (auto& ready : children_ready) {
                ready.get();
            }
        }
        return;
    }

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

auto split_mesh_id(const std::string& id) {
    const auto separator = id.find(":/");
    const auto slash = separator == std::string::npos ? std::string::npos : id.find('/', separator + 2);
    if (separator == std::string::npos || slash == std::string::npos || slash == separator + 2) {
        throw std::runtime_error("Malformed Silo mesh reference: " + id);
    }
    const std::string digits = id.substr(separator + 2, slash - separator - 2);
    if (digits.find_first_not_of("01234567") != std::string::npos) {
        throw std::runtime_error("Malformed Silo node identifier: " + digits);
    }
    const auto node = static_cast<node_location::node_id>(std::stoull(digits, nullptr, 8));
    return std::make_pair(node, id.substr(0, separator));
}

void load_data_from_silo(std::string fname, std::shared_ptr<node_server> root_ptr, hpx::id_type root) {
    timings::scope ts(root_ptr->timings_, timings::time_io);
    printf("Reading %s\n", fname.c_str());
    const auto tstart = time(NULL);
    const integer nprocs = opts().all_localities.size();
    struct index_data {
        bool exact;
        std::vector<node_location::node_id> nodes;
        std::vector<integer> positions;
        std::vector<integer> groups;
        std::map<node_location::node_id, std::string> leaves;
    };
    const auto index = hpx::threads::run_as_os_thread([&]() {
        std::unique_ptr<DBfile, decltype(&DBClose)> db(
            DBOpenReal(fname.c_str(), DB_UNKNOWN, DB_READ), &DBClose);
        if (!db) {
            throw std::runtime_error("Unable to open Silo master file " + fname);
        }
        index_data data;
        data.exact = silo_restart::present(db.get());
        const integer count = data.exact
            ? silo_restart::read_array<integer>(db.get(), "node_count", 1)[0]
            : read_silo_var<integer>()(db.get(), "node_count");
        if (count <= 0 || count > 10000000) {
            throw std::runtime_error("Invalid checkpoint node count");
        }
        const size_t size = static_cast<size_t>(count);
        data.nodes.resize(size);
        if (data.exact) {
            // node_id is unsigned but the existing Silo field uses DB_LONG_LONG.
            static_assert(sizeof(node_location::node_id) == sizeof(integer), "node identifier size");
            const auto ids = silo_restart::read_array<integer>(db.get(), "node_list", size);
            std::memcpy(data.nodes.data(), ids.data(), size * sizeof(integer));
            data.positions = silo_restart::read_array<integer>(db.get(), "node_positions", size);
            data.groups = silo_restart::read_array<integer>(db.get(), "restart_groups", size);
        } else {
            data.positions.resize(size);
            silo_restart::check(DBReadVar(db.get(), "node_list", data.nodes.data()), "unable to read node_list");
            silo_restart::check(DBReadVar(db.get(), "node_positions", data.positions.data()), "unable to read node_positions");
        }
        std::set<node_location::node_id> unique_nodes(data.nodes.begin(), data.nodes.end());
        std::set<integer> unique_positions(data.positions.begin(), data.positions.end());
        if (unique_nodes.size() != size || unique_positions.size() != size ||
                *unique_positions.begin() != 0 || *unique_positions.rbegin() != count - 1) {
            throw std::runtime_error("Invalid checkpoint node/position mapping");
        }
        for (const auto node : unique_nodes) {
            auto encoded = node;
            integer level = 0;
            while (encoded > 1) {
                encoded >>= NDIM;
                ++level;
            }
            if (encoded != 1 || (data.exact && level > opts().max_level)) {
                throw std::runtime_error("Invalid checkpoint node identifier or refinement level");
            }
        }
        std::unique_ptr<DBmultimesh, decltype(&DBFreeMultimesh)> mesh(
            DBGetMultimesh(db.get(), "quadmesh"), &DBFreeMultimesh);
        if (!mesh || mesh->nblocks <= 0 || mesh->nblocks > count || mesh->meshnames == nullptr) {
            throw std::runtime_error("Invalid checkpoint mesh index");
        }
        for (int i = 0; i < mesh->nblocks; ++i) {
            if (mesh->meshnames[i] == nullptr) {
                throw std::runtime_error("Missing checkpoint mesh reference");
            }
            const auto entry = split_mesh_id(mesh->meshnames[i]);
            if (unique_nodes.count(entry.first) == 0 || !data.leaves.insert(entry).second) {
                throw std::runtime_error("Invalid checkpoint leaf mapping");
            }
        }
        silo_epoch() = data.exact
            ? silo_restart::read_array<integer>(db.get(), "epoch", 1)[0] + 1
            : read_silo_var<integer>()(db.get(), "epoch") + 1;
        return data;
    }).get();

    node_dir_.clear();
    for (size_t i = 0; i < index.nodes.size(); ++i) {
        node_entry_t entry;
        entry.position = index.positions[i];
        const auto leaf = index.leaves.find(index.nodes[i]);
        entry.load = leaf != index.leaves.end();
        entry.exact = index.exact;
        entry.locality_id = entry.position * nprocs / index.nodes.size();
        if (entry.exact) {
            if (index.groups[i] < 0 || index.groups[i] > std::numeric_limits<int>::max()) {
                throw std::runtime_error("Invalid checkpoint data group");
            }
            entry.filename = fname + ".data/" + std::to_string(index.groups[i]) + ".silo";
        } else if (entry.load) {
            entry.filename = leaf->second;
        }
        node_dir_.emplace(index.nodes[i], std::move(entry));
    }
    // Validate the full tree before creating any distributed components.
    const node_location root_location;
    if (node_dir_.count(root_location.to_id()) == 0) {
        throw std::runtime_error("Checkpoint has no root node");
    }
    for (const auto& item : node_dir_) {
        node_location location(item.first);
        if (location.level() != 0 && node_dir_.count(location.get_parent().to_id()) == 0) {
            throw std::runtime_error("Checkpoint has an orphan node");
        }
        for (integer ci = 0; ci < NCHILD; ++ci) {
            const bool has_child = node_dir_.count(location.get_child(ci).to_id()) != 0;
            if (has_child == item.second.load) {
                throw std::runtime_error("Checkpoint has inconsistent leaf/child topology");
            }
        }
    }
    const auto directory = std::move(node_dir_);
    std::vector<hpx::future<void>> futs;
    for (integer i = 0; i < nprocs; ++i) {
        futs.push_back(hpx::async<load_open_action>(opts().all_localities[i], fname, directory));
    }
    for (auto& future : futs) {
        future.get();
    }
    root_ptr->reconstruct_tree();
    node_registry::clear();
    futs.clear();
    for (integer i = 0; i < nprocs; ++i) {
        futs.push_back(hpx::async<load_close_action>(opts().all_localities[i]));
    }
    for (auto& future : futs) {
        future.get();
    }
    printf("Read took %li seconds\n", time(NULL) - tstart);
}

void node_server::reconstruct_tree() {
    const auto root_entry = node_dir_.find(my_location.to_id());
    if (root_entry == node_dir_.end()) {
        throw std::runtime_error("Checkpoint is missing its root");
    }
    if (exact_restart_) {
        load_checkpoint(read_node_checkpoint(my_location, root_entry->second));
        if (bool(is_refined) == root_entry->second.load) {
            throw std::runtime_error("Checkpoint root refinement flag disagrees with mesh topology");
        }
    } else if (root_entry->second.load) {
        load_silo_leaf(*grid_ptr, my_location, root_entry->second.filename);
        is_refined = false;
    } else {
        is_refined = true;
    }
    if (is_refined) {
        std::vector<hpx::future<void>> futs;
        for (integer ci = 0; ci < NCHILD; ++ci) {
            futs.push_back(hpx::async([this, ci]() {
                const auto location = my_location.get_child(ci);
                const auto entry = node_dir_.find(location.to_id());
                if (entry == node_dir_.end()) {
                    throw std::runtime_error("Checkpoint root has a missing child");
                }
                children[ci] = hpx::new_<node_server>(localities[entry->second.locality_id], location);
            }));
        }
        for (auto& future : futs) {
            future.get();
        }
    }
    if (!exact_restart_) {
        current_time = silo_output_time();
        rotational_time = silo_output_rotation_time();
    }
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
