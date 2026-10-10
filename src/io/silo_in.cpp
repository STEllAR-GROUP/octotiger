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

#include <future>
#include <memory>
#include <cstring>
#include <mutex>
#include <stdexcept>
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
            std::vector<integer> exact_layout;
            if (exact) {
                const auto version = silo_restart::read_array<integer>(db, "restart_version", 1)[0];
                exact_layout = silo_restart::read_array<integer>(db, "restart_layout", 7);
                if (restore_integrator && version == silo_restart::format_version) {
                    opts().time_integrator = SSP_MULTISTEP;
                }
                if (version != silo_restart::format_version || opts().time_integrator != SSP_MULTISTEP) {
                    throw std::runtime_error("Exact checkpoint requires format 1 and --time_integrator=SSP_MULTISTEP");
                }
                if (exact_layout[0] != NDIM || exact_layout[1] != INX || exact_layout[2] != H_NX ||
                        exact_layout[4] <= 0 || exact_layout[4] > 1024 ||
                        exact_layout[3] != exact_layout[4] + 10 ||
                        exact_layout[5] != integer(sizeof(real)) || exact_layout[6] != SSP_MULTISTEP) {
                    throw std::runtime_error("Exact checkpoint has an incompatible field layout");
                }
            }

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
            if (exact && opts().n_species != exact_layout[4]) {
                throw std::runtime_error("Exact checkpoint species metadata is inconsistent");
            }
            opts().n_fields = opts().n_species + 10;
			opts().eos = eos_type(ri(db, "eos"));
			opts().gravity = ri(db, "gravity");
			opts().hydro = ri(db, "hydro");
			opts().omega = rr(db, "omega") * opts().code_to_s;
			opts().output_dt = rr(db, "output_frequency");
			opts().problem = problem_type(ri(db, "problem"));
			if (DBInqVarExists(db, "radiation") && ri(db, "radiation") != 0) {
				throw std::runtime_error("This branch does not support radiation transport restarts");
			}
			opts().refinement_floor = rr(db, "refinement_floor");
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

node_server::node_server(const node_location &loc) :
		my_location(loc) {
	const auto &localities = opts().all_localities;
	initialize(0.0, 0.0);
	step_num = gcycle = hcycle = 0;

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
	const auto &localities = opts().all_localities;
	initialize(0.0, 0.0);
	step_num = gcycle = hcycle = 0;
	int nc = 0;
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
