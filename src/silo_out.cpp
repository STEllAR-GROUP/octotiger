#include "octotiger/silo.hpp"
#include "octotiger/node_registry.hpp"

#include <hpx/runtime/threads/run_as_os_thread.hpp>

static const auto &localities = options::all_localities;

template<class T>
struct write_silo_var {
	void operator()(DBfile *db, const char *name, T var) const {
		int one = 1;
		DBWrite(db, name, &var, &one, 1, db_type<T>::d);
	}
};

struct node_list_t;

void output_stage1(std::string fname, int cycle);
node_list_t output_stage2(std::string fname, int cycle);
void output_stage3(std::string fname, int cycle, int gn, int gb, int ge);
void output_stage4(std::string fname, int cycle);

HPX_PLAIN_ACTION(output_stage1, output_stage1_action);
HPX_PLAIN_ACTION(output_stage2, output_stage2_action);
HPX_PLAIN_ACTION(output_stage3, output_stage3_action);

struct node_list_t {
	std::vector<node_location::node_id> silo_leaves;
	std::vector<int> group_num;
	std::vector<node_location::node_id> all;
	std::vector<integer> positions;
	std::vector<std::vector<double>> extents;
	std::vector<int> zone_count;
	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & silo_leaves;
		arc & all;
		arc & positions;
		arc & extents;
		arc & zone_count;
	}
};

struct mesh_vars_t {
	std::vector<silo_var_t> vars;
	std::vector<std::string> var_names;
	std::vector<std::pair<std::string, real>> outflow;
	std::string mesh_name;
	std::vector<std::vector<real>> X;
	std::array<int, NDIM> X_dims;
	std::array<int, NDIM> var_dims;
	node_location location;
	mesh_vars_t(mesh_vars_t&&) = default;
	int compression_level() {
		int lev = 0;
		int i = var_dims[0];
		while (i > INX) {
			i >>= 1;
			lev++;
		}
		return lev;
	}
	mesh_vars_t(const node_location &loc, int compression = 0) :
			X(NDIM), location(loc) {
		const int nx = (INX << compression);
		X_dims[0] = X_dims[1] = X_dims[2] = nx + 1;
		var_dims[0] = var_dims[1] = var_dims[2] = nx;
		const real dx = 2.0 * opts().xscale / nx / (1 << loc.level());
		for (int d = 0; d < NDIM; d++) {
			const int d0 = d;
			X[d0].resize(X_dims[d0]);
			const real o = loc.x_location(d) * opts().xscale;
			for (int i = 0; i < X_dims[d0]; i++) {
				X[d0][i] = (o + i * dx) * opts().code_to_cm;
			}
		}
		mesh_name = oct_to_str(loc.to_id());
	}
};

struct mesh_vars_t;

static std::vector<mesh_vars_t> all_mesh_vars;
static std::vector<hpx::future<mesh_vars_t>> futs_;
static node_list_t node_list_;
static int nsteps;
static int time_elapsed;

static time_t start_time = time(nullptr);
static integer start_step = 0;
static int timestamp;
static int steps_elapsed;
static const int HOST_NAME_LEN = 100;

std::vector<mesh_vars_t> compress(std::vector<mesh_vars_t> &&mesh_vars) {

//	printf("Compressing %i grids...\n", int(mesh_vars.size()));

	using table_type = std::unordered_map<node_location::node_id, std::shared_ptr<mesh_vars_t>>;
	table_type table;
	std::vector<mesh_vars_t> done;
	for (auto &mv : mesh_vars) {
		table.insert(std::make_pair(mv.location.to_id(), std::make_shared<mesh_vars_t>(std::move(mv))));
	}
	int ll = 0;
	std::vector<table_type::iterator> iters;
	while (table.size()) {
		iters.clear();
		auto i = table.begin();
		node_location tmp;
		tmp.from_id(i->first);
		int first_nx = i->second->X_dims[0];
		const integer shift = (tmp.level() - 1) * NDIM;
		if (shift > 0) {
			auto loc_bits = i->first & node_location::node_id(~(0x7 << shift));
			bool levels_match = true;
			for (int j = 0; j < NCHILD; j++) {
				auto this_iter = table.find(loc_bits | (j << shift));
				if (this_iter != table.end()) {
					levels_match = levels_match && (this_iter->second->X_dims[0] == first_nx);
					iters.push_back(this_iter);
				}
			}
			const auto field_names = grid::get_field_names();
			const int nfields = field_names.size();
			if (iters.size() == NCHILD && levels_match) {
				node_location ploc;
				ploc.from_id(loc_bits);
				ploc = ploc.get_parent();
				auto new_mesh_ptr = std::make_shared<mesh_vars_t>(ploc, iters[0]->second->compression_level() + 1);
				for (int f = 0; f < nfields; f++) {
					const bool is_hydro = grid::is_hydro_field(field_names[f]);
					if (is_hydro) {
						new_mesh_ptr->outflow.push_back(std::make_pair(field_names[f], 0.0));
					}
					silo_var_t new_var(field_names[f], new_mesh_ptr->var_dims[0]);
					for (integer ci = 0; ci != NCHILD; ci++) {
						if (is_hydro) {
							new_mesh_ptr->outflow[f].second += iters[ci]->second->outflow[f].second;
						}
						const auto &old_var = iters[ci]->second->vars[f];
						const int nx = new_mesh_ptr->var_dims[0] / 2;
						const int ib = ((ci >> 0) & 1) * nx;
						const int jb = ((ci >> 1) & 1) * nx;
						const int kb = ((ci >> 2) & 1) * nx;
						const int ie = ib + nx;
						const int je = jb + nx;
						const int ke = kb + nx;
						for (int i = ib; i < ie; i++) {
							for (int j = jb; j < je; j++) {
								for (int k = kb; k < ke; k++) {
									const int iiip = i * (4 * nx * nx) + j * 2 * nx + k;
									const int iiic = (i - ib) * (nx * nx) + (j - jb) * nx + (k - kb);
									new_var(iiip) = old_var(iiic);
								}
							}
						}
						new_var.set_range(old_var.min());
						new_var.set_range(old_var.max());
					}
					new_mesh_ptr->vars.push_back(std::move(new_var));
				}
				for (auto this_iter : iters) {
					table.erase(this_iter);
				}
				table.insert(std::make_pair(ploc.to_id(), new_mesh_ptr));
			} else {
				for (auto this_iter : iters) {
					node_location loc;
					loc.from_id(this_iter->first);
					done.push_back(std::move(*this_iter->second));
					table.erase(this_iter);
				}
			}
		} else {
			done.push_back(std::move(*table.begin()->second));
			table.erase(table.begin());
		}
	}

//	printf("Compressed to %i grids...\n", int(done.size()));

	return std::move(done);
}

void output_stage1(std::string fname, int cycle) {
	std::vector<node_location::node_id> ids;
	futs_.clear();
	const auto *node_ptr_ = node_registry::begin()->second.get_ptr().get();
	silo_output_time() = node_ptr_->get_time() * opts().code_to_s;
	silo_output_rotation_time() = node_ptr_->get_rotation_count();
	for (auto i = node_registry::begin(); i != node_registry::end(); ++i) {
		const auto *node_ptr_ = i->second.get_ptr().get();
		if (!node_ptr_->refined()) {
			futs_.push_back(hpx::async([](node_location loc, node_registry::node_ptr ptr) {
				const auto *this_ptr = ptr.get_ptr().get();
				assert(this_ptr);
				const real dx = TWO / real(1 << loc.level()) / real(INX);
				mesh_vars_t rc(loc);
				const std::string suffix = oct_to_str(loc.to_id());
				const grid &gridref = this_ptr->get_hydro_grid();
				rc.vars = gridref.var_data();
				rc.outflow = gridref.get_outflows();
				return std::move(rc);
			}, i->first, i->second));
		}
	}
}

node_list_t output_stage2(std::string fname, int cycle) {
	const int this_id = hpx::get_locality_id();
	const int nfields = grid::get_field_names().size();
	std::string this_fname = fname + std::string(".") + std::to_string(INX) + std::string(".silo");
	all_mesh_vars.clear();
	all_mesh_vars.reserve(futs_.size());
	for (auto &this_fut : futs_) {
		all_mesh_vars.push_back(std::move(GET(this_fut)));
	}
	std::vector<node_location::node_id> ids;
	if (opts().compress_silo) {
		all_mesh_vars = compress(std::move(all_mesh_vars));
	}
	node_list_t nl;
	nl.extents.resize(nfields);
	ids.reserve(all_mesh_vars.size());
	for (int f = 0; f < nfields; f++) {
		nl.extents[f].reserve(all_mesh_vars.size() * 2);
	}
	for (const auto &mv : all_mesh_vars) {
		ids.push_back(mv.location.to_id());
		nl.zone_count.push_back(mv.var_dims[0] * mv.var_dims[1] * mv.var_dims[2]);
		for (int f = 0; f < nfields; f++) {
			//		printf( "%s %e %e\n", mv.vars[f].name(), mv.vars[f].min(), mv.vars[f].max());
			nl.extents[f].push_back(mv.vars[f].min());
			nl.extents[f].push_back(mv.vars[f].max());
		}

	}
	std::vector<node_location::node_id> all;
	std::vector<integer> positions;
	all.reserve(node_registry::size());
	positions.reserve(node_registry::size());
	for (auto i = node_registry::begin(); i != node_registry::end(); ++i) {
		all.push_back(i->first.to_id());
		positions.push_back(i->second.get_ptr().get()->get_position());
	}
	nl.silo_leaves = std::move(ids);
	nl.all = std::move(all);
	nl.positions = std::move(positions);
	return std::move(nl);
}

void output_stage3(std::string fname, int cycle, int gn, int gb, int ge) {
	const int this_id = hpx::get_locality_id();
	const int nfields = grid::get_field_names().size();
	std::string this_fname = fname + ".silo.data/" + std::to_string(gn) + std::string(".silo");
	double dtime = silo_output_rotation_time();
	hpx::threads::run_as_os_thread([&this_fname, this_id, &dtime, gb, gn](integer cycle) {
		DBfile *db;
		if (this_id == gb) {
			printf( "%s\n", this_fname.c_str());
			sleep(10);
			db = DBCreateReal(this_fname.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-tiger", SILO_DRIVER);
		} else {
			db = DBOpenReal(this_fname.c_str(), SILO_DRIVER, DB_APPEND);
		}
		float ftime = dtime;
		int one = 1;
		int opt1 = DB_CARTESIAN;
		int planar = DB_VOLUME;
		char xstr[2] = { 'x', '\0' };
		char ystr[2] = { 'y', '\0' };
		char zstr[2] = { 'z', '\0' };

		auto optlist_mesh = DBMakeOptlist(100);
		DBAddOption(optlist_mesh, DBOPT_COORDSYS, &opt1);
		DBAddOption(optlist_mesh, DBOPT_CYCLE, &cycle);
		DBAddOption(optlist_mesh, DBOPT_XLABEL, xstr);
		DBAddOption(optlist_mesh, DBOPT_YLABEL, ystr);
		DBAddOption(optlist_mesh, DBOPT_ZLABEL, zstr);
		DBAddOption(optlist_mesh, DBOPT_PLANAR, &planar);
		DBAddOption(optlist_mesh, DBOPT_TIME, &ftime);
		DBAddOption(optlist_mesh, DBOPT_DTIME, &dtime);
		// TODO: XUNITS. YUNITS, ZUNITS
		DBAddOption(optlist_mesh, DBOPT_HIDE_FROM_GUI, &one);

		const char *coord_names[] = { "x", "y", "z" };
		constexpr int data_type = DB_DOUBLE;
		constexpr int ndim = NDIM;
		constexpr int coord_type = DB_COLLINEAR;
		int count = 0;
		for (const auto &mesh_vars : all_mesh_vars) {
			const auto &X = mesh_vars.X;
			const real *coords[NDIM];
			for (int d = 0; d < NDIM; d++) {
				coords[d] = X[d].data();
			}
			const auto &dir_name = mesh_vars.mesh_name.c_str();
			DBMkDir(db, dir_name);
			DBSetDir(db, dir_name);
			DBPutQuadmesh(db, "quadmesh", coord_names, coords, mesh_vars.X_dims.data(), ndim, data_type, coord_type, optlist_mesh);

			for (integer m = 0; m != mesh_vars.vars.size(); m++) {
				auto optlist_var = DBMakeOptlist(100);
				DBAddOption(optlist_var, DBOPT_COORDSYS, &opt1);
				DBAddOption(optlist_var, DBOPT_CYCLE, &cycle);
				DBAddOption(optlist_var, DBOPT_TIME, &ftime);
				DBAddOption(optlist_var, DBOPT_DTIME, &dtime);
				// TODO: UNITS
				DBAddOption(optlist_var, DBOPT_HIDE_FROM_GUI, &one);

				const auto &o = mesh_vars.vars[m];
				const bool is_hydro = grid::is_hydro_field(o.name());
				if (is_hydro) {
					real outflow = mesh_vars.outflow[m].second;
					write_silo_var<real> f;
					f(db, outflow_name(o.name()).c_str(), outflow);
					DBAddOption(optlist_var, DBOPT_CONSERVED, &one);
					DBAddOption(optlist_var, DBOPT_EXTENSIVE, &one);
				}
				//			if( std::strcmp(o.name(),"fx")==0 ) {
//							for( int i = 0; i < INX*INX*INX; i++) {
//								printf( "%e\n", o(i));
//							}
				//				}
				DBPutQuadvar1(db, o.name(), "quadmesh", o.data(), mesh_vars.var_dims.data(), ndim, nullptr, 0, DB_DOUBLE, DB_ZONECENT, optlist_var);
				count++;
				DBFreeOptlist(optlist_var);
			}
			DBSetDir(db, "/");
		}
		DBFreeOptlist(optlist_mesh);
		DBClose(db);
	}, cycle).get();
	if (this_id < ge - 1) {
		output_stage3_action func;
		func(localities[this_id + 1], fname, cycle, gn, gb, ge);
	}
}

void output_stage4(std::string fname, int cycle) {
	const int nfields = grid::get_field_names().size();
	std::string this_fname = fname + std::string(".silo");
	double dtime = silo_output_rotation_time();
	double rtime = silo_output_rotation_time();
	hpx::threads::run_as_os_thread([&this_fname, fname, nfields, &rtime](int cycle) {
		auto *db = DBCreateReal(this_fname.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-tiger", SILO_DRIVER);
		double dtime = silo_output_time();
		float ftime = dtime;
		std::vector<std::pair<int, node_location>> node_locs;
		std::vector<char*> mesh_names;
		std::vector<std::vector<char*>> field_names(nfields);
		node_locs.reserve(node_list_.silo_leaves.size());
		int j = 0;
		for (auto &i : node_list_.silo_leaves) {
			node_location nloc;
			nloc.from_id(i);
			node_locs.push_back(std::make_pair(node_list_.group_num[j], nloc));
			j++;
		}
		const auto top_field_names = grid::get_field_names();
		mesh_names.reserve(node_locs.size());
		for (int f = 0; f < nfields; f++)
			field_names[f].reserve(node_locs.size());
		for (int i = 0; i < node_locs.size(); i++) {
			const auto prefix = fname + ".silo.data/" + std::to_string(node_locs[i].first) + ".silo:/" + oct_to_str(node_locs[i].second.to_id()) + "/";
			const auto str = prefix + "quadmesh";
			char *ptr = new char[str.size() + 1];
			std::strcpy(ptr, str.c_str());
			mesh_names.push_back(ptr);
			for (int f = 0; f < nfields; f++) {
				const auto str = prefix + top_field_names[f];
				char *ptr = new char[str.size() + 1];
				strcpy(ptr, str.c_str());
				field_names[f].push_back(ptr);
			}
		}

		const int n_total_domains = mesh_names.size();

		int opt1 = DB_CARTESIAN;
		int mesh_type = DB_QUADMESH;
		int one = 1;
		int dj = DB_ABUTTING;
		int six = 2 * NDIM;
		assert(n_total_domains > 0);
		std::vector<double> extents;
		extents.reserve(node_locs.size() * 6);
		for (const auto &n0 : node_locs) {
			const auto &n = n0.second;
			const real scale = opts().xscale * opts().code_to_cm;
			const double xmin = n.x_location(0) * scale;
			const double ymin = n.x_location(1) * scale;
			const double zmin = n.x_location(2) * scale;
			const double d = TWO / real(1 << n.level()) * scale;
			const double xmax = xmin + d;
			const double ymax = ymin + d;
			const double zmax = zmin + d;
			extents.push_back(xmin);
			extents.push_back(ymin);
			extents.push_back(zmin);
			extents.push_back(xmax);
			extents.push_back(ymax);
			extents.push_back(zmax);
		}
		int three = 3;
		int two = 2;
		auto optlist = DBMakeOptlist(100);
		DBAddOption(optlist, DBOPT_CYCLE, &cycle);
		DBAddOption(optlist, DBOPT_TIME, &ftime);
		DBAddOption(optlist, DBOPT_DTIME, &dtime);
		DBAddOption(optlist, DBOPT_EXTENTS_SIZE, &six);
		DBAddOption(optlist, DBOPT_EXTENTS, extents.data());
		DBAddOption(optlist, DBOPT_ZONECOUNTS, node_list_.zone_count.data());
//			DBAddOption(optlist, DBOPT_TV_CONNECTIVITY, &one);
		DBAddOption(optlist, DBOPT_DISJOINT_MODE, &dj);
		DBAddOption(optlist, DBOPT_TOPO_DIM, &three);
		DBAddOption(optlist, DBOPT_MB_BLOCK_TYPE, &mesh_type);
		printf("Putting %i\n", n_total_domains);
		DBPutMultimesh(db, "quadmesh", n_total_domains, mesh_names.data(), nullptr, optlist);
		DBFreeOptlist(optlist);
		char mmesh[] = "quadmesh";
		for (int f = 0; f < nfields; f++) {
			optlist = DBMakeOptlist(100);
			DBAddOption(optlist, DBOPT_CYCLE, &cycle);
			DBAddOption(optlist, DBOPT_TIME, &ftime);
			DBAddOption(optlist, DBOPT_DTIME, &dtime);
			const bool is_hydro = grid::is_hydro_field(top_field_names[f]);
			if (is_hydro) {
				DBAddOption(optlist, DBOPT_CONSERVED, &one);
				DBAddOption(optlist, DBOPT_EXTENSIVE, &one);
			}
			DBAddOption(optlist, DBOPT_EXTENTS_SIZE, &two);
			DBAddOption(optlist, DBOPT_EXTENTS, node_list_.extents[f].data());
			DBAddOption(optlist, DBOPT_MMESH_NAME, mmesh);
			DBPutMultivar(db, top_field_names[f].c_str(), n_total_domains, field_names[f].data(), std::vector<int>(n_total_domains, DB_QUADVAR).data(), optlist);
			DBFreeOptlist(optlist);
		}
		write_silo_var<integer> fi;
		write_silo_var<real> fr;
		fi(db, "version", SILO_VERSION);
		fr(db, "code_to_g", opts().code_to_g);
		fr(db, "code_to_s", opts().code_to_s);
		fr(db, "code_to_cm", opts().code_to_cm);
		fi(db, "n_species", integer(opts().n_species));
		fi(db, "eos", integer(opts().eos));
		fi(db, "gravity", integer(opts().gravity));
		fi(db, "hydro", integer(opts().hydro));
		fr(db, "omega", grid::get_omega() / opts().code_to_s);
		fr(db, "output_frequency", opts().output_dt);
		fi(db, "problem", integer(opts().problem));
		fi(db, "radiation", integer(opts().radiation));
		fr(db, "refinement_floor", opts().refinement_floor);
		fr(db, "cgs_time", dtime);
		fr(db, "rotational_time", rtime);
		fr(db, "xscale", opts().xscale);
		char hostname[HOST_NAME_LEN];
		gethostname(hostname, HOST_NAME_LEN);
		DBWrite(db, "hostname", hostname, &HOST_NAME_LEN, 1, DB_CHAR);
		int nnodes = node_list_.all.size();
		DBWrite(db, "node_list", node_list_.all.data(), &nnodes, 1, DB_LONG_LONG);
		DBWrite(db, "node_positions", node_list_.positions.data(), &nnodes, 1, db_type<integer>::d);
		int nspc = opts().n_species;
		DBWrite(db, "X", opts().X.data(), &nspc, 1, db_type<real>::d);
		DBWrite(db, "Z", opts().Z.data(), &nspc, 1, db_type<real>::d);
		DBWrite(db, "atomic_mass", opts().atomic_mass.data(), &nspc, 1, db_type<real>::d);
		DBWrite(db, "atomic_number", opts().atomic_number.data(), &nspc, 1, db_type<real>::d);
		fi(db, "node_count", integer(nnodes));
		fi(db, "leaf_count", integer(node_list_.silo_leaves.size()));
		write_silo_var<integer>()(db, "timestamp", timestamp);
		write_silo_var<integer>()(db, "epoch", silo_epoch());
		write_silo_var<integer>()(db, "locality_count", localities.size());
		write_silo_var<integer>()(db, "thread_count", localities.size() * std::thread::hardware_concurrency());
		write_silo_var<integer>()(db, "step_count", nsteps);
		write_silo_var<integer>()(db, "time_elapsed", time_elapsed);
		write_silo_var<integer>()(db, "steps_elapsed", steps_elapsed);

		// mesh adjacency information
		int nleaves = node_locs.size();
		std::vector<int> neighbor_count(nleaves, 0);
		std::vector<std::vector<int>> neighbor_lists(nleaves);
		std::vector<std::vector<int>> back_lists(nleaves);
		std::vector<int> linear_neighbor_list;
		std::vector<int> linear_back_list;
		std::vector<std::vector<int*>> connections(nleaves);
		std::vector<int*> linear_connections;
		std::vector<std::vector<int>> tmp;
		tmp.reserve(nleaves);
		for (int n = 0; n < nleaves; n++) {
			for (int m = n + 1; m < nleaves; m++) {
				range_type rn, rm, i;
				rn = node_locs[n].second.abs_range();
				rm = node_locs[m].second.abs_range();
				i = intersection(rn, rm);
				if (i[0].first != -1) {
					neighbor_count[n]++;
					neighbor_count[m]++;
					back_lists[m].push_back(neighbor_lists[n].size());
					back_lists[n].push_back(neighbor_lists[m].size());
					neighbor_lists[m].push_back(n); //
					neighbor_lists[n].push_back(m);
					std::vector<int> adj1(15);
					std::vector<int> adj2(15);
					for (int d = 0; d < NDIM; d++) {
						int d0 = NDIM - d - 1;
						adj1[2 * d + 0] = rn[d].first;
						adj1[2 * d + 1] = rn[d].second;
						adj1[2 * d + 6] = i[d].first;
						adj1[2 * d + 7] = i[d].second;
						adj1[12 + d] = d + 1;
						adj2[2 * d + 0] = rm[d].first;
						adj2[2 * d + 1] = rm[d].second;
						adj2[2 * d + 6] = i[d].first;
						adj2[2 * d + 7] = i[d].second;
						adj2[12 + d] = d + 1;
					}
					tmp.push_back(std::move(adj1));
					tmp.push_back(std::move(adj2));
					connections[n].push_back(tmp[tmp.size() - 2].data());
					connections[m].push_back(tmp[tmp.size() - 1].data());
				}
			}
		}
		linear_neighbor_list.reserve(nleaves);
		linear_back_list.reserve(nleaves);
		linear_connections.reserve(nleaves);
		for (int n = 0; n < nleaves; n++) {
			for (int m = 0; m < neighbor_count[n]; m++) {
				linear_neighbor_list.push_back(neighbor_lists[n][m]);
				linear_back_list.push_back(back_lists[n][m]);
				linear_connections.push_back(connections[n][m]);
			}
		}
		std::vector<int> fifteen(linear_connections.size(), 15);
		std::vector<int> mesh_types(nleaves, mesh_type);
		int isTimeVarying = 1;
		int n = 1;
		DBWrite(db, "ConnectivityIsTimeVarying", &isTimeVarying, &n, 1, DB_INT);
		DBMkDir(db, "Decomposition");
		DBSetDir(db, "Decomposition");
		DBPutMultimeshadj(db, "Domain_Decomposition", nleaves, mesh_types.data(), neighbor_count.data(), linear_neighbor_list.data(), linear_back_list.data(),
				fifteen.data(), linear_connections.data(), nullptr, nullptr, nullptr);
		DBWrite(db, "NumDomains", &nleaves, &one, 1, DB_INT);

		// Expressions

		DBSetDir(db, "/");
		std::vector<char*> names;
		std::vector<char*> defs;
		std::vector<int> types;
		auto exps1 = grid::get_scalar_expressions();
		auto exps2 = grid::get_vector_expressions();
//				decltype(exps1) exps;
		types.reserve(exps1.size() + exps2.size());
		names.reserve(exps1.size() + exps2.size());
		defs.reserve(exps1.size() + exps2.size());
//				for( auto& e : exps1 ) {
//					exps.push_back(std::move(e));
//				}
//				for( auto& e : exps2 ) {
//					exps.push_back(std::move(e));
//				}
		for (integer i = 0; i < exps1.size(); i++) {
			types.push_back(DB_VARTYPE_SCALAR);
			names.push_back(const_cast<char*>(exps1[i].first.c_str()));
			defs.push_back(const_cast<char*>(exps1[i].second.c_str()));
		}
		for (integer i = 0; i < exps2.size(); i++) {
			types.push_back(DB_VARTYPE_VECTOR);
			names.push_back(const_cast<char*>(exps2[i].first.c_str()));
			defs.push_back(const_cast<char*>(exps2[i].second.c_str()));
		}
//				FILE* fp = fopen( "expressions.dat", "at");
		for (int i = 0; i < names.size(); i++) {
			//	fprintf( fp, "%s %i %s\n", names[i], types[i], defs[i]);
		}
//				fprintf( fp, "%e %e\n", physcon().mh, opts().code_to_g);
//				fclose(fp);
		DBPutDefvars(db, "expressions", types.size(), names.data(), types.data(), defs.data(), nullptr);
		DBClose(db);
		for (auto ptr : mesh_names) {
			delete[] ptr;
		}
		for (int f = 0; f < nfields; f++) {
			for (auto &s : field_names[f]) {
				delete[] s;
			}
		}
	}, cycle).get();
}

void output_all(std::string fname, int cycle, bool block) {

	if (opts().disable_output) {
		printf("Skipping SILO output\n");
		return;
	}

	std::string command = "mkdir -p " + fname + ".silo.data\n";
	system(command.c_str());

	static hpx::future<void> barrier(hpx::make_ready_future<void>());
	GET(barrier);
	nsteps = node_registry::begin()->second.get_ptr().get()->get_step_num();
	timestamp = time(nullptr);
	steps_elapsed = nsteps - start_step;
	time_elapsed = time(nullptr) - start_time;
	start_time = timestamp;
	start_step = nsteps;
	std::vector<hpx::future<void>> futs1;
	for (auto &id : localities) {
		futs1.push_back(hpx::async<output_stage1_action>(id, fname, cycle));
	}
	GET(hpx::when_all(futs1));

	std::vector<hpx::future<node_list_t>> id_futs;
	for (auto &id : localities) {
		id_futs.push_back(hpx::async<output_stage2_action>(id, fname, cycle));
	}
	node_list_.silo_leaves.clear();
	node_list_.group_num.clear();
	node_list_.all.clear();
	node_list_.positions.clear();
	node_list_.extents.clear();
	int id = 0;
	for (auto &f : id_futs) {
		const int gn = id * opts().silo_num_groups / localities.size();
		node_list_t this_list = GET(f);
		const int leaf_cnt = this_list.silo_leaves.size();
		for (auto i = 0; i < leaf_cnt; i++) {
			node_list_.group_num.push_back(gn);
		}
		node_list_.silo_leaves.insert(node_list_.silo_leaves.end(), this_list.silo_leaves.begin(), this_list.silo_leaves.end());
		node_list_.all.insert(node_list_.all.end(), this_list.all.begin(), this_list.all.end());
		node_list_.positions.insert(node_list_.positions.end(), this_list.positions.begin(), this_list.positions.end());
		node_list_.zone_count.insert(node_list_.zone_count.end(), this_list.zone_count.begin(), this_list.zone_count.end());
		const int nfields = grid::get_field_names().size();
		node_list_.extents.resize(nfields);
		for (int f = 0; f < this_list.extents.size(); f++) {
			node_list_.extents[f].insert(node_list_.extents[f].end(), this_list.extents[f].begin(), this_list.extents[f].end());
		}
		id++;
	}
	const auto ng = opts().silo_num_groups;

	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < ng; i++) {
		int gb = (i * localities.size()) / ng;
		int ge = ((i + 1) * localities.size()) / ng;
		futs.push_back(hpx::async<output_stage3_action>(localities[0], fname, cycle, i, gb, ge));
	}

	barrier = hpx::async([&futs, fname, cycle]() {
		for (auto &f : futs) {
			GET(f);
		}
		output_stage4(fname, cycle);
	}
	);

	if (block) {
		GET(barrier);
		barrier = hpx::make_ready_future<void>();
	}
}
