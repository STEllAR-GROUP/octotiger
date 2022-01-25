//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/config.hpp"
#include <fenv.h>

#include "octotiger/diagnostics.hpp"
#include "octotiger/future.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_registry.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/profiler.hpp"
#include "octotiger/taylor.hpp"

#include <hpx/include/lcos.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/runtime/get_colocation_id.hpp>
#include <hpx/serialization/list.hpp>

#include <cerrno>
#include <cstdint>
#include <cstdio>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

using check_for_refinement_action_type = node_server::check_for_refinement_action;
HPX_REGISTER_ACTION(check_for_refinement_action_type);

future<void> node_client::check_for_refinement(real omega, real r) const {
	return hpx::async<typename node_server::check_for_refinement_action>(get_unmanaged_gid(), omega, r);
}

void node_server::check_for_refinement(real omega, real new_floor) {
	static hpx::mutex mtx;
	{
		std::lock_guard<hpx::mutex> lock(mtx);
		grid::omega = omega;
		if (new_floor > 0) {
			opts().refinement_floor = new_floor;
		}
	}
	node_registry::delete_(my_location);
	bool rc = false;
	std::array<future<void>, NCHILD + 1> futs;
	for (integer i = 0; i != NCHILD + 1; ++i) {
		futs[i] = hpx::make_ready_future();
	}
	integer index = 0;
	if (is_refined) {
		for (auto &child : children) {
			futs[index++] = child.check_for_refinement(omega, new_floor);
		}
	}
	if (opts().hydro || opts().problem == AMR_TEST) {
		all_hydro_bounds();
	}
	if (!rc) {
		rc = grid_ptr->refine_me(my_location.level(), new_floor);
	}
	if (rc) {
		if (refinement_flag++ == 0) {
			if (!parent.empty()) {
				futs[index++] = parent.force_nodes_to_exist(my_location.get_neighbors());
			}
		}
	}
	for (auto &f : futs) {
		GET(f);
	}

}

using enforce_bc_action_type = node_server::enforce_bc_action;
HPX_REGISTER_ACTION(enforce_bc_action_type);

future<void> node_client::enforce_bc() const {
	return hpx::async<typename node_server::enforce_bc_action>(get_unmanaged_gid());
}

void node_server::enforce_bc() {
	std::array<future<void>, NCHILD + 1> futs;
	for (integer i = 0; i != NCHILD + 1; ++i) {
		futs[i] = hpx::make_ready_future();
	}
	integer index = 0;
	if (is_refined) {
		for (auto &child : children) {
			futs[index++] = child.enforce_bc();
		}
	}
	all_hydro_bounds();
	for (auto &f : futs) {
		GET(f);
	}

}

using kill_action_type = node_server::kill_action;
HPX_REGISTER_ACTION(kill_action_type);

hpx::future<void> node_client::kill() const {
	return hpx::async<typename node_server::kill_action>(get_gid());
}

void node_server::kill() {
	clear_family();

}

using copy_to_locality_action_type = node_server::copy_to_locality_action;
HPX_REGISTER_ACTION(copy_to_locality_action_type);

future<hpx::id_type> node_client::copy_to_locality(const hpx::id_type &id) const {
	return hpx::async<typename node_server::copy_to_locality_action>(get_gid(), id);
}

future<hpx::id_type> node_server::copy_to_locality(const hpx::id_type &id) {

	node_registry::delete_(my_location);

	std::vector<hpx::id_type> cids;
	if (is_refined) {
		cids.resize(NCHILD);
		for (auto &ci : geo::octant::full_set()) {
			cids[ci] = children[ci].get_gid();
		}
	}
	auto rc = hpx::new_<node_server>(id, my_location, step_num, bool(is_refined), current_time, rotational_time, child_descendant_count, std::move(*grid_ptr),
			cids, std::size_t(hcycle), std::size_t(rcycle), std::size_t(gcycle), position);
	clear_family();
	parent = hpx::invalid_id;
	std::fill(neighbors.begin(), neighbors.end(), hpx::invalid_id);
	std::fill(children.begin(), children.end(), hpx::invalid_id);
	return rc;
}

using diagnostics_action_type = node_server::diagnostics_action;
HPX_REGISTER_ACTION(diagnostics_action_type);

future<diagnostics_t> node_client::diagnostics(const diagnostics_t &d) const {
	return hpx::async<typename node_server::diagnostics_action>(get_unmanaged_gid(), d);
}

using compare_analytic_action_type = node_server::compare_analytic_action;
HPX_REGISTER_ACTION(compare_analytic_action_type);

future<analytic_t> node_client::compare_analytic() const {
	return hpx::async<typename node_server::compare_analytic_action>(get_unmanaged_gid());
}

analytic_t node_server::compare_analytic() {
	analytic_t a(opts().n_fields);
	if (!is_refined) {
		a = grid_ptr->compute_analytic(current_time);
	} else {
		std::array<future<analytic_t>, NCHILD> futs;
		integer index = 0;
		for (integer i = 0; i != NCHILD; ++i) {
			futs[index++] = children[i].compare_analytic();
		}
		for (integer i = 0; i != NCHILD; ++i) {
			a += GET(futs[i]);
		}
	}
	if (my_location.level() == 0) {
		print("L1, L2\n");
		real vol = 1.0;
		for (int d = 0; d < NDIM; d++) {
			vol *= 2.0 * opts().xscale;
		}
		for (integer field = 0; field != opts().n_fields; ++field) {
			print("%16s %e %e\n", physics<3>::field_names3[field], a.l1[field] / vol, std::sqrt(a.l2[field] / vol));
		}

		const auto ml = opts().max_level;
		const auto dxmin = 2.0 * opts().xscale / INX / double(1 << ml);
		FILE *fp = fopen("L1.dat", "at");
		fprintf(fp, "%e %i ", dxmin, int(ml));
		for (integer field = 0; field != opts().n_fields; ++field) {
			fprintf(fp, "%e ", a.l1[field] / vol);
		}
		fprintf(fp, "\n");
		fclose(fp);

		fp = fopen("L2.dat", "at");
		fprintf(fp, "%e %i ", dxmin, int(ml));
		for (integer field = 0; field != opts().n_fields; ++field) {
			fprintf(fp, "%e ", std::sqrt(a.l2[field] / vol));
		}
		fprintf(fp, "\n");
		fclose(fp);

		fp = fopen("Linf.dat", "at");
		fprintf(fp, "%e %i ", dxmin, int(ml));
		for (integer field = 0; field != opts().n_fields; ++field) {
			fprintf(fp, "%e ", a.linf[field]);
		}
		fprintf(fp, "\n");
		fclose(fp);
	}
	return a;
}

const diagnostics_t& diagnostics_t::compute() {
	fedisableexcept(FE_DIVBYZERO);
	fedisableexcept(FE_INVALID);
	fedisableexcept(FE_OVERFLOW);

	if (virial_norm != 0.0) {
		virial /= virial_norm;
	}
	if (opts().problem != DWD) {
		return *this;
	}
	real dX[NDIM], V[NDIM];
	for (integer d = 0; d != NDIM; ++d) {
		dX[d] = com[1][d] - com[0][d];
		V[d] = com_dot[1][d] - com_dot[0][d];
	}
	real sep2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
	if (sep2 == 0.0) {
		failed = true;
		return *this;
	}
	omega = std::abs((dX[XDIM] * V[YDIM] - dX[YDIM] * V[XDIM]) * INVERSE(sep2));
	Torb = com[0][XDIM] * g[0][YDIM] - com[0][YDIM] * g[0][XDIM];
	Torb += com[1][XDIM] * g[1][YDIM] - com[1][YDIM] * g[1][XDIM];
//	print( "%e %e %e %e %e\n", dX[XDIM], V[XDIM], dX[YDIM], V[YDIM], omega);
	a = std::sqrt(sep2);
	real mu = m[0] * m[1] / (m[1] + m[0]);
	jorb = mu * omega * sep2;
	if (m[0] > 0.0 && m[1] > 0.0) {
		const real q = m[1] / m[0];
		rL[0] = RL_radius(1.0 / q) * a;
		rL[1] = RL_radius(q) * a;
	}
	for (integer s = 0; s != nspec; ++s) {
		space_vector RdotQ = 0.0;
		for (integer d = 0; d != NDIM; ++d) {
			RdotQ[d] += mom[s](0, d) * dX[0];
			RdotQ[d] += mom[s](1, d) * dX[1];
			RdotQ[d] += mom[s](2, d) * dX[2];
		}
		tidal[s] = RdotQ[0] * dX[1] - RdotQ[1] * dX[0];
		tidal[s] /= std::pow(a, 3.0);
		tidal[s] *= m[1 - s];
	}
	z_mom_orb = mu * sep2;

	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	return *this;
}

diagnostics_t node_server::diagnostics() {

	if (opts().disable_diagnostics) {
		return diagnostics_t();
	}

	diagnostics_t diags;
	for (integer i = 1; i != (opts().problem == DWD ? 5 : 2); ++i) {
//		print( "%i\n", i );
		diags.stage = i;
		diags = diagnostics(diags).compute();
		if (opts().gravity) {
			diags.grid_com = grid_ptr->center_of_mass();

		} else {
			//TODO center of mass for non gravity runs
		}
	}
	if (!diags.failed && !opts().disable_diagnostics) {

		FILE *fp = fopen((opts().data_dir + "binary.dat").c_str(), "at");
		if (fp) {
			fprintf(fp, "%13e ", current_time);
			fprintf(fp, "%13e ", (double) diags.a);
			fprintf(fp, "%13e ", (double) diags.omega);
			fprintf(fp, "%13e ", (double) diags.jorb);
			for (integer s = 0; s != 2; ++s) {
				const auto radius = std::pow(diags.roche_vol[s] / (4.0 / 3.0 * M_PI), 1. / 3.);
				fprintf(fp, "%13e ", (double) diags.m[s]);				// 5 // 19
				fprintf(fp, "%13e ", (double) diags.js[s]);
				fprintf(fp, "%13e ", (double) diags.lz1[s]);
				fprintf(fp, "%13e ", (double) diags.lz2[s]);
				fprintf(fp, "%13e ", (double) diags.ekin[s]);           // 9  // 23
				fprintf(fp, "%13e ", (double) diags.epot[s]);           // 10 // 24
				fprintf(fp, "%13e ", (double) diags.eint[s]);           // 11 // 25
				fprintf(fp, "%13e ", (double) diags.com[s][0]);			// 12 // 26
				fprintf(fp, "%13e ", (double) diags.com[s][1]);			// 13 // 27
				fprintf(fp, "%13e ", (double) diags.com_dot[s][0]);
				fprintf(fp, "%13e ", (double) diags.com_dot[s][1]);
				fprintf(fp, "%13e ", (double) radius);					// 16 // 30
				fprintf(fp, "%13e ", (double) diags.Ts[s]);
				fprintf(fp, "%13e ", (double) diags.z_moment[s]);       // 18  // 32
			}
			fprintf(fp, "%13e ", (double) diags.rho_max[0]); // 33
			fprintf(fp, "%13e ", (double) diags.rho_max[1]); // 34
			fprintf(fp, "%13e ", (double) diags.grid_com[0]); // 35
			fprintf(fp, "%13e ", (double) diags.grid_com[1]); // 36
			fprintf(fp, "%13e ", (double) diags.nonvacj); // 37
			fprintf(fp, "%13e ", (double) diags.nonvacjlz); // 38
			fprintf(fp, "%13e ", (double) diags.Torb); //39
			fprintf(fp, "%13e ", (double) diags.grid_sum[rho_i]); //40
			fprintf(fp, "%13e ", (double) diags.munbound1); //41
			fprintf(fp, "%13e ", (double) diags.munbound2); //4@
			fprintf(fp, "\n");
			fclose(fp);
			fp = fopen((opts().data_dir + "sums.dat").c_str(), "at");
			fprintf(fp, "%.13e ", current_time);
			for (integer i = 0; i != opts().n_fields; ++i) {
				fprintf(fp, "%.13e ", (double) diags.grid_sum[i] + (double) diags.grid_out[i]);
				fprintf(fp, "%.13e ", (double) diags.grid_out[i]);
			}
			for (integer i = 0; i != 3; ++i) {
				fprintf(fp, "%.13e ", (double) diags.lsum[i]);
			}
			fprintf(fp, "\n");
			fclose(fp);

		} else {
			print("Failed to write binary.dat %s\n", std::strerror(errno));
		}
	} else {
		print("Failed to compute Roche geometry\n");
	}
	return diags;
}

diagnostics_t node_server::root_diagnostics(const diagnostics_t &diags) {
	return diags;
}

diagnostics_t node_server::diagnostics(const diagnostics_t &diags) {
	if (is_refined) {
		auto rc = hpx::async(hpx::util::annotated_function([&]() {
			return child_diagnostics(diags);
		}, "diagnostics::return_child_diagnostics"));
		all_hydro_bounds();
		auto diags = GET(rc);
		return diags;
	} else {
		all_hydro_bounds();
		return local_diagnostics(diags);
	}
}

diagnostics_t node_server::child_diagnostics(const diagnostics_t &diags) {
	diagnostics_t sums;
	std::array<future<diagnostics_t>, NCHILD> futs;
	integer index = 0;
	for (integer ci = 0; ci != NCHILD; ++ci) {
		futs[index++] = children[ci].diagnostics(diags);
	}
	auto child_sums = hpx::util::unwrap(futs);
	return std::accumulate(child_sums.begin(), child_sums.end(), sums);
}

diagnostics_t node_server::local_diagnostics(const diagnostics_t &diags) {
//	all_hydro_bounds();
	return grid_ptr->diagnostics(diags);
}

using force_nodes_to_exist_action_type = node_server::force_nodes_to_exist_action;
HPX_REGISTER_ACTION(force_nodes_to_exist_action_type);

future<void> node_client::force_nodes_to_exist(std::vector<node_location> &&locs) const {
	return hpx::async<typename node_server::force_nodes_to_exist_action>(get_unmanaged_gid(), std::move(locs));
}

void node_server::force_nodes_to_exist(std::vector<node_location> &&locs) {
	std::vector<future<void>> futs;
	std::vector<node_location> parent_list;
	std::array<std::vector<node_location>, geo::direction::count()> sibling_lists;
	std::vector<std::vector<node_location>> child_lists(NCHILD);

	futs.reserve(geo::octant::count() + 2);
	parent_list.reserve(locs.size());

	integer index = 0;
	for (auto &loc : locs) {
		assert(loc != my_location);
		if (loc.is_child_of(my_location)) {
			if (refinement_flag++ == 0 && !parent.empty()) {
				futs.push_back(parent.force_nodes_to_exist(my_location.get_neighbors()));
			}
			if (is_refined) {
				for (auto &ci : geo::octant::full_set()) {
					if (loc.is_child_of(my_location.get_child(ci))) {
						child_lists[ci].push_back(loc);
						break;
					}
				}
			}
		} else {

			/** BUG HERE ***/
			if (parent.empty()) {
				print("parent empty %s %s\n", my_location.to_str().c_str(), loc.to_str().c_str());
				abort();
			}
			assert(!parent.empty());

			bool found_match = false;
			for (auto &di : geo::direction::full_set()) {
				if (loc.is_child_of(my_location.get_neighbor(di)) && !neighbors[di].empty()) {
					sibling_lists[di].push_back(loc);
					found_match = true;
					break;
				}
			}
			if (!found_match) {
				parent_list.push_back(loc);
			}
		}
	}
	for (auto &ci : geo::octant::full_set()) {
		if (is_refined && child_lists[ci].size()) {
			futs.push_back(children[ci].force_nodes_to_exist(std::move(child_lists[ci])));
		}
	}
	for (auto &di : geo::direction::full_set()) {
		if (sibling_lists[di].size()) {
			futs.push_back(neighbors[di].force_nodes_to_exist(std::move(sibling_lists[di])));
		}
	}
	if (parent_list.size()) {
		futs.push_back(parent.force_nodes_to_exist(std::move(parent_list)));
	}

//	wait_all_and_propagate_exceptions(futs);
	for (auto &f : futs) {
		GET(f);
	}
}

typedef node_server::form_tree_action form_tree_action_type;
HPX_REGISTER_ACTION(form_tree_action_type);

future<int> node_client::form_tree(hpx::id_type &&id1, hpx::id_type &&id2, std::vector<hpx::id_type> &&ids) {
	return hpx::async<typename node_server::form_tree_action>(get_unmanaged_gid(), std::move(id1), std::move(id2), std::move(ids));
}

bool operator!=(const node_client &nc, const hpx::id_type &id) {
	return nc.get_gid() != id;
}

int node_server::form_tree(hpx::id_type self_gid, hpx::id_type parent_gid, std::vector<hpx::id_type> neighbor_gids) {
	int amr_bnd = 0;

	std::fill(nieces.begin(), nieces.end(), 0);
	for (auto &dir : geo::direction::full_set()) {
		neighbors[dir] = std::move(neighbor_gids[dir]);
	}
	me = std::move(self_gid);
	node_registry::add(my_location, me);
	parent = std::move(parent_gid);
	if (is_refined) {
		std::array<future<int>, NCHILD> cfuts;
		integer index = 0;
		amr_flags.resize(NCHILD);
		for (integer cx = 0; cx != 2; ++cx) {
			for (integer cy = 0; cy != 2; ++cy) {
				for (integer cz = 0; cz != 2; ++cz) {
					std::array<future<hpx::id_type>, geo::direction::count()> child_neighbors_f;
					const integer ci = cx + 2 * cy + 4 * cz;
					for (integer dx = -1; dx != 2; ++dx) {
						for (integer dy = -1; dy != 2; ++dy) {
							for (integer dz = -1; dz != 2; ++dz) {
								if (!(dx == 0 && dy == 0 && dz == 0)) {
									const integer x = cx + dx + 2;
									const integer y = cy + dy + 2;
									const integer z = cz + dz + 2;
									geo::direction i;
									i.set(dx, dy, dz);
									auto &ref = child_neighbors_f[i];
									auto other_child = (x % 2) + 2 * (y % 2) + 4 * (z % 2);
									if (x / 2 == 1 && y / 2 == 1 && z / 2 == 1) {
										ref = hpx::make_ready_future < hpx::id_type > (hpx::unmanaged(children[other_child].get_gid()));
									} else {
										geo::direction dir = geo::direction((x / 2) + NDIM * ((y / 2) + NDIM * (z / 2)));
										node_location parent_loc = my_location.get_neighbor(dir);
										ref = neighbors[dir].get_child_client(parent_loc, other_child);
									}
								}
							}
						}
					}
					cfuts[index++] = hpx::dataflow(hpx::launch::sync, [this, ci](std::array<future<hpx::id_type>, geo::direction::count()> &&cns) {
						std::vector<hpx::id_type> child_neighbors(geo::direction::count());
						for (auto &dir : geo::direction::full_set()) {
							child_neighbors[dir] = GET(cns[dir]);
							amr_flags[ci][dir] = bool(child_neighbors[dir] == hpx::invalid_id);
						}
						return GET(children[ci].form_tree(hpx::unmanaged(children[ci].get_gid()), me.get_gid(), std::move(child_neighbors)));
					}, std::move(child_neighbors_f));
				}
			}
		}
		for (auto &f : cfuts) {
			amr_bnd += GET(f);
		}
		constexpr auto full_set = geo::octant::full_set();
		for (auto &ci : full_set) {
			const auto &flags = amr_flags[ci];
			for (auto &dir : geo::direction::full_set()) {
				if (dir.is_face()) {
					if (flags[dir]) {
						amr_bnd++;
					}
				}
			}
		}
	} else {
		std::vector<future<void>> nfuts;
		nfuts.reserve(NFACE);
		for (auto &f : geo::face::full_set()) {
			const auto &neighbor = neighbors[f.to_direction()];
			if (!neighbor.empty()) {
				nfuts.push_back(neighbor.set_child_aunt(me.get_gid(), f ^ 1).then(
                    hpx::util::annotated_function([this, f](future<set_child_aunt_type> &&n) {
					nieces[f] = GET(n);
				}, "node_server::form_tree::sync")));
			} else {
				nieces[f] = -2;
			}
		}
		for (auto &f : nfuts) {
			GET(f);
		}
	}
	return amr_bnd;
}

using get_child_client_action_type = node_server::get_child_client_action;
HPX_REGISTER_ACTION(get_child_client_action_type);

future<hpx::id_type> node_client::get_child_client(const node_location &parent_loc, const geo::octant &ci) {
	future<hpx::id_type> rfut;
#ifdef OCTOTIGER_USE_NODE_CACHE
	hpx::shared_future < hpx::id_type > sfut;
	bool found;
#endif
	if (get_gid() != hpx::invalid_id) {
#ifdef OCTOTIGER_USE_NODE_CACHE
		auto loc = parent_loc.get_child(ci);
		table_type::iterator entry;
		std::unique_lock<hpx::mutex> lock(node_cache_mutex);
		entry = node_cache.find(loc);
		found = bool(entry != node_cache.end());
		if (!found) {
			sfut = hpx::async<typename node_server::get_child_client_action>(get_unmanaged_gid(), ci);
			node_cache[loc] = sfut;
			lock.unlock();
		} else {
			lock.unlock();
			sfut = entry->second;
		}
		if (found) {
			++hits;
			if (sfut.is_ready()) {
				rfut = hpx::make_ready_future(entry->second.get());
			} else {
				found = false;
			}
		} else {
			++misses;
		}
		if (!found) {
			rfut = hpx::async(hpx::util::annotated_function([=]() {
						return sfut.get();
					}), "return_future_get_child_client");
		}
#else
		rfut = hpx::async<typename node_server::get_child_client_action>(get_unmanaged_gid(), ci);
		;
#endif
	} else {
		auto tmp = hpx::invalid_id;
		rfut = hpx::make_ready_future < hpx::id_type > (std::move(tmp));
	}
	return rfut;
}

hpx::id_type node_server::get_child_client(const geo::octant &ci) {
	if (is_refined) {
		return children[ci].get_gid();
	} else {
		return hpx::invalid_id;
	}
}

using set_child_aunt_action_type = node_server::set_child_aunt_action;
HPX_REGISTER_ACTION(set_child_aunt_action_type);

future<set_child_aunt_type> node_client::set_child_aunt(const hpx::id_type &aunt, const geo::face &f) const {
	return hpx::async<typename node_server::set_child_aunt_action>(get_unmanaged_gid(), aunt, f);
}

set_child_aunt_type node_server::set_child_aunt(const hpx::id_type &aunt, const geo::face &face) const {
	if (is_refined) {
		std::array<future<void>, geo::octant::count() / 2> futs;
		integer index = 0;
		for (auto const &ci : geo::octant::face_subset(face)) {
			futs[index++] = children[ci].set_aunt(aunt, face);
		}
//        wait_all_and_propagate_exceptions(futs);
		for (auto &f : futs) {
			GET(f);
		}
	} else {
		for (auto const &ci : geo::octant::face_subset(face)) {
			if (children[ci].get_gid() != hpx::invalid_id) {
				print("CHILD SHOULD NOT EXIST\n");
				abort();
			}
		}
	}
#ifdef NIECE_BOOL
	return is_refined;
#else
	return is_refined ? +1 : -1;
#endif
}

using get_ptr_action_type = node_server::get_ptr_action;
HPX_REGISTER_ACTION(get_ptr_action_type);

std::uintptr_t node_server::get_ptr() {
	return reinterpret_cast<std::uintptr_t>(this);
}

future<node_server*> node_client::get_ptr() const {
	return hpx::async<typename node_server::get_ptr_action>(get_unmanaged_gid()).then(hpx::util::annotated_function([this](future<std::uintptr_t> &&fut) {
		if (hpx::find_here() != hpx::get_colocation_id(get_gid()).get()) {
			print("get_ptr called non-locally\n");
			abort();
		}
		return reinterpret_cast<node_server*>(GET(fut));
	}, "node_client::get_ptr"));
}
#endif
