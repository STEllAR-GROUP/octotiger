/*
 * node_geometry.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"
#include "future.hpp"


bool node_server::check_for_refinement() {
	bool rc = false;
	if (is_refined) {
		std::vector<hpx::future<bool>> futs;
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.check_for_refinement());
		}
		for (auto& fut : futs) {
			if (rc) {
				GET(fut);
			} else {
				rc = GET(fut);
			}
		}
	}
	if (!rc) {
		rc = grid_ptr->refine_me(my_location.level());
	}
	if (rc) {
		if (refinement_flag++ == 0) {
			if (!parent.empty()) {
				const auto neighbors = my_location.get_neighbors();
				parent.force_nodes_to_exist(std::list<node_location>(neighbors.begin(), neighbors.end())).get();
			}
		}
	}
	return refinement_flag;
}

integer node_server::regrid_gather(bool rebalance_only) {
	integer count = integer(1);

	if (is_refined) {
		if (!rebalance_only) {
			/* Turning refinement off */
			if (refinement_flag == 0) {
				children.clear();
				is_refined = false;
				grid_ptr->set_leaf(true);
			}
		}

		if (is_refined) {
			std::list<hpx::future<integer>> futs;
			for (auto& child : children) {
				futs.push_back(child.regrid_gather(rebalance_only));
			}
			for (auto& ci : geo::octant::full_set()) {
				auto child_cnt = futs.begin()->get();
				futs.pop_front();
				child_descendant_count[ci] = child_cnt;
				count += child_cnt;
			}
		} else {
			for (auto& ci : geo::octant::full_set()) {
				child_descendant_count[ci] = 0;
			}
		}
	} else if (!rebalance_only) {
		//		if (grid_ptr->refine_me(my_location.level())) {
		if (refinement_flag != 0) {
			refinement_flag = 0;
			count += NCHILD;

			children.resize(NCHILD);
			std::vector<node_location> clocs(NCHILD);

			/* Turning refinement on*/
			is_refined = true;
			grid_ptr->set_leaf(false);

			for (auto& ci : geo::octant::full_set()) {
				child_descendant_count[ci] = 1;
				children[ci] = hpx::new_<node_server>(hpx::find_here(), my_location.get_child(ci), me, current_time, rotational_time);
				std::array<integer, NDIM> lb = { 2*H_BW, 2*H_BW, 2*H_BW };
				std::array<integer, NDIM> ub;
				lb[XDIM] += (1 & (ci >> 0)) * (INX);
				lb[YDIM] += (1 & (ci >> 1)) * (INX);
				lb[ZDIM] += (1 & (ci >> 2)) * (INX);
				for (integer d = 0; d != NDIM; ++d) {
					ub[d] = lb[d] + (INX);
				}
				std::vector<real> outflows(NF, ZERO);
				if (ci == 0) {
					outflows = grid_ptr->get_outflows();
				}
				if (current_time > ZERO) {
					children[ci].set_grid(grid_ptr->get_prolong(lb, ub), std::move(outflows)).get();
				}
			}
		}
	}

	return count;
}

void node_server::set_grid(const std::vector<real>& data, std::vector<real>&& outflows) {
	grid_ptr->set_prolong(data, std::move(outflows));
}

void node_server::regrid(const hpx::id_type& root_gid, bool rb) {
	assert(grid_ptr != nullptr);
	printf("-----------------------------------------------\n");
	if (!rb) {
		printf("checking for refinement\n");
		check_for_refinement();
	}
	printf("regridding\n");
	integer a = regrid_gather(rb);
	printf("rebalancing %i nodes\n", int(a));
	regrid_scatter(0, a);
	assert(grid_ptr != nullptr);
	std::vector<hpx::id_type> null_neighbors(geo::direction::count());
	printf("forming tree connections\n");
	form_tree(root_gid, hpx::invalid_id, null_neighbors);
	//if (current_time > ZERO) {
	printf("solving gravity\n");
	solve_gravity(true);
	//}
	printf("regrid done\n-----------------------------------------------\n");
}

void node_server::regrid_scatter(integer a_, integer total) {
	refinement_flag = 0;
	std::list<hpx::future<void>> futs;
	if (is_refined) {
		integer a = a_;
		const auto localities = hpx::find_all_localities();
		++a;
		for (auto& ci : geo::octant::full_set()) {
			const integer loc_index = a * localities.size() / total;
			const auto child_loc = localities[loc_index];
			a += child_descendant_count[ci];
			const hpx::id_type id = children[ci].get_gid();
			integer current_child_id = hpx::naming::get_locality_id_from_gid(id.get_gid());
			auto current_child_loc = localities[current_child_id];
			if (child_loc != current_child_loc) {
				children[ci] = children[ci].copy_to_locality(child_loc);
			}
		}
		a = a_ + 1;
		for (auto& ci : geo::octant::full_set()) {
			futs.push_back(children[ci].regrid_scatter(a, total));
			a += child_descendant_count[ci];
		}
	}
	clear_family();
	for (auto&& fut : futs) {
		GET(fut);
	}
}

void node_server::clear_family() {
	parent = hpx::invalid_id;
	me = hpx::invalid_id;
	std::fill(aunts.begin(), aunts.end(), hpx::invalid_id);
	std::fill(siblings.begin(), siblings.end(), hpx::invalid_id);
	std::fill(neighbors.begin(), neighbors.end(), hpx::invalid_id);
	std::fill(nieces.begin(), nieces.end(), std::vector<node_client>());
}

void node_server::form_tree(const hpx::id_type& self_gid, const hpx::id_type& parent_gid,
		const std::vector<hpx::id_type>& neighbor_gids) {
	for (auto& dir : geo::direction::full_set()) {
		neighbors[dir] = neighbor_gids[dir];
	}
	for (auto& face : geo::face::full_set()) {
		siblings[face] = neighbors[face.to_direction()];
	}

	std::list<hpx::future<void>> cfuts;
	me = self_gid;
	parent = parent_gid;
	if (is_refined) {
		amr_flags.resize(NCHILD);
		for (integer cx = 0; cx != 2; ++cx) {
			for (integer cy = 0; cy != 2; ++cy) {
				for (integer cz = 0; cz != 2; ++cz) {
					std::vector<hpx::future<hpx::id_type>> child_neighbors_f(geo::direction::count());
					std::vector<hpx::id_type> child_neighbors(geo::direction::count());
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
									auto& ref = child_neighbors_f[i];
									auto other_child = (x % 2) + 2 * (y % 2) + 4 * (z % 2);
									if (x / 2 == 1 && y / 2 == 1 && z / 2 == 1) {
										ref = hpx::make_ready_future<hpx::id_type>(children[other_child].get_gid());
									} else {
										geo::direction dir = geo::direction(
												(x / 2) + NDIM * ((y / 2) + NDIM * (z / 2)));
										ref = neighbors[dir].get_child_client(other_child);
									}
								}
							}
						}
					}

					for (auto& dir : geo::direction::full_set()) {
						child_neighbors[dir] = child_neighbors_f[dir].get();
						if( child_neighbors[dir] == hpx::invalid_id) {
							amr_flags[ci][dir] = true;
						} else {
							amr_flags[ci][dir] = false;
						}
					}
					cfuts.push_back(
							children[ci].form_tree(children[ci].get_gid(), me.get_gid(), std::move(child_neighbors)));
				}
			}
		}

		for (auto&& fut : cfuts) {
			GET(fut);
		}

	} else {
		std::vector<hpx::future<std::vector<hpx::id_type>>>nfuts(NFACE);
		for (auto& f : geo::face::full_set()) {
			if( !siblings[f].empty()) {
				nfuts[f] = siblings[f].get_nieces(me.get_gid(), f^1);
			} else {
				nfuts[f] = hpx::make_ready_future(std::vector<hpx::id_type>());
			}
		}
		for (auto& f : geo::face::full_set()) {
			auto ids = nfuts[f].get();
			nieces[f].resize(ids.size());
			for( std::size_t i = 0; i != ids.size(); ++i ) {
				nieces[f][i] = ids[i];
			}
		}
	}

}

hpx::id_type node_server::get_child_client(const geo::octant& ci) {
	if (is_refined) {
		return children[ci].get_gid();
	} else {
		return hpx::invalid_id;
	}
}

hpx::future<hpx::id_type> node_server::copy_to_locality(const hpx::id_type& id) {

	std::vector<hpx::id_type> cids;
	if (is_refined) {
		cids.resize(NCHILD);
		for (auto& ci : geo::octant::full_set()) {
			cids[ci] = children[ci].get_gid();
		}
	}
	auto rc = hpx::new_<node_server>(id, my_location, step_num, is_refined, current_time, rotational_time, child_descendant_count,
			*grid_ptr, cids);
	clear_family();
	return rc;
}

void node_server::force_nodes_to_exist(const std::list<node_location>& locs) {
	std::list<hpx::future<void>> futs;
	std::list<node_location> parent_list;
	std::vector<std::list<node_location>> child_lists(NCHILD);
	for (auto& loc : locs) {
		assert(loc != my_location);
		if (loc.is_child_of(my_location)) {
			if (refinement_flag++ == 0 && !parent.empty()) {
				const auto neighbors = my_location.get_neighbors();
				parent.force_nodes_to_exist(std::list<node_location>(neighbors.begin(), neighbors.end())).get();
			}
			if (is_refined) {
				for (auto& ci : geo::octant::full_set()) {
					if (loc.is_child_of(my_location.get_child(ci))) {
						child_lists[ci].push_back(loc);
						break;
					}
				}
			}
		} else {
			assert(!parent.empty());
			parent_list.push_back(loc);
		}
	}
	for (auto& ci : geo::octant::full_set()) {
		if (is_refined && child_lists[ci].size()) {
			futs.push_back(children[ci].force_nodes_to_exist(std::move(child_lists[ci])));
		}
	}
	if (parent_list.size()) {
		futs.push_back(parent.force_nodes_to_exist(std::move(parent_list)));
	}
	for (auto&& fut : futs) {
		GET(fut);
	}
}

