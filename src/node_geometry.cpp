/*
 * node_geometry.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"

bool node_server::check_for_refinement() {
	bool rc = false;
	if (is_refined) {
		std::vector<hpx::future<bool>> futs;
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.check_for_refinement());
		}
		for (auto& fut : futs) {
			if( rc ) {
				fut.get();
			} else {
				rc = fut.get();
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

integer node_server::regrid_gather() {
	integer count = integer(1);

	if (is_refined) {
		std::list<hpx::future<integer>> futs;

		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].regrid_gather());
		}

		/* Turning refinement off */
		if (refinement_flag == 0) {
			for (integer ci = 0; ci != NCHILD; ++ci) {
				children[ci] = hpx::invalid_id;
			}
			is_refined = false;
			grid_ptr->set_leaf(true);
		}

		for (integer ci = 0; ci != NCHILD; ++ci) {
			auto child_cnt = futs.begin()->get();
			futs.pop_front();
			child_descendant_count[ci] = child_cnt;
			count += child_cnt;
		}

	} else {
		//		if (grid_ptr->refine_me(my_location.level())) {
		if (refinement_flag != 0) {
			refinement_flag = 0;
			count += NCHILD;

			children.resize(NCHILD);
			std::vector<node_location> clocs(NCHILD);

			/* Turning refinement on*/
			is_refined = true;
			grid_ptr->set_leaf(false);

			for (integer ci = 0; ci != NCHILD; ++ci) {
				child_descendant_count[ci] = 1;
				children[ci] = hpx::new_<node_server>(hpx::find_here(), my_location.get_child(ci), me, current_time);
				std::array<integer, NDIM> lb = { HBW, HBW, HBW };
				std::array<integer, NDIM> ub;
				lb[XDIM] += (1 & (ci >> 0)) * (INX / 2);
				lb[YDIM] += (1 & (ci >> 1)) * (INX / 2);
				lb[ZDIM] += (1 & (ci >> 2)) * (INX / 2);
				for (integer d = 0; d != NDIM; ++d) {
					ub[d] = lb[d] + (INX / 2);
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
	grid_ptr->set(data, std::move(outflows));
}

void node_server::regrid(const hpx::id_type& root_gid) {
	assert(grid_ptr!=nullptr);
	printf("--------------------\nchecking for refinement\n");
	check_for_refinement();
	printf("regridding\n");
	integer a = regrid_gather();
	printf("rebalancing %i nodes\n", int(a));
	regrid_scatter(0, a);
	assert(grid_ptr!=nullptr);
	std::vector<hpx::id_type> null_sibs(NFACE);
	printf("forming tree connections\n");
	form_tree(root_gid, hpx::invalid_id, null_sibs);
	if (current_time > ZERO) {
		printf("solving gravity\n");
		solve_gravity(RHO);
	}
	printf("regrid done\n--------------------\n");
}

void node_server::regrid_scatter(integer a_, integer total) {
	refinement_flag = 0;
	std::list<hpx::future<void>> futs;
	if (is_refined) {
		integer a = a_;
		const auto localities = hpx::find_all_localities();
		++a;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			const integer loc_index = a * localities.size() / total;
			const auto child_loc = localities[loc_index];
			a += child_descendant_count[ci];
			const hpx::id_type id = children[ci].get_gid();
			integer current_child_id = hpx::naming::get_locality_id_from_gid(id.get_gid());
			auto current_child_loc = localities[current_child_id];
			if (child_loc != current_child_loc) {
				//		printf( "Moving %s from %i to %i\n", my_location.get_child(ci).to_str().c_str(), hpx::get_locality_id(), int(loc_index));
				children[ci] = children[ci].copy_to_locality(child_loc);
			}
		}
		a = a_ + 1;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].regrid_scatter(a, total));
			a += child_descendant_count[ci];
		}
	}
	clear_family();
	if (is_refined) {
		for (auto i = futs.begin(); i != futs.end(); ++i) {
			i->get();
		}
	}
}

void node_server::clear_family() {
	parent = hpx::invalid_id;
	me = hpx::invalid_id;
	std::fill(aunts.begin(), aunts.end(), hpx::invalid_id);
	std::fill(siblings.begin(), siblings.end(), hpx::invalid_id);
	std::fill(nieces.begin(), nieces.end(), std::vector<node_client>());
}

void node_server::form_tree(const hpx::id_type& self_gid, const hpx::id_type& parent_gid,
		const std::vector<hpx::id_type>& sib_gids) {
	for (integer si = 0; si != NFACE; ++si) {
		siblings[si] = sib_gids[si];
	}
	std::list<hpx::future<void>> cfuts;
	me = self_gid;
	parent = parent_gid;
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			std::vector<std::shared_ptr<hpx::future<hpx::id_type>>>child_sibs_f(NFACE);
			std::vector < hpx::id_type > child_sibs(NFACE);
			for (integer d = 0; d != NDIM; ++d) {
				const integer flip = ci ^ (1 << d);
				const integer bit = (ci >> d) & 1;
				const integer other = 2 * d + bit;
				const integer thisf = 2 * d + (1 - bit);
				child_sibs_f[thisf] = std::make_shared<hpx::future < hpx::id_type >>(hpx::make_ready_future<hpx::id_type>(children[flip].get_gid()));
				child_sibs_f[other] = std::make_shared<hpx::future < hpx::id_type >>(siblings[other].get_child_client(flip));
			}
			for( integer f = 0; f != NFACE; ++f) {
				child_sibs[f] = child_sibs_f[f]->get();
			}
			cfuts.push_back(children[ci].form_tree(children[ci].get_gid(), me.get_gid(), std::move(child_sibs)));
		}
		for( auto&& i : cfuts) {
			i.get();
		}
	} else {
		std::vector<hpx::future<std::vector<hpx::id_type>>> nfuts(NFACE);
		for( integer f = 0; f != NFACE; ++f) {
			if( !siblings[f].empty()) {
				nfuts[f] = siblings[f].get_nieces(me.get_gid(), f^1);
			} else {
				nfuts[f] = hpx::make_ready_future(std::vector<hpx::id_type>());
			}
		}
		for( integer f = 0; f != NFACE; ++f) {
			auto ids = nfuts[f].get();
			nieces[f].resize(ids.size());
			for( std::size_t i = 0; i != ids.size(); ++i ) {
				nieces[f][i] = ids[i];
			}
		}
	}

}

hpx::id_type node_server::get_child_client(integer ci) {
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
		for (integer ci = 0; ci != NCHILD; ++ci) {
			cids[ci] = children[ci].get_gid();
		}
	}
	auto rc = hpx::new_<node_server>(id, my_location, step_num, is_refined, current_time, child_descendant_count,
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
				for (integer ci = 0; ci != NCHILD; ++ci) {
					if (loc.is_child_of(my_location.get_child(ci))) {
						child_lists[ci].push_back(loc);
						break;
					}
				}
			}
		} else {
			assert( !parent.empty());
			parent_list.push_back(loc);
		}
	}
	for (integer ci = 0; ci != NCHILD; ++ci) {
		if (is_refined && child_lists[ci].size()) {
			futs.push_back(children[ci].force_nodes_to_exist(std::move(child_lists[ci])));
		}
	}
	if (parent_list.size()) {
		futs.push_back(parent.force_nodes_to_exist(std::move(parent_list)));
	}
	for (auto&& fut : futs) {
		fut.get();
	}
}

