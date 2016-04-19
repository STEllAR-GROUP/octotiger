/*
 * regrid_scatter.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */





#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::regrid_scatter_action regrid_scatter_action_type;
HPX_REGISTER_ACTION (regrid_scatter_action_type);


hpx::future<void> node_client::regrid_scatter(integer a, integer b) const {
	return hpx::async<typename node_server::regrid_scatter_action>(get_gid(), a, b);
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
