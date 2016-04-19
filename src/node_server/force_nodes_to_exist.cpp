/*
 * force_nodes_to_exist.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */


#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::force_nodes_to_exist_action force_nodes_to_exist_action_type;
HPX_REGISTER_ACTION (force_nodes_to_exist_action_type);


hpx::future<void> node_client::force_nodes_to_exist(std::list<node_location>&& locs) const {
	return hpx::async<typename node_server::force_nodes_to_exist_action>(get_gid(), std::move(locs));
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

