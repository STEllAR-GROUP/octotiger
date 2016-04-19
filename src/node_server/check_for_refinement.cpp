/*
 * check_for_refinement.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */





#include "../node_server.hpp"
#include "../node_client.hpp"


typedef node_server::check_for_refinement_action check_for_refinement_action_type;
HPX_REGISTER_ACTION (check_for_refinement_action_type);


hpx::future<bool> node_client::check_for_refinement() const {
	return hpx::async<typename node_server::check_for_refinement_action>(get_gid());
}


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
