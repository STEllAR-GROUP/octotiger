/*
 * velocity_inc.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */



#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::velocity_inc_action velocity_inc_action_type;
HPX_REGISTER_ACTION (velocity_inc_action_type);


hpx::future<void> node_client::velocity_inc(const space_vector& dv) const {
	return hpx::async<typename node_server::velocity_inc_action>(get_gid(), dv);
}



void node_server::velocity_inc(const space_vector& dv) {
	if (is_refined) {
		std::vector<hpx::future<void>> futs;
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.velocity_inc(dv));
		}
		for (auto&& fut : futs) {
			fut.get();
		}
	} else {
		grid_ptr->velocity_inc(dv);
	}
}


