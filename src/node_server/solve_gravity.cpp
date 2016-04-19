/*
 * solve_gravity.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */

#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::solve_gravity_action solve_gravity_action_type;
HPX_REGISTER_ACTION (solve_gravity_action_type);

hpx::future<void> node_client::solve_gravity(bool ene) const {
	return hpx::async<typename node_server::solve_gravity_action>(get_gid(), ene);
}


void node_server::solve_gravity(bool ene) {
	std::list<hpx::future<void>> child_futs;
	for (auto& child : children) {
		child_futs.push_back(child.solve_gravity(ene));
	}
	compute_fmm(RHO, ene);
	for (auto&& fut : child_futs) {
		GET(fut);
	}
}
