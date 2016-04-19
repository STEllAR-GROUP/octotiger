/*
 * find_omega_part.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */



#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::find_omega_part_action find_omega_part_action_type;
HPX_REGISTER_ACTION (find_omega_part_action_type);


hpx::future<std::pair<real, real>> node_client::find_omega_part(const space_vector& pivot) const {
	return hpx::async<typename node_server::find_omega_part_action>(get_gid(), pivot);
}

std::pair<real, real> node_server::find_omega_part(const space_vector& pivot) const {
	std::pair<real, real> d;
	if (is_refined) {
		std::vector < hpx::future<std::pair<real, real>>>futs;
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.find_omega_part(pivot));
		}
		d.first = d.second = ZERO;
		for (auto&& fut : futs) {
			auto tmp = GET(fut);
			d.first += tmp.first;
			d.second += tmp.second;
		}
	} else {
		d = grid_ptr->omega_part(pivot);
	}
	return d;
}
