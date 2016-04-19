/*
 * get_child_client.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */

#include "../node_server.hpp"
#include "../node_client.hpp"


typedef node_server::get_child_client_action get_child_client_action_type;
HPX_REGISTER_ACTION (get_child_client_action_type);

hpx::future<hpx::id_type> node_client::get_child_client(const geo::octant& ci) {
	if (get_gid() != hpx::invalid_id) {
		return hpx::async<typename node_server::get_child_client_action>(get_gid(), ci);
	} else {
		auto tmp = hpx::invalid_id;
		return hpx::make_ready_future<hpx::id_type>(std::move(tmp));
	}
}


hpx::id_type node_server::get_child_client(const geo::octant& ci) {
	if (is_refined) {
		return children[ci].get_gid();
	} else {
		return hpx::invalid_id;
	}
}
