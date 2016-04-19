/*
 * hydro_children.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */


#include "../node_server.hpp"
#include "../node_client.hpp"


typedef node_server::send_hydro_children_action send_hydro_children_action_type;
HPX_REGISTER_ACTION (send_hydro_children_action_type);

void node_server::recv_hydro_children(std::vector<real>&& data, const geo::octant& ci) {
	child_hydro_channels[ci]->set_value(std::move(data));
}

hpx::future<void> node_client::send_hydro_children(std::vector<real>&& data, const geo::octant& ci) const {
	return hpx::async<typename node_server::send_hydro_children_action>(get_gid(), std::move(data), ci);
}
