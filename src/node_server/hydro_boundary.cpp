/*
 * recv_hydro_boundary.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */




#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::send_hydro_boundary_action send_hydro_boundary_action_type;
HPX_REGISTER_ACTION (send_hydro_boundary_action_type);


hpx::future<void> node_client::send_hydro_boundary(std::vector<real>&& data, const geo::direction& dir) const {
	return hpx::async<typename node_server::send_hydro_boundary_action>(get_gid(), std::move(data), dir);
}


void node_server::recv_hydro_boundary(std::vector<real>&& bdata, const geo::direction& dir) {
	sibling_hydro_channels[dir]->set_value(std::move(bdata));
}
