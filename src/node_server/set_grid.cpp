/*
 * set_grid.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */



#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::set_grid_action set_grid_action_type;
HPX_REGISTER_ACTION (set_grid_action_type);



hpx::future<void> node_client::set_grid(std::vector<real>&& g, std::vector<real>&& o) const {
	return hpx::async<typename node_server::set_grid_action>(get_gid(), g, o);
}


void node_server::set_grid(const std::vector<real>& data, std::vector<real>&& outflows) {
	grid_ptr->set_prolong(data, std::move(outflows));
}
