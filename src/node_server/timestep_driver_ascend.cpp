/*
 * timestep_driver.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */

#include "../node_server.hpp"
#include "../node_client.hpp"


typedef node_server::timestep_driver_ascend_action timestep_driver_ascend_action_type;
HPX_REGISTER_ACTION (timestep_driver_ascend_action_type);



hpx::future<void> node_client::timestep_driver_ascend(real dt) const {
	return hpx::async<typename node_server::timestep_driver_ascend_action>(get_gid(), dt);
}



void node_server::timestep_driver_ascend(real dt) {
	global_timestep_channel->set_value(dt);
	if (is_refined) {
		std::list<hpx::future<void>> futs;
		for (auto i = children.begin(); i != children.end(); ++i) {
			futs.push_back(i->timestep_driver_ascend(dt));
		}
		for (auto i = futs.begin(); i != futs.end(); ++i) {
			GET(*i);
		}
	}
}
