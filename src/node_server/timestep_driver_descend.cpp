/*
 * timestep_driver.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */

#include "../node_server.hpp"
#include "../node_client.hpp"


typedef node_server::timestep_driver_descend_action timestep_driver_descend_action_type;
HPX_REGISTER_ACTION (timestep_driver_descend_action_type);




hpx::future<real> node_client::timestep_driver_descend() const {
	return hpx::async<typename node_server::timestep_driver_descend_action>(get_gid());
}

real node_server::timestep_driver_descend() {
	real dt;
	if (is_refined) {
		dt = std::numeric_limits < real > ::max();
		std::list<hpx::future<real>> futs;
		for (auto i = children.begin(); i != children.end(); ++i) {
			futs.push_back(i->timestep_driver_descend());
		}
		for (auto i = futs.begin(); i != futs.end(); ++i) {
			dt = std::min(dt, GET(*i));
		}
		dt = std::min(GET(local_timestep_channel->get_future()), dt);
	} else {
		dt = GET(local_timestep_channel->get_future());
	}
	return dt;
}
