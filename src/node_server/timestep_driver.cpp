/*
 * timestep_driver.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */

#include "../node_server.hpp"
#include "../node_client.hpp"


typedef node_server::timestep_driver_action timestep_driver_action_type;
HPX_REGISTER_ACTION (timestep_driver_action_type);



hpx::future<real> node_client::timestep_driver() const {
	return hpx::async<typename node_server::timestep_driver_action>(get_gid());
}

real node_server::timestep_driver() {
	const real dt = timestep_driver_descend();
	timestep_driver_ascend(dt);
	return dt;
}

