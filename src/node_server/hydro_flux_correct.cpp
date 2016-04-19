/*
 * hydro_children.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */


#include "../node_server.hpp"
#include "../node_client.hpp"


typedef node_server::send_hydro_flux_correct_action send_hydro_flux_correct_action_type;
HPX_REGISTER_ACTION (send_hydro_flux_correct_action_type);



hpx::future<void> node_client::send_hydro_flux_correct(std::vector<real>&& data, const geo::face& face,
		const geo::octant& ci) const {
	return hpx::async<typename node_server::send_hydro_flux_correct_action>(get_gid(), std::move(data), face, ci);
}

void node_server::recv_hydro_flux_correct(std::vector<real>&& data, const geo::face& face, const geo::octant& ci) {
	const geo::quadrant index(ci, face.get_dimension());
	niece_hydro_channels[face][index]->set_value(std::move(data));
}
