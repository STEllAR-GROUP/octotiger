/*
 * gravity_multipoles.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */





#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::send_gravity_multipoles_action send_gravity_multipoles_action_type;
HPX_REGISTER_ACTION (send_gravity_multipoles_action_type);



#ifdef USE_SPHERICAL

hpx::future<void> node_client::send_gravity_multipoles(std::vector<multipole_type>&& data,
		const geo::octant& ci) const {
	return hpx::async<typename node_server::send_gravity_multipoles_action>(get_gid(), std::move(data), ci);
}
#else

hpx::future<void> node_client::send_gravity_multipoles(multipole_pass_type&& data, const geo::octant& ci) const {
	return hpx::async<typename node_server::send_gravity_multipoles_action>(get_gid(), std::move(data), ci);
}

#endif


#ifdef USE_SPHERICAL
void node_server::recv_gravity_expansions(std::vector<expansion_type>&& v) {
	parent_gravity_channel->set_value(std::move(v));
}

#else

void node_server::recv_gravity_expansions(expansion_pass_type&& v) {
	parent_gravity_channel->set_value(std::move(v));
}
#endif
