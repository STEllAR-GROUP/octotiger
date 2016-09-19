/*
 * gravity_expansions.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */




#include "../node_server.hpp"
#include "../node_client.hpp"
#include "taylor.hpp"



typedef node_server::send_gravity_expansions_action send_gravity_expansions_action_type;
HPX_REGISTER_ACTION (send_gravity_expansions_action_type);



#ifdef USE_SPHERICAL

hpx::future<void> node_client::send_gravity_expansions(std::vector<expansion_type>&& data) const {
	return hpx::async<typename node_server::send_gravity_expansions_action>(get_gid(), std::move(data));
}
#else

hpx::future<void> node_client::send_gravity_expansions(expansion_pass_type&& data) const {
	return hpx::async<typename node_server::send_gravity_expansions_action>(get_gid(), std::move(data));
}

#endif


#ifdef USE_SPHERICAL

void node_server::recv_gravity_boundary(std::vector<multipole_type>&& bdata, const geo::direction& dir) {
	neighbor_gravity_channels[dir]->set_value(std::move(bdata));
}
#else
void node_server::recv_gravity_boundary(std::vector<real>&& bdata, const geo::direction& dir, bool monopole) {
	neighbor_gravity_channels[dir]->set_value(std::make_pair(std::move(bdata), monopole));
}
#endif
