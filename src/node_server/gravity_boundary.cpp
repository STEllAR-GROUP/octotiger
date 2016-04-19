/*
 * gravity_boundary.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */


#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::send_gravity_boundary_action send_gravity_boundary_action_type;
HPX_REGISTER_ACTION (send_gravity_boundary_action_type);




#ifdef USE_SPHERICAL
hpx::future<void> node_client::send_gravity_boundary(std::vector<multipole_type>&& data, const geo::direction& dir) const {
	return hpx::async<typename node_server::send_gravity_boundary_action>(get_gid(), std::move(data), dir);
}

#else
hpx::future<void> node_client::send_gravity_boundary(std::vector<real>&& data, const geo::direction& dir, bool monopole) const {
	return hpx::async<typename node_server::send_gravity_boundary_action>(get_gid(), std::move(data), dir, monopole);
}

#endif



#ifdef USE_SPHERICAL
void node_server::recv_gravity_multipoles(std::vector<multipole_type>&& v, const geo::octant& ci) {
	child_gravity_channels[ci]->set_value(std::move(v));
}
#else
void node_server::recv_gravity_multipoles(multipole_pass_type&& v, const geo::octant& ci) {
	child_gravity_channels[ci]->set_value(std::move(v));
}

#endif
