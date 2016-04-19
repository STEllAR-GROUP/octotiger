/*
 * set_aunt.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */





#include "../node_server.hpp"
#include "../node_client.hpp"



typedef node_server::set_aunt_action set_aunt_action_type;
HPX_REGISTER_ACTION (set_aunt_action_type);


hpx::future<void> node_client::set_aunt(const hpx::id_type& aunt, const geo::face& f) const {
	return hpx::async<typename node_server::set_aunt_action>(get_gid(), aunt, f);
}

void node_server::set_aunt(const hpx::id_type& aunt, const geo::face& face) {
	aunts[face] = aunt;
}

