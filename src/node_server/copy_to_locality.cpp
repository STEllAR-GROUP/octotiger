/*
 * copy_to_locality.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */




#include "../node_server.hpp"
#include "../node_client.hpp"


typedef node_server::copy_to_locality_action copy_to_locality_action_type;

HPX_REGISTER_ACTION (copy_to_locality_action_type);


hpx::future<hpx::id_type> node_client::copy_to_locality(const hpx::id_type& id) const {
	return hpx::async<typename node_server::copy_to_locality_action>(get_gid(), id);
}


hpx::future<hpx::id_type> node_server::copy_to_locality(const hpx::id_type& id) {

	std::vector<hpx::id_type> cids;
	if (is_refined) {
		cids.resize(NCHILD);
		for (auto& ci : geo::octant::full_set()) {
			cids[ci] = children[ci].get_gid();
		}
	}
	auto rc = hpx::new_<node_server>(id, my_location, step_num, is_refined, current_time, rotational_time,
			child_descendant_count, std::move(*grid_ptr), cids);
	clear_family();
	return rc;
}
