/*
 * regrid.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */



#include "../node_server.hpp"
#include "../node_client.hpp"


typedef node_server::regrid_action regrid_action_type;
HPX_REGISTER_ACTION (regrid_action_type);


hpx::future<void> node_client::regrid(const hpx::id_type& g, bool rb) const {
	return hpx::async<typename node_server::regrid_action>(get_gid(), g, rb);
}


void node_server::regrid(const hpx::id_type& root_gid, bool rb) {
	assert(grid_ptr != nullptr);
	printf("-----------------------------------------------\n");
	if (!rb) {
		printf("checking for refinement\n");
		check_for_refinement();
	}
	printf("regridding\n");
	integer a = regrid_gather(rb);
	printf("rebalancing %i nodes\n", int(a));
	regrid_scatter(0, a);
	assert(grid_ptr != nullptr);
	std::vector<hpx::id_type> null_neighbors(geo::direction::count());
	printf("forming tree connections\n");
	form_tree(root_gid, hpx::invalid_id, null_neighbors);
	if (current_time > ZERO) {
		printf("solving gravity\n");
		solve_gravity(true);
	}
	printf("regrid done\n-----------------------------------------------\n");
}
