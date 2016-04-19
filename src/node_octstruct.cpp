/*
 * node_geometry.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"
#include "future.hpp"


void node_server::clear_family() {
	parent = hpx::invalid_id;
	me = hpx::invalid_id;
	std::fill(aunts.begin(), aunts.end(), hpx::invalid_id);
	std::fill(siblings.begin(), siblings.end(), hpx::invalid_id);
	std::fill(neighbors.begin(), neighbors.end(), hpx::invalid_id);
	std::fill(nieces.begin(), nieces.end(), std::vector<node_client>());
}

