/*
 * node_client.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"

hpx::id_type node_client::get_gid() const {
	hpx::id_type i;
	if( valid() ) {
		i = base_type::get_gid();
	} else {
		i = hpx::invalid_id;
	}
	return i;
}

bool node_client::empty() const {
	bool rc;
	if( valid() ) {
		if( get_gid() == hpx::invalid_id) {
			rc = true;
		} else {
			rc = false;
		}
	} else {
		rc = true;
	}
	return rc;
}

