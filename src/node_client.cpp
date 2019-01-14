/*
 * node_client.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: dmarce1
 */

#include "octotiger/future.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"

#include <hpx/include/naming.hpp>
#include <hpx/lcos/broadcast.hpp>

bool node_client::is_local() const {
	return local;
}

hpx::id_type node_client::get_gid() const {
	return id;
}

hpx::id_type node_client::get_unmanaged_gid() const {
	assert(unmanaged != hpx::invalid_id);
	return unmanaged;
}

node_client& node_client::operator=(hpx::future<hpx::id_type>&& fut) {
	id = fut.get();
	if( !empty() ) {
        unmanaged = hpx::id_type(id.get_gid(), hpx::id_type::unmanaged);
        local = hpx::naming::get_locality_from_id(id) == hpx::find_here();
// 		local = bool(hpx::get_colocation_id(id).get() == hpx::find_here());
	}
	return *this;
}

node_client& node_client::operator=(const hpx::id_type& _id) {
	id = _id;
	if (!empty()) {
        unmanaged = hpx::id_type(id.get_gid(), hpx::id_type::unmanaged);
        local = hpx::naming::get_locality_from_id(id) == hpx::find_here();
// 		local = bool(hpx::get_colocation_id(id).get() == hpx::find_here());
	}
	return *this;
}

node_client::node_client(hpx::future<hpx::id_type>&& fut) {
	id = fut.get();
	if( !empty() ) {
        unmanaged = hpx::id_type(id.get_gid(), hpx::id_type::unmanaged);
        local = hpx::naming::get_locality_from_id(id) == hpx::find_here();
// 		local = bool(hpx::get_colocation_id(id).get() == hpx::find_here());
	}
}

node_client::node_client(const hpx::id_type& _id) {
	id = _id;
	if (!empty()) {
        unmanaged = hpx::id_type(id.get_gid(), hpx::id_type::unmanaged);
        local = hpx::naming::get_locality_from_id(id) == hpx::find_here();
	}
}

node_client::node_client() {
	local = true;
}


bool node_client::empty() const {
	return get_gid() == hpx::invalid_id;
}

void node_client::report_timing() const {
	   hpx::async<typename node_server::report_timing_action>(get_gid()).get();
	}

