/*
 * node_client.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"

bool node_client::is_local() {
	return local;
}

hpx::id_type node_client::get_gid() const {
	return id;
}

node_client& node_client::operator=(hpx::future<hpx::id_type>&& fut) {
	id = fut.get();
	if( !empty() ) {
		local = bool(hpx::get_colocation_id(id).get() == hpx::find_here());
	}
	return *this;
}

node_client& node_client::operator=(const hpx::id_type& _id) {
	id = _id;
	if (!empty()) {
		local = bool(hpx::get_colocation_id(id).get() == hpx::find_here());
	}
	return *this;
}

node_client::node_client(hpx::future<hpx::id_type>&& fut) {
	id = fut.get();
	if( !empty() ) {
		local = bool(hpx::get_colocation_id(id).get() == hpx::find_here());
	}
}

node_client::node_client(const hpx::id_type& _id) {
	id = _id;
	if (!empty()) {
		local = bool(hpx::get_colocation_id(id).get() == hpx::find_here());
	}
}

node_client::node_client() {
	local = true;
}

//hpx::future<grid::output_list_type> node_client::output() const {
//	return hpx::async<typename node_server::output_action>(get_gid(), std::string(""));
//}

bool node_client::empty() const {
	return get_gid() == hpx::invalid_id;
}

void node_client::report_timing() const {
    hpx::async<typename node_server::report_timing_action>(get_gid()).get();
}
