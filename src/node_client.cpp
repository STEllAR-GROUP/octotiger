/*
 * node_client.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"
#include "options.hpp"

extern options opts;

#include <hpx/lcos/broadcast.hpp>

#ifdef OCTOTIGER_USE_NODE_CACHE
typename node_client::table_type node_client::node_cache;
hpx::mutex node_client::node_cache_mutex;
std::atomic<integer> node_client::hits(0);
std::atomic<integer> node_client::misses(0);
#endif


bool node_client::is_local() const {
	return local;
}

hpx::id_type node_client::get_gid() const {
	return id;
}

hpx::id_type node_client::get_unmanaged_gid() const {
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
// 		local = bool(hpx::get_colocation_id(id).get() == hpx::find_here());
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



#ifdef OCTOTIGER_USE_NODE_CACHE

HPX_PLAIN_ACTION(node_client::cycle_node_cache, cycle_node_cache_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION (cycle_node_cache_action);
HPX_REGISTER_BROADCAST_ACTION (cycle_node_cache_action);


void node_client::cycle_node_cache() {
	hpx::future<void> fut;
	fut = hpx::make_ready_future();
	if (hpx::get_locality_id() == 0) {
		std::vector<hpx::id_type> remotes;
		remotes.reserve(options::all_localities.size() - 1);
		for (hpx::id_type const& id : options::all_localities) {
			if (id != hpx::find_here())
				remotes.push_back(id);
		}
		if (remotes.size() > 0) {
			hpx::lcos::broadcast < cycle_node_cache_action > (remotes);
		}
	}
	auto func = [&](){
		std::lock_guard<hpx::mutex> lock(node_cache_mutex);
		node_cache.clear();
		hits = misses = 0;
	};
	if (hpx::get_locality_id() == 0) {
		printf( "Node location cache efficiency\n");
		printf( "Hits: %i Misses: %i Hit Rate: %.1f%%\n", int(hits), int(misses), 100.0*real(hits) / real(hits+misses));
		hpx::apply(func);
	} else {
		func();
	}
}

#endif
