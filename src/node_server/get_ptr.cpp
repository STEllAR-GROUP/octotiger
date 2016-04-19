/*
 * get_ptr.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */



#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::get_ptr_action get_ptr_action_type;
HPX_REGISTER_ACTION (get_ptr_action_type);

std::uintptr_t node_server::get_ptr() {
	return reinterpret_cast<std::uintptr_t>(this);
}


hpx::future<node_server*> node_client::get_ptr() const {
	return hpx::async<typename node_server::get_ptr_action>(get_gid()).then([](hpx::future<std::uintptr_t>&& fut) {
		return reinterpret_cast<node_server*>(fut.get());
	});
}
