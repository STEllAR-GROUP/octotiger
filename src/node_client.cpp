/*
 * node_client.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"
//#include <boost/serialization/list.hpp>




hpx::future<void> node_client::form_tree(const hpx::id_type& id1, const hpx::id_type& id2,
		const std::vector<hpx::id_type>& ids) {
	return hpx::async<typename node_server::form_tree_action>(get_gid(), id1, id2, std::move(ids));
}
hpx::future<hpx::id_type> node_client::copy_to_locality(const hpx::id_type& id) const {
	return hpx::async<typename node_server::copy_to_locality_action>(get_gid(), id);
}

hpx::future<hpx::id_type> node_client::load_node(std::size_t filepos, const std::string& fname, const node_location& loc, const hpx::id_type& id ) {
	return hpx::async<typename node_server::load_node_action>(get_gid(),filepos, fname, loc, id);
}

hpx::future<diagnostics_t> node_client::diagnostics() const {
	return hpx::async<typename node_server::diagnostics_action>(get_gid());
}



hpx::future<node_server*> node_client::get_ptr() const	{
	return hpx::async<typename node_server::get_ptr_action>(get_gid()).then([](hpx::future<std::uintptr_t>&& fut){
		return reinterpret_cast<node_server*>(fut.get());
	});
}

node_client::node_client() {
}

hpx::future<grid::output_list_type> node_client::output() const {
	return hpx::async<typename node_server::output_action>(get_gid(), std::string(""));
}
hpx::future<std::pair<std::size_t,std::size_t>> node_client::save(integer loc_id, std::string fname) const {
	return hpx::async<typename node_server::save_action>(get_gid(), loc_id, fname);
}

hpx::future<void> node_client::solve_gravity(bool ene, integer c) const {
	return hpx::async<typename node_server::solve_gravity_action>(get_gid(), ene, c);
}

hpx::future<integer> node_client::regrid_gather() const {
	return hpx::async<typename node_server::regrid_gather_action>(get_gid());
}

hpx::future<void> node_client::regrid_scatter(integer a, integer b) const {
	return hpx::async<typename node_server::regrid_scatter_action>(get_gid(), a, b);
}
/*
 hpx::future<void> node_client::register_(const node_location& nloc) const {
 return hpx::async([=]() {
 auto fut = hpx::register_id_with_basename("node_location", get_gid(), nloc.unique_id());
 if( !fut.get() ) {
 printf( "Failed to register node at location %s\n", nloc.to_str().c_str());
 }
 });
 }*/

hpx::future<hpx::id_type> node_client::get_child_client(integer ci) {
	if (get_gid() != hpx::invalid_id) {
		return hpx::async<typename node_server::get_child_client_action>(get_gid(), ci);
	} else {
		return hpx::make_ready_future(hpx::invalid_id);
	}
}
/*
 hpx::future<void> node_client::unregister(const node_location& nloc) const {
 return hpx::unregister_id_with_basename("node_location", nloc.unique_id());
 }*/

hpx::future<void> node_client::send_hydro_boundary(std::vector<real>&& data, integer rk, integer face) const {
	return hpx::async<typename node_server::send_hydro_boundary_action>(get_gid(), std::move(data), rk, face);
}

hpx::future<void> node_client::send_gravity_boundary(std::vector<real>&& data, integer face, integer c) const {
	return hpx::async<typename node_server::send_gravity_boundary_action>(get_gid(), std::move(data), face, c);
}

hpx::future<void> node_client::send_gravity_multipoles( multipole_pass_type&& data, integer ci, integer c) const {
	return hpx::async<typename node_server::send_gravity_multipoles_action>(get_gid(), std::move(data), ci, c);
}

hpx::future<void>node_client::send_hydro_children( std::vector<real>&& data, integer rk, integer ci) const {
	return hpx::async<typename node_server::send_hydro_children_action>(get_gid(), std::move(data), rk, ci);
}


hpx::future<void> node_client::send_gravity_expansions( expansion_pass_type&& data, integer c) const {
	return hpx::async<typename node_server::send_gravity_expansions_action>(get_gid(), std::move(data), c);
}

hpx::future<void> node_client::step() const {
	return hpx::async<typename node_server::step_action>(get_gid());
}

hpx::future<void> node_client::start_run() const {
	return hpx::async<typename node_server::start_run_action>(get_gid());
}

hpx::future<void> node_client::regrid() const {
	return hpx::async<typename node_server::regrid_action>(get_gid());
}


hpx::future<real> node_client::timestep_driver() const {
	return hpx::async<typename node_server::timestep_driver_action>(get_gid());
}

hpx::future<void> node_client::timestep_driver_ascend(real dt) const {
	return hpx::async<typename node_server::timestep_driver_ascend_action>(get_gid(),dt);
}

hpx::future<real> node_client::timestep_driver_descend() const {
	return hpx::async<typename node_server::timestep_driver_descend_action>(get_gid());
}




