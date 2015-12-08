/*
 * node_client.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"

hpx::id_type node_client::get_gid() const {
	return id;
}

hpx::future<std::pair<real,real>> node_client::find_omega_part(const space_vector& pivot) const {
	return hpx::async<typename node_server::find_omega_part_action>(get_gid(), pivot);
}

node_client& node_client::operator=(hpx::future<hpx::id_type>&& fut ) {
	id = fut.get();
	return *this;
}

hpx::future<std::vector<hpx::id_type>> node_client::get_nieces(const hpx::id_type& aunt, const geo::face& f) const {
	return hpx::async<typename node_server::get_nieces_action>(get_gid(), aunt, f);
}

hpx::future<bool> node_client::check_for_refinement() const {
	return hpx::async<typename node_server::check_for_refinement_action>(get_gid());
}


hpx::future<void> node_client::force_nodes_to_exist(std::list<node_location>&& locs) const {
	return hpx::async<typename node_server::force_nodes_to_exist_action>(get_gid(), std::move(locs));
}

hpx::future<void> node_client::set_aunt(const hpx::id_type& aunt, const geo::face& f) const {
	return hpx::async<typename node_server::set_aunt_action>(get_gid(), aunt, f);
}

node_client& node_client::operator=(const hpx::id_type& _id) {
	id = _id;
	return *this;
}

node_client::node_client(hpx::future<hpx::id_type>&& fut ) {
	id = fut.get();
}

node_client::node_client(const hpx::id_type& _id) {
	id = _id;
}

hpx::future<void> node_client::form_tree(const hpx::id_type& id1, const hpx::id_type& id2,
		const std::vector<hpx::id_type>& ids) {
	return hpx::async<typename node_server::form_tree_action>(get_gid(), id1, id2, std::move(ids));
}

hpx::future<hpx::id_type> node_client::copy_to_locality(const hpx::id_type& id) const {
	return hpx::async<typename node_server::copy_to_locality_action>(get_gid(), id);
}

hpx::future<diagnostics_t> node_client::diagnostics() const {
	return hpx::async<typename node_server::diagnostics_action>(get_gid());
}

hpx::future<node_server*> node_client::get_ptr() const {
	return hpx::async<typename node_server::get_ptr_action>(get_gid()).then([](hpx::future<std::uintptr_t>&& fut) {
		return reinterpret_cast<node_server*>(fut.get());
	});
}

node_client::node_client() {
}

//hpx::future<grid::output_list_type> node_client::output() const {
//	return hpx::async<typename node_server::output_action>(get_gid(), std::string(""));
//}


hpx::future<grid::output_list_type> node_client::load(integer i, const hpx::id_type& _me, bool do_o) const {
	return hpx::async<typename node_server::load_action>(get_gid(), i, _me, do_o);
}

integer node_client::save(integer i) const {
	return hpx::async<typename node_server::save_action>(get_gid(), i).get();
}

hpx::future<void> node_client::solve_gravity(bool ene) const {
	return hpx::async<typename node_server::solve_gravity_action>(get_gid(), ene);
}

hpx::future<integer> node_client::regrid_gather(bool rb) const {
	return hpx::async<typename node_server::regrid_gather_action>(get_gid(), rb);
}

hpx::future<void> node_client::regrid_scatter(integer a, integer b) const {
	return hpx::async<typename node_server::regrid_scatter_action>(get_gid(), a, b);
}

hpx::future<hpx::id_type> node_client::get_child_client(const geo::octant& ci) {
	if (get_gid() != hpx::invalid_id) {
		return hpx::async<typename node_server::get_child_client_action>(get_gid(), ci);
	} else {
		auto tmp = hpx::invalid_id;
		return hpx::make_ready_future<hpx::id_type>(std::move(tmp));
	}
}

hpx::future<void> node_client::send_hydro_boundary(std::vector<real>&& data, const geo::direction& dir) const {
	return hpx::async<typename node_server::send_hydro_boundary_action>(get_gid(), std::move(data), dir);
}

bool node_client::empty() const {
	return get_gid() == hpx::invalid_id;
}

hpx::future<void> node_client::send_gravity_boundary(std::vector<real>&& data, const geo::direction& dir, bool monopole) const {
	return hpx::async<typename node_server::send_gravity_boundary_action>(get_gid(), std::move(data), dir, monopole);
}

hpx::future<void> node_client::send_gravity_multipoles(multipole_pass_type&& data, const geo::octant& ci) const {
	return hpx::async<typename node_server::send_gravity_multipoles_action>(get_gid(), std::move(data), ci);
}

hpx::future<void> node_client::send_hydro_children(std::vector<real>&& data, const geo::octant& ci) const {
	return hpx::async<typename node_server::send_hydro_children_action>(get_gid(), std::move(data), ci);
}

hpx::future<void> node_client::send_hydro_flux_correct(std::vector<real>&& data, const geo::face& face, const geo::octant& ci) const {
	return hpx::async<typename node_server::send_hydro_flux_correct_action>(get_gid(), std::move(data), face, ci);
}

hpx::future<void> node_client::send_gravity_expansions(expansion_pass_type&& data) const {
	return hpx::async<typename node_server::send_gravity_expansions_action>(get_gid(), std::move(data));
}

hpx::future<void> node_client::step() const {
	return hpx::async<typename node_server::step_action>(get_gid());
}

hpx::future<void> node_client::start_run() const {
	return hpx::async<typename node_server::start_run_action>(get_gid());
}

hpx::future<void> node_client::set_grid(std::vector<real>&& g, std::vector<real>&& o) const {
	return hpx::async<typename node_server::set_grid_action>(get_gid(), g, o);
}

hpx::future<void> node_client::regrid(const hpx::id_type& g, bool rb) const {
	return hpx::async<typename node_server::regrid_action>(get_gid(), g, rb);
}

hpx::future<real> node_client::timestep_driver() const {
	return hpx::async<typename node_server::timestep_driver_action>(get_gid());
}

hpx::future<void> node_client::timestep_driver_ascend(real dt) const {
	return hpx::async<typename node_server::timestep_driver_ascend_action>(get_gid(), dt);
}

hpx::future<real> node_client::timestep_driver_descend() const {
	return hpx::async<typename node_server::timestep_driver_descend_action>(get_gid());
}

