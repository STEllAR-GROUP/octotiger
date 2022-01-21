//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef NODE_SERVER_HPP_
#define NODE_SERVER_HPP_

#include "octotiger/print.hpp"
#include "octotiger/config/export_definitions.hpp"
#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/channel.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/future.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_location.hpp"
#include "octotiger/profiler.hpp"
#include "octotiger/io/silo.hpp"
//#include "octotiger/struct_eos.hpp"

#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/mutex.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <map>
#include <vector>


struct node_count_type {
	std::uint64_t total;
	std::uint64_t leaf;
	std::uint64_t amr_bnd;
	template<class A>
	void serialize(A& arc, unsigned) {
		arc & total;
		arc & leaf;
		arc & amr_bnd;
	}
	node_count_type() {
		total = leaf = amr_bnd = std::uint64_t(0);
	}
};

class OCTOTIGER_EXPORT node_server: public hpx::components::managed_component_base<node_server> {



private:
	struct sibling_hydro_type {
		std::vector<real> data;
		geo::direction direction;
	};
	integer position;
	std::atomic<integer> refinement_flag;
	node_location my_location;
	integer step_num;
	std::size_t rcycle;
	std::size_t hcycle;
	std::size_t gcycle;
	real current_time;
	real rotational_time;
	std::shared_ptr<grid> grid_ptr; //
	std::shared_ptr<rad_grid> rad_grid_ptr; //
	std::atomic<bool> is_refined;
	std::array<integer, NVERTEX> child_descendant_count;
	std::array<real, NDIM> xmin;
	real dx;

	/* this node*/
	node_client me;
	/* The parent is the node one level coarser that this node is a child of*/
	node_client parent;
	/* neighbors refers to the up to 26 adjacent neighbors to this node (on the same refinement level). These are in the directions
	 *  of the 6 faces, 12 edges, and 8 vertices of the subgrid cube. If there is an AMR boundary to a coarser level, that neighbor is empty. */
	std::vector<node_client> neighbors;
	/* Child refers to the up to 8 refined children of this node. Either all or none exist.*/
	std::array<node_client, NCHILD> children;
	/* nieces are the children of neighbors that are adjacent to this node. They are one level finer than this node
	 * . Only nieces in the face directions are needed, and in each
	 * face direction there are 4 adjacent neighbors (or zero). This is used for AMR boundary handling - interpolation onto finer boundaries and flux matchinig.*/
	std::vector<integer> nieces;
	/* An aunt is this node's parent's neighbor, so it is one level coarser.
	 *  Only aunts in the 6 face directions are required. Used for AMR boundary handling. */
	std::vector<node_client> aunts;

	std::vector<std::array<bool, geo::direction::count()>> amr_flags;
	hpx::lcos::local::spinlock mtx;
	hpx::lcos::local::spinlock prolong_mtx;
	channel<expansion_pass_type> parent_gravity_channel;
	std::array<semaphore, geo::direction::count()> neighbor_signals;
	std::array<unordered_channel<std::vector<real>>, NCHILD> child_hydro_channels;
	std::array<unordered_channel<neighbor_gravity_type>, geo::direction::count()> neighbor_gravity_channels;
	std::array<unordered_channel<sibling_hydro_type>, geo::direction::count()> sibling_hydro_channels;
	std::array<channel<multipole_pass_type>, NCHILD> child_gravity_channels;
	std::array<std::array<channel<std::vector<real>>, 4>, NFACE> niece_hydro_channels;
	channel<timestep_t> global_timestep_channel;
	std::array<channel<timestep_t>, NCHILD + 1> local_timestep_channels;

	timestep_t dt_;

public:
	timings timings_;

	real get_time() const {
		return current_time;
	}
	const grid& get_hydro_grid() const {
		return *grid_ptr;
	}
	grid& get_hydro_grid() {
		return *grid_ptr;
	}
	real get_rotation_count() const;
	node_server& operator=(node_server&&) = default;
	static std::uint64_t cumulative_nodes_count(bool);
	static std::uint64_t cumulative_leafs_count(bool);
	static std::uint64_t cumulative_amrs_count(bool);
	static void register_counters();
private:
	static hpx::mutex node_count_mtx;
	static node_count_type cumulative_node_count;
	static bool static_initialized;
	static std::atomic<integer> static_initializing;
	void initialize(real, real);
	void send_hydro_amr_boundaries(bool energy_only=false);
	void collect_hydro_boundaries(bool energy_only=false);
	static void static_initialize();
	void clear_family();
	hpx::future<void> exchange_flux_corrections();

	hpx::future<void> nonrefined_step();
	void refined_step();

	diagnostics_t root_diagnostics(const diagnostics_t& diags);
	diagnostics_t child_diagnostics(const diagnostics_t& diags);
	diagnostics_t local_diagnostics(const diagnostics_t& diags);
	hpx::future<real> local_step(integer steps);

public:
	integer get_step_num() const {
		return step_num;
	}
	void exchange_interlevel_hydro_data();
	void all_hydro_bounds();
	void energy_hydro_bounds();
	static bool child_is_on_face(integer ci, integer face) {
		return (((ci >> (face / 2)) & 1) == (face & 1));
	}
	bool refined() const {
		return is_refined;
	}
	void set_time( real t, real r ) {
		current_time = t;
		rotational_time = r;
	}
	node_server() {
		initialize(ZERO, ZERO);
	}
	~node_server();
	node_server(const node_location&);
	node_server(const node_location&, silo_load_t load_vars);
	node_server(const node_location&, const node_client& parent_id, real, real, std::size_t, std::size_t, std::size_t,
			std::size_t);

	integer get_position() const {
		return position;
	}

	void reconstruct_tree();

	/*TODO move radiation to*/
	node_server(const node_location&, integer, bool, real, real, const std::array<integer, NCHILD>&, grid,
			const std::vector<hpx::id_type>&, std::size_t, std::size_t, std::size_t, integer position);

	void report_timing();/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, report_timing, report_timing_action);

	node_count_type regrid_gather(bool rebalance_only);/**/HPX_DEFINE_COMPONENT_ACTION(node_server, regrid_gather, regrid_gather_action);

	hpx::future<hpx::id_type> create_child(hpx::id_type const& locality, integer ci);

	void regrid_scatter(integer, integer);/**/HPX_DEFINE_COMPONENT_ACTION(node_server, regrid_scatter, regrid_scatter_action);

	void recv_flux_check(std::vector<real>&&, const geo::direction&, std::size_t cycle);
	/**/HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_flux_check, send_flux_check_action);

	void recv_hydro_boundary(std::vector<real>&&, const geo::direction&, std::size_t cycle);
	/**/HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_hydro_boundary, send_hydro_boundary_action);

	void recv_hydro_amr_boundary(std::vector<real>&&, const geo::direction&, std::size_t cycle);
	/**/HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_hydro_amr_boundary, send_hydro_amr_boundary_action);

	void recv_rad_amr_boundary(std::vector<real>&&, const geo::direction&, std::size_t cycle);
	/**/HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_rad_amr_boundary, send_rad_amr_boundary_action);

	void recv_hydro_children(std::vector<real>&&, const geo::octant& ci, std::size_t cycle);
	/**/HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_hydro_children, send_hydro_children_action);

	void recv_hydro_flux_correct(std::vector<real>&&, const geo::face& face, const geo::octant& ci);
	/**/HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_hydro_flux_correct, send_hydro_flux_correct_action);

	void recv_gravity_boundary(gravity_boundary_type&&, const geo::direction&, bool monopole, std::size_t cycle);
	void recv_gravity_multipoles(multipole_pass_type&&, const geo::octant&);
	void recv_gravity_expansions(expansion_pass_type&&);

	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_gravity_boundary, send_gravity_boundary_action);/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_gravity_multipoles, send_gravity_multipoles_action);/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_gravity_expansions, send_gravity_expansions_action);

	hpx::future<real> step(integer steps);/**/HPX_DEFINE_COMPONENT_ACTION(node_server, step, step_action);

	void update();

	node_count_type regrid(const hpx::id_type& root_gid, real omega, real new_floor, bool rb, bool grav_energy_comp=true);

	void compute_fmm(gsolve_type gs, bool energy_account, bool allocate_only = false);

	void solve_gravity(bool ene, bool skip_solve);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, solve_gravity, solve_gravity_action);

	void execute_solver(bool scf, node_count_type);

	void set_grid(const std::vector<real>&, std::vector<real>&&);/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, set_grid, set_grid_action);

	hpx::future<void> timestep_driver_descend();

	void set_local_timestep(integer i, timestep_t dt);/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, set_local_timestep, set_local_timestep_action);

	void timestep_driver_ascend(timestep_t);/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, timestep_driver_ascend, timestep_driver_ascend_action);

	hpx::future<hpx::id_type> copy_to_locality(const hpx::id_type&);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, copy_to_locality, copy_to_locality_action);

	hpx::id_type get_child_client(const geo::octant&);/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, get_child_client, get_child_client_action);

	int form_tree(hpx::id_type, hpx::id_type=hpx::invalid_id, std::vector<hpx::id_type> = std::vector<hpx::id_type>(geo::direction::count()));/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, form_tree, form_tree_action);

	std::uintptr_t get_ptr();/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, get_ptr, get_ptr_action);

	analytic_t compare_analytic();/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, compare_analytic, compare_analytic_action);

	std::pair<real,real> amr_error();
	HPX_DEFINE_COMPONENT_ACTION(node_server, amr_error, amr_error_action);

	diagnostics_t diagnostics(const diagnostics_t&);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, diagnostics, diagnostics_action);

	diagnostics_t diagnostics();

	void set_aunt(const hpx::id_type&, const geo::face& face);/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, set_aunt, set_aunt_action);

	set_child_aunt_type set_child_aunt(const hpx::id_type&, const geo::face& face) const;/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, set_child_aunt, set_child_aunt_action);

	void check_for_refinement(real omega, real new_floor);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, check_for_refinement, check_for_refinement_action);

	void enforce_bc();/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, enforce_bc, enforce_bc_action);

	void force_nodes_to_exist(std::vector<node_location>&& loc);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, force_nodes_to_exist, force_nodes_to_exist_action);

	scf_data_t scf_params();/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, scf_params, scf_params_action);

	real scf_update(real, real, real, real, real, real, real, struct_eos, struct_eos);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, scf_update, scf_update_action);

	void velocity_inc(const space_vector& dv);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, velocity_inc, velocity_inc_action);

	line_of_centers_t line_of_centers(const std::pair<space_vector, space_vector>& line) const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, line_of_centers, line_of_centers_action);

	void rho_mult(real factor, real);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server,rho_mult, rho_mult_action);

	void rho_move(real);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server,rho_move, rho_move_action);

	void run_scf(std::string const& data_dir);

private:
	struct sibling_rad_type {
		std::vector<real> data;
		geo::direction direction;
	};

	std::array<unordered_channel<sibling_rad_type>, geo::direction::count()> sibling_rad_channels;
	std::array<unordered_channel<std::vector<real>>, NCHILD> child_rad_channels;
	unordered_channel<expansion_pass_type> parent_rad_channel;
public:
	hpx::future<void> exchange_rad_flux_corrections();
	void compute_radiation(real dt, real omega);
	hpx::future<void> exchange_interlevel_rad_data();
	void all_rad_bounds();

	void collect_radiation_bounds();
	void send_rad_amr_bounds();

	void recv_rad_flux_correct(std::vector<real>&&, const geo::face& face, const geo::octant& ci);/**/
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_rad_flux_correct, send_rad_flux_correct_action);

	void recv_rad_boundary(std::vector<real>&&, const geo::direction&, std::size_t cycle);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_rad_boundary, send_rad_boundary_action);

	void recv_rad_children(std::vector<real>&&, const geo::octant& ci, std::size_t cycle);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_rad_children, send_rad_children_action);

	std::array<std::array<channel<std::vector<real>>, 4>, NFACE> niece_rad_channels;

	void set_rad_grid(const std::vector<real>&/*, std::vector<real>&&*/);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, set_rad_grid, set_rad_grid_action);

	void erad_init();/**/HPX_DEFINE_COMPONENT_ACTION(node_server, erad_init, erad_init_action);

	void kill();
	HPX_DEFINE_COMPONENT_ACTION(node_server,kill);

	void change_units(real m, real l, real t, real k);/**/
	HPX_DEFINE_COMPONENT_ACTION(node_server, change_units, change_units_action);

	void set_cgs(bool change = true);

};

HPX_REGISTER_ACTION_DECLARATION(node_server::kill_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::change_units_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::rho_mult_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::line_of_centers_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::velocity_inc_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::scf_update_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::set_grid_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::force_nodes_to_exist_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::check_for_refinement_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::enforce_bc_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::set_aunt_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::set_child_aunt_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_hydro_children_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_hydro_flux_correct_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::regrid_gather_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::regrid_scatter_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_flux_check_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_hydro_boundary_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_hydro_amr_boundary_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_rad_amr_boundary_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_gravity_boundary_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_gravity_multipoles_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_gravity_expansions_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::step_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::solve_gravity_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::copy_to_locality_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::get_child_client_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::form_tree_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::get_ptr_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::diagnostics_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::timestep_driver_ascend_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::scf_params_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_rad_boundary_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_rad_children_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_rad_flux_correct_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::set_rad_grid_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::erad_init_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::amr_error_action);
//HPX_REGISTER_ACTION_DECLARATION(node_server::set_parent_action);

#endif /* NODE_SERVER_HPP_ */
