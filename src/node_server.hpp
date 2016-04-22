/*
 * node_server.hpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#ifndef NODE_SERVER_HPP_
#define NODE_SERVER_HPP_

#include "defs.hpp"
#include "node_location.hpp"
#include "node_client.hpp"
#include "grid.hpp"
#include "geometry.hpp"
#include "channel.hpp"
#include "future.hpp"
#include <atomic>


class node_server: public hpx::components::managed_component_base<node_server>
#ifdef USE_SPHERICAL
, public fmm
#endif
{
public:
	static void set_gravity(bool);
	static void set_hydro(bool);
private:
	void set_omega_and_pivot();
	std::atomic<integer> refinement_flag;
	static bool hydro_on;
	static bool gravity_on;
	node_location my_location;
	integer step_num;
	real current_time;
	real rotational_time;
	std::shared_ptr<grid> grid_ptr; //
	bool is_refined;
	std::array<integer, NVERTEX> child_descendant_count;
	std::array<real, NDIM> xmin;
	real dx;
	node_client me;
	node_client parent;
	std::vector<node_client> neighbors;
	std::vector<node_client> siblings;
	std::vector<node_client> children;
	std::vector<std::vector<node_client> > nieces;
	std::vector<node_client> aunts;
	std::vector<std::array<bool, geo::direction::count()>> amr_flags;
	hpx::lcos::local::spinlock mtx;
	std::array<std::shared_ptr<channel<std::vector<real>>> ,geo::direction::count()> sibling_hydro_channels;
	std::array<std::shared_ptr<channel<std::vector<real>>>, NCHILD> child_hydro_channels;

#ifdef USE_SPHERICAL
	std::shared_ptr<channel<std::vector<expansion_type>>> parent_gravity_channel;
	std::array<std::shared_ptr<channel<std::vector<multipole_type>>>, geo::direction::count()> neighbor_gravity_channels;
	std::array<std::shared_ptr<channel<std::vector<multipole_type>>>, geo::direction::count()> child_gravity_channels;
#else
	std::shared_ptr<channel<expansion_pass_type>> parent_gravity_channel;
	std::array<std::shared_ptr<channel<std::pair<std::vector<real>, bool>>> ,geo::direction::count()>neighbor_gravity_channels;
	std::array<std::shared_ptr<channel<multipole_pass_type>>, NCHILD> child_gravity_channels;
#endif

	std::array<std::array<std::shared_ptr<channel<std::vector<real>>> , 4>, NFACE> niece_hydro_channels;
	std::shared_ptr<channel<real>> global_timestep_channel;
	std::shared_ptr<channel<real>> local_timestep_channel;
	hpx::mutex load_mutex;
public:
	real get_time() const {
		return current_time;
	}
	node_server& operator=(node_server&&) = default;

	template<class Archive>
	void serialize(Archive& arc, unsigned) {
		integer rf;
		arc & my_location;
		arc & step_num;
		arc & is_refined;
		arc & children;
		arc & parent;
		arc & neighbors;
		arc & siblings;
		arc & nieces;
		arc & aunts;
		arc & child_descendant_count;
		arc & current_time;
		arc & rotational_time;
		arc & xmin;
		arc & dx;
		arc & amr_flags;
		arc & *grid_ptr;
		rf = refinement_flag;
		arc & rf;
		refinement_flag = rf;
	}

	node_server(const node_location&, integer, bool, real, real, const std::array<integer,NCHILD>&, grid, const std::vector<hpx::id_type>&);
	node_server(node_server&& other) = default;
	std::size_t load_me(FILE* fp);
	std::size_t save_me(FILE* fp) const;
private:

	static bool static_initialized;
	static std::atomic<integer> static_initializing;

	void initialize(real, real);
	hpx::future<void> collect_hydro_boundaries();
	void exchange_interlevel_hydro_data();
	static void static_initialize();
	void clear_family();
	hpx::future<void> exchange_flux_corrections();

public:

//	static void output_form();
//	static grid::output_list_type output_collect(const std::string&);

	static bool child_is_on_face(integer ci, integer face);

	std::list<hpx::future<void>> set_nieces_amr(const geo::face&) const;
	node_server();
	~node_server();
	node_server( const node_server& other);
	node_server(const node_location&, const node_client& parent_id, real, real);
	void set_hydro_boundary(const std::vector<real>&, const geo::direction&);
	std::vector<real> get_hydro_boundary(const geo::direction& face);

	void save_to_file(const std::string&) const;
	void load_from_file(const std::string&);
	void load_from_file_and_output(const std::string&, const std::string&);
	void set_gravity_boundary(const std::vector<real>&, const geo::direction&, bool monopole);
	std::vector<real> get_gravity_boundary(const geo::direction& dir);

	grid::output_list_type output(std::string fname) const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, output, output_action);


	integer regrid_gather(bool rebalance_only);
	HPX_DEFINE_COMPONENT_ACTION(node_server, regrid_gather, regrid_gather_action);

	void regrid_scatter(integer, integer);
	HPX_DEFINE_COMPONENT_ACTION(node_server, regrid_scatter, regrid_scatter_action);

	void recv_hydro_boundary(std::vector<real>&&, const geo::direction&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_hydro_boundary, send_hydro_boundary_action);

	void recv_hydro_children(std::vector<real>&&, const geo::octant& ci);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_hydro_children, send_hydro_children_action);

	void recv_hydro_flux_correct(std::vector<real>&&, const geo::face& face, const geo::octant& ci);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_hydro_flux_correct, send_hydro_flux_correct_action);

#ifdef USE_SPHERICAL
	void recv_gravity_multipoles(std::vector<multipole_type>&& v, const geo::octant& ci);
	void recv_gravity_expansions(std::vector<expansion_type>&& v);
	void recv_gravity_boundary(std::vector<multipole_type>&& bdata, const geo::direction& dir);
#else
	void recv_gravity_boundary(std::vector<real>&&, const geo::direction&, bool monopole);
	void recv_gravity_multipoles(multipole_pass_type&&, const geo::octant&);
	void recv_gravity_expansions(expansion_pass_type&&);
#endif

	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_gravity_boundary, send_gravity_boundary_action);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_gravity_multipoles, send_gravity_multipoles_action);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_gravity_expansions, send_gravity_expansions_action);

	void step();
	HPX_DEFINE_COMPONENT_ACTION(node_server, step, step_action);

	void regrid(const hpx::id_type& root_gid, bool rb);
	HPX_DEFINE_COMPONENT_ACTION(node_server, regrid, regrid_action);

	void compute_fmm(gsolve_type gs, bool energy_account);

	void solve_gravity(bool ene);
	HPX_DEFINE_COMPONENT_ACTION(node_server, solve_gravity, solve_gravity_action);

	void start_run(bool scf);
	HPX_DEFINE_COMPONENT_ACTION(node_server, start_run, start_run_action);

	void set_grid(const std::vector<real>&, std::vector<real>&&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, set_grid, set_grid_action);

	real timestep_driver();
	HPX_DEFINE_COMPONENT_ACTION(node_server, timestep_driver, timestep_driver_action);

	real timestep_driver_descend();
	HPX_DEFINE_COMPONENT_ACTION(node_server, timestep_driver_descend, timestep_driver_descend_action);

	void timestep_driver_ascend(real);
	HPX_DEFINE_COMPONENT_ACTION(node_server, timestep_driver_ascend, timestep_driver_ascend_action);

	hpx::future<hpx::id_type> copy_to_locality(const hpx::id_type&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, copy_to_locality, copy_to_locality_action);

	hpx::id_type get_child_client(const geo::octant&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, get_child_client, get_child_client_action);

	void form_tree(const hpx::id_type&, const hpx::id_type&, const std::vector<hpx::id_type>&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, form_tree, form_tree_action);

	std::uintptr_t get_ptr();
	HPX_DEFINE_COMPONENT_ACTION(node_server, get_ptr, get_ptr_action);

	diagnostics_t diagnostics(const std::pair<space_vector,space_vector>& axis, const std::pair<real,real>& l1) const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, diagnostics, diagnostics_action);

	diagnostics_t diagnostics() const;

	grid::output_list_type load(integer, const hpx::id_type& _me, bool do_output, std::string);
	HPX_DEFINE_COMPONENT_ACTION(node_server, load, load_action);

	integer save(integer, std::string) const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, save, save_action);

	void set_aunt(const hpx::id_type&, const geo::face& face);
	HPX_DEFINE_COMPONENT_ACTION(node_server, set_aunt, set_aunt_action);

	std::vector<hpx::id_type> get_nieces(const hpx::id_type&, const geo::face& face) const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, get_nieces, get_nieces_action);

	bool check_for_refinement();
	HPX_DEFINE_COMPONENT_ACTION(node_server, check_for_refinement, check_for_refinement_action);

	void force_nodes_to_exist(const std::list<node_location>& loc);
	HPX_DEFINE_COMPONENT_ACTION(node_server, force_nodes_to_exist, force_nodes_to_exist_action);

	scf_data_t scf_params();
	HPX_DEFINE_COMPONENT_ACTION(node_server, scf_params, scf_params_action);

	real scf_update(bool);
	HPX_DEFINE_COMPONENT_ACTION(node_server, scf_update, scf_update_action);

	real find_omega() const;

	std::pair<real, real> find_omega_part(const space_vector& pivot) const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, find_omega_part, find_omega_part_action);

	void velocity_inc(const space_vector& dv);
	HPX_DEFINE_COMPONENT_ACTION(node_server, velocity_inc, velocity_inc_action);

	line_of_centers_t line_of_centers(const std::pair<space_vector,space_vector>& line) const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, line_of_centers, line_of_centers_action);

	void run_scf();

};

HPX_REGISTER_ACTION_DECLARATION (node_server::output_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::line_of_centers_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::velocity_inc_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::scf_update_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::find_omega_part_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::set_grid_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::force_nodes_to_exist_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::check_for_refinement_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::set_aunt_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::get_nieces_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::load_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::save_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::send_hydro_children_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::send_hydro_flux_correct_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::regrid_gather_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::regrid_scatter_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::send_hydro_boundary_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::send_gravity_boundary_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::send_gravity_multipoles_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::send_gravity_expansions_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::step_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::regrid_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::solve_gravity_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::start_run_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::copy_to_locality_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::get_child_client_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::form_tree_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::get_ptr_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::diagnostics_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::timestep_driver_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::timestep_driver_ascend_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::timestep_driver_descend_action);
HPX_REGISTER_ACTION_DECLARATION (node_server::scf_params_action);

HPX_ACTION_USES_LARGE_STACK (node_server::output_action);
HPX_ACTION_USES_LARGE_STACK (node_server::line_of_centers_action);
HPX_ACTION_USES_LARGE_STACK (node_server::scf_update_action);
HPX_ACTION_USES_LARGE_STACK (node_server::find_omega_part_action);
HPX_ACTION_USES_LARGE_STACK (node_server::set_grid_action);
HPX_ACTION_USES_LARGE_STACK (node_server::force_nodes_to_exist_action);
HPX_ACTION_USES_LARGE_STACK (node_server::check_for_refinement_action);
HPX_ACTION_USES_LARGE_STACK (node_server::set_aunt_action);
HPX_ACTION_USES_LARGE_STACK (node_server::get_nieces_action);
HPX_ACTION_USES_LARGE_STACK (node_server::load_action);
HPX_ACTION_USES_LARGE_STACK (node_server::save_action);
HPX_ACTION_USES_LARGE_STACK (node_server::send_hydro_children_action);
HPX_ACTION_USES_LARGE_STACK (node_server::send_hydro_flux_correct_action);
HPX_ACTION_USES_LARGE_STACK (node_server::regrid_gather_action);
HPX_ACTION_USES_LARGE_STACK (node_server::regrid_scatter_action);
HPX_ACTION_USES_LARGE_STACK (node_server::send_hydro_boundary_action);
HPX_ACTION_USES_LARGE_STACK (node_server::send_gravity_boundary_action);
HPX_ACTION_USES_LARGE_STACK (node_server::send_gravity_multipoles_action);
HPX_ACTION_USES_LARGE_STACK (node_server::send_gravity_expansions_action);
HPX_ACTION_USES_LARGE_STACK (node_server::step_action);
HPX_ACTION_USES_LARGE_STACK (node_server::regrid_action);
HPX_ACTION_USES_LARGE_STACK (node_server::solve_gravity_action);
HPX_ACTION_USES_LARGE_STACK (node_server::start_run_action);
HPX_ACTION_USES_LARGE_STACK (node_server::copy_to_locality_action);
HPX_ACTION_USES_LARGE_STACK (node_server::get_child_client_action);
HPX_ACTION_USES_LARGE_STACK (node_server::form_tree_action);
HPX_ACTION_USES_LARGE_STACK (node_server::get_ptr_action);
HPX_ACTION_USES_LARGE_STACK (node_server::diagnostics_action);
HPX_ACTION_USES_LARGE_STACK (node_server::scf_params_action);
HPX_ACTION_USES_LARGE_STACK (node_server::timestep_driver_action);
HPX_ACTION_USES_LARGE_STACK (node_server::timestep_driver_ascend_action);
HPX_ACTION_USES_LARGE_STACK (node_server::timestep_driver_descend_action);
HPX_ACTION_USES_LARGE_STACK (node_server::velocity_inc_action);

#endif /* NODE_SERVER_HPP_ */
