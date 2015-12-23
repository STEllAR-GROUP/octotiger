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
#include <atomic>

const integer INNER = 0;
const integer OUTER = 1;

struct diagnostics_t {
	std::vector<real> grid_sum;
	std::vector<real> outflow_sum;
	std::vector<real> l_sum;
	std::vector<real> field_max;
	std::vector<real> field_min;
	real donor_mass;
	diagnostics_t() :
			grid_sum(NF, ZERO), outflow_sum(NF, ZERO), l_sum(NDIM, ZERO), field_max(NF,
					-std::numeric_limits<real>::max()), field_min(NF, +std::numeric_limits<real>::max()), donor_mass(ZERO) {
	}
	diagnostics_t& operator+=(const diagnostics_t& other) {
		for (integer f = 0; f != NF; ++f) {
			grid_sum[f] += other.grid_sum[f];
			outflow_sum[f] += other.outflow_sum[f];
			field_max[f] = std::max(field_max[f], other.field_max[f]);
			field_min[f] = std::min(field_min[f], other.field_min[f]);
		}
		for (integer d = 0; d != NDIM; ++d) {
			l_sum[d] += other.l_sum[d];
		}
		donor_mass += other.donor_mass;
		return *this;
	}
	diagnostics_t& operator=(const diagnostics_t& other) {
		field_max = other.field_max;
		field_min = other.field_min;
		grid_sum = other.grid_sum;
		outflow_sum = other.outflow_sum;
		l_sum = other.l_sum;
		donor_mass = other.donor_mass;
		return *this;
	}
	;
	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & grid_sum;
		arc & outflow_sum;
		arc & l_sum;
		arc & field_max;
		arc & field_min;
		arc & donor_mass;
	}
};

class node_server: public hpx::components::managed_component_base<node_server> {
private:
	void set_omega_and_pivot();
	std::atomic<integer> refinement_flag;

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
	std::array<std::shared_ptr<channel<std::vector<real> >>, NCHILD> child_hydro_channels;
	std::shared_ptr<channel<expansion_pass_type>> parent_gravity_channel;
	std::array<std::shared_ptr<channel<std::pair<std::vector<real>, bool>>> ,geo::direction::count()> neighbor_gravity_channels;
	std::array<std::shared_ptr<channel<multipole_pass_type>>, NCHILD> child_gravity_channels;
	std::array<std::array<std::shared_ptr<channel<std::vector<real>>>, 4>, NFACE> niece_hydro_channels;
	std::shared_ptr<channel<real>> global_timestep_channel;
	std::shared_ptr<channel<real>> local_timestep_channel;
	hpx::mutex load_mutex;
public:
	real get_time() const {
		return current_time;
	}
	node_server& operator=( node_server&& ) = default;

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
	std::size_t load_me( FILE* fp );
	std::size_t save_me( FILE* fp ) const;
private:

	static bool static_initialized;
	static std::atomic<integer> static_initializing;

	void initialize(real,real);
	void collect_hydro_boundaries();
	void exchange_interlevel_hydro_data();
	static void static_initialize();
	void clear_family();
	hpx::future<void> exchange_flux_corrections();

public:

//	static void output_form();
//	static grid::output_list_type output_collect(const std::string&);

	static bool child_is_on_face(integer ci, integer face);

	std::list<hpx::future<void>> set_nieces_amr(const geo::face& ) const;
	node_server();
	~node_server();
	node_server( const node_server& other);
	node_server(const node_location&, const node_client& parent_id, real, real);
	integer get_boundary_size(std::array<integer, NDIM>&, std::array<integer, NDIM>&, const geo::direction&, const geo::side&, integer bw) const;
	void set_hydro_boundary(const std::vector<real>&, const geo::direction&);
	std::vector<real> get_hydro_boundary(const geo::direction& face);

	void save_to_file( const std::string&) const;
	void load_from_file( const std::string&);
	void load_from_file_and_output( const std::string&, const std::string& );
	void set_gravity_boundary(const std::vector<real>&, const geo::direction&, bool monopole);
	std::vector<real> get_gravity_boundary(const geo::direction& dir);

	integer regrid_gather(bool rebalance_only);
	HPX_DEFINE_COMPONENT_ACTION(node_server, regrid_gather, regrid_gather_action);

	void regrid_scatter(integer, integer);
	HPX_DEFINE_COMPONENT_ACTION(node_server, regrid_scatter, regrid_scatter_action);

	void recv_hydro_boundary( std::vector<real>&&, const geo::direction&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_hydro_boundary, send_hydro_boundary_action);

	void recv_hydro_children( std::vector<real>&&, const geo::octant& ci);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_hydro_children, send_hydro_children_action);

	void recv_hydro_flux_correct( std::vector<real>&&, const geo::face& face, const geo::octant& ci);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_hydro_flux_correct, send_hydro_flux_correct_action);

	void recv_gravity_boundary( std::vector<real>&&, const geo::direction&, bool monopole);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_gravity_boundary, send_gravity_boundary_action);

	void recv_gravity_multipoles( multipole_pass_type&&, const geo::octant&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_gravity_multipoles, send_gravity_multipoles_action);

	void recv_gravity_expansions(expansion_pass_type&&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_gravity_expansions, send_gravity_expansions_action);

	void step();
	HPX_DEFINE_COMPONENT_ACTION(node_server, step, step_action);

	void regrid(const hpx::id_type& root_gid, bool rb);
	HPX_DEFINE_COMPONENT_ACTION(node_server, regrid, regrid_action);

	void compute_fmm(gsolve_type gs, bool energy_account);

	void solve_gravity(bool ene);
	HPX_DEFINE_COMPONENT_ACTION(node_server, solve_gravity, solve_gravity_action);

	void start_run();
	HPX_DEFINE_COMPONENT_ACTION(node_server, start_run, start_run_action);

	void set_grid(const std::vector<real>&, std::vector<real>&&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, set_grid, set_grid_action);

	real timestep_driver();
	HPX_DEFINE_COMPONENT_ACTION(node_server, timestep_driver, timestep_driver_action);

	real timestep_driver_descend();
	HPX_DEFINE_COMPONENT_ACTION(node_server, timestep_driver_descend, timestep_driver_descend_action);

	void timestep_driver_ascend(real);
	HPX_DEFINE_COMPONENT_ACTION(node_server, timestep_driver_ascend, timestep_driver_ascend_action);

	hpx::future<hpx::id_type> copy_to_locality(const hpx::id_type& );
	HPX_DEFINE_COMPONENT_ACTION(node_server, copy_to_locality, copy_to_locality_action);

	hpx::id_type get_child_client(const geo::octant&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, get_child_client, get_child_client_action);

	void form_tree(const hpx::id_type&, const hpx::id_type&, const std::vector<hpx::id_type>& );
	HPX_DEFINE_COMPONENT_ACTION(node_server, form_tree, form_tree_action);

	std::uintptr_t get_ptr();
	HPX_DEFINE_COMPONENT_ACTION(node_server, get_ptr, get_ptr_action);

	diagnostics_t diagnostics() const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, diagnostics, diagnostics_action);

	grid::output_list_type load(integer, const hpx::id_type& _me, bool do_output);
	HPX_DEFINE_COMPONENT_ACTION(node_server, load, load_action);

	integer save(integer=0) const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, save, save_action);

	void set_aunt(const hpx::id_type&, const geo::face& face );
	HPX_DEFINE_COMPONENT_ACTION(node_server, set_aunt, set_aunt_action);

	std::vector<hpx::id_type> get_nieces(const hpx::id_type&, const geo::face& face ) const;
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

	std::pair<real,real> find_omega_part(const space_vector& pivot) const;
	HPX_DEFINE_COMPONENT_ACTION(node_server, find_omega_part, find_omega_part_action);

	void run_scf();

};

HPX_REGISTER_ACTION_DECLARATION( node_server::scf_update_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::find_omega_part_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::set_grid_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::force_nodes_to_exist_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::check_for_refinement_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::set_aunt_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::get_nieces_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::load_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::save_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::send_hydro_children_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::send_hydro_flux_correct_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::regrid_gather_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::regrid_scatter_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::send_hydro_boundary_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::send_gravity_boundary_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::send_gravity_multipoles_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::send_gravity_expansions_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::step_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::regrid_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::solve_gravity_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::start_run_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::copy_to_locality_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::get_child_client_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::form_tree_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::get_ptr_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::diagnostics_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::timestep_driver_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::timestep_driver_ascend_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::timestep_driver_descend_action);
HPX_REGISTER_ACTION_DECLARATION( node_server::scf_params_action);

HPX_ACTION_USES_MEDIUM_STACK( node_server::scf_update_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::find_omega_part_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::set_grid_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::force_nodes_to_exist_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::check_for_refinement_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::set_aunt_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::get_nieces_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::load_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::save_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::send_hydro_children_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::send_hydro_flux_correct_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::regrid_gather_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::regrid_scatter_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::send_hydro_boundary_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::send_gravity_boundary_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::send_gravity_multipoles_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::send_gravity_expansions_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::step_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::regrid_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::solve_gravity_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::start_run_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::copy_to_locality_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::get_child_client_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::form_tree_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::get_ptr_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::diagnostics_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::scf_params_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::timestep_driver_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::timestep_driver_ascend_action);
HPX_ACTION_USES_MEDIUM_STACK( node_server::timestep_driver_descend_action);



#endif /* NODE_SERVER_HPP_ */
