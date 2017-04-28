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
//#include "struct_eos.hpp"
#include "profiler.hpp"
#include "rad_grid.hpp"

#include <array>
#include <atomic>
#include <iostream>
#include <map>
#include <vector>




#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

class node_server: public hpx::components::managed_component_base<node_server> {
public:
    static void set_gravity(bool b) {
        gravity_on = b;
    }
    static void set_hydro(bool b) {
        hydro_on = b;
}
private:
    struct neighbor_gravity_type {
        gravity_boundary_type data;
        bool is_monopole;
        geo::direction direction;
    };
    struct sibling_hydro_type {
        std::vector<real> data;
        geo::direction direction;
    };
    void set_pivot();
    std::atomic<integer> refinement_flag;
    static bool hydro_on;
    static bool gravity_on;
    static std::stack<grid::output_list_type> pending_output;
    node_location my_location;
    integer step_num;
    std::size_t hcycle;
    std::size_t gcycle;
    real current_time;
    real rotational_time;
    std::shared_ptr<grid> grid_ptr; //
#ifdef RADIATION
    std::shared_ptr<rad_grid> rad_grid_ptr; //
#endif
    bool is_refined;
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
    std::vector<bool> nieces;
    /* An aunt is this node's parent's neighbor, so it is one level coarser.
     *  Only aunts in the 6 face directions are required. Used for AMR boundary handling. */
    std::vector<node_client> aunts;

    std::vector<std::array<bool, geo::direction::count()>> amr_flags;
    hpx::lcos::local::spinlock mtx;
    hpx::lcos::local::spinlock prolong_mtx;
    std::array<channel<std::vector<real>>, NCHILD> child_hydro_channels;
     channel<expansion_pass_type> parent_gravity_channel;
     std::array<channel<neighbor_gravity_type>, geo::direction::count()> neighbor_gravity_channels;
    std::array<channel<sibling_hydro_type>, geo::direction::count()> sibling_hydro_channels;
    std::array<channel<multipole_pass_type>, NCHILD> child_gravity_channels;
    std::array<std::array<channel<std::vector<real>>, 4>, NFACE> niece_hydro_channels;
    channel<real> global_timestep_channel;
    std::array<channel<real>, NCHILD + 1> local_timestep_channels;
    hpx::mutex load_mutex;

    timings timings_;
    real dt_;


public:
    static bool is_gravity_on() {
        return gravity_on;
    }
    real get_time() const {
        return current_time;
    }
    real get_rotation_count() const;
    node_server& operator=(node_server&&) = default;

    template<class Archive>
    void serialize(Archive& arc, unsigned) {
    	std::size_t _hb = hcycle;
    	std::size_t _gb = gcycle;
    	arc & _hb;
    	arc & _gb;
        arc & my_location;
        arc & step_num;
        arc & is_refined;
        arc & children;
        arc & parent;
        arc & neighbors;
        arc & nieces;
        arc & aunts;
        arc & child_descendant_count;
        arc & current_time;
        arc & rotational_time;
        arc & xmin;
        arc & dx;
        arc & amr_flags;
        arc & grid_ptr;
        integer rf = refinement_flag;
        arc & rf;
        refinement_flag = rf;
        arc & timings_;
        hcycle = _hb;
        gcycle = _gb;
    }

    std::size_t load_me(FILE *fp, bool old_format);
    std::size_t save_me(std::ostream& strm) const;
private:

    static bool static_initialized;
    static std::atomic<integer> static_initializing;

    void initialize(real, real);
    void send_hydro_amr_boundaries(bool tau_only = false);
    void collect_hydro_boundaries(bool tau_only = false);
    void exchange_interlevel_hydro_data();
    static void static_initialize();
    void all_hydro_bounds(bool tau_only = false);
    void clear_family();
    hpx::future<void> exchange_flux_corrections();

    hpx::future<void> nonrefined_step();
    void refined_step();

    diagnostics_t root_diagnostics(diagnostics_t && diags,
        std::pair<real, real> rho1, std::pair<real, real> rho2,real phi_1, real phi_2) const;
    diagnostics_t child_diagnostics(const std::pair<space_vector, space_vector>& axis,
        const std::pair<real, real>& l1, real, real) const;
    diagnostics_t local_diagnostics(const std::pair<space_vector, space_vector>& axis,
        const std::pair<real, real>& l1, real, real) const;
    hpx::future<real> local_step(integer steps);

    std::map<integer, std::vector<char> > save_local(integer& cnt, std::string const& filename, hpx::future<void>& child_fut) const;

public:
     static bool child_is_on_face(integer ci, integer face) {
        return (((ci >> (face / 2)) & 1) == (face & 1));
    }

    std::vector<hpx::future<void>> set_nieces_amr(const geo::face&) const;
	node_server() {
	    initialize(ZERO, ZERO);
    }
	~node_server() {}
	node_server(const node_server& other);
	node_server(const node_location&, const node_client& parent_id, real, real, std::size_t, std::size_t, std::size_t);
	node_server(const node_location&, integer, bool, real, real, const std::array<integer, NCHILD>&, grid, const std::vector<hpx::id_type>&, std::size_t,
			std::size_t);
	node_server(node_server&& other) = default;


    void report_timing();
    HPX_DEFINE_COMPONENT_ACTION(node_server, report_timing, report_timing_action);

    void save_to_file(const std::string&, std::string const& data_dir) const;
    void load_from_file(const std::string&, std::string const& data_dir);
    void load_from_file_and_output(const std::string&, const std::string&, std::string const& data_dir);


    grid::output_list_type output(std::string dname, std::string fname, int cycle, bool analytic) const;
    HPX_DEFINE_COMPONENT_ACTION(node_server, output, output_action);

    static void parallel_output_gather(grid::output_list_type&&);
    static void parallel_output_complete(std::string dname, std::string fname, real tm, int cycle, bool analytic);

    integer regrid_gather(bool rebalance_only);
    HPX_DEFINE_COMPONENT_ACTION(node_server, regrid_gather, regrid_gather_action);

    hpx::future<hpx::id_type> create_child(hpx::id_type const& locality, integer ci);

    hpx::future<void> regrid_scatter(integer, integer);
    HPX_DEFINE_COMPONENT_ACTION(node_server, regrid_scatter, regrid_scatter_action);

    void recv_hydro_boundary(std::vector<real>&&, const geo::direction&, std::size_t cycle);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_hydro_boundary, send_hydro_boundary_action);

    void recv_hydro_children(std::vector<real>&&, const geo::octant& ci, std::size_t cycle);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_hydro_children, send_hydro_children_action);

	void recv_hydro_flux_correct(std::vector<real>&&, const geo::face& face, const geo::octant& ci);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_hydro_flux_correct, send_hydro_flux_correct_action);

	void recv_gravity_boundary(gravity_boundary_type&&, const geo::direction&, bool monopole, std::size_t cycle);
	void recv_gravity_multipoles(multipole_pass_type&&, const geo::octant&);
	void recv_gravity_expansions(expansion_pass_type&&);

    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_gravity_boundary, send_gravity_boundary_action);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_gravity_multipoles, send_gravity_multipoles_action);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_gravity_expansions, send_gravity_expansions_action);

    hpx::future<real> step(integer steps);
    HPX_DEFINE_COMPONENT_ACTION(node_server, step, step_action);

    hpx::future<std::pair<real, diagnostics_t> > step_with_diagnostics(integer steps,
        const std::pair<space_vector, space_vector>& axis,
        const std::pair<real, real>& l1, real, real);
    HPX_DEFINE_COMPONENT_ACTION(node_server, step_with_diagnostics, step_with_diagnostics_action);

    hpx::future<std::pair<real, diagnostics_t> > root_step_with_diagnostics(integer steps);

    void update();

    integer regrid(const hpx::id_type& root_gid, real omega, real new_floor, bool rb);

    void compute_fmm(gsolve_type gs, bool energy_account);

    void solve_gravity(bool ene);
    HPX_DEFINE_COMPONENT_ACTION(node_server, solve_gravity, solve_gravity_action);

    void start_run(bool scf, integer);

    void set_grid(const std::vector<real>&, std::vector<real>&&);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, set_grid, set_grid_action);

    hpx::future<real> timestep_driver_descend();

    void set_local_timestep(integer i, real dt);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, set_local_timestep, set_local_timestep_action);

    void timestep_driver_ascend(real);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, timestep_driver_ascend, timestep_driver_ascend_action);

    hpx::future<hpx::id_type> copy_to_locality(const hpx::id_type&);
    HPX_DEFINE_COMPONENT_ACTION(node_server, copy_to_locality, copy_to_locality_action);

    hpx::id_type get_child_client(const geo::octant&);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, get_child_client, get_child_client_action);

    hpx::future<void> form_tree(hpx::id_type, hpx::id_type, std::vector<hpx::id_type>);
    HPX_DEFINE_COMPONENT_ACTION(node_server, form_tree, form_tree_action);

    std::uintptr_t get_ptr();
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, get_ptr, get_ptr_action);

    analytic_t compare_analytic();
    HPX_DEFINE_COMPONENT_ACTION(node_server, compare_analytic, compare_analytic_action);

    diagnostics_t diagnostics(const std::pair<space_vector, space_vector>& axis,
        const std::pair<real, real>& l1, real, real) const;
    HPX_DEFINE_COMPONENT_ACTION(node_server, diagnostics, diagnostics_action);

    diagnostics_t diagnostics() const;

    hpx::future<grid::output_list_type> load(integer, integer, integer, bool do_output,
        std::string);
    HPX_DEFINE_COMPONENT_ACTION(node_server, load, load_action);

    hpx::future<void> save(integer, std::string const&) const;
    HPX_DEFINE_COMPONENT_ACTION(node_server, save, save_action);

    void set_aunt(const hpx::id_type&, const geo::face& face);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, set_aunt, set_aunt_action);

    bool set_child_aunt(const hpx::id_type&,
        const geo::face& face) const;
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, set_child_aunt, set_child_aunt_action);

    hpx::future<void> check_for_refinement(real omega,real new_floor);
    HPX_DEFINE_COMPONENT_ACTION(node_server, check_for_refinement, check_for_refinement_action);

    void force_nodes_to_exist(std::vector<node_location>&& loc);
    HPX_DEFINE_COMPONENT_ACTION(node_server, force_nodes_to_exist, force_nodes_to_exist_action);

    scf_data_t scf_params();
    HPX_DEFINE_COMPONENT_ACTION(node_server, scf_params, scf_params_action);

    real scf_update(real, real, real, real, real, real, real, struct_eos, struct_eos);
    HPX_DEFINE_COMPONENT_ACTION(node_server, scf_update, scf_update_action);

    void velocity_inc(const space_vector& dv);
    HPX_DEFINE_COMPONENT_ACTION(node_server, velocity_inc, velocity_inc_action);

    line_of_centers_t line_of_centers(
        const std::pair<space_vector, space_vector>& line) const;
    HPX_DEFINE_COMPONENT_ACTION(node_server, line_of_centers, line_of_centers_action);

    void rho_mult(real factor, real);
    HPX_DEFINE_COMPONENT_ACTION(node_server,rho_mult, rho_mult_action);

    void rho_move(real);
    HPX_DEFINE_COMPONENT_ACTION(node_server,rho_move, rho_move_action);

    void run_scf(std::string const& data_dir);

#ifdef RADIATION
private:
    struct sibling_rad_type {
        std::vector<rad_type> data;
        geo::direction direction;
    };

	std::array<channel<sibling_rad_type>, geo::direction::count()> sibling_rad_channels;
	std::array<channel<std::vector<real>>, NCHILD> child_rad_channels;
	channel<expansion_pass_type> parent_rad_channel;
public:
	hpx::future<void> exchange_rad_flux_corrections();
	void compute_radiation(real dt);
	hpx::future<void> exchange_interlevel_rad_data();
	void all_rad_bounds();

	void collect_radiation_bounds();
    void send_rad_amr_bounds();

    void recv_rad_flux_correct(std::vector<real>&&, const geo::face& face,
        const geo::octant& ci);
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(node_server, recv_rad_flux_correct, send_rad_flux_correct_action);


	void recv_rad_boundary(std::vector<rad_type>&&, const geo::direction&);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_rad_boundary, send_rad_boundary_action);

	void recv_rad_children(std::vector<real>&&, const geo::octant& ci);
	HPX_DEFINE_COMPONENT_ACTION(node_server, recv_rad_children, send_rad_children_action);

    std::array<std::array<channel<std::vector<real>>, 4>, NFACE> niece_rad_channels;

    void set_rad_grid(const std::vector<real>&/*, std::vector<real>&&*/);
    HPX_DEFINE_COMPONENT_ACTION(node_server, set_rad_grid, set_rad_grid_action);

    void erad_init();
    HPX_DEFINE_COMPONENT_ACTION(node_server, erad_init, erad_init_action);

#endif

    hpx::future<void> change_units(real m, real l, real t, real k);
    HPX_DEFINE_COMPONENT_ACTION(node_server, change_units, change_units_action);

    void set_cgs(bool change = true);


};

// HPX_ACTION_USES_LARGE_STACK(node_server::rho_mult_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::output_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::line_of_centers_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::scf_update_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::set_aunt_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::set_child_aunt_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::load_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::save_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::step_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::solve_gravity_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::copy_to_locality_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::get_child_client_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::get_ptr_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::diagnostics_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::scf_params_action);
// HPX_ACTION_USES_LARGE_STACK(node_server::velocity_inc_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::change_units_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::rho_mult_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::output_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::line_of_centers_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::velocity_inc_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::scf_update_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::set_grid_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::force_nodes_to_exist_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::check_for_refinement_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::set_aunt_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::set_child_aunt_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::load_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::save_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_hydro_children_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_hydro_flux_correct_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::regrid_gather_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::regrid_scatter_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_hydro_boundary_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_gravity_boundary_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_gravity_multipoles_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_gravity_expansions_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::step_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::step_with_diagnostics_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::solve_gravity_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::copy_to_locality_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::get_child_client_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::form_tree_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::get_ptr_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::diagnostics_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::timestep_driver_ascend_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::scf_params_action);

#ifdef RADIATION
HPX_REGISTER_ACTION_DECLARATION(node_server::send_rad_boundary_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_rad_children_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::send_rad_flux_correct_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::set_rad_grid_action);
HPX_REGISTER_ACTION_DECLARATION(node_server::erad_init_action);
#endif

#endif /* NODE_SERVER_HPP_ */
