/*
 * node_client.hpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#ifndef NODE_CLIENT_HPP_
#define NODE_CLIENT_HPP_

#include "defs.hpp"
#include "node_location.hpp"
#include "grid.hpp"
#include "geometry.hpp"
#include "eos.hpp"
#include "diagnostics.hpp"

//#include <boost/mpi/packed_iarchive.hpp>

class node_server;
class analytic_t;

class node_client {
private:
//	hpx::shared_future<hpx::id_type> id_fut;
	hpx::id_type id;
	bool local;
public:
	bool is_local();
	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & id;
	}
	bool empty() const;
	hpx::id_type get_gid() const;
	node_client& operator=(hpx::future<hpx::id_type>&& fut );
	node_client& operator=(const hpx::id_type& _id );
	node_client(hpx::future<hpx::id_type>&& fut );
	node_client(const hpx::id_type& _id );
	hpx::future<scf_data_t> scf_params() const;
	hpx::future<void> rho_mult(real, real) const;
	hpx::future<void> rho_move(real) const;
	hpx::future<real> scf_update(real,real,real,real, real, real, real, struct_eos, struct_eos) const;
	void send_hydro_children( std::vector<real>&&, const geo::octant& ci) const;
	void send_hydro_flux_correct( std::vector<real>&&, const geo::face& face, const geo::octant& ci) const;
	hpx::future<grid::output_list_type> load(integer, const hpx::id_type&, bool do_output,std::string) const;
	hpx::future<diagnostics_t> diagnostics(const std::pair<space_vector,space_vector>& axis, const std::pair<real,real>& l1, real, real) const;
	hpx::future<analytic_t> compare_analytic() const;
	hpx::future<grid::output_list_type> output(std::string fname, int, bool) const;
	node_client();
	hpx::future<std::vector<hpx::id_type>> get_nieces(const hpx::id_type&, const geo::face&) const;
	hpx::future<void> set_aunt(const hpx::id_type&, const geo::face&) const;
	hpx::future<node_server*> get_ptr() const;
	hpx::future<void> form_tree(const hpx::id_type&, const hpx::id_type&, const std::vector<hpx::id_type>& );
	hpx::future<hpx::id_type> get_child_client(const geo::octant&);
	hpx::future<void> regrid_scatter(integer, integer) const;
	hpx::future<integer> regrid_gather(bool) const;
	hpx::future<line_of_centers_t> line_of_centers(const std::pair<space_vector,space_vector>& line) const;
	void send_hydro_boundary(std::vector<real>&&, const geo::direction& dir) const;
	void send_gravity_boundary(gravity_boundary_type&&, const geo::direction&, bool monopole) const;
	void send_gravity_multipoles(multipole_pass_type&&, const geo::octant& ci) const;
	void send_gravity_expansions(expansion_pass_type&&) const;
	hpx::future<void> step() const;
	hpx::future<void> start_run(bool) const;
	hpx::future<void> regrid(const hpx::id_type&, bool rb) const;
	hpx::future<void> solve_gravity(bool ene) const;
	hpx::future<hpx::id_type> copy_to_locality(const hpx::id_type& ) const;
	hpx::future<void> set_grid(std::vector<real>&&,std::vector<real>&&) const;
	void timestep_driver_ascend(real) const;
	hpx::future<real> timestep_driver_descend() const;
	hpx::future<grid::output_list_type> output() const;
	hpx::future<void> velocity_inc(const space_vector&) const;
	integer save(integer,std::string) const;
	hpx::future<bool> check_for_refinement() const;
	hpx::future<void> force_nodes_to_exist(std::vector<node_location>&& loc) const;

    void report_timing() const;

//	hpx::future<void> find_family() const;

	};
#endif /* NODE_CLIENT_HPP_ */
