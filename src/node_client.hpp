/*
 * node_client.hpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#ifndef NODE_CLIENT_HPP_
#define NODE_CLIENT_HPP_

#include "defs.hpp"
#include "future.hpp"
#include "node_location.hpp"
#include "grid.hpp"
#include "geometry.hpp"
#include "eos.hpp"
#include "diagnostics.hpp"
#include "radiation/rad_grid.hpp"

//#include <boost/mpi/packed_iarchive.hpp>

class node_server;
class analytic_t;


#ifdef USE_NIECE_BOOL
typedef bool set_child_aunt_type;
#else
typedef integer set_child_aunt_type;
#endif


namespace hpx {
    using mutex = hpx::lcos::local::spinlock;
}

class node_client {

private:
//	hpx::shared_future<hpx::id_type> id_fut;
	hpx::id_type id;
    hpx::id_type unmanaged;
	bool local;
public:
	bool is_local() const;
	template<class Arc>
	void load(Arc& arc, unsigned)
    {
		arc & id;
        if (!empty())
        {
            unmanaged = hpx::id_type(id.get_gid(), hpx::id_type::unmanaged);
        }
	}

	template<class Arc>
	void save(Arc& arc, unsigned) const
    {
		arc & id;
    }
    HPX_SERIALIZATION_SPLIT_MEMBER();

	bool empty() const;
	hpx::id_type get_gid() const;
	hpx::id_type get_unmanaged_gid() const;
	node_client& operator=(future<hpx::id_type>&& fut );
	node_client& operator=(const hpx::id_type& _id );
	node_client(future<hpx::id_type>&& fut );
	node_client(const hpx::id_type& _id );
	future<scf_data_t> scf_params() const;
	future<void> rho_mult(real, real) const;
	future<void> rho_move(real) const;
	future<void> check_channels() const;
	future<real> scf_update(real,real,real,real, real, real, real, struct_eos, struct_eos) const;
	void send_hydro_children( std::vector<real>&&, const geo::octant& ci, std::size_t cycle) const;
	void send_hydro_flux_correct( std::vector<real>&&, const geo::face& face, const geo::octant& ci) const;
	void send_read_flux_correct( std::vector<real>&&, const geo::face& face, const geo::octant& ci) const;
	void send_rad_flux_correct( std::vector<real>&&, const geo::face& face, const geo::octant& ci) const;
	future<grid::output_list_type> load(integer, integer, integer, bool do_output,std::string) const;
	future<diagnostics_t> diagnostics(const diagnostics_t&) const;
	future<analytic_t> compare_analytic() const;
	future<grid::output_list_type> output(std::string dname, std::string fname, int, bool) const;
	node_client();
	future<set_child_aunt_type> set_child_aunt(const hpx::id_type&, const geo::face&) const;
	future<void> set_aunt(const hpx::id_type&, const geo::face&) const;
	future<node_server*> get_ptr() const;
	future<void> form_tree(hpx::id_type&&, hpx::id_type&&, std::vector<hpx::id_type>&& );
	future<hpx::id_type> get_child_client(const node_location& parent_loc, const geo::octant&);
	future<void> regrid_scatter(integer, integer) const;
	future<integer> regrid_gather(bool) const;
	future<line_of_centers_t> line_of_centers(const std::pair<space_vector,space_vector>& line) const;
	void send_flux_check(std::vector<real>&&, const geo::direction& dir, std::size_t cycle) const;
	void send_hydro_boundary(std::vector<real>&&, const geo::direction& dir, std::size_t cycle) const;
	void send_gravity_boundary(gravity_boundary_type&&, const geo::direction&, bool monopole, std::size_t cycle) const;
	void send_gravity_multipoles(multipole_pass_type&&, const geo::octant& ci) const;
	void send_gravity_expansions(expansion_pass_type&&) const;
	future<real> step(integer) const;
	future<void> solve_gravity(bool ene,bool aonly) const;
	future<hpx::id_type> copy_to_locality(const hpx::id_type& ) const;
	future<void> set_grid(std::vector<real>&&,std::vector<real>&&) const;
	void timestep_driver_ascend(real) const;
    void set_local_timestep(integer, real) const;
	future<grid::output_list_type> output() const;
	future<void> velocity_inc(const space_vector&) const;
    future<void> save(integer,std::string) const;
	future<void> check_for_refinement(real omega, real) const;
	future<void> force_nodes_to_exist(std::vector<node_location>&& loc) const;
#ifdef FIND_AXIS_V2
    future<std::array<std::pair<real,space_vector>,2>> find_axis_tool() const;
#endif

    void report_timing() const;

    future<void> change_units(real,real,real,real) const;

    future<void> erad_init() const ;
    future<void> send_rad_children( std::vector<real>&&, const geo::octant& ci, std::size_t cycle) const;
	future<void> send_rad_boundary(std::vector<rad_type>&&, const geo::direction&, std::size_t cycle) const;
	future<void> set_rad_grid(std::vector<real>&&) const;
#ifdef OCTOTIGER_USE_NODE_CACHE
    struct node_loc_hash_t {
    	std::size_t operator()( const node_location& loc) const {
    		return loc.unique_id();
    	}
    };
 	using table_type = std::unordered_map<node_location, hpx::shared_future<hpx::id_type>, node_loc_hash_t>;
	static table_type node_cache;
	static hpx::mutex node_cache_mutex;
	static std::atomic<integer> hits;
	static std::atomic<integer> misses;
	static void cycle_node_cache();
#endif

	};
#endif /* NODE_CLIENT_HPP_ */
