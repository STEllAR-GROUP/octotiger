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
#include "taylor.hpp"
#include "grid.hpp"

struct diagnostics_t;
//#include <boost/mpi/packed_iarchive.hpp>

class node_server;

class node_client {
private:
	hpx::id_type id;

public:
	template< class Arc >
	void serialize( Arc& arc, unsigned ) {
		arc & id;
	}
	hpx::id_type get_gid() const {
		return id;
	}
	node_client& operator=(hpx::future<hpx::id_type>&& fut ) {
		id = fut.get();
		return *this;
	}
	node_client& operator=(const hpx::shared_future<hpx::id_type>& fut ) {
		id = fut.get();
		return *this;
	}
	node_client& operator=(hpx::id_type _id ) {
		id = _id;
		return *this;
	}
	node_client(hpx::future<hpx::id_type>&& fut ) {
		id =fut.get();
	}
	node_client(const hpx::shared_future<hpx::id_type>& fut ) {
		id =fut.get();
	}
	node_client(hpx::id_type _id ) {
		id = _id;
	}
	hpx::future<void> send_hydro_children( std::vector<real>&&, integer rk, integer ci) const;
	hpx::future<hpx::id_type> load_node(std::size_t fpos, const std::string& fname, const node_location&, const hpx::id_type& );
	hpx::future<diagnostics_t> diagnostics() const;
	node_client();
	hpx::future<node_server*> get_ptr() const;
	hpx::future<void> form_tree(const hpx::id_type&, const hpx::id_type&, const std::vector<hpx::id_type>& );
	hpx::future<hpx::id_type> get_child_client(integer ci);
	node_client( const hpx::id_type& id);
	hpx::future<void> regrid_scatter(integer, integer) const;
	//hpx::future<void> register_(const node_location&) const;
	//hpx::future<void> unregister(const node_location&) const;
	hpx::future<integer> regrid_gather() const;
	hpx::future<void> send_hydro_boundary(std::vector<real>&&, integer rk, integer face) const;
	hpx::future<void> send_gravity_boundary(std::vector<real>&&, integer face, integer) const;
	hpx::future<void> send_gravity_multipoles(multipole_pass_type&&, integer ci, integer) const;
	hpx::future<void> send_gravity_expansions(expansion_pass_type&&, integer) const;
	hpx::future<void> step() const;
	hpx::future<void> start_run() const;
	hpx::future<void> regrid() const;
	hpx::future<void> solve_gravity(bool ene, integer c) const;
	hpx::future<hpx::id_type> copy_to_locality(const hpx::id_type& ) const;

	hpx::future<real> timestep_driver() const;
	hpx::future<void> timestep_driver_ascend(real) const;
	hpx::future<real> timestep_driver_descend() const;
	hpx::future<grid::output_list_type> output() const;
	hpx::future<std::pair<std::size_t,std::size_t>> save(integer loc_id, std::string) const;

//	hpx::future<void> find_family() const;

		};
#endif /* NODE_CLIENT_HPP_ */
