/*
 * grid.hpp
 *
 *  Created on: May 26, 2015
 *      Author: dmarce1
 */

#ifndef GRID_HPP_
#define GRID_HPP_

#include "defs.hpp"
#include "roe.hpp"
#include "taylor.hpp"
#include "space_vector.hpp"
#include <functional>
#include <list>

//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/binary_iarchive.hpp>
#include <list>

struct npair {
	integer lev;
	std::pair<integer, integer> loc;
};

typedef std::pair<integer, integer> dpair;

const integer GRID_IS_ROOT = 0x1;
const integer GRID_IS_LEAF = 0x2;

typedef float xpoint_type;
typedef int zone_int_type;

class grid {
public:
	typedef std::array<xpoint_type, NDIM> xpoint;
	struct node_point;
	static void set_omega(real o);
	static real get_omega();
private:
	static real omega;

	std::vector<std::vector<real>> U;
	std::vector<std::vector<real>> U0;
	std::vector<std::vector<real>> dUdt;
	std::vector<std::array<std::vector<real>, NF>> Uf;
	std::vector<std::array<std::vector<real>, NF>> F;
	std::vector<std::vector<real>> X;
	std::vector<std::vector<real>> G;
	std::vector<std::vector<real>> G0;
	//std::vector<std::vector<real>> S0;
	//std::vector<std::vector<real>> S;
	std::vector<std::vector<real>> src;

	std::vector<std::vector<dpair>> ilist_d_bnd;
	std::vector<std::vector<npair>> ilist_n_bnd;
	bool is_root;
	bool is_leaf;
	std::vector<std::vector<multipole> > M;
	std::vector<std::vector<expansion> > L;
	std::vector<std::vector<expansion> > L_c;
	real dx, t;
	std::array<real, NDIM> xmin;
	integer step_num;
	integer nlevel;
	std::vector<real> U_out;
	std::vector<real> U_out0;
//	std::vector<real> S_out;
//	std::vector<real> S_out0;
	std::vector<real> dphi_dt;
	std::vector<std::vector<space_vector> > com;
	std::vector<npair> ilist_n;
	std::vector<dpair> ilist_d;
	static bool xpoint_eq(const xpoint& a, const xpoint& b);
public:

	std::pair<std::vector<real>,std::vector<real> > field_range() const;
	struct output_list_type;
	static void merge_output_lists(output_list_type& l1, output_list_type& l2);


	real& hydro_value(integer, integer, integer, integer);
	real hydro_value(integer, integer, integer, integer) const;
	multipole& multipole_value(integer, integer, integer, integer);
	const multipole& multipole_value(integer, integer, integer, integer) const;
	const space_vector& center_of_mass_value(integer, integer, integer) const;
	space_vector& center_of_mass_value(integer, integer, integer);
	bool refine_me(integer lev) const;
	integer level_count() const;
	void compute_ilist();
	void compute_dudt();
	void egas_to_etot();
	void etot_to_egas();
	void solve_gravity(gsolve_type = RHO);
	multipole_pass_type compute_multipoles(gsolve_type, const multipole_pass_type* = nullptr);
	void compute_interactions(gsolve_type);
	void compute_boundary_interactions(gsolve_type, integer face);

	expansion_pass_type compute_expansions(gsolve_type, const expansion_pass_type* = nullptr);
	real get_time() const;
	integer get_step() const;

	void diagnostics();
	std::vector<real> conserved_sums() const;
	std::vector<real> l_sums() const;
	std::vector<real> conserved_outflows() const;
	grid(const std::function<std::vector<real>(real, real, real)>&, real dx, std::array<real, NDIM> xmin,
			integer flags);
	grid(real dx, std::array<real, NDIM>, integer flags);
	grid();
	~grid();
	void allocate();
	void reconstruct();
	void store();
	void restore();
	real compute_fluxes();
	void compute_sources();
	void boundaries();
	void set_physical_boundaries(integer);
	void next_u(integer rk, real dt);
	static void output(const output_list_type&, const char*);
	output_list_type get_output_list() const;
	template<class Archive>
	void load(Archive& arc, const unsigned) {
		arc >> is_leaf;
		arc >> is_root;
		arc >> dx;
		arc >> t;
		arc >> step_num;
		arc >> xmin;
		allocate();
		arc >> U;
		arc >> G;
		arc >> U_out;
	}
	template<class Archive>
	void save(Archive& arc, const unsigned) const {
		arc << is_leaf;
		arc << is_root;
		arc << dx;
		arc << t;
		arc << step_num;
		arc << xmin;
		arc << U;
		arc << G;
		arc << U_out;
	}
	HPX_SERIALIZATION_SPLIT_MEMBER()
	;

	std::size_t load(FILE* fp);
	std::size_t save(FILE* fp) const;

};

	struct grid::node_point {
		xpoint pt;
		integer index;
		template<class Arc>
		void serialize(Arc& arc, unsigned) {
			arc & pt[XDIM];
			arc & pt[YDIM];
			arc & pt[ZDIM];
			arc & index;
		}
		bool operator==(const grid::node_point& other) const;
		bool operator<(const grid::node_point& other) const;
	};


	struct grid::output_list_type {
		std::set<node_point> nodes;
		std::vector<zone_int_type> zones;
		std::array<std::vector<real>, NF + NGF> data;
		template<class Arc>
		void serialize(Arc& arc, unsigned int) {
			arc & nodes;
			arc & zones;
			for( integer i = 0; i != NF + NGF; ++i ) {
				arc & data[i];
			}
		}
	};

#endif /* GRID_HPP_ */
