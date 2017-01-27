/*
 * grid.hpp
 *
 *  Created on: May 26, 2015
 *      Author: dmarce1
 */

#ifndef GRID_HPP_
#define GRID_HPP_

#define NPF 5

#include "simd.hpp"
#include "defs.hpp"
#include "roe.hpp"
#include "space_vector.hpp"
#include "geometry.hpp"
#include "problem.hpp"
#include "taylor.hpp"
#include "scf_data.hpp"

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/set.hpp>
#include <hpx/runtime/serialization/array.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <functional>
#include <list>
#include <memory>
#include <set>


class struct_eos;

class analytic_t {
public:
	std::array<real,NF> l1, l2, linf;
	std::array<real,NF> l1a, l2a, linfa;
	template<class Arc>
	void serialize(Arc& a, unsigned) {
		a & l1;
		a & l2;
		a & linf;
		a & l1a;
		a & l2a;
		a & linfa;
	}
	analytic_t() {
		for( integer field = 0; field != NF; ++field) {
			l1[field] = 0.0;
			l2[field] = 0.0;
			l1a[field] = 0.0;
			l2a[field] = 0.0;
			linf[field] = 0.0;
			linfa[field] = 0.0;
		}
	}
	analytic_t& operator+=(const analytic_t& other) {
		for( integer field = 0; field != NF; ++field) {
			l1[field] += other.l1[field];
			l2[field] += other.l2[field];
			l1a[field] += other.l1a[field];
			l2a[field] += other.l2a[field];
			linf[field] = std::max(linf[field], other.linf[field]);
			linfa[field] = std::max(linfa[field], other.linfa[field]);
		}
		return *this;
	}
};

struct interaction_type {
	std::uint16_t first;
	std::uint16_t second;
	space_vector x;
	v4sd four;
};

struct boundary_interaction_type {
	std::uint16_t second;
	std::vector<std::uint16_t> first;
	std::vector<v4sd> four;
	space_vector x;
};

typedef taylor<4, real> multipole;
typedef taylor<4, real> expansion;
typedef std::pair<std::vector<multipole>, std::vector<space_vector>> multipole_pass_type;
typedef std::pair<std::vector<expansion>, std::vector<space_vector>> expansion_pass_type;

struct gravity_boundary_type {
	std::shared_ptr<std::vector<multipole>> M;
	std::shared_ptr<std::vector<real>> m;
	std::shared_ptr<std::vector<space_vector>> x;
	bool is_local;
	gravity_boundary_type() :
		M(nullptr), m(nullptr), x(nullptr) {
	}
	void allocate() {
		if (M == nullptr) {
			M = std::make_shared<std::vector<multipole> >();
			m = std::make_shared<std::vector<real> >();
			x = std::make_shared<std::vector<space_vector> >();
		}
	}
	template<class Archive>
	void serialize(Archive& arc, unsigned) {
		allocate();
		arc & *M;
		arc & *m;
		arc & *x;
		arc & is_local;
	}
};

using line_of_centers_t = std::vector<std::pair<real,std::vector<real>>>;

void output_line_of_centers(FILE* fp, const line_of_centers_t& loc);

void line_of_centers_analyze(const line_of_centers_t& loc, real omega, std::pair<real, real>& rho1_max, std::pair<real, real>& rho2_max,
	std::pair<real, real>& l1_phi, std::pair<real, real>& l2_phi, std::pair<real, real>& l3_phi, real&, real&);

typedef real xpoint_type;
typedef int zone_int_type;


class grid {
public:
	static char const* field_names[];
	typedef std::array<xpoint_type, NDIM> xpoint;
	struct node_point;
	static void set_max_level(integer l);
	static void set_fgamma(real);
	static real get_fgamma();
	static real get_A();
	static real get_B();
	static void set_analytic_func(const analytic_func_type& func);
private:
	static analytic_func_type analytic;
	static real Acons, Bcons;
	static real fgamma;
	static integer max_level;
	static real omega;
	static space_vector pivot;
	static real scaling_factor;

	std::vector<std::vector<real>> U;
	std::vector<std::vector<real>> Ua;
	std::vector<std::vector<real>> U0;
	std::vector<std::vector<real>> dUdt;
	std::vector<std::array<std::vector<real>, NF>> F;
	std::vector<std::vector<real>> X;
	std::vector<v4sd> G;
	std::shared_ptr<std::vector<multipole>> M_ptr;
	std::shared_ptr<std::vector<real>> mon_ptr;
	std::vector<expansion> L;
	std::vector<space_vector> L_c;
	std::vector<real> dphi_dt;
#ifdef USE_GRAV_PAR
    std::unique_ptr<hpx::lcos::local::spinlock> L_mtx;
#endif

//    std::shared_ptr<std::atomic<integer>> Muse_counter;

	bool is_root;
	bool is_leaf;
	real dx;
	std::array<real, NDIM> xmin;
	std::vector<real> U_out;
	std::vector<real> U_out0;
	std::vector<std::shared_ptr<std::vector<space_vector>>> com_ptr;
	static bool xpoint_eq(const xpoint& a, const xpoint& b);
	void compute_boundary_interactions_multipole_multipole(gsolve_type type, const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
	void compute_boundary_interactions_monopole_monopole(gsolve_type type, const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
	void compute_boundary_interactions_monopole_multipole(gsolve_type type, const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
	void compute_boundary_interactions_multipole_monopole(gsolve_type type, const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
public:
	analytic_t compute_analytic(real);
	void compute_boundary_interactions(gsolve_type, const geo::direction&, bool is_monopole, const gravity_boundary_type&);
	static void set_scaling_factor(real f);
	static real get_scaling_factor();
	bool get_leaf() const;
	static space_vector get_pivot();
	real get_source(integer i, integer j, integer k) const;
	std::vector<real> get_outflows();
	void set_root(bool flag = true);
	void set_leaf(bool flag = true);
	bool is_in_star(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, integer frac, integer index) const;
	static void set_omega(real);
	static void set_AB(real, real);
	static real get_omega();
	static void set_pivot(const space_vector& p);
	line_of_centers_t line_of_centers(const std::pair<space_vector, space_vector>& line);
	void compute_conserved_slopes(const std::array<integer, NDIM> lb = { 1, 1, 1 }, const std::array<integer, NDIM> ub = { H_NX - 1, H_NX - 1, H_NX - 1 },
		bool tau_only = false);
	void compute_primitive_slopes(real theta, const std::array<integer, NDIM> lb = { 1, 1, 1 },
		const std::array<integer, NDIM> ub = { H_NX - 1, H_NX - 1, H_NX - 1 }, bool tau_only = false);
	void compute_primitives(const std::array<integer, NDIM> lb = { 1, 1, 1 }, const std::array<integer, NDIM> ub = { H_NX - 1, H_NX - 1, H_NX - 1 },
		bool tau_only = false) const;
	void set_coordinates();
	void set_hydro_boundary(const std::vector<real>&, const geo::direction&, integer width, bool tau_only = false);
	std::vector<real> get_hydro_boundary(const geo::direction& face, integer width, bool tau_only = false);
	scf_data_t scf_params();
	real scf_update(real, real, real, real, real, real, real, struct_eos, struct_eos);
	std::pair<std::vector<real>, std::vector<real> > field_range() const;
	struct output_list_type;
	static void merge_output_lists(output_list_type& l1, output_list_type&& l2);
	void velocity_inc(const space_vector& dv);
	void set_outflows(std::vector<real>&&);
	std::vector<real> get_restrict() const;
	std::vector<real> get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::dimension&) const;
	std::vector<real> get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, bool tau_only=false);
	void set_prolong(const std::vector<real>&, std::vector<real>&&);
	void set_restrict(const std::vector<real>&, const geo::octant&);
	void set_flux_restrict(const std::vector<real>&, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::dimension&);
	space_vector center_of_mass() const;
	bool refine_me(integer lev) const;
	void compute_dudt();
	void egas_to_etot();
	void etot_to_egas();
	void dual_energy_update();
	void solve_gravity(gsolve_type = RHO);
	multipole_pass_type compute_multipoles(gsolve_type, const multipole_pass_type* = nullptr);
	void compute_interactions(gsolve_type);
	void rho_mult(real f0, real f1 );
	void rho_move(real x);
	expansion_pass_type compute_expansions(gsolve_type, const expansion_pass_type* = nullptr);
	integer get_step() const;
	std::vector<real> conserved_sums(space_vector& com,space_vector& com_dot, const std::pair<space_vector,space_vector>& axis, const std::pair<real,real>& l1,integer frac) const;
	std::pair<std::vector<real>, std::vector<real>> diagnostic_error() const;
	void diagnostics();
	real z_moments( const std::pair<space_vector,space_vector>& axis, const std::pair<real,real>& l1, integer frac) const;
	std::vector<real> frac_volumes() const;
	real roche_volume(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, real, bool donor) const;
	std::vector<real> l_sums() const;
	std::vector<real> gforce_sum(bool torque) const;
	std::vector<real> conserved_outflows() const;
	grid(const init_func_type&, real dx, std::array<real, NDIM> xmin);
	grid(real dx, std::array<real, NDIM>);
	grid();
	~grid();
	grid(const grid&) = delete;
	grid(grid&&) = default;
	grid& operator=(const grid&) = delete;
	grid& operator=(grid&&) = default;
	std::pair<space_vector,space_vector> find_axis() const;
	space_vector get_cell_center(integer i, integer j, integer k);
	gravity_boundary_type get_gravity_boundary(const geo::direction& dir, bool is_local);
   void allocate();
   void reconstruct();
   void store();
   real compute_fluxes();
   void compute_sources(real t);
   void set_physical_boundaries(const geo::face&, real t);
   void next_u(integer rk, real t, real dt);
   static void output(const output_list_type&, std::string, real t, int cycle, bool a);
   output_list_type get_output_list(bool analytic) const;
   template<class Archive>
   void load(Archive& arc, const unsigned) {
   	arc >> is_leaf;
   	arc >> is_root;
   	arc >> dx;
   	arc >> xmin;
   	allocate();
   	arc >> U;
   	for( integer i = 0; i != INX*INX*INX; ++i ) {
#if defined(HPX_HAVE_DATAPAR)
        arc >> G[i];
#else
   		arc >> G[i][0];
   		arc >> G[i][1];
   		arc >> G[i][2];
   		arc >> G[i][3];
#endif
   	}
   	arc >> U_out;
   }
   template<class Archive>
   void save(Archive& arc, const unsigned) const {
   	arc << is_leaf;
   	arc << is_root;
   	arc << dx;
   	arc << xmin;
   	arc << U;
   	for( integer i = 0; i != INX*INX*INX; ++i ) {
#if defined(HPX_HAVE_DATAPAR)
        arc << G[i];
#else
   		arc << G[i][0];
   		arc << G[i][1];
   		arc << G[i][2];
   		arc << G[i][3];
#endif
   	}
   	arc << U_out;
   }
   HPX_SERIALIZATION_SPLIT_MEMBER();
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
}
;

struct grid::output_list_type {
	std::set<node_point> nodes;
	std::vector<zone_int_type> zones;
	std::array<std::vector<real>, NF + NGF + NPF> data;
	std::array<std::vector<real>, NF + NGF + NPF> analytic;
	template<class Arc>
	void serialize(Arc& arc, unsigned int) {
		arc & nodes;
		arc & zones;
		for (integer i = 0; i != NF + NGF + NPF; ++i) {
			arc & data[i];
		}
	}
}
;

void scf_binary_init();

#endif /* GRID_HPP_ */
