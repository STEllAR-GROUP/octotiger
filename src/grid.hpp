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
#include "space_vector.hpp"
#include "geometry.hpp"
#include <functional>
#include <list>
#include "eos.hpp"
#include <set>
#include "problem.hpp"
#include "taylor.hpp"

struct interaction_type {
	integer first;
	integer second;
	space_vector x;
};

struct boundary_interaction_type {
	integer second;
	std::vector<integer> first;
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
			M = std::shared_ptr < std::vector<multipole> > (new std::vector<multipole>);
			m = std::shared_ptr < std::vector<real> > (new std::vector<real>);
			x = std::shared_ptr < std::vector<space_vector> > (new std::vector<space_vector>);
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
struct scf_data_t {
	real m_x;
	real m;
	real virial_sum;
	real virial_norm;
	real phiA;
	real phiB;
	real phiC;
	real entC;
	real donor_phi_min;
	real accretor_phi_min;
	real donor_phi_max;
	real accretor_phi_max;
	real donor_x;
	real accretor_x;
	real l1_phi;
	real l1_x;
	real accretor_mass;
	real donor_mass;
	real donor_central_enthalpy;
	real accretor_central_enthalpy;
	real donor_central_density;
	real accretor_central_density;
	real xA, xB, xC;
	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & xA;
		arc & xB;
		arc & xC;
		arc & m_x;
		arc & m;
		arc & virial_sum;
		arc & virial_norm;
		arc & phiA;
		arc & phiB;
		arc & phiC;
		arc & entC;
		arc & donor_phi_min;
		arc & accretor_phi_min;
		arc & donor_phi_max;
		arc & accretor_phi_max;
		arc & donor_x;
		arc & accretor_x;
		arc & l1_phi;
		arc & l1_x;
		arc & accretor_mass;
		arc & donor_mass;
		arc & donor_central_enthalpy;
		arc & accretor_central_enthalpy;
		arc & donor_central_density;
		arc & accretor_central_density;
	}
	scf_data_t() {
		donor_phi_max = accretor_phi_max = l1_phi = -std::numeric_limits < real > ::max();
		accretor_mass = donor_mass = donor_central_enthalpy = accretor_central_enthalpy = ZERO;
		phiA = phiB = phiC = 0.0;
		virial_sum = virial_norm = 0.0;
		donor_central_density = accretor_central_density = 0.0;
		m = m_x = 0.0;
	}
	void accumulate(const scf_data_t& other) {
		if (phiA > other.phiA) {
			phiA = other.phiA;
			xA = other.xA;
		}
		if (phiB > other.phiB) {
			phiB = other.phiB;
			xB = other.xB;
		}
		if (phiC > other.phiC) {
			phiC = other.phiC;
			xC = other.xC;
		}
		m += other.m;
		m_x += other.m_x;
		virial_sum += other.virial_sum;
		virial_norm += other.virial_norm;
		phiA = std::min(phiA, other.phiA);
		phiB = std::min(phiB, other.phiB);
		phiC = std::min(phiC, other.phiC);
		entC = std::max(entC, other.entC);
		donor_phi_max = std::max(donor_phi_max, other.donor_phi_max);
		accretor_phi_max = std::max(accretor_phi_max, other.accretor_phi_max);
		accretor_mass += other.accretor_mass;
		donor_mass += other.donor_mass;
		if (other.donor_central_enthalpy > donor_central_enthalpy) {
			donor_phi_min = other.donor_phi_min;
			donor_x = other.donor_x;
			donor_central_enthalpy = other.donor_central_enthalpy;
			donor_central_density = other.donor_central_density;
		}
		if (other.accretor_central_enthalpy > accretor_central_enthalpy) {
			accretor_phi_min = other.accretor_phi_min;
			accretor_x = other.accretor_x;
			accretor_central_enthalpy = other.accretor_central_enthalpy;
			accretor_central_density = other.accretor_central_density;
		}
		if (other.l1_phi > l1_phi) {
			l1_phi = other.l1_phi;
			l1_x = other.l1_x;
		}
	}
};

class grid {
public:
	typedef std::array<xpoint_type, NDIM> xpoint;
	struct node_point;
	static void set_max_level(integer l);
private:
	static integer max_level;
	static real omega;
	static space_vector pivot;
	static real scaling_factor;

	std::vector<std::vector<real>> U;
	std::vector<std::vector<real>> U0;
	std::vector<std::vector<real>> dUdt;
	std::vector<std::array<std::vector<real>, NF>> F;
	std::vector<std::vector<real>> X;
	std::vector<std::vector<real>> G;
	std::vector<multipole> M;
	std::vector<real> mon;
	std::vector<expansion> L;
	std::vector<space_vector> L_c;
	std::vector<real> dphi_dt;

	bool is_root;
	bool is_leaf;
	real dx;
	std::array<real, NDIM> xmin;
	std::vector<real> U_out;
	std::vector<real> U_out0;
	std::vector<std::vector<space_vector> > com;
	static bool xpoint_eq(const xpoint& a, const xpoint& b);
	void compute_boundary_interactions_multipole_multipole(gsolve_type type, const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
	void compute_boundary_interactions_monopole_monopole(gsolve_type type, const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
	void compute_boundary_interactions_monopole_multipole(gsolve_type type, const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
	void compute_boundary_interactions_multipole_monopole(gsolve_type type, const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
public:
	void compute_boundary_interactions(gsolve_type, const geo::direction&, bool is_monopole, const gravity_boundary_type&);
	static void set_scaling_factor(real f) {
		scaling_factor = f;
	}
	static real get_scaling_factor() {
		return scaling_factor;
	}
	void set_root(bool flag = true);
	void set_leaf(bool flag = true);
	bool get_leaf() const {
		return is_leaf;
	}
	bool is_in_star(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, integer frac, integer index) const;
	static void set_omega(real);
	static real get_omega();
	static void set_pivot(const space_vector& p);
	static space_vector get_pivot() {
		return pivot;
	}
	line_of_centers_t line_of_centers(const std::pair<space_vector, space_vector>& line);
	real get_source(integer i, integer j, integer k) const {
		return U[rho_i][hindex(i + H_BW, j + H_BW, k + H_BW)] * dx * dx * dx;
	}
	void set_4force(integer i, integer j, integer k, const std::array<real, NDIM + 1>& four_force) {
		const auto iii = gindex(i, j, k );
		for (integer f = 0; f != NDIM + 1; ++f) {
			G[f][iii] = four_force[f];
		}
	}
	void compute_conserved_slopes(const std::array<integer, NDIM> lb = { 1, 1, 1 }, const std::array<integer, NDIM> ub = { H_NX - 1, H_NX - 1, H_NX - 1 },
		bool tau_only = false);
	void compute_primitive_slopes(real theta, const std::array<integer, NDIM> lb = { 1, 1, 1 },
		const std::array<integer, NDIM> ub = { H_NX - 1, H_NX - 1, H_NX - 1 }, bool tau_only = false);
	void compute_primitives(const std::array<integer, NDIM> lb = { 1, 1, 1 }, const std::array<integer, NDIM> ub = { H_NX - 1, H_NX - 1, H_NX - 1 },
		bool tau_only = false);
	void set_coordinates();
	void set_hydro_boundary(const std::vector<real>&, const geo::direction&, integer width, bool tau_only = false);
	std::vector<real> get_hydro_boundary(const geo::direction& face, integer width, bool tau_only = false);
	scf_data_t scf_params();
	real scf_update(real, real, real, real, real, real, real, accretor_eos, donor_eos);
	std::pair<std::vector<real>, std::vector<real> > field_range() const;
	struct output_list_type;
	static void merge_output_lists(output_list_type& l1, output_list_type&& l2);
	std::vector<real> get_outflows() {
		return U_out;
	}
	void velocity_inc(const space_vector& dv);
	void set_outflows(std::vector<real>&&);
	std::vector<real> get_restrict() const;
	std::vector<real> get_flux_restrict(const std::array<integer, NDIM>& lb,
		const std::array<integer, NDIM>& ub, const geo::dimension&) const;
	std::vector<real> get_prolong(const std::array<integer, NDIM>& lb,
		const std::array<integer, NDIM>& ub, bool tau_only=false);
	void set_prolong(const std::vector<real>&, std::vector<real>&&);
	void set_restrict(const std::vector<real>&, const geo::octant&);
	void set_flux_restrict(const std::vector<real>&, const std::array<integer, NDIM>& lb,
		const std::array<integer, NDIM>& ub, const geo::dimension&);
	space_vector center_of_mass() const;
//	space_vector& center_of_mass_value(integer i, integer j, integer k);
//	const space_vector& center_of_mass_value(integer i, integer j, integer k) const;
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

		std::pair<std::vector<real>, std::vector<real>> diagnostic_error() const;
		void diagnostics();
		std::vector<real> conserved_sums(space_vector& com,space_vector& com_dot, const std::pair<space_vector,space_vector>& axis, const std::pair<real,real>& l1,integer frac) const;
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
		grid(const grid&) = default;
		grid(grid&&) = default;
		grid& operator=(const grid&) = default;
		grid& operator=(grid&&) = default;

		std::pair<space_vector,space_vector> find_axis() const;

		space_vector get_cell_center(integer i, integer j, integer k);

		gravity_boundary_type get_gravity_boundary(const geo::direction& dir, bool is_local);
		void allocate();
		void reconstruct();
		void store();
		real compute_fluxes();
		void compute_sources(real t);
		void boundaries();
		void set_physical_boundaries(const geo::face&);
		void next_u(integer rk, real t, real dt);
		static void output(const output_list_type&, std::string, real t, int cycle);
		output_list_type get_output_list() const;
		template<class Archive>
		void load(Archive& arc, const unsigned) {
			arc >> is_leaf;
			arc >> is_root;
			arc >> dx;
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
}
;

struct grid::output_list_type {
	std::set<node_point> nodes;
	std::vector<zone_int_type> zones;
	std::array<std::vector<real>, NF + NGF> data;
	template<class Arc>
	void serialize(Arc& arc, unsigned int) {
		arc & nodes;
		arc & zones;
		for (integer i = 0; i != NF + NGF; ++i) {
			arc & data[i];
		}
	}
}
;

void scf_binary_init();

#endif /* GRID_HPP_ */
