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
#include "geometry.hpp"
#include <functional>
#include <list>
#include <set>
#include <list>
//#include "valarray.hpp"

struct npair {
	integer lev;
	std::pair<integer, integer> loc;
};

typedef npair dpair;

typedef real xpoint_type;
typedef int zone_int_type;

struct scf_data_t {
	real phi_eff_acc;
	real phi_eff_don;
	real x_acc;
	real x_don;
	real l1;
	real rho_max_acc;
	real rho_max_don;
	real phi_a;
	real x_com;
	real m_don;
	real m_acc;
	real virial_num;
	real virial_den;
	real lx;
	real phi_l1;
	real sx_sum, sy_sum, sz_sum;
	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & virial_num;
		arc & virial_den;
		arc & phi_eff_acc;
		arc & phi_eff_don;
		arc & rho_max_acc;
		arc & rho_max_don;
		arc & phi_a;
		arc & x_com;
		arc & m_don;
		arc & m_acc;
		arc & x_don;
		arc & x_acc;
		arc & l1;
		arc & lx;
		arc & phi_l1;
		arc & sx_sum;
		arc & sy_sum;
		arc & sz_sum;
	}
	scf_data_t() {
		sz_sum = sy_sum = sx_sum = ZERO;
		rho_max_acc = rho_max_don = -std::numeric_limits<real>::max();
		phi_a = 1.0;
		x_com = ZERO;
		m_don = ZERO;
		m_acc = ZERO;
		virial_num = virial_den = ZERO;
		l1 = -std::numeric_limits<real>::max();
	}
	void accumulate(const scf_data_t& other) {
		if (other.rho_max_acc > rho_max_acc) {
			rho_max_acc = other.rho_max_acc;
			phi_eff_acc = other.phi_eff_acc;
			x_acc = other.x_acc;
		}
		if (other.rho_max_don > rho_max_don) {
			rho_max_don = other.rho_max_don;
			phi_eff_don = other.phi_eff_don;
			x_don = other.x_don;
		}
		sx_sum += other.sx_sum;
		sy_sum += other.sy_sum;
		sz_sum += other.sz_sum;
		phi_a = (other.phi_a < ZERO) ? other.phi_a : phi_a;
		x_com = ((m_don + m_acc) * x_com + (other.m_don + other.m_acc) * other.x_com);
		m_don += other.m_don;
		m_acc += other.m_acc;
		if (m_don + m_acc > ZERO) {
			x_com /= (m_don + m_acc);
		} else {
			printf("No mass!\n");
			exit(0);
		}
		virial_num += other.virial_num;
		virial_den += other.virial_den;
		if (l1 < other.l1) {
			l1 = other.l1;
			lx = other.lx;
			phi_l1 = other.phi_l1;
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

	std::vector<std::vector<real>> U;
	std::vector<std::vector<real>> U0;
	std::vector<std::vector<real>> dUdt;
	std::vector<std::array<std::vector<real>, NF>> Uf;
	std::vector<std::array<std::vector<real>, NF>> F;
	std::vector<std::vector<real>> X;
	std::vector<std::vector<real>> G;
	std::vector<std::vector<real>> G0;
	std::vector<std::vector<real>> src;

	std::vector<std::vector<dpair>> ilist_d_bnd;
	std::vector<std::vector<npair>> ilist_n_bnd;
	bool is_root;
	bool is_leaf;
	std::vector<std::vector<multipole> > M;
	std::vector<std::vector<expansion> > L;
	std::vector<std::vector<expansion> > L_c;
	real dx;
	std::array<real, NDIM> xmin;
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
	void compute_boundary_interactions_multipole(gsolve_type type, const std::vector<npair>&);
	void compute_boundary_interactions_monopole(gsolve_type type, const std::vector<npair>&);

public:
	std::pair<real, real> omega_part(const space_vector& pivot) const;
	void set_root(bool flag = true);
	void set_leaf(bool flag = true);
	bool get_leaf() const {
		return is_leaf;
	}

	static void set_omega(real);
	static void set_pivot(const space_vector& p);
	static real get_omega() {
		return omega;
	}
	static space_vector get_pivot() {
		return pivot;
	}
	void set_coordinates();
	scf_data_t scf_params();
	real scf_update(bool mom_only);
	std::pair<std::vector<real>, std::vector<real> > field_range() const;
	struct output_list_type;
	static void merge_output_lists(output_list_type& l1, output_list_type&& l2);
	std::vector<real> get_outflows() {
		return U_out;
	}
	void set_outflows(std::vector<real>&&);
	std::vector<real> get_restrict() const;
	std::vector<real> get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
			const geo::dimension&) const;
	std::vector<real> get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub);
	void set_prolong(const std::vector<real>&, std::vector<real>&&);
	void set_restrict(const std::vector<real>&, const geo::octant&);
	void set_flux_restrict(const std::vector<real>&, const std::array<integer, NDIM>& lb,
			const std::array<integer, NDIM>& ub, const geo::dimension&);
	real& hydro_value(integer, integer, integer, integer);
	real hydro_value(integer, integer, integer, integer) const;
	multipole& multipole_value(integer, integer, integer, integer);
	const multipole& multipole_value(integer, integer, integer, integer) const;
	space_vector center_of_mass() const;
	space_vector& center_of_mass_value(integer i, integer j, integer k);
	const space_vector& center_of_mass_value(integer i, integer j, integer k) const;
	bool refine_me(integer lev) const;
	integer level_count() const;
	void compute_ilist();
	void compute_dudt();
	void egas_to_etot();
	void etot_to_egas();
	void dual_energy_update();
	void solve_gravity(gsolve_type = RHO);
	multipole_pass_type compute_multipoles(gsolve_type, const multipole_pass_type* = nullptr);
	void compute_interactions(gsolve_type);
	void compute_boundary_interactions(gsolve_type, const geo::direction&, bool is_monopole);

	expansion_pass_type compute_expansions(gsolve_type, const expansion_pass_type* = nullptr);
	integer get_step() const;

	void diagnostics();
	std::vector<real> conserved_sums(std::function<bool(real,real,real)> use = [](real,real,real) {return true;}) const;
	std::vector<real> l_sums() const;
	std::vector<real> conserved_outflows() const;
	grid(const std::function<std::vector<real>(real, real, real)>&, real dx, std::array<real, NDIM> xmin);
	grid(real dx, std::array<real, NDIM>);
	grid();
	~grid();
	grid(const grid&) = default;
	grid(grid&&) = default;
	grid& operator=(const grid&) = default;
	grid& operator=(grid&&) = default;

	void allocate();
	void reconstruct();
	void store();
	void restore();
	real compute_fluxes();
	void compute_sources();
	void boundaries();
	void set_physical_boundaries(const geo::face&);
	void next_u(integer rk, real dt);
	static void output(const output_list_type&, std::string, real t);
	output_list_type get_output_list() const;
	template<class Archive>
	void load(Archive& arc, const unsigned) {
		//	bool leaf, root;
			arc >> is_leaf;
			arc >> is_root;
			//	set_root(root);
			//	set_leaf(leaf);
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
};

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
};

void scf_binary_init();

#endif /* GRID_HPP_ */
