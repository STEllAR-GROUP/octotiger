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
#include "problem.hpp"
#include <list>
//#include "valarray.hpp"

struct npair {
	integer lev;
	std::pair<integer, integer> loc;
};

typedef npair dpair;

typedef real xpoint_type;
typedef int zone_int_type;

#ifdef OLD_SCF

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
	real vol_a_star;
	real vol_d_star;
	real vol_a_lobe;
	real vol_d_lobe;
	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & vol_a_star;
		arc & vol_a_lobe;
		arc & vol_d_star;
		arc & vol_d_lobe;
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
		rho_max_acc = rho_max_don = -std::numeric_limits < real > ::max();
		phi_a = 1.0;
		x_com = ZERO;
		m_don = ZERO;
		m_acc = ZERO;
		vol_a_star = vol_d_star = vol_a_lobe = vol_d_lobe = ZERO;
		virial_num = virial_den = ZERO;
		l1 = -std::numeric_limits < real > ::max();
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
		vol_a_star += other.vol_a_star;
		vol_d_star += other.vol_d_star;
		vol_a_lobe += other.vol_a_lobe;
		vol_d_lobe += other.vol_d_lobe;
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
#else
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
	real omega_int_part_1;
	real omega_int_part_2;
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
		arc & omega_int_part_1;
		arc & omega_int_part_2;
	}
	scf_data_t() {
		donor_phi_max = accretor_phi_max = l1_phi = -std::numeric_limits < real
				> ::max();
		accretor_mass = donor_mass = donor_central_enthalpy =
				accretor_central_enthalpy = omega_int_part_1 =
						omega_int_part_2 = ZERO;
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
		omega_int_part_1 += other.omega_int_part_1;
		omega_int_part_2 += other.omega_int_part_2;
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
#endif

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
	std::vector<std::array<std::vector<real>, NF>> Uf;
	std::vector<std::array<std::vector<real>, NF>> F;
	std::vector<std::vector<real>> X;
	std::vector<std::vector<real>> G;
	std::vector<std::vector<real>> G_analytic;
	std::vector<std::vector<real>> G0;
	std::vector<std::vector<real>> src;

	std::vector<std::vector<dpair>> ilist_d_bnd;
	std::vector<std::vector<npair>> ilist_n_bnd;bool is_root;bool is_leaf;
	std::vector<std::vector<multipole> > M;
	std::vector<std::vector<expansion> > L;
	std::vector<std::vector<expansion> > L_c;
	real dx;
	std::array<real, NDIM> xmin;
	integer nlevel;
	std::vector<real> U_out;
	std::vector<real> U_out0;
	std::vector<real> dphi_dt;
	std::vector<std::vector<space_vector> > com;
	std::vector<npair> ilist_n;
	std::vector<dpair> ilist_d;
	static bool xpoint_eq(const xpoint& a, const xpoint& b);
	void compute_boundary_interactions_multipole(gsolve_type type,
			const std::vector<npair>&);
	void compute_boundary_interactions_monopole(gsolve_type type,
			const std::vector<npair>&);

public:
	static void set_scaling_factor(real f) {
		scaling_factor = f;
	}
	static real get_scaling_factor() {
		return scaling_factor;
	}
	std::pair<real, real> omega_part(const space_vector& pivot) const;
	void set_root(bool flag = true);
	void set_leaf(bool flag = true);bool get_leaf() const {
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
	real get_source(integer i, integer j, integer k) const {
		return U[rho_i][hindex(i + H_BW, j + H_BW, k + H_BW)] * dx * dx * dx;
	}
	void set_4force(integer i, integer j, integer k,
			const std::array<real, NDIM + 1>& four_force) {
		const auto iii = gindex(i + G_BW, j + G_BW, k + G_BW);
		for (integer f = 0; f != NDIM + 1; ++f) {
			G[f][iii] = four_force[f];
		}
	}
	std::vector<std::vector<std::vector<real>>>compute_conserved_slopes(const std::vector<std::vector<real>>& V, const std::vector<std::vector<std::vector<real>>>& dV, const std::array<integer, NDIM> lb = {1,1,1}, const std::array<integer, NDIM> ub = {H_NX -1, H_NX-1, H_NX-1});std::vector < std::vector<std::vector<real>>> compute_primitive_slopes(const std::vector<std::vector<real>>& V, real theta, const std::array<integer, NDIM> lb = {1,1,1}, const std::array<integer, NDIM> ub = {H_NX -1, H_NX-1, H_NX-1});std::vector<std::vector<real>> compute_primitives(const std::array<integer, NDIM> lb = {1,1,1}, const std::array<integer, NDIM> ub = {H_NX -1, H_NX-1, H_NX-1});
	void set_coordinates();
	scf_data_t scf_params();
	real scf_update(bool mom_only);
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
			const std::array<integer, NDIM>& ub);
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

	std::pair<std::vector<real>, std::vector<real>> diagnostic_error() const;
	void diagnostics();
	std::vector<real> conserved_sums(std::function<bool(real, real, real)> use =
			[](real,real,real) {return true;}) const;
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

	space_vector find_axis() const;

	std::vector<real> get_gravity_boundary(const geo::direction& dir) {

		std::array<integer, NDIM> lb, ub;
		std::vector<real> data;
		integer size = get_boundary_size(lb, ub, dir, INNER, G_BW);
		const bool is_refined = !is_leaf;
		if (is_refined) {
			size *= 20 + 3;
		} else {
			size *= 1 + 3;
		}
		data.resize(size);
		integer iter = 0;

		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					const auto& m = multipole_value(0, i, j, k);
					const auto& com = center_of_mass_value(i, j, k);
					const integer top = is_refined ? m.size() : 1;
					for (integer l = 0; l < top; ++l) {
						data[iter] = m.ptr()[l];
						++iter;
					}
					for (integer d = 0; d != NDIM; ++d) {
						data[iter] = com[d];
						++iter;
					}
				}
			}
		}

		return data;
	}

	void set_gravity_boundary(const std::vector<real>& data, const geo::direction& dir,
			bool monopole) {
		std::array<integer, NDIM> lb, ub;
		get_boundary_size(lb, ub, dir, OUTER, G_BW);
		integer iter = 0;
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					auto& m = multipole_value(0, i, j, k);
					auto& com = center_of_mass_value(i, j, k);
					const integer top = monopole ? 1 : m.size();
					for (integer l = 0; l < top; ++l) {
						m.ptr()[l] = data[iter];
						++iter;
					}
					for (integer l = top; l < m.size(); ++l) {
						m.ptr()[l] = ZERO;
					}
					for (integer d = 0; d != NDIM; ++d) {
						com[d] = data[iter];
						++iter;
					}
				}
			}
		}
	}

	void allocate();
	void reconstruct();
	void store();
	void restore();
	real compute_fluxes();
	void compute_sources(real t);
	void boundaries();
	void set_physical_boundaries(const geo::face&);
	void next_u(integer rk, real t, real dt);
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
	bool operator==(const grid::node_point& other) const;bool operator<(
			const grid::node_point& other) const;
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
