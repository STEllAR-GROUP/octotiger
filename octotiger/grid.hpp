//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef GRID_HPP_
#define GRID_HPP_

#define SILO_UNITS

#include "octotiger/config.hpp"
#include "octotiger/config/export_definitions.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/diagnostics.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/problem.hpp"
#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/roe.hpp"
#include "octotiger/scf_data.hpp"
#include "octotiger/io/silo.hpp"
// #include "octotiger/simd_legacy.hpp"
#include "octotiger/space_vector.hpp"
//#include "octotiger/taylor.hpp"
#include "octotiger/math/Real.hpp"

#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <functional>
#include <memory>
#include <utility>
#include <vector>
//#include "octotiger/unitiger/hydro.hpp"

class struct_eos;

class analytic_t {
public:
	std::vector<Real> l1, l2, linf;
	integer nfields_;
	template<class Arc>
	void serialize(Arc &a, unsigned) {
		a & nfields_;
		if (opts().radiation) {
			l1.resize(nfields_ + NRF);
			l2.resize(nfields_ + NRF);
			linf.resize(nfields_ + NRF);
		} else {
			l1.resize(nfields_);
			l2.resize(nfields_);
			linf.resize(nfields_);
		}
		a & l1;
		a & l2;
		a & linf;
	}
	analytic_t() {
		nfields_ = 0;
	}
	analytic_t(integer nfields) {
		nfields_ = nfields;
		l1.resize(nfields_ + NRF);
		l2.resize(nfields_ + NRF);
		linf.resize(nfields_ + NRF);
		for (integer field = 0; field != nfields_; ++field) {
			l1[field] = 0.0;
			l2[field] = 0.0;
			linf[field] = 0.0;
		}
		if (opts().radiation) {
			for (integer field = nfields_; field != nfields_ + NRF; ++field) {
				l1[field] = 0.0;
				l2[field] = 0.0;
				linf[field] = 0.0;
			}
		}
	}
	analytic_t& operator+=(const analytic_t &other) {
		for (integer field = 0; field != nfields_; ++field) {
			l1[field] += other.l1[field];
			l2[field] += other.l2[field];
			linf[field] = std::max(linf[field], other.linf[field]);
		}
		if (opts().radiation) {
			for (integer field = nfields_; field != nfields_ + NRF; ++field) {
				l1[field] += other.l1[field];
				l2[field] += other.l2[field];
				linf[field] = std::max(linf[field], other.linf[field]);
			}
		}
		return *this;
	}
};

//HPX_IS_BITWISE_SERIALIZABLE(analytic_t);  commenting this line because it causes failures when futures return analytic_t type from one locality to another

using line_of_centers_t = std::vector<std::pair<Real,std::vector<Real>>>;

void output_line_of_centers(FILE *fp, const line_of_centers_t &loc);

void line_of_centers_analyze(const line_of_centers_t &loc, Real omega, std::pair<Real, Real> &rho1_max,
		std::pair<Real, Real> &rho2_max, std::pair<Real, Real> &l1_phi, std::pair<Real, Real> &l2_phi,
		std::pair<Real, Real> &l3_phi, Real &rho1_phi, Real &rho2_phi);

using xpoint_type = Real;
using zone_int_type = int;

template<int, int, class >
class hydro_computer;

class grid {
public:
	using xpoint = std::array<xpoint_type, NDIM>;
	struct node_point;
	OCTOTIGER_EXPORT static void set_min_level(integer l);
	OCTOTIGER_EXPORT static void set_max_level(integer l);
	OCTOTIGER_EXPORT static void set_fgamma(Real fg) {
		fgamma = fg;
	}
	OCTOTIGER_EXPORT static void static_init();
	OCTOTIGER_EXPORT static Real get_fgamma() {
		return fgamma;
	}
	using roche_type = char;
private:
	static std::vector<int> field_bw;
	static std::vector<int> energy_bw;
	static std::unordered_map<std::string, int> str_to_index_hydro;
	static std::unordered_map<int, std::string> index_to_str_hydro;
	static std::unordered_map<std::string, int> str_to_index_gravity;
	static std::unordered_map<int, std::string> index_to_str_gravity;
	static Real omega;
	static Real fgamma;
	static integer min_level;
	static integer max_level;
	static hpx::spinlock omega_mtx;
	static OCTOTIGER_EXPORT Real scaling_factor;
	static double idle_rate;
	hydro_computer<NDIM, INX, physics<NDIM>> hydro;
	std::shared_ptr<rad_grid> rad_grid_ptr;
	std::vector<roche_type> roche_lobe;
	std::vector<int> is_coarse;
	std::vector<int> has_coarse;
	std::vector<std::vector<Real>> Ushad;
	std::vector<std::vector<Real>> U;
	std::vector<std::vector<Real>> U0;
	std::vector<std::vector<Real>> dUdt;
	std::vector<hydro_state_t<std::vector<Real>>> F;
	std::vector<std::vector<Real>> X;
#if defined(__AVX2__) && defined(OCTOTIGER_LEGACY_VC)
	std::vector<v4sd> G;
#else
	std::vector<std::array<Real,4>> G;
#endif
	std::shared_ptr<std::vector<multipole>> M_ptr;
	std::shared_ptr<std::vector<Real>> mon_ptr;
	std::vector<expansion> L;
	std::vector<space_vector> L_c;
	std::vector<Real> dphi_dt;
#ifdef OCTOTIGER_HAVE_GRAV_PAR
	std::unique_ptr<hpx::spinlock> L_mtx;
#endif

//    std::shared_ptr<std::atomic<integer>> Muse_counter;
	bool is_root;
	bool is_leaf;
	Real dx;
	std::array<Real, NDIM> xmin;
	std::vector<Real> U_out;
	std::vector<Real> U_out0;
	std::vector<std::shared_ptr<std::vector<space_vector>>> com_ptr;
	static bool xpoint_eq(const xpoint &a, const xpoint &b);
	void compute_boundary_interactions_multipole_multipole(gsolve_type type,
			const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
	void compute_boundary_interactions_monopole_monopole(gsolve_type type, const std::vector<boundary_interaction_type>&,
			const gravity_boundary_type&);
	void compute_boundary_interactions_monopole_multipole(gsolve_type type,
			const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
	void compute_boundary_interactions_multipole_monopole(gsolve_type type,
			const std::vector<boundary_interaction_type>&, const gravity_boundary_type&);
public:
	static void set_idle_rate();
	static std::string hydro_units_name(const std::string&);
	static std::string gravity_units_name(const std::string&);
	std::vector<roche_type> get_roche_lobe() const;
	void rho_from_species();
	static bool is_hydro_field(const std::string&);
	static std::vector<std::string> get_field_names();
	static std::vector<std::string> get_hydro_field_names();

	std::vector<multipole>& get_M() {
		return *M_ptr;
	}

	std::vector<Real>& get_mon() {
		return *mon_ptr;
	}

	std::vector<std::shared_ptr<std::vector<space_vector>>>& get_com_ptr() {
		return com_ptr;
	}

	std::vector<expansion>& get_L() {
		return L;
	}

	std::vector<space_vector>& get_L_c() {
		return L_c;
	}

	std::array<Real, NDIM> get_xmin() {
		return xmin;
	}

	Real get_dx() {
		return dx;
	}
	std::vector<std::vector<Real>>& get_X() {
		return X;
	}

	std::shared_ptr<rad_grid> get_rad_grid() {
		return rad_grid_ptr;
	}
	void rad_init();
	void change_units(Real mass, Real length, Real time, Real temp);
	static hpx::future<void> static_change_units(Real mass, Real length, Real time, Real temp);
	Real get_dx() const {
		return dx;
	}
	static std::vector<std::pair<std::string, std::string>> get_scalar_expressions();
	static std::vector<std::pair<std::string, std::string>> get_vector_expressions();
	std::vector<Real>& get_field(integer f) {
		return U[f];
	}
	const std::vector<Real>& get_field(integer f) const {
		return U[f];
	}
	void set_field(std::vector<Real> &&data, integer f) {
		U[f] = std::move(data);
	}
	void set_field(const std::vector<Real> &data, integer f) {
		U[f] = data;
	}
	analytic_t compute_analytic(Real);
	void compute_boundary_interactions(gsolve_type, const geo::direction&, bool is_monopole,
			const gravity_boundary_type&);
	static void set_scaling_factor(Real f) {
		scaling_factor = f;
	}
	diagnostics_t diagnostics(const diagnostics_t &diags);
	static Real get_scaling_factor() {
		return scaling_factor;
	}
	bool get_leaf() const {
		return is_leaf;
	}
	Real get_source(integer i, integer j, integer k) const {
		return U[rho_i][hindex(i + H_BW, j + H_BW, k + H_BW)] * dx * dx * dx;
	}
	//std::vector<Real> const& get_outflows() const {
//		return U_out;
//	}
	std::vector<std::pair<std::string, Real>> get_outflows() const;
	void set_outflows(std::vector<std::pair<std::string, Real>> &&u);
	void set_outflow(std::pair<std::string, Real> &&u);
	void set_outflows(std::vector<Real> &&u) {
		U_out = std::move(u);
	}
	std::vector<Real> get_outflows_raw() {
		return U_out;
	}
	void set_root(bool flag = true) {
		is_root = flag;
	}
	bool get_root() {
		return is_root;
	}
	void set_leaf(bool flag = true) {
		if (is_leaf != flag) {
			is_leaf = flag;
		}
	}
	std::pair<Real, Real> amr_error() const;
	bool is_in_star(const std::pair<space_vector, space_vector> &axis, const std::pair<Real, Real> &l1, integer frac,
			integer index, Real rho_cut) const;
	static void set_omega(Real, bool bcast = true);
	static OCTOTIGER_EXPORT Real& get_omega();
	line_of_centers_t line_of_centers(const std::pair<space_vector, space_vector> &line);
	void set_coordinates();
	std::vector<Real> get_flux_check(const geo::face&);
	void set_flux_check(const std::vector<Real>&, const geo::face&);
	void set_hydro_boundary(const std::vector<Real>&, const geo::direction&, bool energy_only);
	std::vector<Real> get_hydro_boundary(const geo::direction &face, bool energy_only);
	scf_data_t scf_params();
	Real scf_update(Real, Real, Real, Real, Real, Real, Real, struct_eos, struct_eos);
	std::pair<std::vector<Real>, std::vector<Real> > field_range() const;
	void velocity_inc(const space_vector& dv);
	void energy_adj();
	std::vector<Real> get_restrict() const;
	std::vector<Real> get_flux_restrict(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub,
			const geo::dimension&) const;
	std::vector<Real> get_prolong(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub);
	void clear_amr();
	void set_hydro_amr_boundary(const std::vector<Real>&, const geo::direction&, bool energy_only);
	void complete_hydro_amr_boundary(bool energy_only);
	std::vector<Real> get_subset(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub,
			bool energy_only);
	void set_prolong(const std::vector<Real>&, std::vector<Real>&&);
	void set_restrict(const std::vector<Real>&, const geo::octant&);
	void set_flux_restrict(const std::vector<Real>&, const std::array<integer, NDIM> &lb,
			const std::array<integer, NDIM> &ub, const geo::dimension&);
	space_vector center_of_mass() const;
	bool refine_me(integer lev, integer last_ngrids) const;
	void compute_dudt();
	void egas_to_etot();
	void etot_to_egas();
	void dual_energy_update();
	void solve_gravity(gsolve_type = RHO);
	multipole_pass_type compute_multipoles(gsolve_type, const multipole_pass_type* = nullptr);
	void compute_interactions(gsolve_type);
	void rho_mult(Real f0, Real f1);
	void rho_move(Real x);
	expansion_pass_type compute_expansions(gsolve_type, const expansion_pass_type* = nullptr);
	expansion_pass_type compute_expansions_soa(gsolve_type, const expansion_pass_type* = nullptr);
	integer get_step() const;
	std::vector<Real> conserved_sums(space_vector &com, space_vector &com_dot,
			const std::pair<space_vector, space_vector> &axis, const std::pair<Real, Real> &l1, integer frac,
			Real rho_cut) const;
	std::pair<std::vector<Real>, std::vector<Real>> diagnostic_error() const;
	void diagnostics();
	Real z_moments(const std::pair<space_vector, space_vector> &axis, const std::pair<Real, Real> &l1, integer frac,
			Real rho_cut) const;
	std::vector<Real> frac_volumes() const;
	Real roche_volume(const std::pair<space_vector, space_vector> &axis, const std::pair<Real, Real> &l1, Real,
			bool donor) const;
	std::vector<Real> l_sums() const;
	std::vector<Real> gforce_sum(bool torque) const;
	std::vector<Real> conserved_outflows() const;
	void init_z_field();
	grid(const init_func_type&, Real dx, std::array<Real, NDIM> xmin);
	grid(Real dx, std::array<Real, NDIM>);
	grid();
	~grid() {
	}
	grid(const grid&) = delete;
	grid(grid&&) = default;
	grid& operator=(const grid&) = delete;
	grid& operator=(grid&&) = default;
#ifdef FIND_AXIS_V2
	std::array<std::pair<Real, space_vector>,2> find_core_max() const;
#else
	std::pair<space_vector, space_vector> find_axis() const;
#endif
	space_vector get_cell_center(integer i, integer j, integer k);
	gravity_boundary_type get_gravity_boundary(const geo::direction &dir, bool is_local);
	neighbor_gravity_type fill_received_array(neighbor_gravity_type raw_input);

	const std::vector<boundary_interaction_type>& get_ilist_n_bnd(const geo::direction &dir);
	void allocate();
	void store();
	void restore();
	timestep_t compute_fluxes();
	Real compute_positivity_speed_limit() const;
	void compute_sources(Real t, Real);
	void set_physical_boundaries(const geo::face&, Real t);
	void next_u(integer rk, Real t, Real dt);
	template<class Archive>
	void load(Archive &arc, const unsigned);
	static Real convert_gravity_units(int);
	static Real convert_hydro_units(int);

	template<class Archive>
	void save(Archive &arc, const unsigned) const;HPX_SERIALIZATION_SPLIT_MEMBER()
	;
	std::pair<Real, Real> virial() const;

	std::vector<silo_var_t> var_data() const;
	void set(const std::string name, Real *data, int);
	friend class node_server;
};

struct grid::node_point {
	xpoint pt;
	integer index;
	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & pt;
		arc & index;
	}
	bool operator==(const grid::node_point &other) const;
	bool operator<(const grid::node_point &other) const;
};

namespace hpx {
namespace traits {
template<>
struct is_bitwise_serializable<grid::node_point> : std::true_type {
};
}
}

void scf_binary_init();

template<class Archive>
void grid::load(Archive &arc, const unsigned) {
	arc >> roche_lobe;
	arc >> is_leaf;
	arc >> is_root;
	arc >> dx;
	arc >> xmin;
	allocate();
	arc >> U;
	if (rad_grid_ptr != nullptr) {
		arc >> *rad_grid_ptr;
		rad_grid_ptr->set_dx(dx);
	}
	for (integer i = 0; i != INX * INX * INX; ++i) {
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
void grid::save(Archive &arc, const unsigned) const {
	arc << roche_lobe;
	arc << is_leaf;
	arc << is_root;
	arc << dx;
	arc << xmin;
	arc << U;
	if (rad_grid_ptr != nullptr) {
		arc << *rad_grid_ptr;
	}
	for (integer i = 0; i != INX * INX * INX; ++i) {
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

#endif /* GRID_HPP_ */
