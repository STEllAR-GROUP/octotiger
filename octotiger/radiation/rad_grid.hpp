//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef RAD_GRID_HPP_
#define RAD_GRID_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/gas/GasEoS.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/io/silo.hpp"
#include "octotiger/math/Matrix.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/math/Vector.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/RadiationEoS.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/hydro_impl/flux.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct.hpp"
#include "octotiger/unitiger/radiation/radiation_physics.hpp"

#include <string>
#include <unordered_map>

inline static auto almostOne = 1_R - 4_R * std::sqrt(eps_R);

int radiationSubstepCount(Real dt);
Real radiationHydroSignalSpeed(StateVector const &Ur, StateVector const &Ug, Real dx);
void radiationTransportFluxes(std::vector<StateVector>& flux, StateVector const &Ur, StateVector const &Ug, Real dx);
StateVector radiationImplicitSource(StateVector const &Ur, StateVector const &Ug, Real dt);
StateVector radiationExternalSource(std::vector<std::vector<Real>> x, Real t);
void radiationApplyFluxes(StateVector const &U0, StateVector& U, std::vector<StateVector> const &F, Real β, Real h);
void radiationApplyImplicitSource(StateVector &Ur, StateVector &Ug, StateVector const &dUdt, Real dt);
void radiationApplyExternalSource(StateVector &Ur, StateVector const &dUdt, Real dt);

inline auto boundaryCount(std::integral auto... n) {
	using namespace std;
	return (0_R + ... + abs(fmod(Real(n) / Real(INX) + 1_R, 1_R)));
}

class rad_grid {
public:
private:
	static constexpr int DX = RAD_NX * RAD_NX;
	static constexpr int DY = RAD_NX;
	static constexpr int DZ = 1;
	static constexpr int R_DN[NDIM] = {RAD_NX * RAD_NX, RAD_NX, 1};
	static std::unordered_map<std::string, int> str_to_index;
	static std::unordered_map<int, std::string> index_to_str;
	Real dx;
	std::vector<std::atomic<int>> is_coarse;
	std::vector<std::atomic<int>> has_coarse;
	StateVector Ushad;
	StateVector U = StateVector(NRF);
	StateVector U0 = StateVector(NRF);
	std::vector<StateVector> flux;
	std::array<std::array<std::vector<Real> *, NDIM>, NDIM> P;
	StateVector X;
	hydro_computer<NDIM, INX, radiation_physics<NDIM>> hydro;

public:
	static void static_init();
	static std::vector<std::string> get_field_names();
	void set(const std::string name, Real *data);
	std::vector<silo_var_t> var_data() const;
	void set_X(const StateVector &x);
	void restore();
	void store();
	auto const &get_X() const {
		return X;
	}
	template <class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & dx;
		arc & U;
	}
	StateVector computeTransport(std::function<void()> const &);
	std::pair<StateVector, StateVector> computeSource(StateVector &, Real dt) const;
	//	void computeMaterialProperties(const StateVector &);
	void change_units(Real m, Real l, Real t, Real k);
	void sanity_check();
	void initialize_erad(const std::vector<Real> rho, const std::vector<Real> tau);
	void set_dx(Real dx);
	// void compute_fEdd();
	std::vector<Real> get_restrict() const;
	std::vector<Real> get_prolong(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub);
	void set_prolong(const std::vector<Real> &);
	void set_restrict(const std::vector<Real> &, const geo::octant &);
	void set_flux_restrict(const std::vector<Real> &data, const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub,
						   const geo::dimension &dim);
	std::vector<Real> get_flux_restrict(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub, const geo::dimension &dim) const;
	std::vector<Real> get_intensity(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub, const geo::octant &);
	void allocate();
	rad_grid(Real dx);
	rad_grid();
	void set_boundary(const std::vector<Real> &data, const geo::direction &dir);
	Real get_field(int f, int i, int j, int k) const;
	void set_field(Real v, int f, int i, int j, int k);
	void set_physical_boundaries(geo::face f, Real t);
	std::vector<Real> get_boundary(const geo::direction &dir);
	using kappa_type = std::function<Real(Real)>;

	void clear_amr();
	void set_rad_amr_boundary(const std::vector<Real> &, const geo::direction &);
	void complete_rad_amr_boundary();
	std::vector<Real> get_subset(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub);
	auto getStateReferences() {
		return std::tie(U0, U);
	}
	friend class node_server;
};

#endif /* RAD_GRID_HPP_ */
