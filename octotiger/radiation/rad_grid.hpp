//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef RAD_GRID_HPP_
#define RAD_GRID_HPP_

#include "octotiger/math/Real.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/io/silo.hpp"
#include "octotiger/radiation/conservation.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

class rad_grid {
public:
	static constexpr integer er_i = 0;
	static constexpr integer fx_i = 1;
	static constexpr integer fy_i = 2;
	static constexpr integer fz_i = 3;
private:
	static constexpr integer DX = RAD_NX * RAD_NX;
	static constexpr integer DY = RAD_NX;
	static constexpr integer DZ = 1;
	static constexpr int radiationStrides[NDIM] = { RAD_NX * RAD_NX, RAD_NX, 1 };
	static std::unordered_map<std::string, int> str_to_index;
	static std::unordered_map<int, std::string> index_to_str;
	Real dx = 0;
	std::vector<std::atomic<int>> is_coarse;
	std::vector<std::atomic<int>> has_coarse;
	std::vector<std::vector<Real>> Ushad;
	// Structure of arrays; only E and physical Fx,Fy,Fz are evolved/serialized.
	// Divide flux components by c when creating an M1 ConservedState=(E,Q).
	std::vector<std::vector<Real>> U;
	// Finite-volume fluxes/reflux messages retain the same physical storage units.
	std::vector<std::vector<std::vector<Real>>> flux;
	// VL predictor states use (E,Q=F/c), including the two-cell ghost ring.
	std::array<std::vector<Real>, NRF> Uhalf;
	std::array<std::array<std::vector<Real>, NRF>, 2> faces;
	bool sourceEnabled = false;
	std::vector<Real> chiAbsorption, chiTotal;
	std::array<std::vector<Real>, NDIM> gasVelocity;
	// Accumulated gas feedback; the material state stays fixed during subcycles.
	std::array<std::vector<Real>, NRF> gasDelta;
	std::vector<std::vector<Real>> X;
	std::vector<Real> mmw, X_spc, Z_spc;
	// Interval budgets use physical (E,F) units and are drained before regridding.
	// Diagnostic intervals are not serialized; checkpoint field layout stays intact.
	radiationConservation::Moments conservationBoundary{};
	radiationConservation::Moments conservationSource{};
public:
	static void static_init();
	static std::vector<std::string> get_field_names();
	void set(const std::string name, Real* data);
	std::vector<silo_var_t> var_data() const;
	void set_X( const std::vector<std::vector<Real>>& x );

	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & dx;
		arc & U;
	}
	void compute_mmw(const std::vector<std::vector<Real>>& U);
	void change_units(Real m, Real l, Real t, Real k);
	void sanity_check();
	void compute_flux(Real dt, Real omega, Real time = 0,
        std::array<bool, 2 * NDIM> physicalFaces = {});
    void prepareSources(const std::vector<std::vector<Real>>& gas);
    void finishSources(std::vector<std::vector<Real>>& gas);
	Real maxTimestep(Real omega) const;
	void initialize_erad(const std::vector<Real>& rho, const std::vector<Real>& tau);
	void set_dx(Real dx);
	void advance(Real dt, Real omega);
	void applyRegressionSource(Real dt);
	radiationConservation::Totals takeConservation();
	void accountBoundaryFlux(Real dt, geo::face face);
	void rad_imp(std::vector<Real>& egas, std::vector<Real>& tau, std::vector<Real>& sx, std::vector<Real>& sy, std::vector<Real>& sz,
			const std::vector<Real>& rho, Real dt);
	std::vector<Real> get_restrict() const;
	std::vector<Real> get_prolong(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub);
	void set_prolong(const std::vector<Real>&);
	void set_restrict(const std::vector<Real>&, const geo::octant&);
	void set_flux_restrict(const std::vector<Real>& data, const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub,
			const geo::dimension& dim);
	std::vector<Real> get_flux_restrict(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::dimension& dim) const;
	std::vector<Real> get_intensity(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub, const geo::octant&);
	void allocate();
	rad_grid(Real dx);
	rad_grid();
	void set_boundary(const std::vector<Real>& data, const geo::direction& dir);
	Real get_field(integer f, integer i, integer j, integer k) const;
	void set_field(Real v, integer f, integer i, integer j, integer k);
	void set_physical_boundaries(geo::face f, Real t, bool half = false);
	std::vector<Real> get_boundary(const geo::direction& dir);
	using kappa_type = std::function<Real(Real)>;

	Real hydroSignalSpeed(const std::vector<Real>& egas, const std::vector<Real>& tau, const std::vector<Real>& sx, const std::vector<Real>& sy, const std::vector<Real>& sz,
			const std::vector<Real>& rho);

	void clear_amr();
	void set_rad_amr_boundary(const std::vector<Real>&, const geo::direction&);
	void complete_rad_amr_boundary();
	std::vector<Real> get_subset(const std::array<integer, NDIM>& lb, const std::array<integer, NDIM>& ub);

	friend class node_server;
};




#endif /* RAD_GRID_HPP_ */
