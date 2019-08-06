/*
 * hydro.hpp
 *
 *  Created on: Jul 31, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_HYDRO_HPP_
#define OCTOTIGER_UNITIGER_HYDRO_HPP_
#include <vector>

#include "safe_real.hpp"
#include "basis.hpp"

#define SAFE_MATH_ON
#ifdef NOHPX
#include "/home/dmarce1/workspace/octotiger/octotiger/safe_math.hpp"
#else
#include "../../octotiger/safe_math.hpp"
#endif

#ifdef NOHPX
#include <future>
using std::future;
using std::async;
using std::launch;
#endif
namespace hydro {

void filter_cell1d(std::array<safe_real, 3> &C, safe_real C0);
void filter_cell2d(std::array<safe_real, 9> &C, safe_real C0);
void filter_cell3d(std::array<safe_real, 27> &C, safe_real C0);
void output_cell2d(FILE *fp, const std::array<safe_real, 9> &C, int joff, int ioff);
}

#include "./cell_geometry.hpp"

template<int NDIM, int INX>
struct hydro_computer: public cell_geometry<NDIM, INX> {
	using geo = cell_geometry<NDIM,INX>;
	const std::vector<std::vector<std::array<safe_real, geo::NDIR>>> reconstruct(std::vector<std::vector<safe_real>> U, safe_real dx);
	safe_real flux(const std::vector<std::vector<std::array<safe_real, geo::NDIR>>> &Q, std::vector<std::vector<std::vector<safe_real>>> &F,
			std::vector<std::array<safe_real, NDIM>> &X, safe_real omega);
	void update_tau(std::vector<std::vector<safe_real>> &U, safe_real dx);

	inline static safe_real minmod(safe_real a, safe_real b);
	inline static safe_real minmod_theta(safe_real a, safe_real b, safe_real c);
	inline static safe_real bound_width();
	void boundaries(std::vector<std::vector<safe_real>> &U);
	void advance(const std::vector<std::vector<safe_real>> &U0, std::vector<std::vector<safe_real>> &U,
			const std::vector<std::vector<std::vector<safe_real>>> &F, const std::vector<std::array<safe_real, NDIM>> &X, safe_real dx, safe_real dt,
			safe_real beta, safe_real omega);
	void output(const std::vector<std::vector<safe_real>> &U, const std::vector<std::array<safe_real, NDIM>> &X, int num);

	void use_angmom_correction(int index, int count) {
		angmom_index_ = index;
		angmom_count_ = count;
	}

	static constexpr int rho_i = 0;
	static constexpr int egas_i = 1;
	static constexpr int tau_i = 2;
	static constexpr int pot_i = 3;
	static constexpr int sx_i = 4;
	static constexpr int sy_i = 5;
	static constexpr int sz_i = 6;
	static constexpr int zx_i = 4 + NDIM;
	static constexpr int zy_i = 5 + NDIM;
	static constexpr int zz_i = 6 + NDIM;

	int nf;

	hydro_computer();

private:

	template<int BW, int PAD_DIR = geo::NDIR / 2>
	static std::array<int, int(std::pow<int, int>(geo::H_NX - 2 * BW, NDIM))> find_interior_indices() {
		std::array<int, int(std::pow<int, int>(geo::H_NX - 2 * BW, NDIM))> indexes;
		int l = 0;
		for (int i = 0; i < geo::H_N3; i++) {
			int k = i;
			bool interior = true;
			for (int dim = 0; dim < NDIM; dim++) {
				const int tmp = k % geo::H_NX;
				const int lb = geo::face_locs[NDIM-1][PAD_DIR][dim] == +1 ? BW - 1 : BW;
				const int ub = geo::face_locs[NDIM-1][PAD_DIR][dim] == -1 ? geo::H_NX - BW + 1 : geo::H_NX - BW;
				interior = interior && tmp >= lb;
				interior = interior && tmp < ub;
				k /= geo::H_NX;
			}
			if (interior) {
				indexes[l++] = i;
			}
		}
		return indexes;
	}

	int angmom_index_, angmom_count_;
	static std::vector<int> find_indices(int lb, int ub);
	std::vector<std::array<safe_real, geo::NDIR / 2>> D1;
	std::vector<std::vector<std::array<safe_real, geo::NDIR>>> Q;
	std::vector<std::vector<std::array<safe_real, geo::NDIR>>> L;
	std::vector<std::vector<std::vector<std::array<safe_real, geo::NFACEDIR>>>> fluxes;

	void filter_cell(std::array<safe_real, geo::NDIR> &C, safe_real c0) {
		if constexpr (NDIM == 1) {
			hydro::filter_cell1d(C, c0);
		} else if constexpr (NDIM == 2) {
			hydro::filter_cell2d(C, c0);
		} else {
			hydro::filter_cell3d(C, c0);
		}
	}

	safe_real z_error(const std::vector<std::vector<safe_real>> &U) {
		safe_real err = 0.0;
		for (auto &i : find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
			err += std::abs(U[zx_i][i]);
		}
		return err;
	}

}
;

#include "impl/impl.hpp"

#endif /* OCTOTIGER_UNITIGER_HYDRO_HPP_ */
