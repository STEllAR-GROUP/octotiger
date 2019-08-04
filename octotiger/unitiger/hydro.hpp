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

template<int NDIM, int INX, int ORDER>
struct hydro_computer {
	safe_real hydro_flux(std::vector<std::vector<safe_real>> U, std::vector<std::vector<std::vector<safe_real>>> &F,
			std::vector<std::array<safe_real, NDIM>> &X, safe_real omega);

	void update_tau(std::vector<std::vector<safe_real>> &U);

	template<class VECTOR>
	void to_prim(VECTOR u, safe_real &p, safe_real &v, int dim);
	template<class VECTOR>
	void flux(const VECTOR &UL, const VECTOR &UR, VECTOR &F, int dim, safe_real &a, std::array<safe_real, NDIM> &vg);
	inline static safe_real minmod(safe_real a, safe_real b);
	inline static safe_real minmod_theta(safe_real a, safe_real b, safe_real c);
	inline static safe_real bound_width();

	void boundaries(std::vector<std::vector<safe_real>> &U);
	void advance(const std::vector<std::vector<safe_real>> &U0, std::vector<std::vector<safe_real>> &U,
			const std::vector<std::vector<std::vector<safe_real>>> &F, safe_real dx, safe_real dt, safe_real beta, safe_real omega);
	void output(const std::vector<std::vector<safe_real>> &U, const std::vector<std::array<safe_real, NDIM>> &X, int num);

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
	static constexpr int spc_i = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));

	int nf;

	hydro_computer(int nspecies);

private:
	static std::vector<int> find_indices(int lb, int ub);
	int ns;
	static constexpr int H_BW = 3;
	static constexpr int H_NX = (2 * H_BW + INX);
	static constexpr int H_DNX = 1;
	static constexpr int H_DN[3] = { 1, H_NX, H_NX * H_NX };
	static constexpr int H_DNY = H_NX;
	static constexpr int H_DNZ = (H_NX * H_NX);
	static constexpr int H_N3 = std::pow(H_NX, NDIM);
	static constexpr int H_DN0 = 0;
	static constexpr int NDIR = std::pow(3, NDIM);
	static constexpr int NANGMOM = NDIM == 1 ? 0 : std::pow(3, NDIM - 2);
	static constexpr int kdeltas[3][3][3][3] = { { { { } } }, { { { 0, 1 }, { -1, 0 } } }, { { { 0, 0, 0 }, { 0, 0, 1 }, { 0, -1, 0 } }, { { 0, 0, -1 }, { 0, 0,
			0 }, { 1, 0, 0 } }, { { 0, 1, 0 }, { -1, 0, 0 }, { 0, 0, 0 } } } };
	static constexpr int NFACEDIR = std::pow(3, NDIM - 1);
	static constexpr int lower_face_members[3][3][9] = { { { 0 } }, { { 3, 0, 6 }, { 1, 0, 2 } }, { { 12, 0, 3, 6, 9, 15, 18 },
			{ 10, 0, 1, 2, 9, 11, 18, 19, 20 }, { 4, 0, 1, 2, 3, 5, 6, 7, 8 } } };

	static constexpr safe_real quad_weights[3][9] = { { 1.0 }, { 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0 }, { 16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 16.
			/ 36., 1. / 36., 4. / 36., 1. / 36. } };

	static constexpr safe_real vol_weights[3][27] = {
	/**/{ 1.0 },
	/**/{ 1. / 36., 4. / 36., 1. / 36., 4. / 36., 16. / 36., 4. / 36., 1. / 36., 4. / 36., 1. / 36. },
	/**/{ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216., 1. / 216.,
	/****/4. / 216., 16. / 216., 4. / 216., 16. / 216., 64. / 216., 16. / 216., 4. / 216., 16. / 216., 4. / 216.,
	/****/1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216., 1. / 216. } };

	static constexpr int face_locs[3][27][3] = {
	/**/{ { -1 }, { 0 }, { 1 } },



	/**/{ { -1, -1 }, {  0, -1 }, { 1, -1 },
	/**/  { -1,  0 }, { +0,  0 }, { 1,  0 },
	/**/{   -1,  1 }, {  0,  1 }, { 1,  1 } },


	/**/{ { -1, -1, -1 }, { -1, -1, +0 }, { -1, -1, +1 },
	/**/{ -1, +0, -1 }, { -1, +0, +0 }, { -1, +0, +1 },
	/**/{ -1, +1, -1 }, { -1, +1, +0 }, { -1, +1, +1 },
	/**/{ +0, -1, -1 }, { +0, -1, +0 }, { +0, -1, +1 },
	/**/{ +0, +0, -1 }, { +0, +0, +0 }, { +0, +0, +1 },
	/**/{ +0, +1, -1 }, { +0, +1, +0 }, { +0, +1, +1 },
	/**/{ +1, -1, -1 }, { +1, -1, +0 }, { +1, -1, +1 },
	/**/{ +1, +0, -1 }, { +1, +0, +0 }, { +1, +0, +1 },
	/**/{ +1, +1, -1 }, { +1, +1, +0 }, { +1, +1, +1 } } };

	static constexpr int directions[3][27] = { {
	/**/-H_DNX, +H_DN0, +H_DNX /**/
	}, {
	/**/-H_DNX - H_DNY, +H_DN0 - H_DNY, +H_DNX - H_DNY,/**/
	/**/-H_DNX + H_DN0, +H_DN0 + H_DN0, +H_DNX + H_DN0,/**/
	/**/-H_DNX + H_DNY, +H_DN0 + H_DNY, +H_DNX + H_DNY, /**/
	}, {
	/**/-H_DNX - H_DNY - H_DNZ, +H_DN0 - H_DNY - H_DNZ, +H_DNX - H_DNY - H_DNZ,/**/
	/**/-H_DNX + H_DN0 - H_DNZ, +H_DN0 + H_DN0 - H_DNZ, +H_DNX + H_DN0 - H_DNZ,/**/
	/**/-H_DNX + H_DNY - H_DNZ, +H_DN0 + H_DNY - H_DNZ, +H_DNX + H_DNY - H_DNZ,/**/
	/**/-H_DNX - H_DNY + H_DN0, +H_DN0 - H_DNY + H_DN0, +H_DNX - H_DNY + H_DN0,/**/
	/**/-H_DNX + H_DN0 + H_DN0, +H_DN0 + H_DN0 + H_DN0, +H_DNX + H_DN0 + H_DN0,/**/
	/**/-H_DNX + H_DNY + H_DN0, +H_DN0 + H_DNY + H_DN0, +H_DNX + H_DNY + H_DN0,/**/
	/**/-H_DNX - H_DNY + H_DNZ, +H_DN0 - H_DNY + H_DNZ, +H_DNX - H_DNY + H_DNZ,/**/
	/**/-H_DNX + H_DN0 + H_DNZ, +H_DN0 + H_DN0 + H_DNZ, +H_DNX + H_DN0 + H_DNZ,/**/
	/**/-H_DNX + H_DNY + H_DNZ, +H_DN0 + H_DNY + H_DNZ, +H_DNX + H_DNY + H_DNZ/**/

	} };
	std::vector<std::array<safe_real, NDIR / 2>> D1;
	std::vector<std::vector<std::array<safe_real, NDIR>>> Q;
	std::vector<std::vector<std::array<safe_real, NDIR>>> L;
	std::vector<std::vector<std::vector<std::array<safe_real, NFACEDIR>>>> fluxes;

	void filter_cell(std::array<safe_real,NDIR> &C, safe_real c0) {
		if constexpr (NDIM == 1) {
			hydro::filter_cell1d(C,c0);
		} else if constexpr (NDIM == 2) {
			hydro::filter_cell2d(C,c0);
		} else {
			hydro::filter_cell3d(C,c0);
		}
	}

	safe_real z_error(const std::vector<std::vector<safe_real>>& U) {
		safe_real err = 0.0;
		for( auto& i : find_indices(H_BW,H_NX-H_BW) ) {
			err += std::abs(U[zx_i][i]);
		}
		return err;
	}

}
;

#include "impl/impl.hpp"

#endif /* OCTOTIGER_UNITIGER_HYDRO_HPP_ */
