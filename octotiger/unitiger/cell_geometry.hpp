//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_UNITIGER_CELL_GEOMETRY_HPP_
#define OCTOTIGER_UNITIGER_CELL_GEOMETRY_HPP_

#include <mutex>

#include "octotiger/unitiger/util.hpp"

template<int NDIM, int INX>
struct cell_geometry {

	static constexpr int H_BW = OCTOTIGER_BW;
	static constexpr int H_NX = (2 * H_BW + INX);

	static constexpr int H_NX_X = cell_geometry::H_NX;
	static constexpr int H_NX_Y = NDIM > 1 ? cell_geometry::H_NX : 1;
	static constexpr int H_NX_Z = NDIM > 2 ? cell_geometry::H_NX : 1;

	static constexpr int H_NX_XM2 = cell_geometry::H_NX - 2;
	static constexpr int H_NX_YM2 = NDIM > 1 ? cell_geometry::H_NX - 2 : 1;
	static constexpr int H_NX_ZM2 = NDIM > 2 ? cell_geometry::H_NX - 2 : 1;

	static constexpr int H_NX_XM3 = cell_geometry::H_NX - 3;
	static constexpr int H_NX_YM3 = NDIM > 1 ? cell_geometry::H_NX - 3 : 1;
	static constexpr int H_NX_ZM3 = NDIM > 2 ? cell_geometry::H_NX - 3 : 1;

	static constexpr int H_NX_XM4 = cell_geometry::H_NX - 4;
	static constexpr int H_NX_YM4 = NDIM > 1 ? cell_geometry::H_NX - 4 : 1;
	static constexpr int H_NX_ZM4 = NDIM > 2 ? cell_geometry::H_NX - 4 : 1;

	static constexpr int H_NX_XM6 = cell_geometry::H_NX - 6;
	static constexpr int H_NX_YM6 = NDIM > 1 ? cell_geometry::H_NX - 6 : 1;
	static constexpr int H_NX_ZM6 = NDIM > 2 ? cell_geometry::H_NX - 6 : 1;

	static constexpr int H_NX_XM8 = cell_geometry::H_NX - 8;
	static constexpr int H_NX_YM8 = NDIM > 1 ? cell_geometry::H_NX - 8 : 1;
	static constexpr int H_NX_ZM8 = NDIM > 2 ? cell_geometry::H_NX - 8 : 1;

	static constexpr int H_DNX = NDIM == 3 ? cell_geometry::H_NX * cell_geometry::H_NX : (NDIM == 2 ? cell_geometry::H_NX : 1);
	static constexpr int H_DNY = NDIM == 3 ? cell_geometry::H_NX : 1;
	static constexpr int H_DNZ = 1;
	static constexpr int H_N3 = std::pow(cell_geometry::H_NX, NDIM);
	static constexpr int H_DN0 = 0;
	static constexpr int NDIR = std::pow(3, NDIM);
	static constexpr int NANGMOM = NDIM == 1 ? 0 : std::pow(3, NDIM - 2);
	static constexpr int NFACEDIR = std::pow(3, NDIM - 1);
	static constexpr int H_DN[3] = { H_DNX, H_DNY, H_DNZ };

	static constexpr int group_count() {
		return ngroups_[NDIM - 1];
	}

	static int group_size(int gi) {
		return group_size_[NDIM - 1][gi];
	}

	static std::pair<int, int> group_pair(int gi, int ni) {
		return groups3d_[NDIM - 1][gi][ni];
	}

private:

	static constexpr int ngroups_[3] = { 0, 1, 4 };
	static constexpr int group_size_[3][4] = { { }, { 4 }, { 8, 4, 4, 4 } };
	static constexpr std::pair<int, int> groups3d_[3][4][8] = { { { } }, { {
	/**/{ -H_DNX - H_DNY, 8 },
	/**/{ -H_DNX, 2 },
	/**/{ -H_DNY, 6 },
	/**/{ -H_DN0, 0 } } },
	/**/{ {
	/* 0  1  2 */
	/* 3  4  5 */
	/* 6  7  8 */

	/* 9 10 11 */
	/*12 13 14 */
	/*15 16 17 */

	/*18 19 20 */
	/*21 22 23 */
	/*24 25 26 */

	/**/{ (-H_DNX - H_DNY - H_DNZ), 26 },
	/**/{ (-H_DNY - H_DNZ), 24 },
	/**/{ (-H_DNX - H_DNZ), 20 },
	/**/{ (-H_DNX - H_DNY), 8 },
	/**/{ -H_DNX, 2 },
	/**/{ -H_DNY, 6 },
	/**/{ -H_DNZ, 18 },
	/**/{ -H_DN0, 0 } }, {

	/**/{ (-H_DNX - H_DNY), 17 },
	/**/{ -H_DNX, 11 },
	/**/{ -H_DNY, 15 },
	/**/{ -H_DN0, 9 }, }, {

	/**/{ (-H_DNX - H_DNZ), 23 },
	/**/{ -H_DNX, 5 },
	/**/{ -H_DNZ, 21 },
	/**/{ -H_DN0, 3 }, }, {

	/**/{ (-H_DNZ - H_DNY), 25 },
	/**/{ -H_DNY, 7 },
	/**/{ -H_DNZ, 19 },
	/**/{ -H_DN0, 1 } } } };
	static constexpr bool is_lower_face[3][3][27] = { { 1, 0, 0 },
	/**/{ { 1, 0, 0, 1, 0, 0, 1, 0, 0 }, { 1, 1, 1, 0, 0, 0, 0, 0, 0 } }, {
	/**/{ 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0 },
	/**/{ 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 },
	/**/{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } };

	static constexpr int levi_civitas[3][3][3][3] = { { { { } } }, { { { 0, 1 }, { -1, 0 } } }, { { { 0, 0, 0 }, { 0, 0, 1 }, { 0, -1, 0 } }, { { 0, 0, -1 }, {
			0, 0, 0 }, { 1, 0, 0 } }, { { 0, 1, 0 }, { -1, 0, 0 }, { 0, 0, 0 } } } };
	static constexpr int lower_face_members[3][3][9] = { { { 0 } }, { { 3, 0, 6 }, { 1, 0, 2 } }, { { 12, 0, 3, 6, 9, 15, 18, 21, 24 }, { 10, 0, 1, 2, 9, 11,
			18, 19, 20 }, { 4, 0, 1, 2, 3, 5, 6, 7, 8 } } };

	static constexpr safe_real quad_weights[3][9] = { { 1.0 }, { 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0 }, { 16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4.
			/ 36., 1. / 36., 4. / 36., 1. / 36. } };

	static constexpr safe_real vol_weights[3][27] = {
	/**/{ 1. / 6., 4. / 6., 1. / 6. },
	/**/{ 1. / 36., 4. / 36., 1. / 36., 4. / 36., 16. / 36., 4. / 36., 1. / 36., 4. / 36., 1. / 36. },
	/**/{ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216., 1. / 216.,
	/****/4. / 216., 16. / 216., 4. / 216., 16. / 216., 64. / 216., 16. / 216., 4. / 216., 16. / 216., 4. / 216.,
	/****/1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216., 1. / 216. } };

	static constexpr int face_locs[3][27][3] = {
	/**/{ { -1 }, { 0 }, { 1 } },

	/**/{
	/**/{ -1, -1 }, { +0, -1 }, { +1, -1 },
	/**/{ -1, +0 }, { +0, +0 }, { +1, +0 },
	/**/{ -1, +1 }, { +0, +1 }, { +1, +1 } },

	/**/{
	/**/{ -1, -1, -1 }, { +0, -1, -1 }, { +1, -1, -1 },
	/**/{ -1, +0, -1 }, { +0, +0, -1 }, { 1, +0, -1 },
	/**/{ -1, +1, -1 }, { +0, +1, -1 }, { +1, +1, -1 },
	/**/{ -1, -1, +0 }, { +0, -1, +0 }, { +1, -1, +0 },
	/**/{ -1, +0, +0 }, { +0, +0, +0 }, { +1, +0, +0 },
	/**/{ -1, +1, +0 }, { +0, +1, +0 }, { +1, +1, +0 },
	/**/{ -1, -1, +1 }, { +0, -1, +1 }, { +1, -1, +1 },
	/**/{ -1, +0, +1 }, { +0, +0, +1 }, { +1, +0, +1 },
	/**/{ -1, +1, +1 }, { +0, +1, +1 }, { +1, +1, +1 } } };

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

	static std::array<std::array<std::vector<int>, NDIR>, H_BW> all_indices;

	static void verify_3d_constdefs() {
		for (int i = -1; i < 2; i++) {
			for (int j = -1; j < 2; j++) {
				for (int k = -1; k < 2; k++) {
					const int index = (i + 1) + 3 * (j + 1) + 9 * (k + 1);
					safe_real sum = H_DN[0] * i;
					sum += H_DN[1] * j;
					sum += H_DN[2] * k;
					if (directions[2][index] != sum) {
						printf("directions failed verification at %i %i %i\n", i, j, k);
						abort();
					}
					bool cond = false;
					cond = cond || (face_locs[2][index][0] != i);
					cond = cond || (face_locs[2][index][1] != j);
					cond = cond || (face_locs[2][index][2] != k);
					if (cond) {
						printf("xlocs failed verification at %i %i %i %i\n", index, i, j, k);
						for (int dim = 0; dim < NDIM; dim++) {
							printf("%i ", face_locs[2][index][dim]);
						}
						printf("\n");
						abort();
					}
				}
			}
		}

		bool fail = false;
		/* corners */
		int gi = 0;
		for (int n = 0; n < group_size_[2][0]; n++) {
			const auto pair = group_pair(gi, n);
			const int x = pair.second % 3;
			const int y = (pair.second / 3) % 3;
			const int z = pair.second / 9;
			const int index = -((x / 2) * H_DNX + (y / 2) * H_DNY + (z / 2) * H_DNZ);
			if (index != pair.first) {
				fail = true;
			}
		}
		gi = 1;
		for (int n = 0; n < group_size_[2][1]; n++) {
			const auto pair = group_pair(gi, n);
			const int x = pair.second % 3;
			const int y = (pair.second / 3) % 3;
			const int index = -((x / 2) * H_DNX + (y / 2) * H_DNY);
			if (index != pair.first) {
				fail = true;
			}
		}
		gi = 2;
		for (int n = 0; n < group_size_[2][2]; n++) {
			const auto pair = group_pair(gi, n);
			const int x = pair.second % 3;
			const int z = pair.second / 9;
			const int index = -((x / 2) * H_DNX + (z / 2) * H_DNZ);
			if (index != pair.first) {
				fail = true;
			}
		}
		gi = 3;
		for (int n = 0; n < group_size_[2][2]; n++) {
			const auto pair = group_pair(gi, n);
			const int y = (pair.second / 3) % 3;
			const int z = pair.second / 9;
			const int index = -((y / 2) * H_DNY + (z / 2) * H_DNZ);
			if (index != pair.first) {
				fail = true;
			}
		}
		if (fail) {
			printf("Corners/edges indexes failed\n");
			abort();
		}
//		printf("3D geometry constdefs passed verification\n");
	}

public:

	cell_geometry() {
		static std::once_flag flag;
		std::call_once(flag, []() {
			printf( "Initializing cell_geometry %i %i %i\n", NDIM, INX, cell_geometry::H_NX);
			verify_3d_constdefs();
			for (int bw = 1; bw <= H_BW; bw++) {
				for (int d = 0; d < NDIR; d++) {
					all_indices[bw - 1][d] = find_indices(bw, cell_geometry::H_NX - bw, d);
				}
			}
		});
	}

	inline const auto& get_indexes(int bw, int d) const {
		return all_indices[bw - 1][d];
	}

	inline static constexpr auto levi_civita() {
		return levi_civitas[NDIM - 1];
	}
	inline static constexpr auto direction() {
		return directions[NDIM - 1];
	}

	inline static constexpr auto xloc() {
		return face_locs[NDIM - 1];
	}

	inline static constexpr auto is_lface(int dim, int d) {
		return is_lower_face[NDIM - 1][dim][d];
	}

	inline static constexpr auto volume_weight() {
		return vol_weights[NDIM - 1];
	}

	inline static constexpr auto face_weight() {
		return quad_weights[NDIM - 1];
	}

	inline static constexpr auto face_pts() {
		return lower_face_members[NDIM - 1];
	}

	inline constexpr int flip(const int d) const {
		return NDIR - 1 - d;
	}

	static inline int flip_dim(const int d, int flip_dim) {
		std::array<int, NDIM> dims;
		int k = d;
		for (int dim = 0; dim < NDIM; dim++) {
			dims[dim] = k % 3;
			k /= 3;
		}
		k = 0;
		dims[flip_dim] = 2 - dims[flip_dim];
		for (int dim = 0; dim < NDIM; dim++) {
			k *= 3;
			k += dims[NDIM - 1 - dim];
		}
		return k;
	}

	static auto to_index(int j, int k, int l) {
		if /*constexpr*/(NDIM == 1) {
			return j;
		} else if /*constexpr*/(NDIM == 2) {
			return (j * cell_geometry::H_NX + k);
		} else {
			return (j * cell_geometry::H_NX + k) * cell_geometry::H_NX + l;
		}
	}

	static inline std::vector<int> find_indices(int lb, int ub, int d = NDIR / 2) {
		std::vector<int> I;
		std::array<int, NDIM> lbs;
		std::array<int, NDIM> ubs;
		for (int dim = 0; dim < NDIM; dim++) {
			ubs[dim] = xloc()[d][dim] == -1 ? (ub + 1) : ub;
			lbs[dim] = xloc()[d][dim] == +1 ? (lb - 1) : lb;
		}
		for (int i = 0; i < H_N3; i++) {
			bool interior = true;
			const auto dims = index_to_dims<NDIM, cell_geometry::H_NX>(i);
			for (int dim = 0; dim < NDIM; dim++) {
				int this_i = dims[dim];
				if (this_i < lbs[dim] || this_i >= ubs[dim]) {
					interior = false;
					break;
				}
			}
			if (interior) {
				I.push_back(i);
			}
		}
		return I;
	}

};

template<int NDIM, int INX>
std::array<std::array<std::vector<int>, cell_geometry<NDIM, INX>::NDIR>, cell_geometry<NDIM, INX>::H_BW> cell_geometry<NDIM, INX>::all_indices;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::ngroups_[3];

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::group_size_[3][4];

template<int NDIM, int INX>
constexpr std::pair<int, int> cell_geometry<NDIM, INX>::groups3d_[3][4][8];

template<int NDIM, int INX>
constexpr bool cell_geometry<NDIM, INX>::is_lower_face[3][3][27];

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::levi_civitas[3][3][3][3];

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::lower_face_members[3][3][9];

template<int NDIM, int INX>
constexpr safe_real cell_geometry<NDIM, INX>::quad_weights[3][9];

template<int NDIM, int INX>
constexpr safe_real cell_geometry<NDIM, INX>::vol_weights[3][27];

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::face_locs[3][27][3];

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::directions[3][27];

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_DN[3];

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_BW;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_X;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_Y;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_Z;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_XM2;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_YM2;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_ZM2;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_XM4;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_YM4;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_ZM4;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_XM6;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_YM6;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_NX_ZM6;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_DNX;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_DNY;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_DNZ;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_N3;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::H_DN0;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::NDIR;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::NANGMOM;

template<int NDIM, int INX>
constexpr int cell_geometry<NDIM, INX>::NFACEDIR;

#endif /* OCTOTIGER_UNITIGER_CELL_GEOMETRY_HPP_ */
