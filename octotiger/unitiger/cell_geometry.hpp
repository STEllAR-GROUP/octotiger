/*
 * cell_geometry.hpp
 *
 *  Created on: Aug 5, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_CELL_GEOMETRY_HPP_
#define OCTOTIGER_UNITIGER_CELL_GEOMETRY_HPP_

/* 0  1  2 */
/* 3  4  5 */
/* 6  7  8 */

/* 9 10 11 */
/*12 13 14 */
/*15 16 17 */

/*18 19 20 */
/*21 22 23 */
/*24 25 26 */

template<int NDIM, int INX>
struct cell_geometry {
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
	static constexpr int kdeltas[3][3][3][3] = { { { { } } }, { { { 0, 1 }, { -1, 0 } } }, { { { 0, 0, 0 }, { 0, 0, 1 },
			{ 0, -1, 0 } }, { { 0, 0, -1 }, { 0, 0, 0 }, { 1, 0, 0 } }, { { 0, 1, 0 }, { -1, 0, 0 }, { 0, 0, 0 } } } };
	static constexpr int NFACEDIR = std::pow(3, NDIM - 1);
	static constexpr int lower_face_members[3][3][9] = { { { 0 } }, { { 3, 0, 6 }, { 1, 0, 2 } }, { { 12, 0, 3, 6, 9, 15,
			18, 21, 24 }, { 10, 0, 1, 2, 9, 11, 18, 19, 20 }, { 4, 0, 1, 2, 3, 5, 6, 7, 8 } } };

	static constexpr safe_real quad_weights[3][9] = { { 1.0 }, { 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0 }, { 16. / 36., 1.
			/ 36., 4. / 36., 1. / 36., 4. / 36., 4. / 36., 1. / 36., 4. / 36., 1. / 36. } };

	static constexpr safe_real vol_weights[3][27] = {
	/**/{ 1.0 },
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

	constexpr int flip(const int d) {
		return NDIR - 1 - d;
	}

};

template<int NDIM, int INX>
static inline std::vector<int> find_indices(int lb, int ub) {
	static constexpr cell_geometry<NDIM, INX> geo;
	std::vector<int> I;
	for (int i = 0; i < geo.H_N3; i++) {
		int k = i;
		bool interior = true;
		for (int dim = 0; dim < NDIM; dim++) {
			int this_i = k % geo.H_NX;
			if (this_i < lb || this_i >= ub) {
				interior = false;
				break;
			} else {
				k /= geo.H_NX;
			}
		}
		if (interior) {
			I.push_back(i);
		}
	}
	return I;
}

#endif /* OCTOTIGER_UNITIGER_CELL_GEOMETRY_HPP_ */
