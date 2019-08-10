/*
 * cell_geometry.hpp
 *
 *  Created on: Aug 5, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_CELL_GEOMETRY_HPP_
#define OCTOTIGER_UNITIGER_CELL_GEOMETRY_HPP_

#include <mutex>

/* 0  1  2 */
/* 3  4  5 */
/* 6  7  8 */

/* 9 10 11 */
/*12 13 14 */
/*15 16 17 */

/*18 19 20 */
/*21 22 23 */
/*24 25 26 */

#include "./util.hpp"

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
	static constexpr int NFACEDIR = std::pow(3, NDIM - 1);

private:
	static constexpr int kdeltas[3][3][3][3] = { { { { } } }, { { { 0, 1 }, { -1, 0 } } }, { { { 0, 0, 0 }, { 0, 0, 1 }, { 0, -1, 0 } }, { { 0, 0, -1 }, { 0, 0,
			0 }, { 1, 0, 0 } }, { { 0, 1, 0 }, { -1, 0, 0 }, { 0, 0, 0 } } } };
	static constexpr int lower_face_members[3][3][9] = { { { 0 } }, { { 3, 0, 6 }, { 1, 0, 2 } }, { { 12, 0, 3, 6, 9, 15, 18, 21, 24 }, { 10, 0, 1, 2, 9, 11,
			18, 19, 20 }, { 4, 0, 1, 2, 3, 5, 6, 7, 8 } } };

	static constexpr safe_real quad_weights[3][9] = { { 1.0 }, { 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0 }, { 16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4.
			/ 36., 1. / 36., 4. / 36., 1. / 36. } };

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

	static std::array<std::array<std::vector<int>, NDIR>, H_BW> all_indices;

public:

	cell_geometry() {
		static std::once_flag flag;
		std::call_once(flag, []() {
			for (int bw = 1; bw <= H_BW; bw++) {
				for (int d = 0; d < NDIR; d++) {
					all_indices[bw - 1][d] = find_indices(bw, H_NX - bw, d);
				}
			}
		});
	}

	inline const auto& get_indexes(int bw, int d) const {
		return all_indices[bw - 1][d];
	}

	inline static constexpr auto kronecker_delta() {
		return kdeltas[NDIM - 1];
	}
	inline static constexpr auto direction() {
		return directions[NDIM - 1];
	}

	inline static constexpr auto xloc() {
		return face_locs[NDIM - 1];
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

	inline constexpr int flip(const int d) {
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
	;

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
			const auto dims = index_to_dims<NDIM, H_NX>(i);
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

#endif /* OCTOTIGER_UNITIGER_CELL_GEOMETRY_HPP_ */
