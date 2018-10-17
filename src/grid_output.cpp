#include "defs.hpp"
#include "grid.hpp"
#ifdef DO_OUTPUT
#include <silo.h>
#endif
#include <atomic>
#include <ios>
#include <fstream>
#include <thread>
#include <cmath>
#include "physcon.hpp"
#include "options.hpp"

extern options opts;
#include "radiation/rad_grid.hpp"

#include <hpx/include/lcos.hpp>

namespace hpx {
using mutex = hpx::lcos::local::spinlock;
}

std::vector<std::vector<real>>& TLS_V();

#include <unordered_map>




inline bool float_eq(xpoint_type a, xpoint_type b) {
	constexpr static xpoint_type eps = 0.00000011920928955078125; // std::pow(xpoint_type(2), -23);
// 	const xpoint_type eps = std::pow(xpoint_type(2), -23);
	return std::abs(a - b) < eps;
}

bool grid::xpoint_eq(const xpoint& a, const xpoint& b) {
	bool rc = true;
	for (integer d = 0; d != NDIM; ++d) {
		if (!float_eq(a[d], b[d])) {
			rc = false;
			break;
		}
	}
	return rc;
}

bool grid::node_point::operator==(const node_point& other) const {
	return xpoint_eq(other.pt, pt);
}

bool grid::node_point::operator<(const node_point& other) const {
	bool rc = false;
	for (integer d = 0; d != NDIM; ++d) {
		if (!float_eq(pt[d], other.pt[d])) {
			rc = (pt[d] < other.pt[d]);
			break;
		}
	}
	return rc;
}

const std::array<const char*,OUTPUT_COUNT>& grid::field_names(  ) {
	static std::array<const char*,OUTPUT_COUNT> field_names;
	if( opts.radiation) {
		field_names = {{"rho", "egas", "sx", "sy", "sz", "tau", "pot", "zx", "zy", "zz", "primary_core", "primary_envelope", "secondary_core",
			"secondary_envelope", "vacuum", "er", "fx", "fy", "fz", "phi", "gx", "gy", "gz", "vx", "vy", "vz", "eint",
			"zzs", "roche"}};
	} else {
		field_names = {{"rho", "egas", "sx", "sy", "sz", "tau", "pot", "zx", "zy", "zz", "primary_core",
			"primary_envelope", "secondary_core", "secondary_envelope", "vacuum", "phi", "gx", "gy", "gz", "vx", "vy", "vz", "eint",
			"zzs", "roche"}};
	}
	return field_names;
}

std::size_t grid::load(std::istream& strm, bool old_format) {
	static hpx::mutex mtx;
	std::size_t cnt = 0;
	{
		static std::atomic<bool> statics_loaded(false);
        bool expected = false;
		if(statics_loaded.compare_exchange_strong(expected, true)) {
			cnt += read(strm, &scaling_factor, 1);
			if( opts.ngrids > -1 && opts.refinement_floor < 0.0) {
				cnt += read( strm, &opts.refinement_floor, 1);
			} else {
				real dummy;
				cnt += read( strm, &dummy, 1);
			}
			if( !old_format ) {
				cnt += read( strm, &physcon.A, 1);
				cnt += read( strm, &physcon.B, 1);
			}
			statics_loaded = true;
		} else {
			std::size_t offset = (old_format ? 2 : 4) * sizeof(real);
                        strm.seekg(offset, std::ios_base::cur);
			cnt += offset;
		}
	}

	cnt += read( strm, &is_leaf, sizeof(bool)) * sizeof(bool);
	cnt += read( strm, &is_root, sizeof(bool)) * sizeof(bool);

	allocate();

	for (integer f = 0; f != NF; ++f) {
		for (integer i = H_BW; i < H_NX - H_BW; ++i) {
			for (integer j = H_BW; j < H_NX - H_BW; ++j) {
				if( f < NF - 3 || !old_format) {
					const integer iii = hindex(i, j, H_BW);
					cnt += read( strm, &(U[f][iii]), INX);
				} else {
					for( integer k = H_BW; k != H_NX - H_BW; ++k) {
						const integer iii = hindex(i, j, k);
						U[f][iii] = 0.0;
					}
				}
			}
		}
	}
	for (integer i = 0; i < G_NX; ++i) {
		for (integer j = 0; j < G_NX; ++j) {
			for (integer k = 0; k < G_NX; ++k) {
				const integer iii = gindex(i, j, k);
				for( integer f = 0; f != NGF; ++f ) {
					real tmp;
					cnt += read( strm, &tmp, 1);
					G[iii][f] = tmp;
				}
			}
		}
	}
	if (!old_format) {
		cnt += read( strm, U_out.data(), U_out.size());
	} else {
		std::fill(U_out.begin(), U_out.end(), 0.0);
		cnt += read( strm, U_out.data(), U_out.size() - 3);
	}
	if( opts.radiation) {
		cnt += rad_grid_ptr->load(strm);
	}
	set_coordinates();
	return cnt;
}

std::size_t grid::save(std::ostream& strm) const {
    std::size_t cnt = 0;

    cnt += write(strm, scaling_factor);
    cnt += write(strm, opts.refinement_floor);
    cnt += write(strm, physcon.A);
    cnt += write(strm, physcon.B);

    cnt += write(strm, is_leaf);
    cnt += write(strm, is_root);

    for (integer f = 0; f != NF; ++f) {
        for (integer i = H_BW; i < H_NX - H_BW; ++i) {
            for (integer j = H_BW; j < H_NX - H_BW; ++j) {
                const integer iii = hindex(i, j, H_BW);
                cnt += write(strm, &U[f][iii], INX);
            }
        }
    }
    for (integer i = 0; i < G_NX; ++i) {
        for (integer j = 0; j < G_NX; ++j) {
            for (integer k = 0; k < G_NX; ++k) {
                const integer iii = gindex(i, j, k);
                for( integer f = 0; f != NGF; ++f ) {
                    real tmp = G[iii][f];
                    cnt += write(strm, tmp);
                }
            }
        }
    }
    cnt += write(strm, U_out.data(), U_out.size());
    if( opts.radiation) {
    	cnt += rad_grid_ptr->save(strm);
    }
    return cnt;
}

