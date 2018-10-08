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
//#define EQ_ONLY
//#define RHO_ONLY

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

void grid::merge_output_lists(grid::output_list_type& l1, grid::output_list_type&& l2) {

	std::unordered_map<zone_int_type, zone_int_type> index_map;

	if (l2.zones.size() > l1.zones.size()) {
		std::swap(l1, l2);
	}
	for (auto i = l2.nodes.begin(); i != l2.nodes.end(); ++i) {
		zone_int_type index, oindex;
		auto this_x = *i;
		oindex = this_x.index;
		auto j = l1.nodes.find(this_x);
		if (j != l1.nodes.end()) {
			index = j->index;
		} else {
			index = l1.nodes.size();
			this_x.index = index;
			l1.nodes.insert(this_x);
		}
		index_map[oindex] = index;
	}
	integer zzz = l1.zones.size();
	l1.zones.resize(zzz + l2.zones.size());
	for (auto i = l2.zones.begin(); i != l2.zones.end(); ++i) {
		l1.zones[zzz] = index_map[*i];
		++zzz;
	}
	for (integer field = 0; field < OUTPUT_COUNT; ++field) {
		const auto l1sz = l1.data[field].size();
		l1.data[field].resize(l1sz + l2.data[field].size());
		std::move(l2.data[field].begin(), l2.data[field].end(), l1.data[field].begin() + l1sz);
	}
	if (l1.analytic.size()) {
		for (integer field = 0; field < NF; ++field) {
			const auto l1sz = l1.analytic[field].size();
			l1.analytic[field].resize(l1sz + l2.analytic[field].size());
			std::move(l2.analytic[field].begin(), l2.analytic[field].end(), l1.analytic[field].begin() + l1sz);
		}
	}
}

grid::output_list_type grid::get_output_list(bool analytic) const {
	auto& V = TLS_V();
	compute_primitives( { { H_BW+1, H_BW+1, H_BW+1 } }, { { H_NX - H_BW-1, H_NX - H_BW-1, H_NX - H_BW-1 } });
	output_list_type rc;
	const integer vertex_order[8] = { 0, 1, 3, 2, 4, 5, 7, 6 };

	std::set<node_point>& node_list = rc.nodes;
	std::vector<zone_int_type>& zone_list = rc.zones;
	std::array<std::vector<real>, OUTPUT_COUNT> &data = rc.data;
	std::array<std::vector<real>, OUTPUT_COUNT> &A = rc.analytic;

	for (integer field = 0; field != OUTPUT_COUNT; ++field) {
		data[field].reserve(INX * INX * INX);
	}
	const integer this_bw = H_BW;
	zone_list.reserve(cube(H_NX - 2 * this_bw) * NCHILD);
	for (integer i = this_bw; i != H_NX - this_bw; ++i) {
		for (integer j = this_bw; j != H_NX - this_bw; ++j) {
			for (integer k = this_bw; k != H_NX - this_bw; ++k) {
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i - H_BW, j - H_BW, k - H_BW);
				if (opts.silo_planes_only) {
					if (!(std::abs(X[ZDIM][iii]) < dx) && !(std::abs(X[YDIM][iii]) < dx)) {
						continue;
					}
				}
				for (integer ci = 0; ci != NVERTEX; ++ci) {
					const integer vi = vertex_order[ci];
					const integer xi = (vi >> 0) & 1;
					const integer yi = (vi >> 1) & 1;

					const integer zi = (vi >> 2) & 1;
					node_point this_x;
					this_x.pt[XDIM] = X[XDIM][iii] + (real(xi) - HALF) * dx;
					this_x.pt[YDIM] = X[YDIM][iii] + (real(yi) - HALF) * dx;
					this_x.pt[ZDIM] = X[ZDIM][iii] + (real(zi) - HALF) * dx;
					auto iter = node_list.find(this_x);
					integer index;
					if (iter != node_list.end()) {
						index = iter->index;
					} else {
						index = node_list.size();
						this_x.index = index;
						node_list.insert(this_x);
					}
					zone_list.push_back(index);
				}
				for (integer field = 0; field != NF; ++field) {
					data[field].push_back(U[field][iii]);
				}
				if( opts.radiation ) {
					const integer d = H_BW - R_BW;
					rad_grid_ptr->get_output(data, i - d, j - d, k - d);
				}
				for (integer field = 0; field != NGF; ++field) {
					data[field + NRF + NF].push_back(G[iiig][field]);
				}
				data[NGF + NRF + NF + 0].push_back(V[vx_i][iii]);
				data[NGF + NRF + NF + 1].push_back(V[vy_i][iii]);
				data[NGF + NRF + NF + 2].push_back(V[vz_i][iii]);
				if (V[egas_i][iii] < de_switch2 * U[egas_i][iii]) {
					data[NGF + NRF + NF + 3].push_back(std::pow(V[tau_i][iii], fgamma));
				} else {
					data[NGF + NRF + NF + 3].push_back(V[egas_i][iii]);
				}
				data[NGF + NRF + NF + 4].push_back(V[zz_i][iii]);
				if( roche_lobe.size()) {
					data[OUTPUT_COUNT - 1].push_back(roche_lobe[h0index(i - H_BW, j - H_BW, k - H_BW)]);
				} else {
					data[OUTPUT_COUNT - 1].push_back(0.0);

				}
			}
		}
	}

	return rc;
}


static const std::array<const char*,OUTPUT_COUNT>& field_names(  ) {
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

void grid::output_header(std::string dirname, std::string base, real t, int cycle, bool a, int procs) {
#ifdef DO_OUTPUT
	std::thread([&]() {

				auto olist = DBMakeOptlist(1);
				double time = double(t);
				int ndim = 3;
				DBAddOption(olist, DBOPT_DTIME, &time);
				std::string filename = dirname + base + std::string(".silo");
//		printf("grid::output_header: filename('%s') dirname('%s') base('%s')\n",
//            filename.c_str(), dirname.c_str(), base.c_str());
				DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Euler Mesh", DB_PDB);
				assert(db);
				std::vector<int> types(procs);
				DBPutMultimesh(db, "mesh", procs, field_names().data(), types.data(), olist);
				for (int field = 0; field != OUTPUT_COUNT; ++field) {
					DBPutMultivar(db, field_names()[field], procs, field_names().data(), types.data(), olist);
				}
				DBFreeOptlist(olist);
				DBClose(db);
			}).join();
#endif
}

void grid::output(const output_list_type& olists,
    std::string _dirname, std::string _base,
    real _t, int cycle, bool analytic) {
	assert(!analytic);
#ifdef DO_OUTPUT
	std::thread(
			[&](const std::string& dirname, const std::string& base, real t) {
				const std::set<node_point>& node_list = olists.nodes;
				const std::vector<zone_int_type>& zone_list = olists.zones;

				const int nzones = zone_list.size() / NVERTEX;
				std::vector<int> zone_nodes;
				zone_nodes = std::move(zone_list);

				const int nnodes = node_list.size();
				std::vector<double> x_coord(nnodes);
				std::vector<double> y_coord(nnodes);
				std::vector<double> z_coord(nnodes);
				std::array<double*, NDIM> node_coords = {x_coord.data(), y_coord.data(), z_coord.data()};
				for (auto iter = node_list.begin(); iter != node_list.end(); ++iter) {
					const integer i = iter->index;
					x_coord[i] = iter->pt[0];
					y_coord[i] = iter->pt[1];
					z_coord[i] = iter->pt[2];
				}

				constexpr int nshapes = 1;
				int shapesize[1] = {NVERTEX};
				int shapetype[1] = {DB_ZONETYPE_HEX};
				int shapecnt[1] = {nzones};
				const char* coord_names[NDIM] = {"x", "y", "z"};
				auto olist = DBMakeOptlist(1);
				double time = double(t);
				int ndim = 3;
				//DBAddOption(olist, DBOPT_CYCLE, &cycle);
				DBAddOption(olist, DBOPT_DTIME, &time);
				//DBAddOption(olist, DBOPT_NSPACE, &ndim );
                std::string filename = dirname + base;
//        		printf("grid::output: filename('%s') dirname('%s') base('%s')\n",
//                    filename.c_str(), dirname.c_str(), base.c_str());
				DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Euler Mesh", DB_PDB);
				assert(db);
				DBPutZonelist2(db, "zones", nzones, int(NDIM), zone_nodes.data(), nzones * NVERTEX, 0, 0, 0, shapetype, shapesize,
						shapecnt, nshapes, olist);
				DBPutUcdmesh(db, "mesh", int(NDIM), const_cast<char**>(coord_names), node_coords.data(), nnodes, nzones, "zones", nullptr, DB_DOUBLE,
						olist);
				DBFreeOptlist(olist);
				for (int field = 0; field != OUTPUT_COUNT; ++field) {
					auto olist = DBMakeOptlist(1);
					double time = double(t);
					int istrue = 1;
					int isfalse = 0;
					DBAddOption(olist, DBOPT_DTIME, &time);
				//	printf( "%lli\n", reinterpret_cast<long long int>(olists.data[field].data()));
					DBPutUcdvar1(db, field_names()[field], "mesh", const_cast<void*>(reinterpret_cast<const void*>(olists.data[field].data())), nzones, nullptr, 0, DB_DOUBLE, DB_ZONECENT,
							olist);
				//	printf( "%s\n", field_names()[field]);
					DBFreeOptlist(olist);
				}
				DBClose(db);
			}, _dirname, _base, _t).join();
#endif
}

std::size_t grid::load(std::istream& strm, bool old_format) {
	static hpx::mutex mtx;
	std::size_t cnt = 0;
	{
		static std::atomic<bool> statics_loaded(false);
        bool expected = false;
		if(statics_loaded.compare_exchange_strong(expected, true)) {
			cnt += read(strm, &scaling_factor, 1);
			if( opts.ngrids > -1 && !opts.refinement_floor_specified) {
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

