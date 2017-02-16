#include "grid.hpp"
#ifdef DO_OUTPUT
#include <silo.h>
#endif
#include <fstream>
#include <thread>
#include <cmath>

#ifdef RADIATION
#include "rad_grid.hpp"
#endif

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
	for (integer field = 0; field < NF + NRF + NGF + NPF; ++field) {
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
	compute_primitives( { { H_BW, H_BW, H_BW } }, { { H_NX - H_BW, H_NX - H_BW, H_NX - H_BW } });
	output_list_type rc;
	const integer vertex_order[8] = { 0, 1, 3, 2, 4, 5, 7, 6 };

	std::set<node_point>& node_list = rc.nodes;
	std::vector<zone_int_type>& zone_list = rc.zones;
	std::array<std::vector<real>, NF + NRF + NGF + NPF> &data = rc.data;
	std::array<std::vector<real>, NF + NRF + NGF + NPF> &A = rc.analytic;

	for (integer field = 0; field != NF + NRF + NGF + NPF; ++field) {
		data[field].reserve(INX * INX * INX);
		if (analytic) {
			A[field].reserve(INX * INX * INX);
		}
	}
	const integer this_bw = H_BW;
	zone_list.reserve(cube(H_NX - 2 * this_bw) * NCHILD);
	for (integer i = this_bw; i != H_NX - this_bw; ++i) {
		for (integer j = this_bw; j != H_NX - this_bw; ++j) {
			for (integer k = this_bw; k != H_NX - this_bw; ++k) {
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i - H_BW, j - H_BW, k - H_BW);
#ifdef EQ_ONLY
				if (!(std::abs(X[ZDIM][iii]) < dx) && !(std::abs(X[YDIM][iii]) < dx)) {
					continue;
				}
#endif
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
				if (analytic) {
					for (integer field = 0; field != NF; ++field) {
						A[field].push_back(Ua[field][iii]);
					}
				}
				for (integer field = 0; field != NF; ++field) {
					data[field].push_back(U[field][iii]);
				}
#ifdef RADIATION
				const integer d = H_BW - R_BW;
				rad_grid_ptr->get_output(data, i - d, j - d, k - d);
#endif
				for (integer field = 0; field != NGF; ++field) {
					data[field + NRF + NF].push_back(G[iiig][field]);
				}
				data[NGF + NRF + NF + 0].push_back(V[vx_i][iii]);
				data[NGF + NRF + NF + 1].push_back(V[vy_i][iii]);
				data[NGF + NRF + NF + 2].push_back(V[vz_i][iii]);
				if (V[egas_i][iii] * V[rho_i][iii] < de_switch2 * U[egas_i][iii]) {
					data[NGF + NRF + NF + 3].push_back(std::pow(V[tau_i][iii], fgamma) / V[rho_i][iii]);
				} else {
					data[NGF + NRF + NF + 3].push_back(V[egas_i][iii]);
				}
				data[NGF + NRF + NF + 4].push_back(V[zz_i][iii]);
			}
		}
	}

	return rc;
}

void grid::output(const output_list_type& olists, std::string _filename, real _t, int cycle, bool analytic) {
#ifdef DO_OUTPUT

	std::thread(
			[&](const std::string& filename, real t) {
				printf( "t = %e\n", t);
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

				constexpr
				int nshapes = 1;
				int shapesize[1] = {NVERTEX};
				int shapetype[1] = {DB_ZONETYPE_HEX};
				int shapecnt[1] = {nzones};
				const char* coord_names[NDIM] = {"x", "y", "z"};

#ifndef	__MIC__
				auto olist = DBMakeOptlist(1);
				double time = double(t);
				int ndim = 3;
				DBAddOption(olist, DBOPT_CYCLE, &cycle);
				DBAddOption(olist, DBOPT_DTIME, &time);
				DBAddOption(olist, DBOPT_NSPACE, &ndim );
				DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Euler Mesh", DB_PDB);
				assert(db);
				DBPutZonelist2(db, "zones", nzones, int(NDIM), zone_nodes.data(), nzones * NVERTEX, 0, 0, 0, shapetype, shapesize,
						shapecnt, nshapes, olist);
				DBPutUcdmesh(db, "mesh", int(NDIM), const_cast<char**>(coord_names), node_coords.data(), nnodes, nzones, "zones", nullptr, DB_DOUBLE,
						olist);
				const char* analytic_names[] = {"rho_a", "egas_a", "sx_a", "sy_a", "sz_a", "tau_a"};
				DBFreeOptlist(olist);
				for (int field = 0; field != NF + NGF + NPF + NRF; ++field) {
					auto olist = DBMakeOptlist(1);
					double time = double(t);
					int istrue = 1;
					int isfalse = 0;
					DBAddOption(olist, DBOPT_CYCLE, &cycle);
					DBAddOption(olist, DBOPT_DTIME, &time);
					DBAddOption(olist, DBOPT_NSPACE, &ndim );
					if( field == rho_i || field == sx_i || field == sy_i || field == sz_i || field == spc_ac_i || field == spc_ae_i || field == spc_dc_i || field == spc_de_i || field == spc_vac_i ) {
						DBAddOption(olist, DBOPT_CONSERVED, &istrue);
					} else {
						DBAddOption(olist, DBOPT_CONSERVED, &isfalse );
					}
					if( field < NF ) {
						DBAddOption(olist, DBOPT_EXTENSIVE, &istrue);
					} else {
						DBAddOption(olist, DBOPT_EXTENSIVE, &isfalse);
					}
					DBAddOption(olist, DBOPT_EXTENSIVE, &istrue);
					DBPutUcdvar1(db, field_names[field], "mesh", const_cast<void*>(reinterpret_cast<const void*>(olists.data[field].data())), nzones, nullptr, 0, DB_DOUBLE, DB_ZONECENT,
							olist);
					if( analytic && field < 6) {
						DBPutUcdvar1(db, analytic_names[field], "mesh", const_cast<void*>(reinterpret_cast<const void*>(olists.analytic[field].data())), nzones, nullptr, 0, DB_DOUBLE, DB_ZONECENT,
								olist);
					}
					DBFreeOptlist(olist);
#ifdef RHO_ONLY
					break;
#endif
				}
				DBClose(db);
#endif
			}, _filename, _t).join();
#endif
}

std::size_t grid::load(FILE* fp) {
	std::size_t cnt = 0;
	auto foo = std::fread;
	{
		static hpx::mutex mtx;
		std::lock_guard<hpx::mutex> lock(mtx);
		cnt += foo(&scaling_factor, sizeof(real), 1, fp) * sizeof(real);
		cnt += foo(&max_level, sizeof(integer), 1, fp) * sizeof(integer);
		cnt += foo(&Acons, sizeof(real), 1, fp) * sizeof(real);
		cnt += foo(&Bcons, sizeof(integer), 1, fp) * sizeof(integer);
	}
	cnt += foo(&is_leaf, sizeof(bool), 1, fp) * sizeof(bool);
	cnt += foo(&is_root, sizeof(bool), 1, fp) * sizeof(bool);

	allocate();

	for (integer f = 0; f != NF; ++f) {
		for (integer i = H_BW; i < H_NX - H_BW; ++i) {
			for (integer j = H_BW; j < H_NX - H_BW; ++j) {
				const integer iii = hindex(i, j, H_BW);
				cnt += foo(&(U[f][iii]), sizeof(real), INX, fp) * sizeof(real);
			}
		}
	}
	for (integer i = 0; i < G_NX; ++i) {
		for (integer j = 0; j < G_NX; ++j) {
			for (integer k = 0; k < G_NX; ++k) {
				const integer iii = gindex(i, j, k);
				real g[NGF];
				cnt += foo(g, sizeof(real), NGF, fp) * sizeof(real);
				for (integer f = 0; f != NGF; ++f) {
					G[iii][f] = g[f];
				}
			}
		}
	}
	cnt += foo(U_out.data(), sizeof(real), U_out.size(), fp) * sizeof(real);
#ifdef RADIATION
	cnt += rad_grid_ptr->load(fp);
#endif
	set_coordinates();
	return cnt;
}

std::size_t grid::save(FILE* fp) const {
	std::size_t cnt = 0;
	auto foo = std::fwrite;
	{
		static hpx::mutex mtx;
		std::lock_guard<hpx::mutex> lock(mtx);
		cnt += foo(&scaling_factor, sizeof(real), 1, fp) * sizeof(real);
		cnt += foo(&max_level, sizeof(integer), 1, fp) * sizeof(integer);
		cnt += foo(&Acons, sizeof(real), 1, fp) * sizeof(real);
		cnt += foo(&Bcons, sizeof(integer), 1, fp) * sizeof(integer);
	}
	cnt += foo(&is_leaf, sizeof(bool), 1, fp) * sizeof(bool);
	cnt += foo(&is_root, sizeof(bool), 1, fp) * sizeof(bool);
	for (integer f = 0; f != NF; ++f) {
		for (integer i = H_BW; i < H_NX - H_BW; ++i) {
			for (integer j = H_BW; j < H_NX - H_BW; ++j) {
				const integer iii = hindex(i, j, H_BW);
				cnt += foo(&U[f][iii], sizeof(real), INX, fp) * sizeof(real);
			}
		}
	}
	for (integer i = 0; i < G_NX; ++i) {
		for (integer j = 0; j < G_NX; ++j) {
			for (integer k = 0; k < G_NX; ++k) {
				const integer iii = gindex(i, j, k);
				const auto d = G[iii][0];
				cnt += foo(&d, sizeof(real), NGF, fp) * sizeof(real);
			}
		}
	}
	cnt += foo(U_out.data(), sizeof(real), U_out.size(), fp) * sizeof(real);
#ifdef RADIATION
	cnt += rad_grid_ptr->save(fp);
#endif
	return cnt;
}

