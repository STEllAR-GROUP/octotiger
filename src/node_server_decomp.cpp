/*
 * node_server_decomp.cpp
 *
 *  Created on: Jun 26, 2015
 *      Author: dmarce1
 */

/*
 *
 * TODO: Sx, Sy, Sz boundaries!!!
 */
#include "node_server.hpp"
#include "future.hpp"


hpx::future<void> node_server::exchange_flux_corrections() {
	const geo::octant ci = my_location.get_child_index();
	auto ptr_futs = std::make_shared<std::list<hpx::future<void>>>();
	for (auto& f : geo::face::full_set()) {
		const auto face_dim = f.get_dimension();
		auto& this_aunt = aunts[f];
		if (!this_aunt.empty()) {
			std::array<integer, NDIM> lb, ub;
			lb[XDIM] = lb[YDIM] = lb[ZDIM] = H_BW;
			ub[XDIM] = ub[YDIM] = ub[ZDIM] = H_NX - H_BW;
			if (f.get_side() == geo::MINUS) {
				lb[face_dim] = H_BW;
			} else {
				lb[face_dim] = H_NX - H_BW;
			}
			ub[face_dim] = lb[face_dim] + 1;
			auto data = grid_ptr->get_flux_restrict(lb, ub, face_dim);
			ptr_futs->push_back(this_aunt.send_hydro_flux_correct(std::move(data), f.flip(), ci));
		}
	}
	return hpx::async([=]() {
		for (auto& f : geo::face::full_set()) {
			if (nieces[f].size()) {
				const auto face_dim = f.get_dimension();
				for (auto& quadrant : geo::quadrant::full_set()) {
					std::array<integer, NDIM> lb, ub;
					switch (face_dim) {
						case XDIM:
						lb[XDIM] = f.get_side() == geo::MINUS ? H_BW : H_NX - H_BW;
						lb[YDIM] = H_BW + quadrant.get_side(0) * (INX / 2);
						lb[ZDIM] = H_BW + quadrant.get_side(1) * (INX / 2);
						ub[XDIM] = lb[XDIM] + 1;
						ub[YDIM] = lb[YDIM] + (INX / 2);
						ub[ZDIM] = lb[ZDIM] + (INX / 2);
						break;
						case YDIM:
						lb[XDIM] = H_BW + quadrant.get_side(0) * (INX / 2);
						lb[YDIM] = f.get_side() == geo::MINUS ? H_BW : H_NX - H_BW;
						lb[ZDIM] = H_BW + quadrant.get_side(1) * (INX / 2);
						ub[XDIM] = lb[XDIM] + (INX / 2);
						ub[YDIM] = lb[YDIM] + 1;
						ub[ZDIM] = lb[ZDIM] + (INX / 2);
						break;
						case ZDIM:
						lb[XDIM] = H_BW + quadrant.get_side(0) * (INX / 2);
						lb[YDIM] = H_BW + quadrant.get_side(1) * (INX / 2);
						lb[ZDIM] = f.get_side() == geo::MINUS ? H_BW : H_NX - H_BW;
						ub[XDIM] = lb[XDIM] + (INX / 2);
						ub[YDIM] = lb[YDIM] + (INX / 2);
						ub[ZDIM] = lb[ZDIM] + 1;
						break;
					}
					std::vector<real> data = GET(niece_hydro_channels[f][quadrant]->get_future());
					grid_ptr->set_flux_restrict(data, lb, ub, face_dim);
				}
			}
		}
		for (auto&& f : *ptr_futs) {
			GET(f);
		}
	});
}

void node_server::exchange_interlevel_hydro_data() {

	hpx::future<void> fut;
	std::vector<real> outflow(NF, ZERO);
	if (is_refined) {
		for (auto& ci : geo::octant::full_set()) {
			std::vector<real> data = GET(child_hydro_channels[ci]->get_future());
			grid_ptr->set_restrict(data, ci);
			integer fi = 0;
			for (auto i = data.end() - NF; i != data.end(); ++i) {
				outflow[fi] += *i;
				++fi;
			}
		}
		grid_ptr->set_outflows(std::move(outflow));
	}
	if (my_location.level() > 0) {
		std::vector<real> data = grid_ptr->get_restrict();
		integer ci = my_location.get_child_index();
		fut = parent.send_hydro_children(std::move(data), ci);
	} else {
		fut = hpx::make_ready_future();
	}
	GET(fut);
}

void node_server::collect_hydro_boundaries() {
	std::list<hpx::future<void>> futs;
	for (auto& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			auto bdata = get_hydro_boundary(dir);
			futs.push_back(neighbors[dir].send_hydro_boundary(std::move(bdata), dir.flip()));
		}
	}

	for (auto& dir : geo::direction::full_set()) {
		if (!(neighbors[dir].empty() && my_location.level() == 0)) {
			std::vector<real> bdata;
			bdata = GET(sibling_hydro_channels[dir]->get_future());
			set_hydro_boundary(bdata, dir);
		}
	}

	for (auto& face : geo::face::full_set()) {
		if (my_location.is_physical_boundary(face)) {
			grid_ptr->set_physical_boundaries(face);
		}
	}

	if (is_refined) {
		for (auto& ci : geo::octant::full_set()) {
			const auto& flags = amr_flags[ci];
			for (auto& dir : geo::direction::full_set()) {
				if (flags[dir]) {
					std::array<integer, NDIM> lb, ub;
					std::vector<real> data;
					get_boundary_size(lb, ub, dir, OUTER, H_BW);
					for (integer dim = 0; dim != NDIM; ++dim) {
						lb[dim] = ((lb[dim] - H_BW)) + 2 * H_BW + ci.get_side(dim) * (INX);
						ub[dim] = ((ub[dim] - H_BW)) + 2 * H_BW + ci.get_side(dim) * (INX);
					}
					data = grid_ptr->get_prolong(lb, ub);
					futs.push_back(children[ci].send_hydro_boundary(std::move(data), dir));
				}
			}
		}
	}

	for (auto&& fut : futs) {
		GET(fut);
	}
}

integer node_server::get_boundary_size(std::array<integer, NDIM>& lb, std::array<integer, NDIM>& ub,
		const geo::direction& dir, const geo::side& side, integer bw) const {
	integer hsize, size;
	size = 0;
	const integer nx = 2 * bw + INX;
	const integer off = (side == OUTER) ? bw : 0;
	hsize = 1;
	for (auto& d : geo::dimension::full_set()) {
		auto this_dir = dir[d];
		if (this_dir == 0) {
			lb[d] = bw;
			ub[d] = nx - bw;
		} else if (this_dir < 0) {
			lb[d] = bw - off;
			ub[d] = 2 * bw - off;
		} else /*if (this_dir > 0) */{
			lb[d] = nx - 2 * bw + off;
			ub[d] = nx - bw + off;
		}
		const integer width = ub[d] - lb[d];
		hsize *= width;
	}
	size += hsize;
	return size;
}

std::vector<real> node_server::get_hydro_boundary(const geo::direction& dir) {

	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	const integer size = NF * get_boundary_size(lb, ub, dir, INNER, H_BW);
	data.resize(size);
	integer iter = 0;

	for (integer field = 0; field != NF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					data[iter] = grid_ptr->hydro_value(field, i, j, k);
					++iter;
				}
			}
		}
	}

	return data;
}

std::vector<real> node_server::get_gravity_boundary(const geo::direction& dir) {

	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size = get_boundary_size(lb, ub, dir, INNER, G_BW);
	if (is_refined) {
		size *= 20 + 3;
	} else {
		size *= 1 + 3;
	}
	data.resize(size);
	integer iter = 0;

	for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
			for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
				const auto& m = grid_ptr->multipole_value(0, i, j, k);
				const auto& com = grid_ptr->center_of_mass_value(i, j, k);
				const integer top = is_refined ? 20 : 1;
				for (integer l = 0; l < top; ++l) {
					data[iter] = m.ptr()[l];
					++iter;
				}
				for (integer d = 0; d != NDIM; ++d) {
					data[iter] = com[d];
					++iter;
				}
			}
		}
	}

	return data;
}

void node_server::set_gravity_boundary(const std::vector<real>& data, const geo::direction& dir, bool monopole) {
	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, dir, OUTER, G_BW);
	integer iter = 0;

	for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
			for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
				auto& m = grid_ptr->multipole_value(0, i, j, k);
				auto& com = grid_ptr->center_of_mass_value(i, j, k);
				const integer top = monopole ? 1 : 20;
				for (integer l = 0; l < top; ++l) {
					m.ptr()[l] = data[iter];
					++iter;
				}
				for (integer l = top; l < 20; ++l) {
					m.ptr()[l] = ZERO;
				}
				for (integer d = 0; d != NDIM; ++d) {
					com[d] = data[iter];
					++iter;
				}
			}
		}
	}
}

void node_server::set_hydro_boundary(const std::vector<real>& data, const geo::direction& dir) {
	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, dir, OUTER, H_BW);
	integer iter = 0;

	for (integer field = 0; field != NF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					grid_ptr->hydro_value(field, i, j, k) = data[iter];
					++iter;
				}
			}
		}
	}
}

void node_server::recv_gravity_multipoles(multipole_pass_type&& v, const geo::octant& ci) {
	child_gravity_channels[ci]->set_value(std::move(v));
}

void node_server::recv_gravity_expansions(expansion_pass_type&& v) {
	parent_gravity_channel->set_value(std::move(v));
}

void node_server::recv_hydro_boundary(std::vector<real>&& bdata, const geo::direction& dir) {
	sibling_hydro_channels[dir]->set_value(std::move(bdata));
}

void node_server::recv_gravity_boundary(std::vector<real>&& bdata, const geo::direction& dir, bool monopole) {
	neighbor_gravity_channels[dir]->set_value(std::make_pair(std::move(bdata), monopole));
}

