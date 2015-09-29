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
#include "future.hpp"
#include "node_server.hpp"

void node_server::collect_hydro_boundaries(integer rk) {
	std::vector<hpx::future<void>> sibling_futs;
    sibling_futs.reserve(NDIM*NDIM);
	std::array<bool, NFACE> is_physical;
	for (integer dim = 0; dim != NDIM; ++dim) {
		for (integer face = 2 * dim; face != 2 * dim + 2; ++face) {
			is_physical[face] = my_location.is_physical_boundary(face);
			if (!is_physical[face]) {
				auto bdata = get_hydro_boundary(face);
				sibling_futs.push_back(siblings[face].send_hydro_boundary(std::move(bdata), rk, face ^ 1));
			}
		}
		for (integer face = 2 * dim; face != 2 * dim + 2; ++face) {
			if (is_physical[face]) {
				grid_ptr->set_physical_boundaries(face);
			} else {
				const std::vector<real> bdata = GET(sibling_hydro_channels[rk][face]->get_future());
				set_hydro_boundary(bdata, face);
			}
		}
	}
    hpx::wait_all(sibling_futs);
}

integer node_server::get_boundary_size(std::array<integer, NDIM>& lb, std::array<integer, NDIM>& ub, integer face,
		integer side) const {
	integer hsize, size, offset;
	size = 0;
	offset = (side == OUTER) ? HBW : 0;
	hsize = 1;
	for (integer d = 0; d != NDIM; ++d) {
		const integer nx = INX + 2 * HBW;
		if (d < face / 2) {
			lb[d] = 0;
			ub[d] = nx;
		} else if (d > face / 2) {
			lb[d] = HBW;
			ub[d] = nx - HBW;
		} else if (face % 2 == 0) {
			lb[d] = HBW - offset;
			ub[d] = 2 * HBW - offset;
		} else {
			lb[d] = nx - 2 * HBW + offset;
			ub[d] = nx - HBW + offset;
		}
		const integer width = ub[d] - lb[d];
		hsize *= width;
	}
	size += hsize;
	return size;
}

std::vector<real> node_server::get_hydro_boundary(integer face) {

	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	const integer size = (NF + NDIM)* get_boundary_size(lb, ub, face, INNER);
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

std::vector<real> node_server::get_gravity_boundary(integer face) {

	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size = get_boundary_size(lb, ub, face, INNER);
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

void node_server::set_gravity_boundary(const std::vector<real>& data, integer face) {
	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, face, OUTER);
	integer iter = 0;

	for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
		for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
			for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
				auto& m = grid_ptr->multipole_value(0, i, j, k);
				auto& com = grid_ptr->center_of_mass_value(i, j, k);
				const integer top = is_refined ? 20 : 1;
				for (integer l = 0; l < top; ++l) {
					m.ptr()[l] = data[iter];
					++iter;
				}
				for (integer d = 0; d != NDIM; ++d) {
					com[d] = data[iter];
					++iter;
				}
			}
		}
	}
}

void node_server::set_hydro_boundary(const std::vector<real>& data, integer face) {
	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, face, OUTER);
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

void node_server::recv_gravity_multipoles(multipole_pass_type&& v, integer ci, integer chan) {
	child_gravity_channels[chan][ci]->set_value(std::move(v));
}

void node_server::recv_gravity_expansions(expansion_pass_type&& v,integer chan ){
	parent_gravity_channel[chan]->set_value(std::move(v));
}

void node_server::recv_hydro_boundary(std::vector<real>&& bdata, integer rk, integer face ) {
	sibling_hydro_channels[rk][face]->set_value(std::move(bdata));
}


void node_server::recv_gravity_boundary(std::vector<real>&& bdata, integer face,integer chan ) {
	sibling_gravity_channels[chan][face]->set_value(std::move(bdata));
}

