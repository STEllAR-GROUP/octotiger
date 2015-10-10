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

hpx::future<void> node_server::exchange_flux_corrections() {
	const integer ci = my_location.get_child_index();
	auto ptr_futs = std::make_shared<std::list<hpx::future<void>>>();
	for (integer f = 0; f != NFACE; ++f) {
		const integer face_dim = f / 2;
		auto& this_aunt = aunts[f];
		if (!this_aunt.empty()) {
			std::array<integer, NDIM> lb, ub;
			lb[XDIM] = lb[YDIM] = lb[ZDIM] = HBW;
			ub[XDIM] = ub[YDIM] = ub[ZDIM] = HNX - HBW;
			if (f % 2 == 0) {
				lb[face_dim] = HBW;
			} else {
				lb[face_dim] = HNX - HBW;
			}
			ub[face_dim] = lb[face_dim] + 1;
			auto data = grid_ptr->get_flux_restrict(lb, ub, face_dim);
			ptr_futs->push_back(this_aunt.send_hydro_flux_correct(std::move(data), f ^ 1, ci));
		}
	}
	return hpx::async([=]() {
		for (integer f = 0; f != NFACE; ++f) {
			if (nieces[f].size()) {
				const integer face_dim = f / 2;
				for (integer quadrant = 0; quadrant != 4; ++quadrant) {
					std::array<integer, NDIM> lb, ub;
					switch (face_dim) {
						case XDIM:
						lb[XDIM] = f % 2 == 0 ? HBW : HNX - HBW;
						lb[YDIM] = HBW + ((quadrant >> 0) & 1) * (INX / 2);
						lb[ZDIM] = HBW + ((quadrant >> 1) & 1) * (INX / 2);
						ub[XDIM] = lb[XDIM] + 1;
						ub[YDIM] = lb[YDIM] + (INX / 2);
						ub[ZDIM] = lb[ZDIM] + (INX / 2);
						break;
						case YDIM:
						lb[XDIM] = HBW + ((quadrant >> 0) & 1) * (INX / 2);
						lb[YDIM] = f % 2 == 0 ? HBW : HNX - HBW;
						lb[ZDIM] = HBW + ((quadrant >> 1) & 1) * (INX / 2);
						ub[XDIM] = lb[XDIM] + (INX / 2);
						ub[YDIM] = lb[YDIM] + 1;
						ub[ZDIM] = lb[ZDIM] + (INX / 2);
						break;
						case ZDIM:
						lb[XDIM] = HBW + ((quadrant >> 0) & 1) * (INX / 2);
						lb[YDIM] = HBW + ((quadrant >> 1) & 1) * (INX / 2);
						lb[ZDIM] = f % 2 == 0 ? HBW : HNX - HBW;
						ub[XDIM] = lb[XDIM] + (INX / 2);
						ub[YDIM] = lb[YDIM] + (INX / 2);
						ub[ZDIM] = lb[ZDIM] + 1;
						break;
					}
					std::vector<real> data = niece_hydro_channels[f][quadrant]->get();
					grid_ptr->set_flux_restrict(data, lb, ub, face_dim);
				}
			}
		}
		for (auto&& f : *ptr_futs) {
			f.get();
		}
	});
}

void node_server::exchange_interlevel_hydro_data() {

	if (my_location.level() > 0) {
		std::vector<real> data = restricted_grid();
		integer ci = my_location.get_child_index();
		parent.send_hydro_children(std::move(data), ci);
	}
	std::list<hpx::future<void>> futs;
	std::vector<real> outflow(NF,ZERO);
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			std::vector<real> data = child_hydro_channels[ci]->get();
			load_from_restricted_child(data, ci);
			integer fi = 0;
			for( auto i = data.end() - NF; i != data.end(); ++i) {
				outflow[fi] += *i;
			    ++fi;
			}
		}
	}
	for (auto&& f : futs ) {
		f.get();
	}
}

std::list<hpx::future<void>> node_server::set_nieces_amr(integer f) const {
	std::list<hpx::future<void>> futs;
	if (nieces[f].size()) {
		integer nindex = 0;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			if (child_is_on_face(ci, f)) {
				std::array<integer, NDIM> lb, ub;
				std::vector<real> data;
				get_boundary_size(lb, ub, f, INNER);
				for (integer dim = 0; dim != NDIM; ++dim) {
					lb[dim] = ((lb[dim] - HBW) / 2) + HBW + ((ci >> dim) & 1) * (INX / 2);
					ub[dim] = ((ub[dim] - HBW) / 2) + HBW + ((ci >> dim) & 1) * (INX / 2);
				}
				data = grid_ptr->get_prolong(lb, ub);
				assert(!nieces[f][nindex].empty());
				futs.push_back(nieces[f][nindex].send_hydro_boundary(std::move(data), f ^ 1));
				++nindex;
			}
		}
	}
	return futs;
}

void node_server::collect_hydro_boundaries() {
	std::list<hpx::future<void>> sibling_futs;
	for (integer dim = 0; dim != NDIM; ++dim) {
		for (integer face = 2 * dim; face != 2 * dim + 2; ++face) {
			if (!siblings[face].empty()) {
				auto bdata = get_hydro_boundary(face);
				sibling_futs.push_back(siblings[face].send_hydro_boundary(std::move(bdata), face ^ 1));
			}
			auto tmp = set_nieces_amr(face);
			sibling_futs.splice(sibling_futs.end(), std::move(tmp));
		}
		for (integer face = 2 * dim; face != 2 * dim + 2; ++face) {
			if (my_location.is_physical_boundary(face)) {
				grid_ptr->set_physical_boundaries(face);
			} else {
				const std::vector<real> bdata = sibling_hydro_channels[face]->get();
				set_hydro_boundary(bdata, face);
			}
		}
	}
	for (auto i = sibling_futs.begin(); i != sibling_futs.end(); ++i) {
		i->get();
	}
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
	const integer size = NF * get_boundary_size(lb, ub, face, INNER);
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

void node_server::set_gravity_boundary(const std::vector<real>& data, integer face, bool monopole) {
	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, face, OUTER);
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

void node_server::recv_gravity_multipoles(multipole_pass_type&& v, integer ci) {
	child_gravity_channels[ci]->set_value(std::move(v));
}

void node_server::recv_gravity_expansions(expansion_pass_type&& v) {
	parent_gravity_channel->set_value(std::move(v));
}

void node_server::recv_hydro_boundary(std::vector<real>&& bdata, integer face ) {
	sibling_hydro_channels[face]->set_value(std::move(bdata));
}

void node_server::recv_gravity_boundary(std::vector<real>&& bdata, integer face, bool monopole ) {
	sibling_gravity_channels[face]->set_value(std::make_pair(std::move(bdata), monopole));
}

