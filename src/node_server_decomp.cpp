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

hpx::future<void> node_server::collect_hydro_boundaries() {
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
		fut.get();
	}
	return hpx::make_ready_future();
	/*	auto allfuts = hpx::when_all(futs.begin(), futs.end());
	 return allfuts.then([](hpx::future<std::vector<hpx::future<void>>>&& futs){
	 return;
	 });*/
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
	return grid_ptr->get_gravity_boundary(dir);
}

void node_server::set_gravity_boundary(const std::vector<real>& data, const geo::direction& dir, bool monopole) {
	grid_ptr->set_gravity_boundary(data, dir, monopole);
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

