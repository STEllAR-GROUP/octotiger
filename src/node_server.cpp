/*
 * node_server.cpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"
#include "problem.hpp"
#include "future.hpp"
#include <streambuf>
#include <fstream>
#include <iostream>
#include "options.hpp"

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(hpx::components::managed_component<node_server>, node_server);


bool node_server::static_initialized(false);
std::atomic<integer> node_server::static_initializing(0);

bool node_server::hydro_on = true;
bool node_server::gravity_on = true;

void node_server::set_gravity(bool b) {
	gravity_on = b;
}

void node_server::set_hydro(bool b) {
	hydro_on = b;
}

real node_server::find_omega() const {
	const auto this_com = grid_ptr->center_of_mass();
//	printf( "%e %e %e\n", this_com[0], this_com[1], this_com[2]);
	auto d = find_omega_part(this_com);
//	printf( "%e %e\n", d.first, d.second);
	return d.first / d.second;
}


integer child_index_to_quadrant_index(integer ci, integer dim) {
	integer index;
	if (dim == XDIM) {
		index = ci >> 1;
	} else if (dim == ZDIM) {
		index = ci & 0x3;
	} else {
		index = (ci & 1) | ((ci >> 1) & 0x2);
	}
	return index;
}

node_server::node_server(const node_location& _my_location, integer _step_num, bool _is_refined, real _current_time,
		real _rotational_time, const std::array<integer, NCHILD>& _child_d, grid _grid,
		const std::vector<hpx::id_type>& _c) {
	my_location = _my_location;
	initialize(_current_time, _rotational_time);
	is_refined = _is_refined;
	step_num = _step_num;
	current_time = _current_time;
	rotational_time = _rotational_time;
	grid test;
	grid_ptr = std::make_shared < grid > (std::move(_grid));
	if (is_refined) {
		children.resize(NCHILD);
		std::copy(_c.begin(), _c.end(), children.begin());
	}
	child_descendant_count = _child_d;
}

bool node_server::child_is_on_face(integer ci, integer face) {
	return (((ci >> (face / 2)) & 1) == (face & 1));
}


void node_server::static_initialize() {
	if (!static_initialized) {
		bool test = static_initializing++;
		if (!test) {
			static_initialized = true;
		}
		while (!static_initialized) {
			hpx::this_thread::yield();
		}
	}
}

void node_server::initialize(real t, real rt) {
	step_num = 0;
	refinement_flag = 0;
	static_initialize();
	global_timestep_channel = std::make_shared<channel<real>>();
	local_timestep_channel = std::make_shared<channel<real>>();
	is_refined = false;
	siblings.resize(NFACE);
	neighbors.resize(geo::direction::count());
	nieces.resize(NFACE);
	aunts.resize(NFACE);
#ifdef USE_SPHERICAL
	for (auto& dir : geo::direction::full_set()) {
		neighbor_gravity_channels[dir] = std::make_shared<channel<std::vector<multipole_type>>>();
	}
	for (auto& ci : geo::octant::full_set()) {
		child_gravity_channels[ci] = std::make_shared<channel<std::vector<multipole_type>>>();
	}
	parent_gravity_channel = std::make_shared<channel<std::vector<expansion_type>>>();
#else
	for (auto& dir : geo::direction::full_set()) {
		neighbor_gravity_channels[dir] = std::make_shared<channel<std::pair<std::vector<real>, bool>> >();
	}
	for (auto& ci : geo::octant::full_set()) {
		child_gravity_channels[ci] = std::make_shared<channel<multipole_pass_type>>();
	}
	parent_gravity_channel = std::make_shared<channel<expansion_pass_type>>();
#endif
	for (auto& ci : geo::octant::full_set()) {
		child_hydro_channels[ci] = std::make_shared<channel<std::vector<real>>>();
	}
	for (auto& dir : geo::direction::full_set()) {
		sibling_hydro_channels[dir] = std::make_shared<channel<std::vector<real>> >();
	}
	for (auto& face : geo::face::full_set()) {
		for (integer i = 0; i != 4; ++i) {
			niece_hydro_channels[face][i] = std::make_shared<channel<std::vector<real>> >();
		}
	}
	current_time = t;
	rotational_time = rt;
	dx = TWO * grid::get_scaling_factor() / real(INX << my_location.level());
	for (auto& d : geo::dimension::full_set()) {
		xmin[d] = grid::get_scaling_factor() * my_location.x_location(d);
	}
	if (current_time == ZERO) {
		const auto p = get_problem();
		grid_ptr = std::make_shared < grid > (p, dx, xmin);
	} else {
		grid_ptr = std::make_shared < grid > (dx, xmin);
	}
	if (my_location.level() == 0) {
		grid_ptr->set_root();
	}
}
node_server::~node_server() {
}

node_server::node_server() {
	initialize(ZERO, ZERO);
}

node_server::node_server(const node_location& loc, const node_client& parent_id, real t, real rt) :
		my_location(loc), parent(parent_id) {
	initialize(t, rt);
}


void node_server::compute_fmm(gsolve_type type, bool energy_account) {

#ifdef USE_SPHERICAL
	fmm::set_dx(dx);
	std::list<hpx::future<void>> child_futs;
	std::list<hpx::future<void>> neighbor_futs;
	hpx::future<void> parent_fut;
	if (energy_account) {
		grid_ptr->egas_to_etot();
	}
	if (!is_refined) {
		for (integer i = 0; i != INX; ++i) {
			for (integer j = 0; j != INX; ++j) {
				for (integer k = 0; k != INX; ++k) {
					fmm::set_source(grid_ptr->get_source(i, j, k), i, j, k);
				}
			}
		}
	} else {
		for (auto& octant : geo::octant::full_set()) {
			fmm::set_multipoles(child_gravity_channels[octant]->get_future().get(), octant);
		}
	}
	const auto myci = my_location.get_child_index();
	const bool is_root = my_location.level() == 0;
	parent_fut = !is_root ? parent.send_gravity_multipoles(fmm::M2M(), myci) : hpx::make_ready_future();

	std::array<integer, NDIM> lb, ub;
	for (auto& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			get_boundary_size(lb, ub, dir, INNER, G_BW);
			for (integer d = 0; d != NDIM; ++d) {
				lb[d] -= G_BW;
				ub[d] -= G_BW;
			}
			neighbor_futs.push_back(neighbors[dir].send_gravity_boundary(fmm::get_multipoles(lb, ub), dir.flip()));
		}
	}

	fmm::self_M2L(is_root, !is_refined);

	for (auto& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			auto tmp = neighbor_gravity_channels[dir]->get_future().get();
			get_boundary_size(lb, ub, dir, OUTER, G_BW);
			for (integer d = 0; d != NDIM; ++d) {
				lb[d] -= G_BW;
				ub[d] -= G_BW;
			}
			fmm::other_M2L(std::move(tmp), lb, ub, !is_refined);
		}
	}
	parent_fut.get();
	if (!is_root) {
		fmm::L2L(parent_gravity_channel->get_future().get());
	}
	if (is_refined) {
		for (auto& ci : geo::octant::full_set()) {
			child_futs.push_back(children[ci].send_gravity_expansions(fmm::get_expansions(ci)));
		}
	}

	if (energy_account) {
		grid_ptr->etot_to_egas();
	}
	for (auto&& fut : child_futs) {
		GET(fut);
	}
	for (auto&& fut : neighbor_futs) {
		GET(fut);
	}

	for (integer i = 0; i != INX; ++i) {
		for (integer j = 0; j != INX; ++j) {
			for (integer k = 0; k != INX; ++k) {
				const auto ff = fmm::four_force(i, j, k);
				grid_ptr->set_4force(i, j, k, ff);
			}
		}
	}
#else
	std::list<hpx::future<void>> child_futs;
	std::list<hpx::future<void>> neighbor_futs;
	hpx::future<void> parent_fut;

	if (energy_account) {
		grid_ptr->egas_to_etot();
	}
	multipole_pass_type m_in, m_out;
	m_out.first.resize(INX * INX * INX);
	m_out.second.resize(INX * INX * INX);
	if (is_refined) {
		for (auto& ci : geo::octant::full_set()) {
			const integer x0 = ci.get_side(XDIM) * INX / 2;
			const integer y0 = ci.get_side(YDIM) * INX / 2;
			const integer z0 = ci.get_side(ZDIM) * INX / 2;
			m_in = GET(child_gravity_channels[ci]->get_future());
			for (integer i = 0; i != INX / 2; ++i) {
				for (integer j = 0; j != INX / 2; ++j) {
					for (integer k = 0; k != INX / 2; ++k) {
						const integer ii = i * INX * INX / 4 + j * INX / 2 + k;
						const integer io = (i + x0) * INX * INX + (j + y0) * INX + k + z0;
						m_out.first[io] = m_in.first[ii];
						m_out.second[io] = m_in.second[ii];
					}
				}
			}
		}
		m_out = grid_ptr->compute_multipoles(type, &m_out);
	} else {
		m_out = grid_ptr->compute_multipoles(type);
	}

	if (my_location.level() != 0) {
		parent_fut = parent.send_gravity_multipoles(std::move(m_out), my_location.get_child_index());
	} else {
		parent_fut = hpx::make_ready_future();
	}

	grid_ptr->compute_interactions(type);

	std::array<bool, geo::direction::count()> is_monopole;
	for (auto& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			auto ndir = dir.flip();
			const bool monopole = !is_refined;
			assert(neighbors[dir].get_gid() != me.get_gid());
			neighbor_futs.push_back(neighbors[dir].send_gravity_boundary(get_gravity_boundary(dir), ndir, monopole));
		}
	}
	for (auto& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			auto tmp = GET(this->neighbor_gravity_channels[dir]->get_future());
			is_monopole[dir] = tmp.second;
			this->set_gravity_boundary(std::move(tmp.first), dir, tmp.second);
		}
	}
	for (auto& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			this->grid_ptr->compute_boundary_interactions(type, dir, is_monopole[dir]);
		}
	}
	GET(parent_fut);

	expansion_pass_type l_in;
	if (my_location.level() != 0) {

		/*****PROBLEM FUTURE******/
		l_in = GET(parent_gravity_channel->get_future());
	}
	const expansion_pass_type ltmp = grid_ptr->compute_expansions(type, my_location.level() == 0 ? nullptr : &l_in);
	if (is_refined) {
		for (auto& ci : geo::octant::full_set()) {
			expansion_pass_type l_out;
			l_out.first.resize(INX * INX * INX / NCHILD);
			if (type == RHO) {
				l_out.second.resize(INX * INX * INX / NCHILD);
			}
			const integer x0 = ci.get_side(XDIM) * INX / 2;
			const integer y0 = ci.get_side(YDIM) * INX / 2;
			const integer z0 = ci.get_side(ZDIM) * INX / 2;
			for (integer i = 0; i != INX / 2; ++i) {
				for (integer j = 0; j != INX / 2; ++j) {
					for (integer k = 0; k != INX / 2; ++k) {
						const integer io = i * INX * INX / 4 + j * INX / 2 + k;
						const integer ii = (i + x0) * INX * INX + (j + y0) * INX + k + z0;
						auto t = ltmp.first[ii];
						l_out.first[io] = t;
						if (type == RHO) {
							l_out.second[io] = ltmp.second[ii];
						}
					}
				}
			}
			child_futs.push_back(children[ci].send_gravity_expansions(std::move(l_out)));
		}
	}

	if (energy_account) {
		grid_ptr->etot_to_egas();
	}

	for (auto&& fut : child_futs) {
		GET(fut);
	}
	for (auto&& fut : neighbor_futs) {
		GET(fut);
	}
#endif
}
