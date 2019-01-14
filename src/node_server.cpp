/*
 * node_server.cpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#include "octotiger/node_server.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/future.hpp"
#include "octotiger/node_registry.hpp"
#include "octotiger/options.hpp"
#include "octotiger/problem.hpp"
#include "octotiger/taylor.hpp"
#include "octotiger/util.hpp"

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

#include <array>
#include <streambuf>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#if !defined(_MSC_VER)
#include <unistd.h>
#endif

HPX_REGISTER_COMPONENT(hpx::components::managed_component<node_server>, node_server);

bool node_server::static_initialized(false);
std::atomic<integer> node_server::static_initializing(0);

real node_server::get_rotation_count() const {
	if (opts().problem == DWD) {
		return rotational_time / (2.0 * M_PI);
	} else {
		return current_time;
	}
}

future<void> node_server::exchange_flux_corrections() {
	const geo::octant ci = my_location.get_child_index();
	constexpr auto full_set = geo::face::full_set();
	for (auto& f : full_set) {
		const auto face_dim = f.get_dimension();
		auto const& this_aunt = aunts[f];
		if (!this_aunt.empty()) {
			std::array<integer, NDIM> lb, ub;
			lb[XDIM] = lb[YDIM] = lb[ZDIM] = 0;
			ub[XDIM] = ub[YDIM] = ub[ZDIM] = INX;
			if (f.get_side() == geo::MINUS) {
				lb[face_dim] = 0;
			} else {
				lb[face_dim] = INX;
			}
			ub[face_dim] = lb[face_dim] + 1;
			auto data = grid_ptr->get_flux_restrict(lb, ub, face_dim);
			this_aunt.send_hydro_flux_correct(std::move(data), f.flip(), ci);
		}
	}

	constexpr integer size = geo::face::count() * geo::quadrant::count();
	std::array<future<void>, size> futs;
	for (auto& f : futs) {
		f = hpx::make_ready_future();
	}
	integer index = 0;
	for (auto const& f : geo::face::full_set()) {
		if (this->nieces[f] == +1) {
			for (auto const& quadrant : geo::quadrant::full_set()) {
				futs[index++] = niece_hydro_channels[f][quadrant].get_future().then(
						hpx::util::annotated_function([this, f, quadrant](future<std::vector<real> > && fdata) -> void
						{
							const auto face_dim = f.get_dimension();
							std::array<integer, NDIM> lb, ub;
							switch (face_dim) {
								case XDIM:
								lb[XDIM] = f.get_side() == geo::MINUS ? 0 : INX;
								lb[YDIM] = quadrant.get_side(0) * (INX / 2);
								lb[ZDIM] = quadrant.get_side(1) * (INX / 2);
								ub[XDIM] = lb[XDIM] + 1;
								ub[YDIM] = lb[YDIM] + (INX / 2);
								ub[ZDIM] = lb[ZDIM] + (INX / 2);
								break;
								case YDIM:
								lb[XDIM] = quadrant.get_side(0) * (INX / 2);
								lb[YDIM] = f.get_side() == geo::MINUS ? 0 : INX;
								lb[ZDIM] = quadrant.get_side(1) * (INX / 2);
								ub[XDIM] = lb[XDIM] + (INX / 2);
								ub[YDIM] = lb[YDIM] + 1;
								ub[ZDIM] = lb[ZDIM] + (INX / 2);
								break;
								case ZDIM:
								lb[XDIM] = quadrant.get_side(0) * (INX / 2);
								lb[YDIM] = quadrant.get_side(1) * (INX / 2);
								lb[ZDIM] = f.get_side() == geo::MINUS ? 0 : INX;
								ub[XDIM] = lb[XDIM] + (INX / 2);
								ub[YDIM] = lb[YDIM] + (INX / 2);
								ub[ZDIM] = lb[ZDIM] + 1;
								break;
							}
							grid_ptr->set_flux_restrict(GET(fdata), lb, ub, face_dim);
						}, "node_server::exchange_flux_corrections::set_flux_restrict"));
			}
		}
	}
	return hpx::when_all(std::move(futs)).then([](future<decltype(futs)> fout) {
		auto fin = GET(fout);
		for( auto& f : fin ) {
			GET(f);
		}
	});
}

void node_server::all_hydro_bounds(bool tau_only) {
	exchange_interlevel_hydro_data();
	collect_hydro_boundaries(tau_only);
	send_hydro_amr_boundaries(tau_only);
	++hcycle;
}

void node_server::exchange_interlevel_hydro_data() {

	if (is_refined) {
		std::vector<real> outflow(opts().n_fields, ZERO);
		for (auto const& ci : geo::octant::full_set()) {
			auto data = GET(child_hydro_channels[ci].get_future(hcycle));
			grid_ptr->set_restrict(data, ci);
			integer fi = 0;
			for (auto i = data.end() - opts().n_fields; i != data.end(); ++i) {
				outflow[fi] += *i;
				++fi;
			}
		}
		grid_ptr->set_outflows(std::move(outflow));
	}
	auto data = grid_ptr->get_restrict();
	integer ci = my_location.get_child_index();
	if (my_location.level() != 0) {
		parent.send_hydro_children(std::move(data), ci, hcycle);
	}
}

void node_server::collect_hydro_boundaries(bool tau_only) {
	for (auto const& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			const integer width = H_BW;
			auto bdata = grid_ptr->get_hydro_boundary(dir, width, tau_only);
			neighbors[dir].send_hydro_boundary(std::move(bdata), dir.flip(), hcycle);
		}
	}

	std::array<future<void>, geo::direction::count()> results;
	integer index = 0;
	for (auto const& dir : geo::direction::full_set()) {
		if (!(neighbors[dir].empty() && my_location.level() == 0)) {
			results[index++] = sibling_hydro_channels[dir].get_future(hcycle).then(
					hpx::util::annotated_function([this, tau_only](future<sibling_hydro_type> && f) -> void
					{
							auto&& tmp = GET(f);
							grid_ptr->set_hydro_boundary(tmp.data, tmp.direction,
									H_BW, tau_only);
						}, "node_server::collect_hydro_boundaries::set_hydro_boundary"));
		}
	}
	while (index < geo::direction::count()) {
		results[index++] = hpx::make_ready_future();
	}
//	wait_all_and_propagate_exceptions(std::move(results));
	for (auto& f : results) {
		GET(f);
	}

	for (auto& face : geo::face::full_set()) {
		if (my_location.is_physical_boundary(face)) {
			grid_ptr->set_physical_boundaries(face, current_time);
		}
	}
}

void node_server::send_hydro_amr_boundaries(bool tau_only) {
	if (is_refined) {
		constexpr auto full_set = geo::octant::full_set();
		for (auto& ci : full_set) {
			const auto& flags = amr_flags[ci];
			for (auto& dir : geo::direction::full_set()) {
				//			if (!dir.is_vertex()) {
				if (flags[dir]) {
					std::array<integer, NDIM> lb, ub;
					std::vector<real> data;
//						const integer width = dir.is_face() ? H_BW : 1;
					const integer width = H_BW;
					if (!tau_only) {
						get_boundary_size(lb, ub, dir, OUTER, INX, width);
					} else {
						get_boundary_size(lb, ub, dir, OUTER, INX, width);
					}
					for (integer dim = 0; dim != NDIM; ++dim) {
						lb[dim] = ((lb[dim] - H_BW)) + 2 * H_BW + ci.get_side(dim) * (INX);
						ub[dim] = ((ub[dim] - H_BW)) + 2 * H_BW + ci.get_side(dim) * (INX);
					}
					data = grid_ptr->get_prolong(lb, ub, tau_only);
					children[ci].send_hydro_boundary(std::move(data), dir, hcycle);
				}
			}
//			}
		}
	}
}

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp) {
	// the machine epsilon has to be scaled to the magnitude of the values used
	// and multiplied by the desired precision in ULPs (units in the last place)
	return std::abs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
	// unless the result is subnormal
			|| std::abs(x - y) < std::numeric_limits<T>::min();
}

inline bool file_exists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

void node_server::clear_family() {
	parent = me = hpx::invalid_id;
	std::fill(aunts.begin(), aunts.end(), hpx::invalid_id);
	std::fill(nieces.begin(), nieces.end(), 0);
	std::fill(neighbors.begin(), neighbors.end(),hpx::invalid_id);
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

void node_server::static_initialize() {
	if (!static_initialized) {
		bool test = (static_initializing++ != 0) ? true : false;
		if (!test) {
			static_initialized = true;
		}
		while (!static_initialized) {
			hpx::this_thread::yield();
		}
	}
}

void node_server::initialize(real t, real rt) {
	for (auto const& dir : geo::direction::full_set()) {
		neighbor_signals[dir].signal();
	}
	gcycle = hcycle = rcycle = 0;
	step_num = 0;
	refinement_flag = 0;
	static_initialize();
	is_refined = false;
	neighbors.resize(geo::direction::count());
	nieces.resize(NFACE);
	current_time = t;
	rotational_time = rt;
	dx = TWO * grid::get_scaling_factor() / real(INX << my_location.level());
	for (auto& d : geo::dimension::full_set()) {
		xmin[d] = grid::get_scaling_factor() * my_location.x_location(d);
	}
	if (current_time == ZERO) {
		const auto p = get_problem();
		grid_ptr = std::make_shared<grid>(p, dx, xmin);
	} else {
		grid_ptr = std::make_shared<grid>(dx, xmin);
	}
	if (opts().radiation) {
		rad_grid_ptr = grid_ptr->get_rad_grid();
		rad_grid_ptr->set_dx(dx);
	}
	if (my_location.level() == 0) {
		grid_ptr->set_root();
	}
	aunts.resize(NFACE);
}

node_server::~node_server() {
}

node_server::node_server(const node_location& loc, const node_client& parent_id, real t, real rt, std::size_t _step_num,
		std::size_t _hcycle, std::size_t _rcycle, std::size_t _gcycle) :
		my_location(loc), parent(parent_id) {
	initialize(t, rt);
	step_num = _step_num;
	gcycle = _gcycle;
	hcycle = _hcycle;
	rcycle = _rcycle;
}

node_server::node_server(const node_location& _my_location, integer _step_num, bool _is_refined, real _current_time,
		real _rotational_time, const std::array<integer, NCHILD>& _child_d, grid _grid, const std::vector<hpx::id_type>& _c,
		std::size_t _hcycle, std::size_t _rcycle, std::size_t _gcycle, integer position_) {
	my_location = _my_location;
	initialize(_current_time, _rotational_time);
	position = position_;
	hcycle = _hcycle;
	gcycle = _gcycle;
	rcycle = _rcycle;
	is_refined = _is_refined;
	step_num = _step_num;
	current_time = _current_time;
	rotational_time = _rotational_time;
//     grid test;
	grid_ptr = std::make_shared<grid>(std::move(_grid));
	if (is_refined) {
		std::copy(_c.begin(), _c.end(), children.begin());
	}
	child_descendant_count = _child_d;
}

void node_server::compute_fmm(gsolve_type type, bool energy_account, bool aonly) {
	if (!opts().gravity) {
		return;
	}

	future<void> parent_fut;
	if (energy_account) {
		grid_ptr->egas_to_etot();
	}
	multipole_pass_type m_out;
	m_out.first.resize(INX * INX * INX);
	m_out.second.resize(INX * INX * INX);

	for (auto const& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			neighbor_signals[dir].wait();
		}
	}

	if (is_refined) {
		std::array<future<void>, geo::octant::count()> futs;
		integer index = 0;
		for (auto& ci : geo::octant::full_set()) {
			future<multipole_pass_type> m_in_future = child_gravity_channels[ci].get_future();

			futs[index++] = m_in_future.then(hpx::util::annotated_function([&m_out, ci](future<multipole_pass_type>&& fut)
			{
				const integer x0 = ci.get_side(XDIM) * INX / 2;
				const integer y0 = ci.get_side(YDIM) * INX / 2;
				const integer z0 = ci.get_side(ZDIM) * INX / 2;
				auto m_in = fut.get();
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
			}, "node_server::compute_fmm::gather_from::child_gravity_channels"));
		}
		wait_all_and_propagate_exceptions(std::move(futs));
		m_out = grid_ptr->compute_multipoles(type, &m_out);
	} else {
		m_out = grid_ptr->compute_multipoles(type);
	}

	if (my_location.level() != 0) {
		parent.send_gravity_multipoles(std::move(m_out), my_location.get_child_index());
	}

	if (!aonly) {
		std::vector<future<void>> send_futs;
		for (auto const& dir : geo::direction::full_set()) {
			if (!neighbors[dir].empty()) {
				auto ndir = dir.flip();
				const bool is_monopole = !is_refined;
//             const auto gid = neighbors[dir].get_gid();
				const bool is_local = neighbors[dir].is_local();
				auto data = grid_ptr->get_gravity_boundary(dir, is_local);
				if (is_local) {
					data.local_semaphore = &neighbor_signals[dir];
				} else {
					neighbor_signals[dir].signal();
					data.local_semaphore = nullptr;
				}
				neighbors[dir].send_gravity_boundary(std::move(data), ndir, is_monopole, gcycle);
			}
		}
	}

	/****************************************************************************/
	// data managemenet for old and new version of interaction computation
	// all neighbors and placeholder for yourself
	bool contains_multipole = false;
	std::vector<neighbor_gravity_type> all_neighbor_interaction_data;
	for (geo::direction const& dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			all_neighbor_interaction_data.push_back(neighbor_gravity_channels[dir].get_future(gcycle).get());
			if (!all_neighbor_interaction_data[dir].is_monopole)
				contains_multipole = true;
		} else {
			all_neighbor_interaction_data.emplace_back();
		}
	}

	std::array<bool, geo::direction::count()> is_direction_empty;
	for (geo::direction const& dir : geo::direction::full_set()) {
		if (neighbors[dir].empty()) {
			is_direction_empty[dir] = true;
		} else {
			is_direction_empty[dir] = false;
		}
	}

	bool new_style_enabled = true;
	/***************************************************************************/
	// new-style interaction calculation (both cannot be active at the same time)
	//if (new_style_enabled && !grid_ptr->get_leaf() && !grid_ptr->get_root()) {
	if (new_style_enabled && !grid_ptr->get_root()) {

		// Get all input structures we need as input
		std::vector<multipole>& M_ptr = grid_ptr->get_M();
		std::vector<real>& mon_ptr = grid_ptr->get_mon();
		std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr = grid_ptr->get_com_ptr();

		// initialize to zero
		std::vector<expansion>& L = grid_ptr->get_L();
		std::vector<space_vector>& L_c = grid_ptr->get_L_c();
		std::fill(std::begin(L), std::end(L), ZERO);
		std::fill(std::begin(L_c), std::end(L_c), ZERO);

		// Check if we are a multipole
		if (!grid_ptr->get_leaf()) {
			// Input structure, needed for multipole-monopole interactions
			std::array<real, NDIM> Xbase = { grid_ptr->get_X()[0][hindex(H_BW, H_BW, H_BW)], grid_ptr->get_X()[1][hindex(H_BW,
					H_BW, H_BW)], grid_ptr->get_X()[2][hindex(H_BW, H_BW, H_BW)] };
			// Make sure we have the right pointer
			multipole_interactor.set_grid_ptr(grid_ptr);
			// Run unified multipole-multipole multipole-monopole FMM interaction kernel
			// This will be either run on a cuda device or the cpu (depending on build type and
			// device load)
			multipole_interactor.compute_multipole_interactions(mon_ptr, M_ptr, com_ptr, all_neighbor_interaction_data, type,
					grid_ptr->get_dx(), is_direction_empty, Xbase);
		} else { // ... we are a monopole
			p2p_interactor.set_grid_ptr(grid_ptr);
			p2p_interactor.compute_p2p_interactions(mon_ptr, all_neighbor_interaction_data, type, grid_ptr->get_dx(),
					is_direction_empty);
			if (contains_multipole) {
				p2m_interactor.set_grid_ptr(grid_ptr);
				p2m_interactor.compute_p2m_interactions(mon_ptr, M_ptr, com_ptr, all_neighbor_interaction_data, type,
						is_direction_empty);
			}
		}
	} else {
		// old-style interaction calculation
		// computes inner interactions
		grid_ptr->compute_interactions(type);
		// waits for boundary data and then computes boundary interactions
		for (auto const& dir : geo::direction::full_set()) {
			if (!is_direction_empty[dir]) {
				neighbor_gravity_type & neighbor_data = all_neighbor_interaction_data[dir];
				grid_ptr->compute_boundary_interactions(type, neighbor_data.direction, neighbor_data.is_monopole,
						neighbor_data.data);
			}
		}
	}

	/**************************************************************************/
	// now that all boundary information has been processed, signal all non-empty neighbors
	// note that this was done before during boundary calculations
	for (auto const& dir : geo::direction::full_set()) {

		if (!neighbors[dir].empty()) {
			neighbor_gravity_type & neighbor_data = all_neighbor_interaction_data[dir];
			if (neighbor_data.data.local_semaphore != nullptr) {
				neighbor_data.data.local_semaphore->signal();
			}
		}
	}
	/***************************************************************************/

	expansion_pass_type l_in;
	if (my_location.level() != 0) {
		l_in = parent_gravity_channel.get_future().get();
	}
	const expansion_pass_type ltmp = grid_ptr->compute_expansions(type, my_location.level() == 0 ? nullptr : &l_in);

	if (is_refined) {
		for (auto const& ci : geo::octant::full_set()) {
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
						l_out.first[io] = ltmp.first[ii];
						if (type == RHO) {
							l_out.second[io] = ltmp.second[ii];
						}
					}
				}
			}
			children[ci].send_gravity_expansions(std::move(l_out));
		}
	}

	if (energy_account) {
		grid_ptr->etot_to_egas();
	}
	++gcycle;
}

void node_server::report_timing() {
	timings_.report("...");
}
