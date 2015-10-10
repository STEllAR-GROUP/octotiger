/*
 * node_server.cpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"
#include "problem.hpp"
#include <streambuf>
#include <fstream>
#include <iostream>

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(hpx::components::managed_component<node_server>, node_server);
typedef node_server::load_node_action load_node_action_type;
typedef node_server::save_action save_action_type;
typedef node_server::timestep_driver_action timestep_driver_action_type;
typedef node_server::timestep_driver_ascend_action timestep_driver_ascend_action_type;
typedef node_server::timestep_driver_descend_action timestep_driver_descend_action_type;
typedef node_server::regrid_gather_action regrid_gather_action_type;
typedef node_server::regrid_scatter_action regrid_scatter_action_type;
typedef node_server::send_hydro_boundary_action send_hydro_boundary_action_type;
typedef node_server::send_gravity_boundary_action send_gravity_boundary_action_type;
typedef node_server::send_gravity_multipoles_action send_gravity_multipoles_action_type;
typedef node_server::send_gravity_expansions_action send_gravity_expansions_action_type;
typedef node_server::step_action step_action_type;
typedef node_server::regrid_action regrid_action_type;
typedef node_server::solve_gravity_action solve_gravity_action_type;
typedef node_server::start_run_action start_run_action_type;
typedef node_server::copy_to_locality_action copy_to_locality_action_type;
typedef node_server::get_child_client_action get_child_client_action_type;
typedef node_server::form_tree_action form_tree_action_type;
typedef node_server::get_ptr_action get_ptr_action_type;
typedef node_server::diagnostics_action diagnostics_action_type;
typedef node_server::send_hydro_children_action send_hydro_children_action_type;
typedef node_server::send_hydro_flux_correct_action send_hydro_flux_correct_action_type;
typedef node_server::output_action output_action_type;
typedef node_server::get_nieces_action get_nieces_action_type;
typedef node_server::set_aunt_action set_aunt_action_type;
typedef node_server::check_for_refinement_action check_for_refinement_action_type;
typedef node_server::force_nodes_to_exist_action force_nodes_to_exist_action_type;
typedef node_server::set_grid_action set_grid_action_type;

HPX_REGISTER_ACTION( set_grid_action_type);
HPX_REGISTER_ACTION( force_nodes_to_exist_action_type);
HPX_REGISTER_ACTION( check_for_refinement_action_type);
HPX_REGISTER_ACTION( set_aunt_action_type);
HPX_REGISTER_ACTION( get_nieces_action_type);
HPX_REGISTER_ACTION( load_node_action_type);
HPX_REGISTER_ACTION( save_action_type);
HPX_REGISTER_ACTION( output_action_type);
HPX_REGISTER_ACTION( send_hydro_children_action_type);
HPX_REGISTER_ACTION( send_hydro_flux_correct_action_type);
HPX_REGISTER_ACTION(regrid_gather_action_type);
HPX_REGISTER_ACTION(regrid_scatter_action_type);
HPX_REGISTER_ACTION(send_hydro_boundary_action_type);
HPX_REGISTER_ACTION(send_gravity_boundary_action_type);
HPX_REGISTER_ACTION(send_gravity_multipoles_action_type);
HPX_REGISTER_ACTION(send_gravity_expansions_action_type);
HPX_REGISTER_ACTION(step_action_type);
HPX_REGISTER_ACTION(regrid_action_type);
HPX_REGISTER_ACTION(solve_gravity_action_type);
HPX_REGISTER_ACTION(start_run_action_type);
HPX_REGISTER_ACTION(copy_to_locality_action_type);
HPX_REGISTER_ACTION(get_child_client_action_type);
HPX_REGISTER_ACTION(form_tree_action_type);
HPX_REGISTER_ACTION(get_ptr_action_type);
HPX_REGISTER_ACTION(diagnostics_action_type);
HPX_REGISTER_ACTION(timestep_driver_action_type);
HPX_REGISTER_ACTION(timestep_driver_ascend_action_type);
HPX_REGISTER_ACTION(timestep_driver_descend_action_type);

bool node_server::static_initialized(false);
std::atomic<integer> node_server::static_initializing(0);

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

real node_server::timestep_driver() {
	const real dt = timestep_driver_descend();
	timestep_driver_ascend(dt);
	return dt;
}

real node_server::timestep_driver_descend() {
	real dt;
	if (is_refined) {
		dt = std::numeric_limits<real>::max();
		std::list<hpx::future<real>> futs;
		for (auto i = children.begin(); i != children.end(); ++i) {
			futs.push_back(i->timestep_driver_descend());
		}
		for (auto i = futs.begin(); i != futs.end(); ++i) {
			dt = std::min(dt, i->get());
		}
		dt = std::min(local_timestep_channel->get(), dt);
	} else {
		dt = local_timestep_channel->get();
	}
	return dt;
}

void node_server::timestep_driver_ascend(real dt) {
	global_timestep_channel->set_value(dt);
	if (is_refined) {
		std::list<hpx::future<void>> futs;
		for (auto i = children.begin(); i != children.end(); ++i) {
			futs.push_back(i->timestep_driver_ascend(dt));
		}
		for (auto i = futs.begin(); i != futs.end(); ++i) {
			i->get();
		}
	}
}

std::uintptr_t node_server::get_ptr() {
	return reinterpret_cast<std::uintptr_t>(this);
}

node_server::node_server(node_server&& other) {
	*this = std::move(other);

}

diagnostics_t node_server::diagnostics() const {
	diagnostics_t sums;
	if (is_refined) {
		std::list<hpx::future<diagnostics_t>> futs;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].diagnostics());
		}
		for (auto ci = futs.begin(); ci != futs.end(); ++ci) {
			auto this_sum = ci->get();
			sums += this_sum;
		}
	} else {
		sums.grid_sum = grid_ptr->conserved_sums();
		sums.outflow_sum = grid_ptr->conserved_outflows();
		sums.l_sum = grid_ptr->l_sums();
		auto tmp = grid_ptr->field_range();
		sums.field_min = std::move(tmp.first);
		sums.field_max = std::move(tmp.second);
	}
	return sums;
}

node_server::node_server(const node_location& _my_location, integer _step_num, bool _is_refined, real _current_time,
		const std::array<integer, NCHILD>& _child_d, grid _grid, const std::vector<hpx::id_type>& _c) {
	my_location = _my_location;
	initialize(_current_time);
	is_refined = _is_refined;
	step_num = _step_num;
	current_time = _current_time;
	grid_ptr = std::make_shared < grid > (std::move(_grid));
	if (is_refined) {
		children.resize(NCHILD);
		for (integer ci = 0; ci != NCHILD; ++ci) {
			children[ci] = _c[ci];
		}
	}
	child_descendant_count = _child_d;
}

void node_server::step() {
	real dt = ZERO;

	std::list<hpx::future<void>> child_futs;
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			child_futs.push_back(children[ci].step());
		}
	}

	real a;
	const real dx = TWO / real(INX << my_location.level());
	real cfl0 = cfl;

	exchange_interlevel_hydro_data();
	collect_hydro_boundaries();
	grid_ptr->store();

	for (integer rk = 0; rk < NRK; ++rk) {
		grid_ptr->reconstruct();
		a = grid_ptr->compute_fluxes();
		auto flux_fut = exchange_flux_corrections();
		if (rk == 0) {
			dt = cfl0 * dx / a;
			local_timestep_channel->set_value(dt);
		}
		flux_fut.get();
		grid_ptr->compute_sources();
		grid_ptr->compute_dudt();
		compute_fmm(DRHODT, false);

		if (rk == 0) {
			dt = global_timestep_channel->get();
		}
		grid_ptr->next_u(rk, dt);

		compute_fmm(RHO, true);
//		if (rk != NRK - 1) {
			exchange_interlevel_hydro_data();
			collect_hydro_boundaries();
//		}
	}
	for (auto i = child_futs.begin(); i != child_futs.end(); ++i) {
		i->get();
	}
	current_time += dt;
	++step_num;
}

bool node_server::child_is_on_face(integer ci, integer face) {
	return (((ci >> (face / 2)) & 1) == (face & 1));
}

std::vector<hpx::id_type> node_server::get_nieces(const hpx::id_type& aunt, integer face) const {
	std::vector<hpx::id_type> nieces;
	if (is_refined) {
		std::vector<hpx::future<void>> futs;
		nieces.reserve(NCHILD / 4);
		futs.reserve(NCHILD / 4);
		for (integer ci = 0; ci != NCHILD; ++ci) {
			if (child_is_on_face(ci, face)) {
				nieces.push_back(children[ci].get_gid());
				futs.push_back(children[ci].set_aunt(aunt, face));
			}
		}
		for (auto&& this_fut : futs) {
			this_fut.get();
		}
	}
	return nieces;
}

void node_server::set_aunt(const hpx::id_type& aunt, integer face) {
	aunts[face] = aunt;
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

void node_server::initialize(real t) {
	step_num = 0;
	refinement_flag = 0;
	static_initialize();
	global_timestep_channel = std::make_shared<channel<real>>();
	local_timestep_channel = std::make_shared<channel<real>>();
	is_refined = false;
	siblings.resize(NFACE);
	nieces.resize(NFACE);
	aunts.resize(NFACE);
	for (integer face = 0; face != NFACE; ++face) {
		sibling_hydro_channels[face] = std::make_shared<channel<std::vector<real>> >();
		sibling_gravity_channels[face] = std::make_shared<channel<std::pair<std::vector<real>, bool>> >();
		for (integer i = 0; i != 4; ++i) {
			niece_hydro_channels[face][i] = std::make_shared<channel<std::vector<real>> >();
		}
	}
	for (integer ci = 0; ci != NCHILD; ++ci) {
		child_hydro_channels[ci] = std::make_shared<channel<std::vector<real>>>();
	}
	for (integer face = 0; face != NFACE; ++face) {
	}
	for (integer ci = 0; ci != NCHILD; ++ci) {
		child_gravity_channels[ci] = std::make_shared<channel<multipole_pass_type>>();
	}
	parent_gravity_channel = std::make_shared<channel<expansion_pass_type>>();
	current_time = t;
	dx = TWO / real(INX << my_location.level());
	for (integer d = 0; d != NDIM; ++d) {
		xmin[d] = my_location.x_location(d);
	}
	if (current_time == ZERO) {
		grid_ptr = std::make_shared < grid > (problem, dx, xmin);
	} else {
		grid_ptr = std::make_shared < grid > (dx, xmin);
	}
	if (my_location.level() == 0) {
		grid_ptr->set_root();
	}
}

std::vector<real> node_server::restricted_grid() const {
	std::vector<real> data(INX * INX * INX / NCHILD * NF + NF);
	integer index = 0;
	for (integer i = HBW; i != HNX - HBW; i += 2) {
		for (integer j = HBW; j != HNX - HBW; j += 2) {
			for (integer k = HBW; k != HNX - HBW; k += 2) {
				for (integer field = 0; field != NF; ++field) {
					real& v = data[index];
					v = ZERO;
					for (integer di = 0; di != 2; ++di) {
						for (integer dj = 0; dj != 2; ++dj) {
							for (integer dk = 0; dk != 2; ++dk) {
								v += grid_ptr->hydro_value(field, i + di, j + dj, k + dk) / real(NCHILD);
							}
						}
					}
					++index;
				}
			}
		}
	}
	const auto U_out = grid_ptr->get_outflows();
	for (integer field = 0; field != NF; ++field) {
		data[index] = U_out[field];
	}
	return data;
}

void node_server::recv_hydro_children(std::vector<real>&& data, integer ci) {
	child_hydro_channels[ci]->set_value(std::move(data));
}

void node_server::recv_hydro_flux_correct(std::vector<real>&& data, integer face, integer ci) {
	const integer dim = face / 2;
	const integer index = child_index_to_quadrant_index(ci,dim);
	niece_hydro_channels[face][index]->set_value(std::move(data));
}

void node_server::load_from_restricted_child(const std::vector<real>& data, integer ci) {
	integer index = 0;
	const integer di = ((ci >> 0) & 1) * INX / 2;
	const integer dj = ((ci >> 1) & 1) * INX / 2;
	const integer dk = ((ci >> 2) & 1) * INX / 2;
	for (integer i = HBW; i != HNX / 2; ++i) {
		for (integer j = HBW; j != HNX / 2; ++j) {
			for (integer k = HBW; k != HNX / 2; ++k) {
				for (integer field = 0; field != NF; ++field) {
					grid_ptr->hydro_value(field, i + di, j + dj, k + dk) = data[index];
					++index;
				}
			}
		}
	}
}

node_server::~node_server() {
}

node_server::node_server() {
	initialize(ZERO);
}

node_server::node_server(const node_location& loc, const node_client& parent_id, real t) :
		my_location(loc), parent(parent_id) {
	initialize(t);
}

node_server& node_server::operator=(node_server&& other ) {

	my_location = std::move(other.my_location);
	step_num = std::move(other.step_num);
	current_time = std::move(other.current_time);
	grid_ptr = std::move(other.grid_ptr);
	is_refined = std::move(other.is_refined);
	child_descendant_count = std::move(other.child_descendant_count);
	xmin = std::move(other.xmin);
	dx = std::move(other.dx);
	me = std::move(other.me);
	parent = std::move(other.parent);
	siblings = std::move(other.siblings);
	children = std::move(other.children);
	child_hydro_channels = std::move(other.child_hydro_channels);
	sibling_hydro_channels = std::move(other.sibling_hydro_channels);
	parent_gravity_channel = std::move(other.parent_gravity_channel);
	sibling_gravity_channels = std::move(other.sibling_gravity_channels);
	child_gravity_channels = std::move(other.child_gravity_channels);
	global_timestep_channel = std::move(other.global_timestep_channel);

	return *this;
}

void node_server::solve_gravity(bool ene) {
//	printf("%s\n", my_location.to_str().c_str());

	std::list<hpx::future<void>> child_futs;
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			child_futs.push_back(children[ci].solve_gravity(ene));
		}
	}
	compute_fmm(RHO, ene);
	if (is_refined) {
		for (auto i = child_futs.begin(); i != child_futs.end(); ++i) {
			i->get();
		}
	}
}

void node_server::compute_fmm(gsolve_type type, bool energy_account) {
	std::list<hpx::future<void>> child_futs;
	std::list<hpx::future<void>> sibling_futs;
	hpx::future<void> parent_fut;

	if (energy_account) {
		grid_ptr->egas_to_etot();
	}
	multipole_pass_type m_in, m_out;
	m_out.first.resize(INX * INX * INX);
	m_out.second.resize(INX * INX * INX);
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			const integer x0 = ((ci >> 0) & 1) * INX / 2;
			const integer y0 = ((ci >> 1) & 1) * INX / 2;
			const integer z0 = ((ci >> 2) & 1) * INX / 2;
			m_in = child_gravity_channels[ci]->get();
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

	std::array<bool, NFACE> is_monopole;
	for (integer dim = 0; dim != NDIM; ++dim) {
		for (integer si = 2 * dim; si != 2 * (dim + 1); ++si) {
			if (!siblings[si].empty()) {
				const bool monopole = !is_refined;
				sibling_futs.push_back(siblings[si].send_gravity_boundary(get_gravity_boundary(si), si ^ 1, monopole));
			}
		}
		for (integer si = 2 * dim; si != 2 * (dim + 1); ++si) {
			if (!siblings[si].empty()) {
				auto tmp = this->sibling_gravity_channels[si]->get();
				is_monopole[si] = tmp.second;
				this->set_gravity_boundary(std::move(tmp.first), si, tmp.second);
			}
		}
	}
	for (integer dim = 0; dim != NDIM; ++dim) {
		for (integer si = 2 * dim; si != 2 * (dim + 1); ++si) {
			if (!siblings[si].empty()) {
				this->grid_ptr->compute_boundary_interactions(type, si, is_monopole[si]);
			}
		}
	}

	parent_fut.get();

	expansion_pass_type l_in;
	if (my_location.level() != 0) {
		l_in = parent_gravity_channel->get();
	}
	const expansion_pass_type ltmp = grid_ptr->compute_expansions(type, my_location.level() == 0 ? nullptr : &l_in);
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			expansion_pass_type l_out;
			l_out.first.resize(INX * INX * INX / NCHILD);
			if (type == RHO) {
				l_out.second.resize(INX * INX * INX / NCHILD);
			}
			const integer x0 = ((ci >> 0) & 1) * INX / 2;
			const integer y0 = ((ci >> 1) & 1) * INX / 2;
			const integer z0 = ((ci >> 2) & 1) * INX / 2;
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

	for (auto i = child_futs.begin(); i != child_futs.end(); ++i) {
		i->get();
	}
	for (auto i = sibling_futs.begin(); i != sibling_futs.end(); ++i) {
		i->get();
	}
}

