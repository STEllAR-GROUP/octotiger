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



HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(hpx::components::managed_component<node_server>, node_server);

typedef node_server::load_action load_action_type;
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
typedef node_server::get_nieces_action get_nieces_action_type;
typedef node_server::set_aunt_action set_aunt_action_type;
typedef node_server::check_for_refinement_action check_for_refinement_action_type;
typedef node_server::force_nodes_to_exist_action force_nodes_to_exist_action_type;
typedef node_server::set_grid_action set_grid_action_type;
typedef node_server::find_omega_part_action find_omega_part_action_type;
typedef node_server::scf_params_action scf_params_action_type;
typedef node_server::scf_update_action scf_update_action_type;

HPX_REGISTER_ACTION(scf_update_action_type);
HPX_REGISTER_ACTION(scf_params_action_type);
HPX_REGISTER_ACTION(find_omega_part_action_type);
HPX_REGISTER_ACTION(set_grid_action_type);
HPX_REGISTER_ACTION(force_nodes_to_exist_action_type);
HPX_REGISTER_ACTION(check_for_refinement_action_type);
HPX_REGISTER_ACTION(set_aunt_action_type);
HPX_REGISTER_ACTION(get_nieces_action_type);
HPX_REGISTER_ACTION(load_action_type);
HPX_REGISTER_ACTION(save_action_type);
HPX_REGISTER_ACTION(send_hydro_children_action_type);
HPX_REGISTER_ACTION(send_hydro_flux_correct_action_type);
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

real node_server::find_omega() const {
	const auto this_com = grid_ptr->center_of_mass();
//	printf( "%e %e %e\n", this_com[0], this_com[1], this_com[2]);
	auto d = find_omega_part(this_com);
//	printf( "%e %e\n", d.first, d.second);
	return d.first / d.second;
}

std::pair<real, real> node_server::find_omega_part(const space_vector& pivot) const {
	std::pair<real, real> d;
	if (is_refined) {
		std::vector<hpx::future<std::pair<real, real>>>futs;
		futs.reserve(NCHILD);
		for( auto& child : children) {
			futs.push_back(child.find_omega_part(pivot));
		}
		d.first = d.second = ZERO;
		for( auto&& fut : futs) {
			auto tmp = GET(fut);
			d.first += tmp.first;
			d.second += tmp.second;
		}
	} else {
		d = grid_ptr->omega_part(pivot);
	}
	return d;
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
			dt = std::min(dt, GET(*i));
		}
		dt = std::min(GET(local_timestep_channel->get_future()), dt);
	} else {
		dt = GET(local_timestep_channel->get_future());
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
			GET(*i);
		}
	}
}

std::uintptr_t node_server::get_ptr() {
	return reinterpret_cast<std::uintptr_t>(this);
}
/*
 node_server::node_server(node_server&& other) {
 *this = std::move(other);

 }*/

diagnostics_t node_server::diagnostics() const {
	diagnostics_t sums;
	if (is_refined) {
		std::list<hpx::future<diagnostics_t>> futs;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].diagnostics());
		}
		for (auto ci = futs.begin(); ci != futs.end(); ++ci) {
			auto this_sum = GET(*ci);
			sums += this_sum;
		}
	} else {
		sums.grid_sum = grid_ptr->conserved_sums();
		sums.outflow_sum = grid_ptr->conserved_outflows();
		sums.donor_mass = grid_ptr->conserved_sums([](real x, real, real) {return x > 0.09;})[rho_i];
		sums.l_sum = grid_ptr->l_sums();
		auto tmp = grid_ptr->field_range();
		sums.field_min = std::move(tmp.first);
		sums.field_max = std::move(tmp.second);
	}

	if (my_location.level() == 0) {
		auto diags = sums;
		FILE* fp = fopen("diag.dat", "at");
		fprintf(fp, "%23.16e ", double(current_time));
		for (integer f = 0; f != NF; ++f) {
			fprintf(fp, "%23.16e ", double(diags.grid_sum[f] + diags.outflow_sum[f]));
			fprintf(fp, "%23.16e ", double(diags.outflow_sum[f]));
		}
		for (integer f = 0; f != NDIM; ++f) {
			fprintf(fp, "%23.16e ", double(diags.l_sum[f]));
		}
		fprintf(fp, "\n");
		fclose(fp);

		fp = fopen("minmax.dat", "at");
		fprintf(fp, "%23.16e ", double(current_time));
		for (integer f = 0; f != NF; ++f) {
			fprintf(fp, "%23.16e ", double(diags.field_min[f]));
			fprintf(fp, "%23.16e ", double(diags.field_max[f]));
		}
		fprintf(fp, "\n");
		fclose(fp);

		fp = fopen("m_don.dat", "at");
		fprintf(fp, "%23.16e ", double(current_time));
		fprintf(fp, "%23.16e ", double(diags.grid_sum[rho_i] - diags.donor_mass));
		fprintf(fp, "%23.16e ", double(diags.donor_mass));
		fprintf(fp, "\n");
		fclose(fp);
}

	return sums;
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

void node_server::step() {
	grid_ptr->set_coordinates();
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
		GET(flux_fut);
		grid_ptr->compute_sources();
		grid_ptr->compute_dudt();
		compute_fmm(DRHODT, false);

		if (rk == 0) {
			dt = GET(global_timestep_channel->get_future());
		}
		grid_ptr->next_u(rk, dt);

		compute_fmm(RHO, true);
		exchange_interlevel_hydro_data();
		collect_hydro_boundaries();
	}
	grid_ptr->dual_energy_update();
	for (auto i = child_futs.begin(); i != child_futs.end(); ++i) {
		GET(*i);
	}
	current_time += dt;
	rotational_time += grid::get_omega() * dt;
	++step_num;
}

bool node_server::child_is_on_face(integer ci, integer face) {
	return (((ci >> (face / 2)) & 1) == (face & 1));
}

std::vector<hpx::id_type> node_server::get_nieces(const hpx::id_type& aunt, const geo::face& face) const {
	std::vector<hpx::id_type> nieces;
	if (is_refined) {
		std::vector<hpx::future<void>> futs;
		nieces.reserve(geo::quadrant::count());
		futs.reserve(geo::quadrant::count());
		for (auto& ci : geo::octant::face_subset(face)) {
			nieces.push_back(children[ci].get_gid());
			futs.push_back(children[ci].set_aunt(aunt, face));
		}
		for (auto&& this_fut : futs) {
			GET(this_fut);
		}
	}
	return nieces;
}

void node_server::set_aunt(const hpx::id_type& aunt, const geo::face& face) {
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
	for (auto& dir : geo::direction::full_set()) {
		neighbor_gravity_channels[dir] = std::make_shared<channel<std::pair<std::vector<real>, bool>> >();
		sibling_hydro_channels[dir] = std::make_shared<channel<std::vector<real>> >();
	}
	for (auto& face : geo::face::full_set()) {
		for (integer i = 0; i != 4; ++i) {
			niece_hydro_channels[face][i] = std::make_shared<channel<std::vector<real>> >();
		}
	}
	for (auto& ci : geo::octant::full_set()) {
		child_hydro_channels[ci] = std::make_shared<channel<std::vector<real>>>();
		child_gravity_channels[ci] = std::make_shared<channel<multipole_pass_type>>();
	}
	parent_gravity_channel = std::make_shared<channel<expansion_pass_type>>();
	current_time = t;
	rotational_time = rt;
	dx = TWO / real(INX << my_location.level());
	for (auto& d : geo::dimension::full_set()) {
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

void node_server::recv_hydro_children(std::vector<real>&& data, const geo::octant& ci) {
	child_hydro_channels[ci]->set_value(std::move(data));
}

void node_server::recv_hydro_flux_correct(std::vector<real>&& data, const geo::face& face, const geo::octant& ci) {
	const geo::quadrant index(ci, face.get_dimension());
	niece_hydro_channels[face][index]->set_value(std::move(data));
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
void node_server::solve_gravity(bool ene) {
	std::list<hpx::future<void>> child_futs;
	for (auto& child : children) {
		child_futs.push_back(child.solve_gravity(ene));
	}
	compute_fmm(RHO, ene);
	for (auto&& fut : child_futs) {
		GET(fut);
	}
}

void node_server::compute_fmm(gsolve_type type, bool energy_account) {
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
}

