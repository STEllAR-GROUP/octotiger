/*
 * node_server.cpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#include "future.hpp"

#include "node_server.hpp"
#include "future.hpp"
#include "problem.hpp"
#include <streambuf>
#include <fstream>
#include <iostream>

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(hpx::components::managed_component<node_server>, node_server);

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
typedef node_server::output_action output_action_type;

HPX_REGISTER_ACTION( save_action_type );
HPX_REGISTER_ACTION( output_action_type );
HPX_REGISTER_ACTION( send_hydro_children_action_type );
HPX_REGISTER_ACTION (regrid_gather_action_type);
HPX_REGISTER_ACTION (regrid_scatter_action_type);
HPX_REGISTER_ACTION (send_hydro_boundary_action_type);
HPX_REGISTER_ACTION (send_gravity_boundary_action_type);
HPX_REGISTER_ACTION (send_gravity_multipoles_action_type);
HPX_REGISTER_ACTION (send_gravity_expansions_action_type);
HPX_REGISTER_ACTION (step_action_type);
HPX_REGISTER_ACTION (regrid_action_type);
HPX_REGISTER_ACTION (solve_gravity_action_type);
HPX_REGISTER_ACTION (start_run_action_type);
HPX_REGISTER_ACTION (copy_to_locality_action_type);
HPX_REGISTER_ACTION (get_child_client_action_type);
HPX_REGISTER_ACTION (form_tree_action_type);
HPX_REGISTER_ACTION (get_ptr_action_type);
HPX_REGISTER_ACTION (diagnostics_action_type);
HPX_REGISTER_ACTION (timestep_driver_action_type);
HPX_REGISTER_ACTION (timestep_driver_ascend_action_type);
HPX_REGISTER_ACTION (timestep_driver_descend_action_type);

bool node_server::static_initialized(false);
std::atomic<integer> node_server::static_initializing(0);


real node_server::timestep_driver() {
	const real dt = timestep_driver_descend();
	timestep_driver_ascend(dt);
	return dt;
}


real node_server::timestep_driver_descend() {
	real dt;
	if( is_refined ) {
		dt = std::numeric_limits<real>::max();
		std::vector<hpx::future<real>> futs;
        futs.reserve(children.size());
		for( auto i = children.begin(); i != children.end(); ++i) {
			futs.push_back(i->timestep_driver_descend());
		}
		for( auto i = futs.begin(); i != futs.end(); ++i) {
			dt = std::min(dt, GET(*i));
		}
	} else {
//         std::cout << "descend: local_timestep_channel\n";
		dt = GET(local_timestep_channel->get_future());
//         std::cout << "descend: local_timestep_channel ... done\n";
	}
	return dt;
}


void node_server::timestep_driver_ascend(real dt) {
	if( is_refined ) {
		std::vector<hpx::future<void>> futs;
        futs.reserve(children.size());
		for( auto i = children.begin(); i != children.end(); ++i ) {
			futs.push_back(i->timestep_driver_ascend(dt));
		}
        hpx::wait_all(futs);
	} else {
//         std::cout << "ascend: global_timestep_channel\n";
		global_timestep_channel->set_value(dt);
//         std::cout << "ascend: global_timestep_channel ... done\n";
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
	if( is_refined ) {
		std::vector<hpx::future<diagnostics_t>> futs;
        futs.reserve(NCHILD);
		for( integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back( children[ci].diagnostics());
		}
        for(auto && this_sum : hpx::util::unwrapped(futs))
        {
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


void node_server::form_tree(const hpx::id_type& self_gid, const hpx::id_type& parent_gid,
		const std::vector<hpx::id_type>& sib_gids) {
	for (integer si = 0; si != NFACE; ++si) {
		siblings[si] = sib_gids[si];
	}
	std::vector<hpx::future<void>> cfuts;
    cfuts.reserve(NCHILD);
	me = self_gid;
	parent = parent_gid;
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			std::vector <hpx::future < hpx::id_type >> child_sibs_f(NFACE);
			std::vector < hpx::id_type > child_sibs(NFACE);
			for (integer d = 0; d != NDIM; ++d) {
				const integer flip = ci ^ (1 << d);
				const integer bit = (ci >> d) & 1;
				const integer other = 2 * d + bit;
				const integer thisf = 2 * d + (1 - bit);
				child_sibs_f[thisf] = hpx::make_ready_future(children[flip].get_gid());
				child_sibs_f[other] = siblings[other].get_child_client(flip);
			}
			for( integer f = 0; f != NFACE; ++f) {
				child_sibs[f] = GET(child_sibs_f[f]);
			}
			cfuts.push_back(children[ci].form_tree(children[ci].get_gid(), me.get_gid(), std::move(child_sibs)));
		}
        hpx::wait_all(cfuts);
	}

}

hpx::id_type node_server::get_child_client(integer ci) {
	return children[ci].get_gid();
}

hpx::future<hpx::id_type> node_server::copy_to_locality(const hpx::id_type& id) {

	std::vector<hpx::id_type> cids;
	if (is_refined) {
		cids.resize(NCHILD);
		for (integer ci = 0; ci != NCHILD; ++ci) {
			cids[ci] = children[ci].get_gid();
		}
	}
	auto rc = hpx::new_ < node_server
			> (id, my_location, step_num, is_refined, current_time, child_descendant_count, *grid_ptr, cids);
	clear_family();
	return rc;
}

node_server::node_server(node_location&& _my_location, integer _step_num, bool _is_refined, real _current_time, std::array<integer,NCHILD>&& _child_d, grid&& _grid,const std::vector<hpx::id_type>& _c) {
	my_location = std::move(_my_location);
	initialize(_current_time);
	is_refined = _is_refined;
	step_num = _step_num;
	current_time = _current_time;
	grid_ptr = std::make_shared<grid>(std::move(_grid));
	if( is_refined ) {
		children.resize(NCHILD);
		for( integer ci = 0; ci != NCHILD; ++ci) {
			children[ci] = _c[ci];
		}
	}
	child_descendant_count = _child_d;
}

void node_server::step() {
	real dt = ZERO;

	std::vector<hpx::future<void>> child_futs;
	if (is_refined) {
        child_futs.reserve(NCHILD);
		for (integer ci = 0; ci != NCHILD; ++ci) {
			child_futs.push_back(children[ci].step());
		}
	}

	real a;
	const real dx = TWO / real(INX << my_location.level());
	real cfl0 = cfl;
	grid_ptr->store();

	for (integer rk = 0; rk < NRK; ++rk) {
		if (!is_refined) {
			grid_ptr->reconstruct();
			a = grid_ptr->compute_fluxes();
			if( rk == 0 ) {
				dt = cfl0 * dx / a;
				local_timestep_channel->set_value(dt);
			}
			grid_ptr->compute_sources();
			grid_ptr->compute_dudt();
		}

		compute_fmm(DRHODT, false, 2 * rk + 0);

		if (!is_refined) {
			if (rk == 0) {
				dt = GET(global_timestep_channel->get_future());
			}
			grid_ptr->next_u(rk, dt);
		}

		compute_fmm(RHO, true, 2 * rk + 1);
	//	auto interlevel_fut = exchange_interlevel_hydro_data(rk);
	//	interlevel_fut.get();
		if( !is_refined ) {
			collect_hydro_boundaries(rk);
		}
	}
    hpx::wait_all(child_futs);

	++step_num;
}
/*

hpx::future<void> node_server::exchange_interlevel_hydro_data(integer rk) {

	if( my_location.level() > 0 ) {
		std::vector<real> data = restricted_grid();
		integer ci = my_location.get_child_index();
		parent.send_hydro_children(std::move(data), rk, ci);
	}
	std::list<hpx::future<void>> futs;
	if( is_refined ) {
		for( integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back( hpx::async([=]() {
				std::vector<real> data = child_hydro_channels[rk][ci]->get();
				load_from_restricted_child(data, ci);
			}));
		}
	}
	hpx::future<void> rfut = hpx::when_all(futs.begin(), futs.end());
	return rfut;
}
*/
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
	static_initialize();
	global_timestep_channel = std::make_shared<channel<real>>();
	local_timestep_channel = std::make_shared<channel<real>>();
	is_refined = false;
	siblings.resize(NFACE);
	for (integer rk = 0; rk != NRK; ++rk) {
		for (integer face = 0; face != NFACE; ++face) {
			sibling_hydro_channels[rk][face] = std::make_shared<channel<std::vector<real>> >();
		}
		for( integer ci = 0; ci != NCHILD; ++ci) {
			child_hydro_channels[rk][ci] = std::make_shared<channel<std::vector<real>>>();
		}
	}
	for (integer c = 0; c != 4; ++c) {
		for (integer face = 0; face != NFACE; ++face) {
			sibling_gravity_channels[c][face] = std::make_shared<channel<std::vector<real>> >();
		}
		for (integer ci = 0; ci != NCHILD; ++ci) {
			child_gravity_channels[c][ci] = std::make_shared<channel<multipole_pass_type>>();
		}
		parent_gravity_channel[c] = std::make_shared<channel<expansion_pass_type>>();
	}
	current_time = t;
	dx = TWO / real(INX << my_location.level());
	for (integer d = 0; d != NDIM; ++d) {
		xmin[d] = my_location.x_location(d);
	}
	const integer flags = ((my_location.level() == 0) ? GRID_IS_ROOT : 0) | GRID_IS_LEAF;
	if (current_time == ZERO) {
		grid_ptr = std::make_shared < grid > (problem, dx, xmin, flags);
	} else {
		grid_ptr = std::make_shared < grid > (dx, xmin, flags);
	}
//	printf("Creating grid at %i: %i %i %i w on locality %i\n", int(my_location.level()), int(my_location[0]),
//			int(my_location[1]), int(my_location[2]), int(hpx::get_locality_id()));
}

std::vector<real> node_server::restricted_grid() const {
	std::vector<real> data(INX*INX*INX/NCHILD*(NF+NDIM));
	integer index = 0;
	for( integer i = HBW; i != HNX - HBW; i += 2) {
		for( integer j = HBW; j != HNX - HBW; j += 2) {
			for( integer k = HBW; k != HNX - HBW; k += 2) {
				for( integer field = 0; field != NF; ++field ) {
					real& v = data[index];
					v = ZERO;
					for( integer di = 0; di != 2; ++di ) {
						for( integer dj = 0; dj != 2; ++dj) {
							for( integer dk = 0; dk != 2; ++dk) {
								v += grid_ptr->hydro_value(field, i + di, j + dj, k + dk) / real(NCHILD);
							}
						}
					}
					++index;
				}
			}
		}
	}
	return data;
}

void node_server::recv_hydro_children( std::vector<real>&& data, integer rk, integer ci) {
	child_hydro_channels[rk][ci]->set_value(std::move(data));
}

void node_server::load_from_restricted_child(const std::vector<real>& data, integer ci)  {
	integer index = 0;
	const integer di = ((ci >> 0) & 1) * INX / 2;
	const integer dj = ((ci >> 1) & 1) * INX / 2;
	const integer dk = ((ci >> 2) & 1) * INX / 2;
	for( integer i = HBW; i != HNX/2; ++i) {
		for( integer j = HBW; j != HNX/2; ++j) {
			for( integer k = HBW; k != HNX/2; ++k) {
				for( integer field = 0; field != NF; ++field ) {
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

integer node_server::regrid_gather() {
	integer count = integer(1);

	if (is_refined) {
		std::vector<hpx::future<integer>> futs;
        futs.reserve(NCHILD);

		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].regrid_gather());
		}

        auto futs_it = futs.begin();
		for (integer ci = 0; ci != NCHILD; ++ci) {
			auto child_cnt = GET(*futs_it);
			++futs_it;
			child_descendant_count[ci] = child_cnt;
			count += child_cnt;
		}

		if (count == 1) {
			for (integer ci = 0; ci != NCHILD; ++ci) {
				children[ci] = hpx::invalid_id;
			}
			is_refined = false;
			const integer flags = ((my_location.level() == 0) ? GRID_IS_ROOT : 0) | GRID_IS_LEAF;
			grid_ptr = std::make_shared < grid > (dx, xmin, flags);
		}

	} else {
		if (grid_ptr->refine_me(my_location.level())) {
			count += NCHILD;


			children.resize(NCHILD);
			std::vector<node_location> clocs(NCHILD);
			for (integer ci = 0; ci != NCHILD; ++ci) {
				child_descendant_count[ci] = 1;
				clocs[ci] = my_location.get_child(ci);
				children[ci] = hpx::new_<node_server>(hpx::find_here(), clocs[ci], me, current_time);
			}
			is_refined = true;
			const integer flags = (my_location.level() == 0) ? GRID_IS_ROOT : 0;
			grid_ptr = std::make_shared < grid > (dx, xmin, flags);
		}
	}
	return count;
}

node_server& node_server::operator=( node_server&& other ) {

	my_location = std::move(other.my_location);
	step_num = std::move(other.step_num);
	current_time = std::move(other.current_time);
	grid_ptr  = std::move(other.grid_ptr);
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

void node_server::regrid() {
	assert(grid_ptr!=nullptr);
	printf( "Regrid gather\n");
	integer a = regrid_gather();
	printf( "Regrid scatter\n");
	regrid_scatter(0, a);
	printf( "Regrid done\n");
	assert(grid_ptr!=nullptr);
}

void node_server::regrid_scatter(integer a_, integer total) {
	std::vector<hpx::future<void>> futs;
	if (is_refined) {
        futs.reserve(NCHILD);
		integer a = a_;
		const auto localities = hpx::find_all_localities();
		++a;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			const integer loc_index = a * localities.size() / total;
			const auto child_loc = localities[loc_index];
			a += child_descendant_count[ci];
			const hpx::id_type id = children[ci].get_gid();
			integer current_child_id = hpx::naming::get_locality_id_from_gid(id.get_gid());
			auto current_child_loc = localities[current_child_id];
			if (child_loc != current_child_loc) {
		//		printf( "Moving %s from %i to %i\n", my_location.get_child(ci).to_str().c_str(), hpx::get_locality_id(), int(loc_index));
				children[ci] = children[ci].copy_to_locality(child_loc);
			}
		}
		a = a_ + 1;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].regrid_scatter(a, total));
			a += child_descendant_count[ci];
		}
	}
	clear_family();
    hpx::wait_all(futs);
}

void node_server::clear_family() {
	for (integer si = 0; si != NFACE; ++si) {
		siblings[si] = hpx::invalid_id;
	}
	parent = hpx::invalid_id;
	me = hpx::invalid_id;
}

void node_server::solve_gravity(bool ene, integer c) {
//	printf("%s\n", my_location.to_str().c_str());

	std::vector<hpx::future<void>> child_futs;
	if (is_refined) {
        child_futs.reserve(NCHILD);
		for (integer ci = 0; ci != NCHILD; ++ci) {
			child_futs.push_back(children[ci].solve_gravity(ene, c));
		}
	}
	compute_fmm(RHO, ene, c);
    hpx::wait_all(child_futs);
}

void node_server::compute_fmm(gsolve_type type, bool energy_account, integer gchannel) {

//	if( my_location.level() == 3 ) printf( "0\n");

	std::vector<hpx::future<void>> child_futs;
	std::vector<hpx::future<void>> sibling_futs;
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
			m_in = GET(child_gravity_channels[gchannel][ci]->get_future());
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
		parent_fut = parent.send_gravity_multipoles(std::move(m_out), my_location.get_child_index(), gchannel);
	} else {
		parent_fut = hpx::make_ready_future();
	}

	grid_ptr->compute_interactions(type);

//	if( my_location.level() == 3 ) printf( "1\n");


    sibling_futs.reserve(NDIM*NDIM);
	for (integer dim = 0; dim != NDIM; ++dim) {
		for (integer si = 2 * dim; si != 2 * (dim + 1); ++si) {
			if (!my_location.is_physical_boundary(si)) {
				sibling_futs.push_back(siblings[si].send_gravity_boundary(get_gravity_boundary(si), si ^ 1, gchannel));
			}
		}
		for (integer si = 2 * dim; si != 2 * (dim + 1); ++si) {
			if (!my_location.is_physical_boundary(si)) {
				const std::vector<real> tmp = GET(this->sibling_gravity_channels[gchannel][si]->get_future());
				this->set_gravity_boundary(std::move(tmp), si);
			}
		}
	}
	for (integer dim = 0; dim != NDIM; ++dim) {
		for (integer si = 2 * dim; si != 2 * (dim + 1); ++si) {
			if (!my_location.is_physical_boundary(si)) {
				this->grid_ptr->compute_boundary_interactions(type, si);
			}
		}
	}

	GET(parent_fut);


	expansion_pass_type l_in;
	if (my_location.level() != 0) {
		l_in = GET(parent_gravity_channel[gchannel]->get_future());
	}
	const expansion_pass_type ltmp = grid_ptr->compute_expansions(type, my_location.level() == 0 ? nullptr : &l_in);
	if (is_refined) {
        child_futs.reserve(NCHILD);
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
						auto t =  ltmp.first[ii];
						l_out.first[io] = t;
						if (type == RHO) {
							l_out.second[io] = ltmp.second[ii];
						}
					}
				}
			}
			child_futs.push_back(children[ci].send_gravity_expansions(std::move(l_out), gchannel));
		}
	}

	if (energy_account) {
		grid_ptr->etot_to_egas();
	}

    hpx::wait_all(child_futs);
    hpx::wait_all(sibling_futs);
}

