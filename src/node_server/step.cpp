/*
 * step.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */


#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::step_action step_action_type;
HPX_REGISTER_ACTION (step_action_type);



hpx::future<void> node_client::step() const {
	return hpx::async<typename node_server::step_action>(get_gid());
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
	const real dx = TWO * grid::get_scaling_factor() / real(INX << my_location.level());
	real cfl0 = cfl;

	exchange_interlevel_hydro_data();
	collect_hydro_boundaries().get();
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
		grid_ptr->compute_sources(current_time);
		grid_ptr->compute_dudt();
		compute_fmm(DRHODT, false);

		if (rk == 0) {
			dt = GET(global_timestep_channel->get_future());
		}
		grid_ptr->next_u(rk, current_time, dt);

		compute_fmm(RHO, true);
		exchange_interlevel_hydro_data();
		collect_hydro_boundaries().get();
	}
	grid_ptr->dual_energy_update();
	for (auto i = child_futs.begin(); i != child_futs.end(); ++i) {
		GET(*i);
	}
	current_time += dt;
	rotational_time += grid::get_omega() * dt;
	++step_num;
}
