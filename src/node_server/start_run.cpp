/*
 * start_run.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */


#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::start_run_action start_run_action_type;
HPX_REGISTER_ACTION (start_run_action_type);


hpx::future<void> node_client::start_run(bool b) const {
	return hpx::async<typename node_server::start_run_action>(get_gid(), b);
}


void node_server::start_run(bool scf) {
	integer output_cnt;

	if (!hydro_on) {
		save_to_file("X.chk");
		diagnostics();
		return;
	}
	if (scf) {
		run_scf();
		set_omega_and_pivot();
		save_to_file("scf.chk");
	}

	printf("Starting...\n");
	regrid(me.get_gid(), false);
	solve_gravity(false);
	if (current_time == 0) {
//		run_scf();
		//	if (system("mkdir dat_back\n")) {
		//	}
		printf("Adjusting velocities:\n");
		auto diag = diagnostics();
		space_vector dv;
		dv[XDIM] = -diag.grid_sum[sx_i] / diag.grid_sum[rho_i];
		dv[YDIM] = -diag.grid_sum[sy_i] / diag.grid_sum[rho_i];
		dv[ZDIM] = -diag.grid_sum[sz_i] / diag.grid_sum[rho_i];
		printf("%e %e %e\n", dv[XDIM], dv[YDIM], dv[ZDIM]);
		this->velocity_inc(dv);
	}

	real output_dt = 2.0 * M_PI / grid::get_omega() / 100.0;

	printf("OMEGA = %e\n", grid::get_omega());
	real& t = current_time;
	integer step_num = 0;

	auto fut_ptr = me.get_ptr();
	node_server* root_ptr = fut_ptr.get();

	output_cnt = root_ptr->get_time() / output_dt;
	hpx::future<void> diag_fut = hpx::make_ready_future();
	hpx::future<void> step_fut = hpx::make_ready_future();
	while (true) {
		auto time_start = std::chrono::high_resolution_clock::now();

		auto axis = grid_ptr->find_axis();
		printf( "%e %e %e\n", axis[0], axis[1], axis[2]);
		diagnostics();

		if (t / output_dt >= output_cnt) {
			char* fname;

			if (asprintf(&fname, "X.%i.chk", int(output_cnt))) {
			}
			save_to_file(fname);
			free(fname);
			//	SYSTEM(std::string("cp *.dat ./dat_back/\n"));
			++output_cnt;

		}
		//	break;
		auto ts_fut = hpx::async([=]() {return timestep_driver();});
		step();
		real dt = GET(ts_fut);

		double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
				std::chrono::high_resolution_clock::now() - time_start).count();
		GET(step_fut);
		step_fut =
				hpx::async(
						[=]() {
							FILE* fp = fopen( "step.dat", "at");
							fprintf(fp, "%i %e %e %e %e %e\n", int(step_num), double(t), double(dt), time_elapsed, grid::get_omega(),rotational_time);
							fclose(fp);
						});
		printf("%i %e %e %e %e %e\n", int(step_num), double(t), double(dt), time_elapsed, grid::get_omega(),
				rotational_time);
//		t += dt;
		++step_num;

		const integer refinement_freq = integer(R_BW / cfl + 0.5);
		if (step_num % refinement_freq == 0) {
			regrid(me.get_gid(), false);
		}
//		set_omega_and_pivot();
	}
}
