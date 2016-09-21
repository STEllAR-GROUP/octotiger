/*
 * start_run.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */

#include "../node_server.hpp"
#include "../node_client.hpp"
#include "../profiler.hpp"
#include "../util.hpp"
#include <mpi.h>
#include "options.hpp"
extern options opts;

typedef node_server::start_run_action start_run_action_type;
HPX_REGISTER_ACTION(start_run_action_type);

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
		set_pivot();
		printf("Adjusting velocities:\n");
		auto diag = diagnostics();
		space_vector dv;
		dv[XDIM] = -diag.grid_sum[sx_i] / diag.grid_sum[rho_i];
		dv[YDIM] = -diag.grid_sum[sy_i] / diag.grid_sum[rho_i];
		dv[ZDIM] = -diag.grid_sum[sz_i] / diag.grid_sum[rho_i];
		this->velocity_inc(dv);
		save_to_file("scf.chk");
	}

	printf("Starting...\n");
	solve_gravity(false);

	real output_dt = 1.0 / OUTPUT_FREQ;

	printf("OMEGA = %e\n", grid::get_omega());
	real& t = current_time;
	integer step_num = 0;

	auto fut_ptr = me.get_ptr();
	node_server* root_ptr = fut_ptr.get();

	output_cnt = root_ptr->get_rotation_count() / output_dt;
	hpx::future<void> diag_fut = hpx::make_ready_future();
	hpx::future<void> step_fut = hpx::make_ready_future();
	profiler_output(stdout);
	real bench_start, bench_stop;
	while (true) {
		auto time_start = std::chrono::high_resolution_clock::now();
		if (root_ptr->get_rotation_count() / output_dt >= output_cnt) {
			if (step_num != 0) {

				char* fname;

				if (asprintf(&fname, "X.%i.chk", int(output_cnt))) {
				}
				save_to_file(fname);
				free(fname);
				if (asprintf(&fname, "X.%i.silo", int(output_cnt))) {
				}
				output(fname, output_cnt);
				free(fname);
				//	SYSTEM(std::string("cp *.dat ./dat_back/\n"));
			}
			++output_cnt;

		}
		if (step_num == 0) {
			bench_start = MPI_Wtime();
		}

		//	break;
		auto ts_fut = hpx::async([=]() {return timestep_driver();});
		step();
		real dt = GET(ts_fut);
		auto diags = diagnostics();

		const real dx = diags.secondary_com[XDIM] - diags.primary_com[XDIM];
		const real dy = diags.secondary_com[YDIM] - diags.primary_com[YDIM];
		const real dx_dot = diags.secondary_com_dot[XDIM] - diags.primary_com_dot[XDIM];
		const real dy_dot = diags.secondary_com_dot[YDIM] - diags.primary_com_dot[YDIM];
		const real theta = atan2(dy, dx);
		real omega = grid::get_omega();
		const real theta_dot = (dy_dot * dx - dx_dot * dy) / (dx * dx + dy * dy) - omega;
		const real w0 = grid::get_omega() * 100.0;
		const real theta_dot_dot = (2.0 * w0 * theta_dot + w0 * w0 * theta);
		real omega_dot;
		omega_dot = theta_dot_dot;
		omega += omega_dot * dt;
//		omega_dot += theta_dot_dot*dt;
		grid::set_omega(omega);

		double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - time_start).count();
		GET(step_fut);
		step_fut = hpx::async([=]() {
			FILE* fp = fopen( "step.dat", "at");
			fprintf(fp, "%i %e %e %e %e %e %e %e %e\n", int(step_num), double(t), double(dt), time_elapsed, rotational_time, theta, theta_dot, omega, omega_dot);
			fclose(fp);
		});
		printf("%i %e %e %e %e %e %e %e %e\n", int(step_num), double(t), double(dt), time_elapsed, rotational_time, theta, theta_dot, omega, omega_dot);

//		t += dt;
		++step_num;

		if (step_num % refinement_freq() == 0) {
			regrid(me.get_gid(), false);
			FILE* fp = fopen("profile.txt", "wt");
			profiler_output(fp);
			fclose(fp);
			//		set_omega_and_pivot();
			bench_stop = MPI_Wtime();
			if (scf || opts.bench) {
				printf("Total time = %e s\n", double(bench_stop - bench_start));
				break;
			}
		}
		//		set_omega_and_pivot();
		if (scf) {
			bench_stop = MPI_Wtime();
			printf("Total time = %e s\n", double(bench_stop - bench_start));
			break;
		}
	}
}
