#include <fenv.h>
#include "defs.hpp"

#include "node_server.hpp"
#include "node_client.hpp"
#include "future.hpp"
#include <chrono>
#include "options.hpp"
#include <unistd.h>
#include "problem.hpp"

options opts;
integer output_cnt;

bool gravity_on = true;
bool hydro_on = true;

HPX_PLAIN_ACTION(grid::set_omega, set_omega_action);
HPX_PLAIN_ACTION(grid::set_pivot, set_pivot_action);

void initialize(options _opts) {
	opts = _opts;
//#ifndef NDEBUG
	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);
//#endif
	grid::set_scaling_factor(opts.xscale);
	grid::set_max_level(opts.max_level);

	if (opts.problem == DWD) {
		set_problem(scf_binary);
		set_refine_test(refine_test_bibi);
	} else if (opts.problem == STAR) {
		set_problem(star);
		set_refine_test(refine_test_bibi);
		/*} else if (opts.problem == OLD_SCF) {
		set_refine_test(refine_test_bibi);
		set_problem(init_func_type([=](real a, real b, real c, real dx) {
			return old_scf(a,b,c,opts.omega,opts.core_thresh_1,opts.core_thresh_2, dx);
		}));
		if (!opts.found_restart_file) {
			if (opts.omega < ZERO) {
				printf("Must specify omega for bibi polytrope\n");
				throw;
			}
			if (opts.core_thresh_1 < ZERO) {
				printf("Must specify core_thresh_1 for bibi polytrope\n");
				throw;
			}
			if (opts.core_thresh_2 < ZERO) {
				printf("Must specify core_thresh_2 for bibi polytrope\n");
				throw;
			}
		}*/
	} else if (opts.problem == SOLID_SPHERE) {
		hydro_on = false;
		set_problem(init_func_type([](real x, real y, real z, real dx) {
			return solid_sphere(x,y,z,dx,0.5);
		}));
	} else {
		printf("No problem specified\n");
		throw;
	}
	node_server::set_gravity(gravity_on);
	node_server::set_hydro(hydro_on);
}

HPX_PLAIN_ACTION(initialize, initialize_action);

real OMEGA;
void node_server::set_omega_and_pivot() {
	auto localities = hpx::find_all_localities();
	space_vector pivot = grid_ptr->center_of_mass();
	std::vector<hpx::future<void>> futs;
	futs.reserve(localities.size());
	for (auto& locality : localities) {
		if (current_time == ZERO) {
			futs.push_back(hpx::async<set_pivot_action>(locality, pivot));
		}
	}
	for (auto&& fut : futs) {
		GET(fut);
	}
	real this_omega = find_omega();
	OMEGA = this_omega;
	futs.clear();
	futs.reserve(localities.size());
	for (auto& locality : localities) {
		futs.push_back(hpx::async<set_omega_action>(locality, this_omega));
	}
	for (auto&& fut : futs) {
		GET(fut);
	}
}

void node_server::start_run(bool scf) {
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

int hpx_main(int argc, char* argv[]) {
	printf("Running\n");
	auto test_fut = hpx::async([]() {
//		while(1){hpx::this_thread::yield();}
		});
	test_fut.get();

	try {
		if (opts.process_options(argc, argv)) {

			auto all_locs = hpx::find_all_localities();
			std::list<hpx::future<void>> futs;
			for (auto i = all_locs.begin(); i != all_locs.end(); ++i) {
				futs.push_back(hpx::async<initialize_action>(*i, opts));
			}
			for (auto i = futs.begin(); i != futs.end(); ++i) {
				GET(*i);
			}

			node_client root_id = hpx::new_<node_server>(hpx::find_here());
			node_client root_client(root_id);

			if (opts.found_restart_file) {
				set_problem(null_problem);
				const std::string fname = opts.restart_filename;
				printf("Loading from %s...\n", fname.c_str());
				if (opts.output_only) {
					const std::string oname = opts.output_filename;
					GET(root_client.get_ptr())->load_from_file_and_output(fname, oname);
				} else {
					GET(root_client.get_ptr())->load_from_file(fname);
					GET(root_client.regrid(root_client.get_gid(), true));
				}
				printf("Done. \n");
			} else {
				for (integer l = 0; l < opts.max_level; ++l) {
					GET(root_client.regrid(root_client.get_gid(), false));
					printf("---------------Created Level %i---------------\n", int(l + 1));
				}
			}

			std::vector<hpx::id_type> null_sibs(geo::direction::count());
			printf("Forming tree connections------------\n");
			GET(root_client.form_tree(root_client.get_gid(), hpx::invalid_id, null_sibs));
			if (gravity_on) {
				//real tstart = MPI_Wtime();
				GET(root_client.solve_gravity(false));
			//	printf("Gravity Solve Time = %e\n", MPI_Wtime() - tstart);
			}
			printf("...done\n");

			if (!opts.output_only) {
				set_problem(null_problem);
				root_client.start_run(opts.problem == DWD && !opts.found_restart_file).get();
			}
		}
	} catch (...) {

	}
	printf("Exiting...\n");
	return hpx::finalize();
}

