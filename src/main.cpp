#include <fenv.h>
#include "defs.hpp"

#include "node_server.hpp"
#include "node_client.hpp"
#include "future.hpp"
#include <chrono>
//#include "options.hpp"
#include <unistd.h>


HPX_PLAIN_ACTION(grid::set_omega, set_omega_action);
HPX_PLAIN_ACTION(grid::set_pivot, set_pivot_action);

void initialize() {
//#ifndef NDEBUG
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);
//#endif
}

HPX_PLAIN_ACTION(initialize, initialize_action);

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
	futs.clear();
	futs.reserve(localities.size());
	for (auto& locality : localities) {
		futs.push_back(hpx::async<set_omega_action>(locality, this_omega));
	}
	for (auto&& fut : futs) {
		GET(fut);
	}
}

void node_server::start_run() {
	if (current_time == 0) {
			run_scf();
		//	if (system("mkdir dat_back\n")) {
		//	}
	}

	printf("Starting...\n");
	solve_gravity(false);

	double output_dt = (real(2) * real(M_PI) / 1.323670e-01) / 100.0;
	integer output_cnt;

	real& t = current_time;
	integer step_num = 0;

	auto fut_ptr = me.get_ptr();
	node_server* root_ptr = fut_ptr.get();

	output_cnt = root_ptr->get_time() / output_dt;
	hpx::future<void> diag_fut = hpx::make_ready_future();
	hpx::future<void> step_fut = hpx::make_ready_future();
	while (true) {
		set_omega_and_pivot();
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
		t += dt;
		++step_num;

		const integer refinement_freq = integer(R_BW / cfl + 0.5);
		if (step_num % refinement_freq == 0) {
			regrid(me.get_gid(), false);
		}

	}
}

int hpx_main(int argc, char* argv[]) {
	printf( "Running\n");
//	options opts;

//	if (opts.process_options(argc, argv)) {

		auto all_locs = hpx::find_all_localities();
		std::list<hpx::future<void>> futs;
		for (auto i = all_locs.begin(); i != all_locs.end(); ++i) {
			futs.push_back(hpx::async<initialize_action>(*i));
		}
		for (auto i = futs.begin(); i != futs.end(); ++i) {
			GET(*i);
		}
		//#ifndef NDEBUG
//#endif
		node_client root_id = hpx::new_<node_server>(hpx::find_here());
		node_client root_client(root_id);

		if (argc == 1) {
			for (integer l = 0; l < MAX_LEVEL; ++l) {
				GET(root_client.regrid(root_client.get_gid(), false));
				printf("---------------Created Level %i---------------\n", int(l + 1));
			}
		} else {
			std::string fname(argv[1]);
			printf("Loading from %s...\n", fname.c_str());
			if (argc == 2) {
				GET(root_client.get_ptr())->load_from_file(fname);
			} else {
				std::string oname(argv[2]);
				GET(root_client.get_ptr())->load_from_file_and_output(fname, oname);
				printf("Converted %s to %s\n", fname.c_str(), oname.c_str());
				return hpx::finalize();
			}
			GET(root_client.regrid(root_client.get_gid(), true));
			printf("Done. \n");
		}

		std::vector<hpx::id_type> null_sibs(geo::direction::count());
		printf("Forming tree connections------------\n");
		GET(root_client.form_tree(root_client.get_gid(), hpx::invalid_id, null_sibs));
		printf("...done\n");

//	sleep(3);

		root_client.start_run().get();
	//}
	//root_client.unregister(node_location()).get();
	printf("Exiting...\n");
	return hpx::finalize();
}

