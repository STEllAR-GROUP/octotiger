#include <fenv.h>
#include "defs.hpp"

#include "node_server.hpp"
#include "node_client.hpp"
#include <chrono>
#include <unistd.h>

void initialize() {
#ifndef NDEBUG
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);
#endif
}

HPX_PLAIN_ACTION( initialize, initialize_action);

void node_server::start_run() {

	printf("Starting...\n");
	solve_gravity(false);

	double output_dt = (real(2) * real(M_PI) / DEFAULT_OMEGA) / 100.0;
//	double output_dt = 0.1;
	integer output_cnt;

	real& t = current_time;
	integer step_num = 0;

	auto fut_ptr = me.get_ptr();
	node_server* root_ptr = fut_ptr.get();

	output_cnt = root_ptr->get_time() / output_dt;
	hpx::future<void> diag_fut = hpx::make_ready_future();
	hpx::future<void> step_fut = hpx::make_ready_future();
	while (true) {
		if( step_num % 15 == 0 ) {
			regrid(me.get_gid());
		}
		auto time_start = std::chrono::high_resolution_clock::now();
		auto diags = diagnostics();
		diag_fut.get();
		diag_fut = hpx::async([=]() {
			FILE* fp = fopen( "diag.dat", "at");
			fprintf( fp, "%19.12e ", double(t));
			for( integer f = 0; f != NF; ++f) {
				fprintf( fp, "%19.12e ", double( diags.grid_sum[f] + diags.outflow_sum[f]));
				fprintf( fp, "%19.12e ", double(diags.outflow_sum[f]));
			}
			for( integer f = 0; f != NDIM; ++f) {
				fprintf( fp, "%19.12e ",double( diags.l_sum[f]));
			}
			fprintf( fp, "\n");
			fclose(fp);

			fp = fopen( "minmax.dat", "at");
			fprintf( fp, "%19.12e ", double(t));
			for( integer f = 0; f != NF; ++f) {
				fprintf( fp, "%19.12e ", double(diags.field_min[f]));
				fprintf( fp, "%19.12e ", double(diags.field_max[f]));
			}
			fprintf( fp, "\n");
			fclose(fp);
		});

//		if( step_num % 10 == 0 ) {
		if (t / output_dt >= output_cnt) {
			char* fname;
	//		if (output_cnt % 4 == 0) {
				printf("--- begin checkpoint ---\n");
				if (asprintf(&fname, "X.%i.chk", int(output_cnt))) {
				}
				save(0, std::string(fname));
				printf("--- end checkpoint ---\n");
				free(fname);

				if (asprintf(&fname, "X.%i.silo", int(output_cnt))) {
				}
				printf("--- begin output ---\n");
				output(std::string(fname));
				printf("--- end output ---\n");
				free(fname);
		//	}
			++output_cnt;

		}
		//	break;
		auto ts_fut = hpx::async([=]() {return timestep_driver();});
		step();
		real dt = ts_fut.get();

		double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
				std::chrono::high_resolution_clock::now() - time_start).count();
		step_fut.get();
		step_fut = hpx::async([=]() {
			FILE* fp = fopen( "step.dat", "at");
			fprintf(fp, "%i %e %e %e\n", int(step_num), double(t), double(dt), time_elapsed);
			fclose(fp);
		});
		printf("%i %e %e %e\n", int(step_num), double(t), double(dt), time_elapsed);
		t += dt;
		++step_num;

	}
}

int hpx_main(int argc, char* argv[]) {
	auto all_locs = hpx::find_all_localities();
	std::list<hpx::future<void>> futs;
	for (auto i = all_locs.begin(); i != all_locs.end(); ++i) {
		futs.push_back(hpx::async<initialize_action>(*i));
	}
	for (auto i = futs.begin(); i != futs.end(); ++i) {
		i->get();
	}
	//#ifndef NDEBUG
//#endif
	node_client root_id = hpx::new_<node_server>(hpx::find_here());
	node_client root_client(root_id);

	if (argc == 1) {
		for (integer l = 0; l < MAX_LEVEL; ++l) {
			root_client.regrid(root_client.get_gid()).get();
			printf("---------------Created Level %i---------------\n", int(l + 1));
		}
	} else {
		std::string fname(argv[1]);
		printf("Loading from %s...\n", fname.c_str());
		root_client.get_ptr().get()->load(fname, root_client);
		root_client.regrid(root_client.get_gid()).get();
		printf("Done. \n");
	}

	std::vector<hpx::id_type> null_sibs(NFACE);
	printf("Forming tree connections------------\n");
	root_client.form_tree(root_client.get_gid(), hpx::invalid_id, null_sibs).get();
	printf("...done\n");

//	sleep(3);

	root_client.start_run().get();

	//root_client.unregister(node_location()).get();
	printf("Exiting...\n");
	return hpx::finalize();
}

