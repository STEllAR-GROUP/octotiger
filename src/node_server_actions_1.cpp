/*
 * node_server_actions_1.cpp
 *
 *  Created on: Sep 23, 2016
 *      Author: dmarce1
 */

#include "node_server.hpp"
#include "node_client.hpp"
#include "diagnostics.hpp"
#include "options.hpp"
#include "taylor.hpp"
#include "util.hpp"
#include "profiler.hpp"
#include <mpi.h>

typedef node_server::load_action load_action_type;
HPX_REGISTER_ACTION(load_action_type);

hpx::mutex rec_size_mutex;
integer rec_size = -1;

void set_locality_data(real omega, space_vector pivot, integer record_size) {
	grid::set_omega(omega);
	grid::set_pivot(pivot);
	rec_size = record_size;
}

hpx::id_type make_new_node(const node_location& loc, const hpx::id_type& _parent) {
	return hpx::new_<node_server>(hpx::find_here(), loc, _parent, ZERO, ZERO).get();
}

HPX_PLAIN_ACTION(make_new_node, make_new_node_action);
HPX_PLAIN_ACTION(set_locality_data, set_locality_data_action);

hpx::future<grid::output_list_type> node_client::load(integer i, const hpx::id_type& _me, bool do_o, std::string s) const {
	return hpx::async<typename node_server::load_action>(get_gid(), i, _me, do_o, s);
}

grid::output_list_type node_server::load(integer cnt, const hpx::id_type& _me, bool do_output, std::string filename) {
	FILE* fp;
	std::size_t read_cnt = 0;

	if (rec_size == -1 && my_location.level() == 0) {
		fp = fopen(filename.c_str(), "rb");
		if (fp == NULL) {
			printf("Failed to open file\n");
			abort();
		}
		fseek(fp, -sizeof(integer), SEEK_END);
		read_cnt += fread(&rec_size, sizeof(integer), 1, fp);
		fseek(fp, -4 * sizeof(real) - sizeof(integer), SEEK_END);
		real omega;
		space_vector pivot;
		read_cnt += fread(&omega, sizeof(real), 1, fp);
		for (auto& d : geo::dimension::full_set()) {
			read_cnt += fread(&(pivot[d]), sizeof(real), 1, fp);
		}
		fclose(fp);
		auto localities = hpx::find_all_localities();
		std::vector<hpx::future<void>> futs;
		futs.reserve(localities.size());
		for (auto& locality : localities) {
			futs.push_back(hpx::async<set_locality_data_action>(locality, omega, pivot, rec_size));
		}
		for (auto&& fut : futs) {
			fut.get();
		}
	}

	static auto localities = hpx::find_all_localities();
	me = _me;
	fp = fopen(filename.c_str(), "rb");
	char flag;
	fseek(fp, cnt * rec_size, SEEK_SET);
	read_cnt += fread(&flag, sizeof(char), 1, fp);
	std::vector<integer> counts(NCHILD);
	for (auto& this_cnt : counts) {
		read_cnt += fread(&this_cnt, sizeof(integer), 1, fp);
	}
	load_me(fp);
	fseek(fp, 0L, SEEK_END);
	integer total_nodes = ftell(fp) / rec_size;
	fclose(fp);
	std::list<hpx::future<grid::output_list_type>> futs;
	//printf( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n" );
	if (flag == '1') {
		is_refined = true;
		children.resize(NCHILD);
		for (auto& ci : geo::octant::full_set()) {
			integer loc_id = ((cnt * localities.size()) / (total_nodes + 1));
			children[ci] = hpx::async<make_new_node_action>(localities[loc_id], my_location.get_child(ci), me.get_gid());
			futs.push_back(children[ci].load(counts[ci], children[ci].get_gid(), do_output, filename));
		}
	} else if (flag == '0') {
		is_refined = false;
		children.clear();
	} else {
		printf("Corrupt checkpoint file\n");
		sleep(10);
		abort();
	}
	grid::output_list_type my_list;
	for (auto&& fut : futs) {
		if (do_output) {
			grid::merge_output_lists(my_list,fut.get());
		} else {
			fut.get();
		}
	}
	//printf( "***************************************\n" );
	if (!is_refined && do_output) {
		my_list = grid_ptr->get_output_list();
		//	grid_ptr = nullptr;
	}
//	hpx::async<inc_grids_loaded_action>(localities[0]).get();
	if (my_location.level() == 0) {
		if (do_output) {
			diagnostics();
			grid::output(my_list, "data.silo", current_time, get_rotation_count() * OUTPUT_FREQ);
		}
		printf("Loaded checkpoint file\n");
		my_list = decltype(my_list)();

	}
	return my_list;
}

typedef node_server::output_action output_action_type;
HPX_REGISTER_ACTION(output_action_type);

hpx::future<grid::output_list_type> node_client::output(std::string fname, int cycle) const {
	return hpx::async<typename node_server::output_action>(get_gid(), fname, cycle);
}

grid::output_list_type node_server::output(std::string fname, int cycle) const {
	if (is_refined) {
		std::list<hpx::future<grid::output_list_type>> futs;
		for (auto i = children.begin(); i != children.end(); ++i) {
			futs.push_back(i->output(fname, cycle));
		}
		auto i = futs.begin();
		grid::output_list_type my_list = i->get();
		for (++i; i != futs.end(); ++i) {
			grid::merge_output_lists(my_list, i->get());
		}

		if (my_location.level() == 0) {
//			hpx::apply([](const grid::output_list_type& olists, const char* filename) {
			printf("Outputing...\n");
			grid::output(my_list, fname, get_time(), cycle);
			printf("Done...\n");
			//		}, std::move(my_list), fname.c_str());
		}
		return my_list;

	} else {
		return grid_ptr->get_output_list();
	}

}

typedef node_server::regrid_gather_action regrid_gather_action_type;
HPX_REGISTER_ACTION(regrid_gather_action_type);

hpx::future<integer> node_client::regrid_gather(bool rb) const {
	return hpx::async<typename node_server::regrid_gather_action>(get_gid(), rb);
}

integer node_server::regrid_gather(bool rebalance_only) {
	integer count = integer(1);

	if (is_refined) {
		if (!rebalance_only) {
			/* Turning refinement off */
			if (refinement_flag == 0) {
				children.clear();
				is_refined = false;
				grid_ptr->set_leaf(true);
			}
		}

		if (is_refined) {
			std::list<hpx::future<integer>> futs;
			for (auto& child : children) {
				futs.push_back(child.regrid_gather(rebalance_only));
			}
			for (auto& ci : geo::octant::full_set()) {
				auto child_cnt = futs.begin()->get();
				futs.pop_front();
				child_descendant_count[ci] = child_cnt;
				count += child_cnt;
			}
		} else {
			for (auto& ci : geo::octant::full_set()) {
				child_descendant_count[ci] = 0;
			}
		}
	} else if (!rebalance_only) {
		//		if (grid_ptr->refine_me(my_location.level())) {
		if (refinement_flag != 0) {
			refinement_flag = 0;
			count += NCHILD;

			children.resize(NCHILD);
			std::vector<node_location> clocs(NCHILD);

			/* Turning refinement on*/
			is_refined = true;
			grid_ptr->set_leaf(false);

			for (auto& ci : geo::octant::full_set()) {
				child_descendant_count[ci] = 1;
				children[ci] = hpx::new_<node_server>(hpx::find_here(), my_location.get_child(ci), me, current_time, rotational_time);
				std::array<integer, NDIM> lb = { 2 * H_BW, 2 * H_BW, 2 * H_BW };
				std::array<integer, NDIM> ub;
				lb[XDIM] += (1 & (ci >> 0)) * (INX);
				lb[YDIM] += (1 & (ci >> 1)) * (INX);
				lb[ZDIM] += (1 & (ci >> 2)) * (INX);
				for (integer d = 0; d != NDIM; ++d) {
					ub[d] = lb[d] + (INX);
				}
				std::vector<real> outflows(NF, ZERO);
				if (ci == 0) {
					outflows = grid_ptr->get_outflows();
				}
				if (current_time > ZERO) {
					children[ci].set_grid(grid_ptr->get_prolong(lb, ub), std::move(outflows)).get();
				}
			}
		}
	}

	return count;
}

typedef node_server::regrid_scatter_action regrid_scatter_action_type;
HPX_REGISTER_ACTION(regrid_scatter_action_type);

hpx::future<void> node_client::regrid_scatter(integer a, integer b) const {
	return hpx::async<typename node_server::regrid_scatter_action>(get_gid(), a, b);
}

void node_server::regrid_scatter(integer a_, integer total) {
	refinement_flag = 0;
	std::list<hpx::future<void>> futs;
	if (is_refined) {
		integer a = a_;
		const auto localities = hpx::find_all_localities();
		++a;
		for (auto& ci : geo::octant::full_set()) {
			const integer loc_index = a * localities.size() / total;
			const auto child_loc = localities[loc_index];
			a += child_descendant_count[ci];
			const hpx::id_type id = children[ci].get_gid();
			integer current_child_id = hpx::naming::get_locality_id_from_gid(id.get_gid());
			auto current_child_loc = localities[current_child_id];
			if (child_loc != current_child_loc) {
				children[ci] = children[ci].copy_to_locality(child_loc);
			}
		}
		a = a_ + 1;
		for (auto& ci : geo::octant::full_set()) {
			futs.push_back(children[ci].regrid_scatter(a, total));
			a += child_descendant_count[ci];
		}
	}
	clear_family();
	for (auto&& fut : futs) {
		fut.get();
	}
}

typedef node_server::regrid_action regrid_action_type;
HPX_REGISTER_ACTION(regrid_action_type);

hpx::future<void> node_client::regrid(const hpx::id_type& g, bool rb) const {
	return hpx::async<typename node_server::regrid_action>(get_gid(), g, rb);
}

void node_server::regrid(const hpx::id_type& root_gid, bool rb) {
	assert(grid_ptr != nullptr);
	printf("-----------------------------------------------\n");
	if (!rb) {
		printf("checking for refinement\n");
		check_for_refinement();
	}
	printf("regridding\n");
	integer a = regrid_gather(rb);
	printf("rebalancing %i nodes\n", int(a));
	regrid_scatter(0, a);
	assert(grid_ptr != nullptr);
	std::vector<hpx::id_type> null_neighbors(geo::direction::count());
	printf("forming tree connections\n");
	form_tree(root_gid, hpx::invalid_id, null_neighbors);
	if (current_time > ZERO) {
		printf("solving gravity\n");
		solve_gravity(true);
	}
	printf("regrid done\n-----------------------------------------------\n");
}

typedef node_server::save_action save_action_type;
HPX_REGISTER_ACTION(save_action_type);

integer node_client::save(integer i, std::string s) const {
	return hpx::async<typename node_server::save_action>(get_gid(), i, s).get();
}

integer node_server::save(integer cnt, std::string filename) const {
	char flag = is_refined ? '1' : '0';
	FILE* fp = fopen(filename.c_str(), (cnt == 0) ? "wb" : "ab");
	fwrite(&flag, sizeof(flag), 1, fp);
	++cnt;
//	printf("                                   \rSaved %li sub-grids\r", (long int) cnt);
	integer value = cnt;
	std::array<integer, NCHILD> values;
	for (auto& ci : geo::octant::full_set()) {
		if (ci != 0 && is_refined) {
			value += child_descendant_count[ci - 1];
		}
		values[ci] = value;
		fwrite(&value, sizeof(value), 1, fp);
	}
	const integer record_size = save_me(fp) + sizeof(flag) + NCHILD * sizeof(integer);
	fclose(fp);
	if (is_refined) {
		for (auto& ci : geo::octant::full_set()) {
			cnt = children[ci].save(cnt, filename);
		}
	}

	if (my_location.level() == 0) {
		FILE* fp = fopen(filename.c_str(), "ab");
		real omega = grid::get_omega();
		space_vector pivot = grid::get_pivot();
		fwrite(&omega, sizeof(real), 1, fp);
		for (auto& d : geo::dimension::full_set()) {
			fwrite(&(pivot[d]), sizeof(real), 1, fp);
		}
		fwrite(&record_size, sizeof(integer), 1, fp);
		fclose(fp);
		printf("Saved %li grids to checkpoint file\n", (long int) cnt);
	}

	return cnt;
}

typedef node_server::set_aunt_action set_aunt_action_type;
HPX_REGISTER_ACTION(set_aunt_action_type);

hpx::future<void> node_client::set_aunt(const hpx::id_type& aunt, const geo::face& f) const {
	return hpx::async<typename node_server::set_aunt_action>(get_gid(), aunt, f);
}

void node_server::set_aunt(const hpx::id_type& aunt, const geo::face& face) {
	aunts[face] = aunt;
}

typedef node_server::set_grid_action set_grid_action_type;
HPX_REGISTER_ACTION(set_grid_action_type);

hpx::future<void> node_client::set_grid(std::vector<real>&& g, std::vector<real>&& o) const {
	return hpx::async<typename node_server::set_grid_action>(get_gid(), g, o);
}

void node_server::set_grid(const std::vector<real>& data, std::vector<real>&& outflows) {
	grid_ptr->set_prolong(data, std::move(outflows));
}

typedef node_server::solve_gravity_action solve_gravity_action_type;
HPX_REGISTER_ACTION(solve_gravity_action_type);

hpx::future<void> node_client::solve_gravity(bool ene) const {
	return hpx::async<typename node_server::solve_gravity_action>(get_gid(), ene);
}

void node_server::solve_gravity(bool ene) {
	std::list<hpx::future<void>> child_futs;
	for (auto& child : children) {
		child_futs.push_back(child.solve_gravity(ene));
	}
	compute_fmm(RHO, ene);
	for (auto&& fut : child_futs) {
		fut.get();
	}
}

typedef node_server::start_run_action start_run_action_type;
HPX_REGISTER_ACTION(start_run_action_type);

hpx::future<void> node_client::start_run(bool b) const {
	return hpx::async<typename node_server::start_run_action>(get_gid(), b);
}

extern options opts;

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
		//	if (step_num != 0) {

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
	//		}
			++output_cnt;

		}
		if (step_num == 0) {
			bench_start = MPI_Wtime();
		}

		//	break;
		auto ts_fut = hpx::async([=]() {return timestep_driver();});
		step();
		real dt = ts_fut.get();
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
		step_fut.get();
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
			                        FILE* fp = fopen( "bench.dat", "at" );
                        fprintf( fp, "%i %e\n", int(hpx::find_all_localities().size()), double(bench_stop - bench_start));
                        fclose(fp);

				break;
			}
		}
		//		set_omega_and_pivot();
		if (scf) {
			bench_stop = MPI_Wtime();
			printf("Total time = %e s\n", double(bench_stop - bench_start));
		//	FILE* fp = fopen( "bench.dat", "at" );
		//	fprintf( fp, "%i %e\n", int(hpx::find_all_localities().size()), double(bench_stop - bench_start));
		//	fclose(fp);
			break;
		}
	}
}

typedef node_server::step_action step_action_type;
HPX_REGISTER_ACTION(step_action_type);

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
	hpx::future<void> fut;
	hpx::future<void> fut_flux;

	fut = all_hydro_bounds();
	grid_ptr->store();

	for (integer rk = 0; rk < NRK; ++rk) {
		grid_ptr->reconstruct();
		a = grid_ptr->compute_fluxes();
		fut_flux = exchange_flux_corrections();
		if (rk == 0) {
			dt = cfl0 * dx / a;
			local_timestep_channel.set_value(dt);
		}
		fut_flux.get();
		grid_ptr->compute_sources(current_time);
		grid_ptr->compute_dudt();
		fut.get();
		compute_fmm(DRHODT, false);

		if (rk == 0) {
			dt = global_timestep_channel.get_future().get();
		}
		grid_ptr->next_u(rk, current_time, dt);

		compute_fmm(RHO, true);
		fut = all_hydro_bounds(rk == NRK - 1);
	}
	fut.get();
	grid_ptr->dual_energy_update();

	for (auto i = child_futs.begin(); i != child_futs.end(); ++i) {
		i->get();
	}
	current_time += dt;
	rotational_time += grid::get_omega() * dt;
	++step_num;
}

typedef node_server::timestep_driver_ascend_action timestep_driver_ascend_action_type;
HPX_REGISTER_ACTION(timestep_driver_ascend_action_type);

hpx::future<void> node_client::timestep_driver_ascend(real dt) const {
	return hpx::async<typename node_server::timestep_driver_ascend_action>(get_gid(), dt);
}

void node_server::timestep_driver_ascend(real dt) {
	global_timestep_channel.set_value(dt);
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

typedef node_server::timestep_driver_descend_action timestep_driver_descend_action_type;
HPX_REGISTER_ACTION(timestep_driver_descend_action_type);

hpx::future<real> node_client::timestep_driver_descend() const {
	return hpx::async<typename node_server::timestep_driver_descend_action>(get_gid());
}

real node_server::timestep_driver_descend() {
	real dt;
	if (is_refined) {
		dt = std::numeric_limits<real>::max();
		std::list < hpx::future < real >> futs;
		for (auto i = children.begin(); i != children.end(); ++i) {
			futs.push_back(i->timestep_driver_descend());
		}
		for (auto i = futs.begin(); i != futs.end(); ++i) {
			dt = std::min(dt, i->get());
		}
		dt = std::min(local_timestep_channel.get_future().get(), dt);
	} else {
		dt = local_timestep_channel.get_future().get();
	}
	return dt;
}

typedef node_server::timestep_driver_action timestep_driver_action_type;
HPX_REGISTER_ACTION(timestep_driver_action_type);

hpx::future<real> node_client::timestep_driver() const {
	return hpx::async<typename node_server::timestep_driver_action>(get_gid());
}

real node_server::timestep_driver() {
	const real dt = timestep_driver_descend();
	timestep_driver_ascend(dt);
	return dt;
}

typedef node_server::velocity_inc_action velocity_inc_action_type;
HPX_REGISTER_ACTION(velocity_inc_action_type);

hpx::future<void> node_client::velocity_inc(const space_vector& dv) const {
	return hpx::async<typename node_server::velocity_inc_action>(get_gid(), dv);
}

void node_server::velocity_inc(const space_vector& dv) {
	if (is_refined) {
		std::vector<hpx::future<void>> futs;
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.velocity_inc(dv));
		}
		for (auto&& fut : futs) {
			fut.get();
		}
	} else {
		grid_ptr->velocity_inc(dv);
	}
}

