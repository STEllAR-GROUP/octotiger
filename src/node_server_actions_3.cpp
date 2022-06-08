//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/future.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/problem.hpp"
#include "octotiger/real.hpp"
#include "octotiger/util.hpp"

#include <cerrno>

#include <hpx/include/lcos.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/include/util.hpp>
#include <hpx/collectives/broadcast.hpp>

#include <algorithm>
#include <array>
#include <cstdio>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

using send_gravity_boundary_action_type = node_server::send_gravity_boundary_action;
HPX_REGISTER_ACTION (send_gravity_boundary_action_type);

void node_client::send_gravity_boundary(gravity_boundary_type &&data, const geo::direction &dir, bool monopole, std::size_t cycle) const {
	hpx::apply<typename node_server::send_gravity_boundary_action>(get_unmanaged_gid(), std::move(data), dir, monopole, cycle);
}

void node_server::recv_gravity_boundary(gravity_boundary_type &&bdata, const geo::direction &dir, bool monopole, std::size_t cycle) {
	neighbor_gravity_type tmp;
	tmp.data = std::move(bdata);
	tmp.is_monopole = monopole;
	tmp.direction = dir;
	neighbor_gravity_channels[dir].set_value(std::move(tmp), cycle);
}

using send_gravity_expansions_action_type = node_server::send_gravity_expansions_action;
HPX_REGISTER_ACTION (send_gravity_expansions_action_type);

void node_server::recv_gravity_expansions(expansion_pass_type &&v) {
	parent_gravity_channel.set_value(std::move(v));
}

void node_client::send_gravity_expansions(expansion_pass_type &&data) const {
	hpx::apply<typename node_server::send_gravity_expansions_action>(get_unmanaged_gid(), std::move(data));
}

using send_gravity_multipoles_action_type = node_server::send_gravity_multipoles_action;
HPX_REGISTER_ACTION (send_gravity_multipoles_action_type);

void node_client::send_gravity_multipoles(multipole_pass_type &&data, const geo::octant &ci) const {
	hpx::apply<typename node_server::send_gravity_multipoles_action>(get_unmanaged_gid(), std::move(data), ci);
}

void node_server::recv_gravity_multipoles(multipole_pass_type &&v, const geo::octant &ci) {
	child_gravity_channels[ci].set_value(std::move(v));
}

using send_hydro_boundary_action_type = node_server::send_hydro_boundary_action;
HPX_REGISTER_ACTION (send_hydro_boundary_action_type);

void node_client::send_hydro_boundary(std::vector<real> &&data, const geo::direction &dir, std::size_t cycle) const {
	hpx::apply<typename node_server::send_hydro_boundary_action>(get_unmanaged_gid(), std::move(data), dir, cycle);
}

void node_server::recv_hydro_boundary(std::vector<real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_hydro_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_hydro_channels[dir].set_value(std::move(tmp), cycle);
}

using send_hydro_amr_boundary_action_type = node_server::send_hydro_amr_boundary_action;
HPX_REGISTER_ACTION (send_hydro_amr_boundary_action_type);

void node_client::send_hydro_amr_boundary(std::vector<real> &&data, const geo::direction &dir, std::size_t cycle) const {
	hpx::apply<typename node_server::send_hydro_amr_boundary_action>(get_unmanaged_gid(), std::move(data), dir, cycle);
}

void node_server::recv_hydro_amr_boundary(std::vector<real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_hydro_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_hydro_channels[dir].set_value(std::move(tmp), cycle);
}

using send_flux_check_action_type = node_server::send_flux_check_action;
HPX_REGISTER_ACTION (send_flux_check_action_type);

void node_client::send_flux_check(std::vector<real> &&data, const geo::direction &dir, std::size_t cycle) const {
	hpx::apply<typename node_server::send_flux_check_action>(get_unmanaged_gid(), std::move(data), dir, cycle);
}

void node_server::recv_flux_check(std::vector<real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_hydro_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_hydro_channels[dir].set_value(std::move(tmp), cycle);
}

using send_hydro_children_action_type = node_server::send_hydro_children_action;
HPX_REGISTER_ACTION (send_hydro_children_action_type);

void node_server::recv_hydro_children(std::vector<real> &&data, const geo::octant &ci, std::size_t cycle) {
	child_hydro_channels[ci].set_value(std::move(data), cycle);
}

void node_client::send_hydro_children(std::vector<real> &&data, const geo::octant &ci, std::size_t cycle) const {
	hpx::apply<typename node_server::send_hydro_children_action>(get_unmanaged_gid(), std::move(data), ci, cycle);
}

using send_hydro_flux_correct_action_type = node_server::send_hydro_flux_correct_action;
HPX_REGISTER_ACTION (send_hydro_flux_correct_action_type);

void node_client::send_hydro_flux_correct(std::vector<real> &&data, const geo::face &face, const geo::octant &ci) const {
	hpx::apply<typename node_server::send_hydro_flux_correct_action>(get_unmanaged_gid(), std::move(data), face, ci);
}

void node_server::recv_hydro_flux_correct(std::vector<real> &&data, const geo::face &face, const geo::octant &ci) {
	const geo::quadrant index(ci, face.get_dimension());
	if (face >= nieces.size()) {
		for (integer i = 0; i != 100; ++i) {
			print("NIECE OVERFLOW\n");
		}
		abort();
	}
	if (nieces[face] != 1) {
		for (integer i = 0; i != 100; ++i) {
			print("Big bad flux error  %c %i\n", is_refined ? 'R' : 'N', int(nieces[face]));
		}
		abort();
	}

	niece_hydro_channels[face][index].set_value(std::move(data));
}

using line_of_centers_action_type = node_server::line_of_centers_action;
HPX_REGISTER_ACTION (line_of_centers_action_type);

future<line_of_centers_t> node_client::line_of_centers(const std::pair<space_vector, space_vector> &line) const {
	return hpx::async<typename node_server::line_of_centers_action>(get_unmanaged_gid(), line);
}

void output_line_of_centers(FILE *fp, const line_of_centers_t &loc) {
	for (integer i = 0; i != loc.size(); ++i) {
		fprintf(fp, "%e ", loc[i].first);
		for (integer j = 0; j != opts().n_fields + NGF; ++j) {
			fprintf(fp, "%e ", loc[i].second[j]);
		}
		fprintf(fp, "\n");
	}
}

line_of_centers_t node_server::line_of_centers(const std::pair<space_vector, space_vector> &line) const {
	line_of_centers_t return_line;
	if (is_refined) {
		std::array<future<line_of_centers_t>, NCHILD> futs;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs[ci] = children[ci].line_of_centers(line);
		}
		std::map<real, std::vector<real>> map;
		for (auto &&fut : futs) {
			auto tmp = fut.get();
			for (integer ii = 0; ii != tmp.size(); ++ii) {
				if (map.find(tmp[ii].first) == map.end()) {
					map.emplace(std::move(tmp[ii]));
				}
			}
		}
		return_line.resize(map.size());
		std::move(map.begin(), map.end(), return_line.begin());
	} else {
		return_line = grid_ptr->line_of_centers(line);
	}

	return return_line;
}

void line_of_centers_analyze(const line_of_centers_t &loc, real omega, std::pair<real, real> &rho1_max, std::pair<real, real> &rho2_max,
		std::pair<real, real> &l1_phi, std::pair<real, real> &l2_phi, std::pair<real, real> &l3_phi, real &rho1_phi, real &rho2_phi) {

	constexpr integer spc_ac_i = spc_i;
	constexpr integer spc_ae_i = spc_i + 1;
	constexpr integer spc_dc_i = spc_i + 2;
	constexpr integer spc_de_i = spc_i + 3;
	constexpr integer spc_vac_i = spc_i + 4;

	for (auto &l : loc) {
		ASSERT_NONAN(l.first);
		for (integer f = 0; f != opts().n_fields + NGF; ++f) {
			ASSERT_NONAN(l.second[f]);
		}
	}

	rho1_max.second = rho2_max.second = 0.0;
	integer rho1_maxi, rho2_maxi;
	///	print( "LOCSIZE %i\n", loc.size());
	for (integer i = 0; i != loc.size(); ++i) {
		const real x = loc[i].first;
		const real rho = loc[i].second[rho_i];
		const real pot = loc[i].second[pot_i];
		if (loc[i].second[spc_ac_i] + loc[i].second[spc_ae_i] > 0.5 * loc[i].second[rho_i]) {
			//		print("%e %e\n", x, rho);
			if (rho1_max.second < rho) {
				//	print( "!\n");
				rho1_max.second = rho;
				rho1_max.first = x;
				rho1_maxi = i;
				real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
				rho1_phi = phi_eff;
			}
		}
	}
	for (integer i = 0; i != loc.size(); ++i) {
		const real x = loc[i].first;
		if (loc[i].second[spc_dc_i] + loc[i].second[spc_de_i] > 0.5 * loc[i].second[rho_i]) {
			const real rho = loc[i].second[rho_i];
			const real pot = loc[i].second[pot_i];
			if (rho2_max.second < rho) {
				rho2_max.second = rho;
				rho2_max.first = x;
				rho2_maxi = i;
				real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
				rho2_phi = phi_eff;
			}
		}
	}
	l1_phi.second = -std::numeric_limits<real>::max();
	l2_phi.second = -std::numeric_limits<real>::max();
	l3_phi.second = -std::numeric_limits<real>::max();
	for (integer i = 0; i != loc.size(); ++i) {
		const real x = loc[i].first;
		const real rho = loc[i].second[rho_i];
		const real pot = loc[i].second[pot_i];
		real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
		if (x > std::min(rho1_max.first, rho2_max.first) && x < std::max(rho1_max.first, rho2_max.first)) {
			if (phi_eff > l1_phi.second) {
				l1_phi.second = phi_eff;
				l1_phi.first = x;
			}
		} else if (std::abs(x) > std::abs(rho2_max.first) && x * rho2_max.first > 0.0) {
			if (phi_eff > l2_phi.second) {
				l2_phi.second = phi_eff;
				l2_phi.first = x;
			}
		} else if (std::abs(x) > std::abs(rho1_max.first)) {
			if (phi_eff > l3_phi.second) {
				l3_phi.second = phi_eff;
				l3_phi.first = x;
			}
		}
	}
}

void node_server::execute_solver(bool scf, node_count_type ngrids) {
	timings_.times_[timings::time_regrid] = 0.0;
	timings_.times_[timings::time_fmm] = 0.0;
	timings_.times_[timings::time_total] = 0.0;
	integer output_cnt { };
//	output_all("X", 0, false);

	if (!opts().hydro && !opts().radiation) {
//		diagnostics();
		if (!opts().disable_output) {
			output_all(this, "final", output_cnt, true);
		}
		if (get_analytic() != nullptr) {
      if (!opts().disable_analytic) { // Pure performance measurements - skip analytics 
          compare_analytic();
      }
      if (opts().gravity) {
        auto start_all_gravity = std::chrono::high_resolution_clock::now(); 
        auto min_duration = std::chrono::milliseconds::max();
        auto max_duration = std::chrono::milliseconds::min();
        for (int iteration = 0; iteration < opts().stop_step; iteration++) {
          std::cout << "Pure-gravity iteration " << iteration << std::endl;
          auto start = std::chrono::high_resolution_clock::now(); 
          solve_gravity(true, false);
          auto stop = std::chrono::high_resolution_clock::now(); 
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
          std::cout << "--> " << iteration + 1 << ". FMM iteration took: " << duration.count() << " ms" << std::endl; 
          if (duration.count() < min_duration.count())
            min_duration = duration;
          if (duration.count() > max_duration.count())
            max_duration = duration;

        }
        auto stop_all_gravity = std::chrono::high_resolution_clock::now(); 
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_all_gravity - start_all_gravity); 
        std::cout << "==> Overall execution time: " << duration.count() << " ms" << std::endl; 
        std::cout << "==> Average iteration execution time: " << duration.count() / opts().stop_step << " ms" << std::endl; 
        std::cout << "==> Minimal iteration execution time: " << min_duration.count() << " ms" << std::endl; 
        std::cout << "==> Maximal iteration execution time: " << max_duration.count() << " ms" << std::endl; 
			}
			if (!opts().disable_output) {
				output_all(this, "analytic", output_cnt, true);
			}
		}
		return;
	}
	if (scf) {
		run_scf(opts().data_dir);
		if (opts().eos == IPR) {
			print("Adjusting energy by SCF pressure:\n");
			this->energy_adj();
		}
		print("Adjusting velocities:\n");
		auto diag = diagnostics();
		space_vector dv;
		dv[XDIM] = -diag.grid_sum[sx_i] / diag.grid_sum[rho_i];
		dv[YDIM] = -diag.grid_sum[sy_i] / diag.grid_sum[rho_i];
		dv[ZDIM] = -diag.grid_sum[sz_i] / diag.grid_sum[rho_i];
		this->velocity_inc(dv);
	}
	if (opts().radiation) {
		if (opts().eos == WD && opts().problem == STAR) {
			print("Initialized radiation and cgs\n");
			set_cgs();
			erad_init();
		}
	}
	print("Starting run...\n");
	auto fut_ptr = me.get_ptr();
	node_server *root_ptr = GET(fut_ptr);
	if (!opts().output_filename.empty()) {
		diagnostics();
		solve_gravity(false, false);
		output_all(this, opts().output_filename, output_cnt, false);
		return;
	}

	if (opts().stop_step != 0) {
		print("Solving gravity\n");
		solve_gravity(false, false);
		ngrids = regrid(me.get_gid(), grid::get_omega(), -1, false);
	}

	real output_dt = opts().output_dt;

	print("OMEGA = %e, output_dt = %e\n", grid::get_omega(), output_dt);
	real &t = current_time;
	integer step_num = 0;

	output_cnt = root_ptr->get_rotation_count() / output_dt;
	print("%e %e\n", root_ptr->get_rotation_count(), output_dt);

	real bench_start, bench_stop;
	while (current_time < opts().stop_time) {
		timings::scope ts(timings_, timings::time_total);
		if (step_num > opts().stop_step)
			break;
		auto time_start = std::chrono::high_resolution_clock::now();
		auto diags = diagnostics();
		if (opts().stop_step == 0) {
			return;
		}
		if (!opts().disable_output && root_ptr->get_rotation_count() / output_dt >= output_cnt) {
			static bool first_call = true;
			if (opts().rewrite_silo || !first_call || (opts().restart_filename == "")) {
				print("doing silo out...\n");
				std::string fname = "X." + std::to_string(int(output_cnt));
				output_all(this, fname, output_cnt, first_call);
				if (opts().rewrite_silo) {
					print("Exiting after rewriting SILO\n");
					return;
				}
			}
			first_call = false;
			++output_cnt;

		}
		if (step_num == 0) {
			bench_start = hpx::chrono::high_resolution_clock::now() / 1e9;
		}

		real dt = 0;
		integer next_step = (std::min)(step_num + refinement_freq(), opts().stop_step + 1);
		real omega_dot = 0.0, omega = 0.0, theta = 0.0, theta_dot = 0.0;

		if ((opts().problem == DWD) && (step_num % refinement_freq() == 0)) {
			print("dwd step...\n");
			auto dt = GET(step(next_step - step_num));
			if (!opts().disable_diagnostics) {
				print("diagnostics...\n");
			}
			omega = grid::get_omega();

			const real dx = diags.com[1][XDIM] - diags.com[0][XDIM];
			const real dy = diags.com[1][YDIM] - diags.com[0][YDIM];
			const real dx_dot = diags.com_dot[1][XDIM] - diags.com_dot[0][XDIM];
			const real dy_dot = diags.com_dot[1][YDIM] - diags.com_dot[0][YDIM];
			theta = atan2(dy, dx);
			omega = grid::get_omega();
//			if (opts().variable_omega) {
//				theta_dot = (dy_dot * dx - dx_dot * dy) / (dx * dx + dy * dy) - omega;
//				const real w0 = grid::get_omega() * 10.0;
//				const real theta_dot_dot = (2.0 * w0 * theta_dot + w0 * w0 * theta);
//				omega_dot = theta_dot_dot;
//				omega += omega_dot * dt;
//			}
			print("New Omega = %e\n", omega);
		} else {
			print("normal step...\n");
			dt = GET(step(next_step - step_num));
			omega = grid::get_omega();
		}

		double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - time_start).count();

		// run output on separate thread
		if (!opts().disable_output) {
			hpx::threads::run_as_os_thread([=]() {
				FILE *fp = fopen((opts().data_dir + "step.dat").c_str(), "at");
				if (fp == NULL) {
					print( "Unable to open step.dat for writing %s\n", std::strerror(errno));
				} else {
					const auto vr = sqrt(sqr(dt_.ur[sx_i]) + sqr(dt_.ur[sy_i]) + sqr(dt_.ur[sz_i])) / dt_.ur[0];
					const auto vl = sqrt(sqr(dt_.ul[sx_i]) + sqr(dt_.ul[sy_i]) + sqr(dt_.ul[sz_i])) / dt_.ul[0];
					fprintf(fp, "%i %e %e %e %e %e %e %e %e %e %e %e %e %i %i %i %i\n", int(next_step - 1), double(t), double(dt_.dt), time_elapsed, rotational_time, dt_.x, dt_.y, dt_.z, dt_.a, dt_.ur[0], dt_.ul[0], vr,vl,dt_.dim, int(ngrids.total), int(ngrids.leaf), int(ngrids.amr_bnd));
					fclose(fp);
				}
			});     // do not wait for it to finish
		}

		hpx::threads::run_as_os_thread(
				[=]() {
					const auto vr = sqrt(sqr(dt_.ur[sx_i]) + sqr(dt_.ur[sy_i]) + sqr(dt_.ur[sz_i])) / dt_.ur[0];
					const auto vl = sqrt(sqr(dt_.ul[sx_i]) + sqr(dt_.ul[sy_i]) + sqr(dt_.ul[sz_i])) / dt_.ul[0];
					print("TS %i:: t: %e, dt: %e, time_elapsed: %e, rotational_time: %e, x: %e, y: %e, z: %e, ",
						int(next_step - 1), double(t), double(dt_.dt), time_elapsed, rotational_time,
						dt_.x, dt_.y, dt_.z);
					print("a: %e, ur: %e, ul: %e, vr: %e, vl: %e, dim: %i, ngrids: %i, leafs: %i, amr_boundaries: %i\n", 
						dt_.a, dt_.ur[0], dt_.ul[0], vr, vl, dt_.dim, int(ngrids.total),
						int(ngrids.leaf), int(ngrids.amr_bnd));
				});     // do not wait for output to finish

		step_num = next_step;

		if (step_num % refinement_freq() == 0) {
			real new_floor = opts().refinement_floor;
			if (opts().ngrids > 0) {
				new_floor *= std::pow(real(ngrids.total) / real(opts().ngrids), 2);
				print("Old refinement floor = %e\n", opts().refinement_floor);
				print("New refinement floor = %e\n", new_floor);
			}

			ngrids = regrid(me.get_gid(), omega, new_floor, false);

			// run output on separate thread
			auto need_break = hpx::threads::run_as_os_thread([&]() {
				//		set_omega_and_pivot();
				bench_stop = hpx::chrono::high_resolution_clock::now() / 1e9;
				if (scf || opts().bench) {
					print("Total time = %e s\n", double(bench_stop - bench_start));
					if (!opts().disable_output) {
						FILE *fp = fopen((opts().data_dir + "bench.dat").c_str(), "at");
						fprintf(fp, "%i %e\n", int(options::all_localities.size()), double(bench_stop - bench_start));
						fclose(fp);
					}
					return true;
				}
				return false;
			});
			if (GET(need_break))
				break;
		}
		if (scf) {
			bench_stop = hpx::chrono::high_resolution_clock::now() / 1e9;
			print("Total time = %e s\n", double(bench_stop - bench_start));
			break;
		}
	}

	bench_stop = hpx::chrono::high_resolution_clock::now() / 1e9;
	{
		timings::scope ts(timings_, timings::time_compare_analytic);

		if (!opts().disable_output) {
			print("doing silo out...\n");
			output_all(this, "final", output_cnt, true);
		}

		if (!opts().disable_analytic && get_analytic() != nullptr) {
			compare_analytic();
			if (opts().gravity) {
				solve_gravity(true, false);
			}
			if (!opts().disable_output) {
				output_all(this, "analytic", output_cnt, true);
			}
		}
	}

	if (opts().bench && !opts().disable_output) {
		hpx::threads::run_as_os_thread([&]() {

			FILE *fp = fopen((opts().data_dir + "scaling.dat").c_str(), "at");
			const auto nproc = options::all_localities.size();
			fprintf(fp, "%i %e\n", int(nproc), float(bench_stop - bench_start));
			fclose(fp);
		}).get();
	}
}

using step_action_type = node_server::step_action;
HPX_REGISTER_ACTION (step_action_type);

future<real> node_client::step(integer steps) const {
	return hpx::async<typename node_server::step_action>(get_unmanaged_gid(), steps);
}

void node_server::refined_step() {

//#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
//	static hpx::util::itt::string_handle sh("node_server::refined_step");
//	hpx::util::itt::task t(hpx::get_thread_itt_domain(), sh);
//#endif

	timings::scope ts(timings_, timings::time_computation);
	const real dx = TWO * grid::get_scaling_factor() / real(INX << my_location.level());
	real cfl0 = opts().cfl;

	real a = std::numeric_limits<real>::min();
	all_hydro_bounds();
	timestep_t tstep;
	tstep.dt = std::numeric_limits<real>::max();
	local_timestep_channels[NCHILD].set_value(tstep);
	auto dt_fut = global_timestep_channel.get_future();

	for (integer rk = 0; rk < NRK; ++rk) {

		{
			timings::scope ts(timings_, timings::time_fmm);
			compute_fmm(DRHODT, false);
			compute_fmm(RHO, true);
		}
		rk == NRK - 1 ? energy_hydro_bounds() : all_hydro_bounds();

	}

	dt_ = GET(dt_fut);
	update();
	if (opts().radiation) {
		compute_radiation(dt_.dt, grid_ptr->get_omega());
		all_hydro_bounds();
	}

}

future<void> node_server::nonrefined_step() {
//#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
//	static hpx::util::itt::string_handle sh("node_server::nonrefined_step");
//	hpx::util::itt::task t(hpx::get_thread_itt_domain(), sh);
//#endif

	timings::scope ts(timings_, timings::time_computation);

	real cfl0 = opts().cfl;
	dt_.dt = ZERO;

	all_hydro_bounds();

	grid_ptr->store();
	future<void> fut = hpx::make_ready_future();

	hpx::shared_future<timestep_t> dt_fut = global_timestep_channel.get_future();

	for (integer rk = 0; rk < NRK; ++rk) {

		fut = fut.then(hpx::launch::async_policy(hpx::threads::thread_priority::boost),
		hpx::util::annotated_function(
				[rk, cfl0, this, dt_fut](future<void> f) {
					GET(f);
					timestep_t a = grid_ptr->compute_fluxes();
					future<void> fut_flux = exchange_flux_corrections();
					fut_flux.get();
//					a = std::max(a, grid_ptr->compute_positivity_speed_limit());
					if (rk == 0) {
						const real dx = TWO * grid::get_scaling_factor() / real(INX << my_location.level());
						dt_ = a;
						dt_.dt = cfl0 * dx / a.a;
						if (opts().stop_time > 0.0) {
							const real maxdt = (opts().stop_time - current_time) / (refinement_freq() - (step_num % refinement_freq()));
							dt_.dt = std::min(dt_.dt, maxdt);
						}
						local_timestep_channels[NCHILD].set_value(dt_);
					}
					grid_ptr->compute_sources(current_time, rotational_time);
					grid_ptr->compute_dudt();
					compute_fmm(DRHODT, false);
					if (rk == 0) {
						dt_ = GET(dt_fut);
					}
					grid_ptr->next_u(rk, current_time, dt_.dt);
					compute_fmm(RHO, true);
					rk == NRK - 1 ? energy_hydro_bounds() : all_hydro_bounds();
				}, "node_server::nonrefined_step::compute_fluxes"));
	}

	return fut.then(hpx::launch::sync, hpx::util::annotated_function( [this](future<void> &&f) {

		GET(f);
		update();
		if (opts().radiation) {
			compute_radiation(dt_.dt, grid_ptr->get_omega());
			all_hydro_bounds();
		}

	}, "node_server::nonrefined_step::update" )
	);
}

void node_server::update() {
	grid_ptr->dual_energy_update();
	current_time += dt_.dt;
	if (grid::get_omega() != 0.0) {
		rotational_time += grid::get_omega() * dt_.dt;
	} else {
		rotational_time = current_time;
	}
}

future<real> node_server::local_step(integer steps) {
	future<real> fut = hpx::make_ready_future(0.0);
	for (integer i = 0; i != steps; ++i) {

		{
			std::lock_guard<hpx::mutex> lock(node_count_mtx);
			cumulative_node_count.total++;
			if (!is_refined) {
				cumulative_node_count.leaf++;
			}
			constexpr auto full_set = geo::octant::full_set();
			if (amr_flags.size()) {
				for (auto &ci : full_set) {
					const auto &flags = amr_flags[ci];
					for (auto &dir : geo::direction::full_set()) {
						if (dir.is_face()) {
							if (flags[dir]) {
								cumulative_node_count.amr_bnd++;
							}
						}
					}
				}
			}
		}

		fut = fut.then(hpx::launch::async_policy(hpx::threads::thread_priority::boost), hpx::util::annotated_function([this, i, steps](future<void> fut) -> real {
			GET(fut);
			auto time_start = std::chrono::high_resolution_clock::now();
			auto next_dt = timestep_driver_descend();

			if (is_refined) {
				refined_step();
			} else {
				GET(nonrefined_step());
			}

			if (my_location.level() == 0) {
				double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - time_start).count();

				hpx::threads::run_as_os_thread([=]() {
					print("%i %e %e %e %e\n", int(step_num), double(current_time), double(dt_.dt), time_elapsed, rotational_time);
				});  // do not wait for output to finish
			}
			++step_num;
			GET(next_dt);
			return dt_.dt;
		}, "local_step::execute_step"));
	}
	return fut;
}

future<real> node_server::step(integer steps) {
	grid_ptr->set_coordinates();

	std::array<future<void>, NCHILD> child_futs;
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			child_futs[ci] = children[ci].step(steps);
		}
	}

	future<real> fut = local_step(steps);

	if (is_refined) {
		return hpx::dataflow(hpx::launch::sync, [this](future<real> dt_fut, future<std::array<future<void>, NCHILD>> &&f) {
			auto fi = GET(f); // propagate exceptions
			for (auto &f : fi) {
				GET(f);
			}
			return GET(dt_fut);
		}, std::move(fut), hpx::when_all(std::move(child_futs)));
	}

	return fut;
}

using timestep_driver_ascend_action_type = node_server::timestep_driver_ascend_action;
HPX_REGISTER_ACTION (timestep_driver_ascend_action_type);

void node_client::timestep_driver_ascend(timestep_t dt) const {
	hpx::apply<typename node_server::timestep_driver_ascend_action>(get_unmanaged_gid(), dt);
}

void node_server::timestep_driver_ascend(timestep_t dt) {
	global_timestep_channel.set_value(dt);
	if (is_refined) {
		for (auto &child : children) {
			child.timestep_driver_ascend(dt);
		}
	}
}

using set_local_timestep_action_type = node_server::set_local_timestep_action;
HPX_REGISTER_ACTION (set_local_timestep_action_type);

void node_client::set_local_timestep(integer idx, timestep_t dt) const {
	hpx::apply<typename node_server::set_local_timestep_action>(get_unmanaged_gid(), idx, dt);
}

void node_server::set_local_timestep(integer idx, timestep_t dt) {
	local_timestep_channels[idx].set_value(dt);
}

future<void> node_server::timestep_driver_descend() {
	if (is_refined) {
		std::array<future<timestep_t>, NCHILD + 1> futs;
		integer index = 0;
		for (auto &local_timestep : local_timestep_channels) {
			futs[index++] = local_timestep.get_future();
		}

		return hpx::dataflow(hpx::launch::sync, /*hpx::util::annotated_function(*/[this](std::array<future<timestep_t>, NCHILD + 1> dts_fut) {

			auto dts = hpx::util::unwrap(dts_fut);
			timestep_t dt;
			dt.dt = 1.0e+99;
			for (const auto &this_dt : dts) {
				if (this_dt.dt < dt.dt) {
					dt = this_dt;
				}
			}

			if (my_location.level() == 0) {
				timestep_driver_ascend(dt);
			} else {
				parent.set_local_timestep(my_location.get_child_index(), dt);
			}

			return;
		}/*, "node_server::timestep_driver_descend")*/, futs);
	} else {
		return local_timestep_channels[NCHILD].get_future().then(hpx::launch::sync, hpx::util::annotated_function([this](future<timestep_t> &&f) {
			timestep_t dt = GET(f);
			parent.set_local_timestep(my_location.get_child_index(), dt);
			return;
		}, "timestep_driver_descend::set_local_timestep")
		);
	}
}

using velocity_inc_action_type = node_server::velocity_inc_action;
HPX_REGISTER_ACTION (velocity_inc_action_type);

future<void> node_client::velocity_inc(const space_vector &dv) const {
	return hpx::async<typename node_server::velocity_inc_action>(get_gid(), dv);
}

void node_server::velocity_inc(const space_vector &dv) {
	if (is_refined) {
		std::array<future<void>, NCHILD> futs;
		integer index = 0;
		for (auto &child : children) {
			futs[index++] = child.velocity_inc(dv);
		}
		//       wait_all_and_propagate_exceptions(futs);
		for (auto &f : futs) {
			GET(f);
		}
	} else {
		grid_ptr->velocity_inc(dv);
	}
}

using energy_adj_action_type = node_server::energy_adj_action;
HPX_REGISTER_ACTION (energy_adj_action_type);

future<void> node_client::energy_adj() const {
        return hpx::async<typename node_server::energy_adj_action>(get_gid());
}

void node_server::energy_adj() {
        if (is_refined) {
                std::array<future<void>, NCHILD> futs;
                integer index = 0;
                for (auto &child : children) {
                        futs[index++] = child.energy_adj();
                }
                //       wait_all_and_propagate_exceptions(futs);
                for (auto &f : futs) {
                        GET(f);
                }
        } else {
                grid_ptr->energy_adj();
        }
}
#endif
