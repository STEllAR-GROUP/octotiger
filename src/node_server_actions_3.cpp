#include "node_server.hpp"
#include "node_client.hpp"
#include "future.hpp"
#include "options.hpp"
#include "util.hpp"

#include <hpx/include/run_as.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

extern options opts;

typedef node_server::send_gravity_boundary_action send_gravity_boundary_action_type;
HPX_REGISTER_ACTION(send_gravity_boundary_action_type);

hpx::future<void> node_client::send_gravity_boundary(gravity_boundary_type&& data,
    const geo::direction& dir, bool monopole) const {
    return hpx::async<typename node_server::send_gravity_boundary_action>(get_gid(),
        std::move(data), dir, monopole);
}

void node_server::recv_gravity_boundary(gravity_boundary_type&& bdata,
    const geo::direction& dir, bool monopole) {
    neighbor_gravity_type tmp;
    tmp.data = std::move(bdata);
    tmp.is_monopole = monopole;
    tmp.direction = dir;
    neighbor_gravity_channels[dir].set_value(std::move(tmp));
}

typedef node_server::send_gravity_expansions_action send_gravity_expansions_action_type;
HPX_REGISTER_ACTION(send_gravity_expansions_action_type);

void node_server::recv_gravity_expansions(expansion_pass_type&& v) {
    parent_gravity_channel.set_value(std::move(v));
}

hpx::future<void> node_client::send_gravity_expansions(expansion_pass_type&& data) const {
    return hpx::async<typename node_server::send_gravity_expansions_action>(get_gid(),
        std::move(data));
}

typedef node_server::send_gravity_multipoles_action send_gravity_multipoles_action_type;
HPX_REGISTER_ACTION(send_gravity_multipoles_action_type);

hpx::future<void> node_client::send_gravity_multipoles(multipole_pass_type&& data,
    const geo::octant& ci) const {
    return hpx::async<typename node_server::send_gravity_multipoles_action>(get_gid(),
        std::move(data), ci);
}

void node_server::recv_gravity_multipoles(multipole_pass_type&& v,
    const geo::octant& ci) {
    child_gravity_channels[ci].set_value(std::move(v));
}

typedef node_server::send_hydro_boundary_action send_hydro_boundary_action_type;
HPX_REGISTER_ACTION(send_hydro_boundary_action_type);

hpx::future<void> node_client::send_hydro_boundary(std::vector<real>&& data,
    const geo::direction& dir) const {
    return hpx::async<typename node_server::send_hydro_boundary_action>(get_gid(),
        std::move(data), dir);
}

void node_server::recv_hydro_boundary(std::vector<real>&& bdata,
    const geo::direction& dir) {
    sibling_hydro_type tmp;
    tmp.data = std::move(bdata);
    tmp.direction = dir;
    sibling_hydro_channels[dir].set_value(std::move(tmp));
}

typedef node_server::send_hydro_children_action send_hydro_children_action_type;
HPX_REGISTER_ACTION(send_hydro_children_action_type);

void node_server::recv_hydro_children(std::vector<real>&& data, const geo::octant& ci) {
    child_hydro_channels[ci].set_value(std::move(data));
}

hpx::future<void> node_client::send_hydro_children(std::vector<real>&& data,
    const geo::octant& ci) const {
    return hpx::async<typename node_server::send_hydro_children_action>(get_gid(),
        std::move(data), ci);
}

typedef node_server::send_hydro_flux_correct_action send_hydro_flux_correct_action_type;
HPX_REGISTER_ACTION(send_hydro_flux_correct_action_type);

hpx::future<void> node_client::send_hydro_flux_correct(std::vector<real>&& data,
    const geo::face& face,
    const geo::octant& ci) const {
    return hpx::async<typename node_server::send_hydro_flux_correct_action>(get_gid(),
        std::move(data), face, ci);
}

void node_server::recv_hydro_flux_correct(std::vector<real>&& data, const geo::face& face,
    const geo::octant& ci) {
    const geo::quadrant index(ci, face.get_dimension());
    niece_hydro_channels[face][index].set_value(std::move(data));
}

typedef node_server::line_of_centers_action line_of_centers_action_type;
HPX_REGISTER_ACTION(line_of_centers_action_type);

hpx::future<line_of_centers_t> node_client::line_of_centers(
    const std::pair<space_vector, space_vector>& line) const {
    return hpx::async<typename node_server::line_of_centers_action>(get_gid(), line);
}

void output_line_of_centers(FILE* fp, const line_of_centers_t& loc) {
    for (integer i = 0; i != loc.size(); ++i) {
        fprintf(fp, "%e ", loc[i].first);
        for (integer j = 0; j != NF + NGF; ++j) {
            fprintf(fp, "%e ", loc[i].second[j]);
        }
        fprintf(fp, "\n");
    }
}

line_of_centers_t node_server::line_of_centers(
    const std::pair<space_vector, space_vector>& line) const {
    std::list < hpx::future < line_of_centers_t >> futs;
    line_of_centers_t return_line;
    if (is_refined) {
        for (integer ci = 0; ci != NCHILD; ++ci) {
            futs.push_back(children[ci].line_of_centers(line));
        }
        std::map<real, std::vector<real>> map;
        for (auto&& fut : futs) {
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

void line_of_centers_analyze(const line_of_centers_t& loc, real omega,
    std::pair<real, real>& rho1_max, std::pair<real, real>& rho2_max,
    std::pair<real, real>& l1_phi, std::pair<real, real>& l2_phi,
    std::pair<real, real>& l3_phi, real& rho1_phi, real& rho2_phi) {

    for (auto& l : loc) {
        ASSERT_NONAN(l.first);
        for (integer f = 0; f != NF + NGF; ++f) {
            ASSERT_NONAN(l.second[f]);
        }
    }

    rho1_max.second = rho2_max.second = 0.0;
    integer rho1_maxi, rho2_maxi;
    ///	printf( "LOCSIZE %i\n", loc.size());
    for (integer i = 0; i != loc.size(); ++i) {
        const real x = loc[i].first;
        const real rho = loc[i].second[rho_i];
        const real pot = loc[i].second[pot_i];
        //	printf( "%e %e\n", x, rho);
        if (rho1_max.second < rho) {
            //	printf( "!\n");
            rho1_max.second = rho;
            rho1_max.first = x;
            rho1_maxi = i;
            real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
            rho1_phi = phi_eff;
        }
    }
    for (integer i = 0; i != loc.size(); ++i) {
        const real x = loc[i].first;
        if (x * rho1_max.first < 0.0) {
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
    l1_phi.second = -std::numeric_limits < real > ::max();
    l2_phi.second = -std::numeric_limits < real > ::max();
    l3_phi.second = -std::numeric_limits < real > ::max();
    for (integer i = 0; i != loc.size(); ++i) {
        const real x = loc[i].first;
        const real rho = loc[i].second[rho_i];
        const real pot = loc[i].second[pot_i];
        real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
        if (x > std::min(rho1_max.first, rho2_max.first)
            && x < std::max(rho1_max.first, rho2_max.first)) {
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

typedef node_server::start_run_action start_run_action_type;
HPX_REGISTER_ACTION(start_run_action_type);

hpx::future<void> node_client::start_run(bool b) const {
    return hpx::async<typename node_server::start_run_action>(get_gid(), b);
}

void node_server::start_run(bool scf) {
    timings::scope ts(timings_, timings::time_total);
    integer output_cnt;

    if (!hydro_on) {
        save_to_file("X.chk");
        diagnostics();
        return;
    }
    printf("%e %e\n", grid::get_A(), grid::get_B());
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
    int ngrids = regrid(me.get_gid(), false);

    real output_dt = opts.output_dt;

    printf("OMEGA = %e, output_dt = %e\n", grid::get_omega(), output_dt);
    real& t = current_time;
    integer step_num = 0;

    auto fut_ptr = me.get_ptr();
    node_server* root_ptr = fut_ptr.get();

    output_cnt = root_ptr->get_rotation_count() / output_dt;
    hpx::future<void> diag_fut = hpx::make_ready_future();
    profiler_output (stdout);
    real bench_start, bench_stop;
    while (current_time < opts.stop_time) {
        if (step_num > opts.stop_step)
            break;

        auto time_start = std::chrono::high_resolution_clock::now();
        if (!opts.disable_output && root_ptr->get_rotation_count() / output_dt >= output_cnt) {
            //	if (step_num != 0) {

            char fname[33];    // 21 bytes for int (max) + some leeway
            sprintf(fname, "X.%i.chk", int(output_cnt));
            save_to_file(fname);

            sprintf(fname, "X.%i.silo", int(output_cnt));
            output(fname, output_cnt, false);

            //	SYSTEM(std::string("cp *.dat ./dat_back/\n"));
            //	}
            ++output_cnt;

        }
        if (step_num == 0) {
            bench_start = hpx::util::high_resolution_clock::now() / 1e9;
        }

        //	break;
        auto ts_fut = hpx::async([=]() {return timestep_driver();});
        step().get();
        real dt = ts_fut.get();
        real omega_dot = 0.0, omega = 0.0, theta = 0.0, theta_dot = 0.0;
        omega = grid::get_omega();
        if (opts.problem == DWD) {
            auto diags = diagnostics();

            const real dx = diags.secondary_com[XDIM] - diags.primary_com[XDIM];
            const real dy = diags.secondary_com[YDIM] - diags.primary_com[YDIM];
            const real dx_dot = diags.secondary_com_dot[XDIM]
                - diags.primary_com_dot[XDIM];
            const real dy_dot = diags.secondary_com_dot[YDIM]
                - diags.primary_com_dot[YDIM];
            theta = atan2(dy, dx);
            omega = grid::get_omega();
            theta_dot = (dy_dot * dx - dx_dot * dy) / (dx * dx + dy * dy) - omega;
            const real w0 = grid::get_omega() * 100.0;
            const real theta_dot_dot = (2.0 * w0 * theta_dot + w0 * w0 * theta);
            omega_dot = theta_dot_dot;
            omega += omega_dot * dt;
//            omega_dot += theta_dot_dot*dt;
            grid::set_omega(omega);
        }
        double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - time_start).count();

        // run output on separate thread
        if (!opts.disable_output)
        {
            hpx::threads::run_as_os_thread([=]()
            {
                FILE* fp = fopen( "step.dat", "at");
                fprintf(fp, "%i %e %e %e %e %e %e %e %e %i\n",
                    int(step_num), double(t), double(dt), time_elapsed, rotational_time,
                    theta, theta_dot, omega, omega_dot, int(ngrids));
                fclose(fp);
            });     // do not wait for it fo finish
        }

        printf("%i %e %e %e %e %e %e %e %e\n", int(step_num), double(t), double(dt),
            time_elapsed, rotational_time, theta, theta_dot, omega, omega_dot);

//		t += dt;
        ++step_num;

        if (step_num % refinement_freq() == 0) {
            ngrids = regrid(me.get_gid(), false);

            // run output on separate thread
            auto need_break = hpx::threads::run_as_os_thread([&]()
            {
                FILE* fp = fopen("profile.txt", "wt");
                profiler_output(fp);
                fclose(fp);

                //		set_omega_and_pivot();
                bench_stop = hpx::util::high_resolution_clock::now() / 1e9;
                if (scf || opts.bench) {
                    printf("Total time = %e s\n", double(bench_stop - bench_start));
                    FILE* fp = fopen("bench.dat", "at");
                    fprintf(fp, "%i %e\n", int(hpx::find_all_localities().size()),
                        double(bench_stop - bench_start));
                    fclose(fp);
                    return true;
                }
                return false;
            });
            if (need_break.get())
                break;
        }
        //		set_omega_and_pivot();
        if (scf) {
            bench_stop = hpx::util::high_resolution_clock::now() / 1e9;
            printf("Total time = %e s\n", double(bench_stop - bench_start));
            //	FILE* fp = fopen( "bench.dat", "at" );
            //	fprintf( fp, "%i %e\n", int(hpx::find_all_localities().size()), double(bench_stop - bench_start));
            //	fclose(fp);
            break;
        }
    }
    compare_analytic();
    output("final.silo", output_cnt, true);
}

typedef node_server::step_action step_action_type;
HPX_REGISTER_ACTION(step_action_type);

hpx::future<void> node_client::step() const {
    return hpx::async<typename node_server::step_action>(get_gid());
}

hpx::future<void> node_server::refined_step(std::vector<hpx::future<void>> child_futs) {
    timings::scope ts(timings_, timings::time_computation);
    const real dx = TWO * grid::get_scaling_factor() / real(INX << my_location.level());
    real cfl0 = cfl;

    // FIXME: is this correct? ('a' was never re-initialized for refined == true)
    real a = std::numeric_limits<real>::min();

    dt_ = cfl0 * dx / a;

    hpx::future<void> fut = all_hydro_bounds();
    for (integer rk = 0; rk < NRK; ++rk) {

        fut = fut.then(
            [rk, this](hpx::future<void> f)
            {
                f.get();        // propagate exceptions

                if (rk == 0) {
                    local_timestep_channel.set_value(dt_);
                }

                compute_fmm(DRHODT, false);

                if (rk == 0) {
                    dt_ = global_timestep_channel.get_future().get();
                }

                compute_fmm(RHO, true);
                return all_hydro_bounds();
            });
    }

    return hpx::dataflow(
        [this](hpx::future<void> f, std::vector<hpx::future<void>> children)
        {
            // propagate exceptions
            f.get();
            for (auto& f: children)
            {
                if (f.has_exception())
                    f.get();
            }

            grid_ptr->dual_energy_update();
            current_time += dt_;
            if (grid::get_omega() != 0.0) {
                rotational_time += grid::get_omega() * dt_;
            } else {
                rotational_time = current_time;
            }
            ++step_num;
        },
        std::move(fut), std::move(child_futs));
}

hpx::future<void> node_server::nonrefined_step() {
    timings::scope ts(timings_, timings::time_computation);

    real cfl0 = cfl;
    dt_ = ZERO;

    hpx::future<void> fut = all_hydro_bounds();

    grid_ptr->store();

    for (integer rk = 0; rk < NRK; ++rk) {

        fut = fut.then(
            [rk, cfl0, this](hpx::future<void> f)
            {
                f.get();        // propagate exceptions

                grid_ptr->reconstruct();
                real a = grid_ptr->compute_fluxes();

                hpx::future<void> fut_flux = exchange_flux_corrections();

                if (rk == 0) {
                    const real dx = TWO * grid::get_scaling_factor() /
                        real(INX << my_location.level());
                    dt_ = cfl0 * dx / a;
                    local_timestep_channel.set_value(dt_);
                }

                return fut_flux.then(
                    [rk, this](hpx::future<void> f)
                    {
                        f.get();        // propagate exceptions

                        grid_ptr->compute_sources(current_time);
                        grid_ptr->compute_dudt();

                        compute_fmm(DRHODT, false);

                        if (rk == 0) {
                            dt_ = global_timestep_channel.get_future().get();
                        }

                        grid_ptr->next_u(rk, current_time, dt_);

                        compute_fmm(RHO, true);
                        return all_hydro_bounds();
                    });
            });
    }

    return fut.then(
        [this](hpx::future<void> f)
        {
            f.get();        // propagate exceptions

            grid_ptr->dual_energy_update();

            current_time += dt_;
            if (grid::get_omega() != 0.0) {
                rotational_time += grid::get_omega() * dt_;
            } else {
                rotational_time = current_time;
            }
            ++step_num;
        });
}

hpx::future<void> node_server::step() {
    grid_ptr->set_coordinates();

    if (is_refined) {
        std::vector<hpx::future<void>> child_futs;
        child_futs.reserve(NCHILD);
        for (integer ci = 0; ci != NCHILD; ++ci) {
            child_futs.push_back(children[ci].step());
        }
        return refined_step(std::move(child_futs));
    }

    return nonrefined_step();
}

typedef node_server::timestep_driver_ascend_action timestep_driver_ascend_action_type;
HPX_REGISTER_ACTION(timestep_driver_ascend_action_type);

hpx::future<void> node_client::timestep_driver_ascend(real dt) const {
    return hpx::async<typename node_server::timestep_driver_ascend_action>(get_gid(), dt);
}

void node_server::timestep_driver_ascend(real dt) {
    global_timestep_channel.set_value(dt);
    if (is_refined) {
        std::vector<hpx::future<void>> futs;
        futs.reserve(children.size());
        for(auto& child: children) {
            futs.push_back(child.timestep_driver_ascend(dt));
        }
        wait_all_and_propagate_exceptions(futs);
    }
}

typedef node_server::timestep_driver_descend_action timestep_driver_descend_action_type;
HPX_REGISTER_ACTION(timestep_driver_descend_action_type);

hpx::future<real> node_client::timestep_driver_descend() const {
    return hpx::async<typename node_server::timestep_driver_descend_action>(get_gid());
}

hpx::future<real> node_server::timestep_driver_descend() {
    if (is_refined) {
        std::vector<hpx::future<real>> futs;
        futs.reserve(children.size() + 1);
        for(auto& child: children) {
            futs.push_back(child.timestep_driver_descend());
        }
        futs.push_back(local_timestep_channel.get_future());

        return hpx::dataflow(
            [](std::vector<hpx::future<real>> dts_fut) -> double
            {
                auto dts = hpx::util::unwrapped(dts_fut);
                return *std::min_element(dts.begin(), dts.end());
            },
            std::move(futs));
    } else {
        return local_timestep_channel.get_future();
    }
}

typedef node_server::timestep_driver_action timestep_driver_action_type;
HPX_REGISTER_ACTION(timestep_driver_action_type);

hpx::future<real> node_client::timestep_driver() const {
    return hpx::async<typename node_server::timestep_driver_action>(get_gid());
}

real node_server::timestep_driver() {
    const real dt = timestep_driver_descend().get();
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
        wait_all_and_propagate_exceptions(futs);
    } else {
        grid_ptr->velocity_inc(dv);
    }
}

