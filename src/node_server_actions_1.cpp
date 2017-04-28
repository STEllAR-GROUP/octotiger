/*
 * node_server_actions_1.cpp
 *
 *  Created on: Sep 23, 2016
 *      Author: dmarce1
 */

#include "defs.hpp"
#include "container_device.hpp"
#include "diagnostics.hpp"
#include "future.hpp"
#include "node_client.hpp"
#include "node_server.hpp"
#include "options.hpp"
#include "profiler.hpp"
#include "taylor.hpp"
#include "set_locality_data.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <vector>

#include <hpx/include/lcos.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/lcos/broadcast.hpp>

#include <boost/iostreams/stream.hpp>

extern options opts;

typedef node_server::load_action load_action_type;
HPX_REGISTER_ACTION(load_action_type);

void set_locality_data(real omega, space_vector pivot)
{
    grid::set_omega(omega,false);
    grid::set_pivot(pivot);
}

HPX_REGISTER_ACTION(set_locality_data_action, set_locality_data_action);
HPX_REGISTER_BROADCAST_ACTION(set_locality_data_action)


void parallel_output_gather(grid::output_list_type);
void parallel_output_complete(std::string dirname, std::string fname, int cycle, bool analytic);
HPX_PLAIN_ACTION(node_server::parallel_output_complete, parallel_output_complete_action);


std::stack<grid::output_list_type> node_server::pending_output;

void node_server::parallel_output_gather(grid::output_list_type&& list) {
	static hpx::mutex mtx;
	if (!list.nodes.empty()) {
		std::lock_guard<hpx::mutex> lock(mtx);
		pending_output.push(std::move(list));
	}
}

void node_server::parallel_output_complete(std::string dirname, std::string fname, real tm, int cycle, bool analytic) {
	grid::output_list_type olist;
	while( !pending_output.empty()) {
		auto next_list = std::move(pending_output.top());
		pending_output.pop();
		grid::merge_output_lists(olist, std::move(next_list));
	}
	grid::output(std::move(olist), dirname, fname, tm, cycle, false);

}

hpx::future<grid::output_list_type> node_client::load(
    integer i, integer total, integer rec_size, bool do_o, std::string s) const {
    return hpx::async<typename node_server::load_action>(get_unmanaged_gid(), i, total, rec_size, do_o, s);
}

hpx::future<grid::output_list_type> node_server::load(
    integer cnt, integer total_nodes, integer rec_size, bool do_output, std::string filename)
{
    if (my_location.level() == 0)
        me = hpx::invalid_id;
    else
        me = this->get_unmanaged_id();

    char flag = '0';
    std::vector<integer> counts(NCHILD);

    // run output on separate thread
    std::size_t read_cnt = 0;
    hpx::threads::run_as_os_thread([&]() {
        FILE* fp = fopen(filename.c_str(), "rb");
        fseek(fp, cnt * rec_size, SEEK_SET);
        read_cnt += fread(&flag, sizeof(char), 1, fp);

        for (auto& this_cnt : counts)
        {
            read_cnt += fread(&this_cnt, sizeof(integer), 1, fp);
        }
      //  printf( "RECSIZE=%i\n", int(rec_size));
        load_me(fp, rec_size==65739);
        fclose(fp);
    }).get();

#ifdef RADIATION
    rad_grid_ptr = grid_ptr->get_rad_grid();
    rad_grid_ptr->sanity_check();
#endif

    std::array<hpx::future<grid::output_list_type>, NCHILD> futs;
    // printf( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n" );
    if (flag == '1') {
        is_refined = true;

        integer index = 0;
        for (auto const& ci : geo::octant::full_set()) {
            integer loc_id = ((cnt * options::all_localities.size()) / (total_nodes + 1));

            futs[index++] =
                hpx::new_<node_server>(options::all_localities[loc_id],
                    my_location.get_child(ci), me.get_gid(), ZERO, ZERO, step_num, hcycle, gcycle).then(
                        [this, ci, counts, do_output, total_nodes, rec_size, filename](hpx::future<hpx::id_type>&& fut)
                        {
                            children[ci] = fut.get();
                            return children[ci].load(counts[ci], total_nodes, rec_size, do_output, filename);
                        }
                    );

//             children[ci] = hpx::new_<node_server>(options::all_localities[loc_id],
//                 my_location.get_child(ci), me.get_gid(), ZERO, ZERO, step_num, hcycle, gcycle);
// #ifdef OCTOTIGER_RESTART_LOAD_SEQ
//             children[ci].load(counts[ci], total_nodes, rec_size, do_output, filename).get();
// #else
//             futs[index++] =
//                 children[ci].load(counts[ci], total_nodes, rec_size, do_output, filename);
// #endif
        }
    } else if (flag == '0') {
        is_refined = false;
        std::fill_n(children.begin(), NCHILD, node_client());
    } else {
        printf("Corrupt checkpoint file\n");
        //		sleep(10);
        hpx::this_thread::sleep_for(std::chrono::seconds(10));
        abort();
    }

    if (!is_refined && do_output)
    {
    	auto l = grid_ptr->get_output_list(false);
    	if( opts.parallel_silo ) {
    		parallel_output_gather(std::move(l));
    	}
        return hpx::make_ready_future(l);
    }


    return hpx::dataflow(
        [this, do_output, filename](std::array<hpx::future<grid::output_list_type>, NCHILD>&& futs)
        {
            grid::output_list_type my_list;
            for (auto&& fut : futs) {
                if (fut.valid()) {
                    if (do_output) {
                        grid::merge_output_lists(my_list, fut.get());
                    } else {
                        fut.get();
                    }
                }
            }
            if (my_location.level() == 0) {
			if (do_output) {
				auto silo_name = opts.output_filename;
				/* Skip for now, more interested in SILO */
				//	if (hydro_on && opts.problem == DWD) {
			//		diagnostics();
			//	}
				std::string this_fname;
				printf("Outputing...\n");
				if (opts.parallel_silo) {
					std::string dir_name = silo_name + std::string(".silo.data");
					if (system((std::string("mkdir -p ") + dir_name + std::string("\n")).c_str()) != 0) {
						abort_error();
					}
					const auto cycle = get_rotation_count();
					const auto sz = opts.all_localities.size();
					std::vector<hpx::future<void>> futs(sz);
					for (integer i = 0; i != sz; ++i) {
						this_fname = dir_name + std::string("/") + silo_name + std::string(".") + std::to_string(i) + std::string(".silo");
						futs[i] = hpx::async < parallel_output_complete_action > (opts.all_localities[i], opts.data_dir, this_fname, get_time(), cycle, false);
					}
					hpx::wait_all(futs);
					grid::output_header(opts.data_dir, silo_name, get_time(), cycle, false, opts.all_localities.size());
				} else {
					this_fname = silo_name + std::string(".silo");
					grid::output(
							my_list, opts.data_dir, this_fname, current_time, get_rotation_count() / opts.output_dt, false);
				}
			}
			printf("Done...\n");

			if( !opts.parallel_silo) {
			}
			printf("Loaded checkpoint file\n");
                my_list = decltype(my_list)();
            }

            return my_list;
        },
        std::move(futs));

//     grid::output_list_type my_list;
//     for (auto&& fut : futs) {
//         if (fut.valid()) {
//             if (do_output) {
//                 grid::merge_output_lists(my_list, fut.get());
//             } else {
//                 fut.get();
//             }
//         }
//     }
//     // printf( "***************************************\n" );
//     if (!is_refined && do_output) {
//         my_list = grid_ptr->get_output_list(false);
//         //	grid_ptr = nullptr;
//     }
    //	hpx::async<inc_grids_loaded_action>(localities[0]).get();
//     if (my_location.level() == 0) {
//         if (do_output) {
//             if (hydro_on && opts.problem == DWD) {
//                 diagnostics();
//             }
//             grid::output(
//                 my_list, "data.silo", current_time, get_rotation_count() / opts.output_dt, false);
//         }
//         printf("Loaded checkpoint file\n");
//         my_list = decltype(my_list)();
//     }
//
//     return my_list;
}

typedef node_server::output_action output_action_type;
HPX_REGISTER_ACTION(output_action_type);

hpx::future<grid::output_list_type> node_client::output(std::string dname,
    std::string fname, int cycle, bool flag) const {
    return hpx::async<typename node_server::output_action>(get_unmanaged_gid(), dname, fname, cycle, flag);
}

grid::output_list_type node_server::output(std::string dname, std::string fname, int cycle, bool analytic) const {
    if (is_refined) {
        std::array<hpx::future<grid::output_list_type>, NCHILD> futs;
        integer index = 0;
        for (auto i = children.begin(); i != children.end(); ++i) {
            futs[index++] = i->output(dname, fname, cycle, analytic);
        }

        auto i = futs.begin();
        grid::output_list_type my_list = i->get();
		if (opts.parallel_silo) {
			parallel_output_gather(std::move(my_list));
			for (++i; i != futs.end(); ++i) {
				i->get();
			}
		} else {
			for (++i; i != futs.end(); ++i) {
				grid::merge_output_lists(my_list, i->get());
			}
		}
		if (my_location.level() == 0) {
			std::string this_fname;
			printf("Outputing...\n");
			if (opts.parallel_silo) {
				std::string this_dname = dname + fname + std::string(".silo.data/");
                //printf("node_server::output (mkdir): this_dname('%s')\n", this_dname.c_str());
				if (system((std::string("mkdir -p ") + this_dname + std::string("\n")).c_str()) != 0) {
					abort_error();
				}
				const auto sz = opts.all_localities.size();
				std::vector<hpx::future<void>> futs(sz);
				for (integer i = 0; i != sz; ++i) {
					this_fname = fname + std::string(".") + std::to_string(i) + std::string(".silo");
					futs[i] = hpx::async<parallel_output_complete_action>(opts.all_localities[i], this_dname, this_fname, get_time(), cycle, analytic);
				}
				hpx::wait_all(futs);
				grid::output_header(this_dname, fname, get_time(), cycle, analytic, opts.all_localities.size());
			} else {
				this_fname = fname + std::string(".silo");
				grid::output(my_list, dname, this_fname, get_time(), cycle, analytic);
			}
			printf("Done...\n");
		}
        return my_list;
    } else {
    	auto l = grid_ptr->get_output_list(analytic);
		if (opts.parallel_silo) {
			parallel_output_gather(std::move(l));
		}
		return l;
    }
}

typedef node_server::regrid_gather_action regrid_gather_action_type;
HPX_REGISTER_ACTION(regrid_gather_action_type);

hpx::future<integer> node_client::regrid_gather(bool rb) const {
    return hpx::async<typename node_server::regrid_gather_action>(get_unmanaged_gid(), rb);
}

integer node_server::regrid_gather(bool rebalance_only) {
    integer count = integer(1);

    if (is_refined) {
        if (!rebalance_only) {
            /* Turning refinement off */
            if (refinement_flag == 0) {
                std::fill_n(children.begin(), NCHILD, node_client());
                is_refined = false;
                grid_ptr->set_leaf(true);
            }
        }

        if (is_refined) {
            std::array<hpx::future<integer>, NCHILD> futs;
            integer index = 0;
            for (auto& child : children) {
                futs[index++] = child.regrid_gather(rebalance_only);
            }
            auto futi = futs.begin();
            for (auto const& ci : geo::octant::full_set()) {
                auto child_cnt = futi->get();
                ++futi;
                child_descendant_count[ci] = child_cnt;
                count += child_cnt;
            }
        } else {
            for (auto const& ci : geo::octant::full_set()) {
                child_descendant_count[ci] = 0;
            }
        }
    } else if (!rebalance_only) {
        //		if (grid_ptr->refine_me(my_location.level())) {
        if (refinement_flag != 0) {
            refinement_flag = 0;
            count += NCHILD;

            /* Turning refinement on*/
            is_refined = true;
            grid_ptr->set_leaf(false);

            for (auto& ci : geo::octant::full_set()) {
                child_descendant_count[ci] = 1;

//                 children[ci] = create_child(hpx::find_here(), ci);

            }
        }
    }

    return count;
}

hpx::future<hpx::id_type> node_server::create_child(hpx::id_type const& locality, integer ci)
{
    return hpx::new_ < node_server
            > (hpx::find_here(), my_location.get_child(ci), me, current_time, rotational_time, step_num, hcycle, gcycle).then(
        [this, ci](hpx::future<hpx::id_type>&& child_idf)
        {
            hpx::id_type child_id = child_idf.get();
            node_client child = child_id;
            {
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
                if (current_time > ZERO)
                {
                    std::vector<real> prolong;
                    {
                        std::unique_lock<hpx::lcos::local::spinlock> lk(prolong_mtx);
                        prolong = grid_ptr->get_prolong(lb, ub);
                    }
                    child.set_grid(std::move(prolong), std::move(outflows)).get();
                }
            }
#ifdef RADIATION
            {
                std::array<integer, NDIM> lb = { 2 * R_BW, 2 * R_BW, 2 * R_BW };
                std::array<integer, NDIM> ub;
                lb[XDIM] += (1 & (ci >> 0)) * (INX);
                lb[YDIM] += (1 & (ci >> 1)) * (INX);
                lb[ZDIM] += (1 & (ci >> 2)) * (INX);
                for (integer d = 0; d != NDIM; ++d) {
                    ub[d] = lb[d] + (INX);
                }
            /*	std::vector<real> outflows(NF, ZERO);
                if (ci == 0) {
                    outflows = grid_ptr->get_outflows();
                }*/
                if (current_time > ZERO) {
                    std::vector<real> prolong;
                    {
                        std::unique_lock<hpx::lcos::local::spinlock> lk(prolong_mtx);
                        prolong = rad_grid_ptr->get_prolong(lb, ub);
                    }
                    child.set_rad_grid(std::move(prolong)/*, std::move(outflows)*/).get();
                }
            }
#endif
            return child_id;
        });
}

typedef node_server::regrid_scatter_action regrid_scatter_action_type;
HPX_REGISTER_ACTION(regrid_scatter_action_type);

hpx::future<void> node_client::regrid_scatter(integer a, integer b) const {
    return hpx::async<typename node_server::regrid_scatter_action>(get_unmanaged_gid(), a, b);
}

hpx::future<void> node_server::regrid_scatter(integer a_, integer total) {
    refinement_flag = 0;
    std::array<hpx::future<void>, geo::octant::count()> futs;
    if (is_refined) {
        integer a = a_;
        ++a;
        integer index = 0;
        for (auto& ci : geo::octant::full_set()) {
            const integer loc_index = a * options::all_localities.size() / total;
            const auto child_loc = options::all_localities[loc_index];
            if (children[ci].empty())
            {
                futs[index++] = create_child(child_loc, ci).then(
                    [this, ci, a, total](hpx::future<hpx::id_type>&& child)
                    {
                        children[ci] = child.get();
                        return children[ci].regrid_scatter(a, total);
                    }
                );
            }
            else
            {
                const hpx::id_type id = children[ci].get_gid();
                integer current_child_id = hpx::naming::get_locality_id_from_gid(id.get_gid());
                auto current_child_loc = options::all_localities[current_child_id];
                if (child_loc != current_child_loc)
                {
                    futs[index++] = children[ci].copy_to_locality(child_loc).then(
                        [this, ci, a, total](hpx::future<hpx::id_type>&& child)
                        {
                            children[ci] = child.get();
                            return children[ci].regrid_scatter(a, total);
                        }
                    );
                }
                else
                {
                    futs[index++] = children[ci].regrid_scatter(a, total);
                }
            }
            a += child_descendant_count[ci];
        }
    }
    clear_family();
    if( is_refined ) {
    	return hpx::when_all(futs);
    } else {
    	return hpx::make_ready_future();
    }
}

integer node_server::regrid(const hpx::id_type& root_gid, real omega, real new_floor, bool rb) {
    timings::scope ts(timings_, timings::time_regrid);
    hpx::util::high_resolution_timer timer;
    assert(grid_ptr != nullptr);
    printf("-----------------------------------------------\n");
    if (!rb) {
        printf("checking for refinement\n");
        check_for_refinement(omega,  new_floor).get();
    }
    printf("regridding\n");
    real tstart = timer.elapsed();
    integer a = regrid_gather(rb);
    real tstop = timer.elapsed();
    printf( "Regridded tree in %f seconds\n", real(tstop - tstart));
    printf("rebalancing %i nodes\n", int(a));
    tstart = timer.elapsed();
    regrid_scatter(0, a).get();
    tstop = timer.elapsed();
    printf( "Rebalanced tree in %f seconds\n", real(tstop - tstart));
    assert(grid_ptr != nullptr);
    std::vector<hpx::id_type> null_neighbors(geo::direction::count());
    tstart = timer.elapsed();
    printf("forming tree connections\n");
    form_tree(hpx::unmanaged(root_gid), hpx::invalid_id, null_neighbors).get();
    tstop = timer.elapsed();
    printf( "Formed tree in %f seconds\n", real(tstop - tstart));
    if (current_time > ZERO) {
        printf("solving gravity\n");
        solve_gravity(true);
    }
    double elapsed = timer.elapsed();
    printf("regrid done in %f seconds\n---------------------------------------\n", elapsed);
#ifdef OCTOTIGER_USE_NODE_CACHE
    node_client::cycle_node_cache();
#endif
    return a;
}

typedef node_server::save_action save_action_type;
HPX_REGISTER_ACTION(save_action_type);

hpx::future<void> node_client::save(integer i, std::string s) const {
    return hpx::async<typename node_server::save_action>(get_unmanaged_gid(), i, s);
}

std::map<integer, std::vector<char> > node_server::save_local(integer& cnt, std::string const& filename, hpx::future<void>& child_fut) const {

    std::map<integer, std::vector<char> > result;
    char flag = is_refined ? '1' : '0';
    integer my_cnt = cnt;

    // Call save on children that are non local, for all
    // locals, get the pointer
    std::vector<hpx::future<void>> child_futs;
    std::vector<hpx::util::tuple<integer, integer, hpx::future<node_server*>>> local_children;
    if (is_refined)
    {
        child_futs.resize(NCHILD);
        local_children.reserve(NCHILD);
        integer i = cnt + 1;
        for (auto& ci : geo::octant::full_set())
        {
            if (!children[ci].is_local())
            {
                child_futs[ci] = children[ci].save(i, filename);
            }
            else
            {
                local_children.emplace_back(ci, i, children[ci].get_ptr());
            }
            i += child_descendant_count[ci];
        }
    }

    std::vector<char> out_buffer;
    out_buffer.reserve(4096 * 5);
    {
        typedef hpx::util::container_device<std::vector<char> > io_device_type;
        boost::iostreams::stream<io_device_type> strm(out_buffer);

        // determine record size
        write(strm, flag);
        integer value = ++cnt;
        std::array<integer, NCHILD> values;
        for (auto const& ci : geo::octant::full_set()) {
            if (ci != 0 && is_refined) {
                value += child_descendant_count[ci - 1];
            }
            values[ci] = value;
            write(strm, value);
        }
        save_me(strm);
    }
    result.emplace(my_cnt, std::move(out_buffer));

    if (is_refined) {
        for(auto& child: local_children)
        {
            auto ci = hpx::util::get<0>(child);
            auto cnt = hpx::util::get<1>(child);
            auto cc = hpx::util::get<2>(child).get();
            auto child_result = cc->save_local(cnt, filename, child_futs[ci]);
            for (auto && d : child_result) {
                result.emplace(std::move(d));
            }
        }
        child_fut = hpx::when_all(child_futs);
    }

    return result;
}

hpx::future<void> node_server::save(integer cnt, std::string const& filename) const
{
    // Create file and save metadata if location == 0
    if (my_location.level() == 0)
    {
        // run output on separate thread
        hpx::threads::run_as_os_thread([&]() {
            FILE *fp = fopen(filename.c_str(), "wb");
            fclose(fp);
        }).get();
    }

    hpx::future<void> child_fut;
    std::map<integer, std::vector<char> > result = save_local(cnt, filename, child_fut);

    // run output on separate thread
    auto fut = hpx::threads::run_as_os_thread([&]() {
        // write all of the buffers to file
        integer record_size = 0;
        FILE* fp = fopen(filename.c_str(), "rb+");
        for (auto const& d : result) {
            if (record_size == 0) {
                record_size = d.second.size();
            }
            else {
                assert(record_size == d.second.size());
            }
            fseek(fp, record_size * d.first, SEEK_SET);
            fwrite(d.second.data(), sizeof(char), d.second.size(), fp);
        }

        if (my_location.level() == 0) {
            std::size_t total = 1;
            for (auto& ci: geo::octant::full_set())
            {
                total += child_descendant_count[ci];
            }
            fseek(fp, record_size * total, SEEK_SET);
            real omega = grid::get_omega();
            fwrite(&omega, sizeof(real), 1, fp);
            space_vector pivot = grid::get_pivot();
            for (auto const& d : geo::dimension::full_set()) {
                real tmp = pivot[d];
                fwrite(&tmp, sizeof(real), 1, fp);
            }
            fwrite(&record_size, sizeof(integer), 1, fp);
            printf("Saved %li grids to checkpoint file\n", (long int) total);
        }
        fclose(fp);
    });

    return hpx::when_all(fut, child_fut);
}

typedef node_server::set_aunt_action set_aunt_action_type;
HPX_REGISTER_ACTION(set_aunt_action_type);

hpx::future<void> node_client::set_aunt(const hpx::id_type& aunt, const geo::face& f) const {
    return hpx::async<typename node_server::set_aunt_action>(get_unmanaged_gid(), aunt, f);
}

void node_server::set_aunt(const hpx::id_type& aunt, const geo::face& face) {
    aunts[face] = aunt;
}

typedef node_server::set_grid_action set_grid_action_type;
HPX_REGISTER_ACTION(set_grid_action_type);

hpx::future<void> node_client::set_grid(std::vector<real>&& g, std::vector<real>&& o) const {
    return hpx::async<typename node_server::set_grid_action>(get_unmanaged_gid(), std::move(g), std::move(o));
}

void node_server::set_grid(const std::vector<real>& data, std::vector<real>&& outflows) {
    grid_ptr->set_prolong(data, std::move(outflows));
}

typedef node_server::solve_gravity_action solve_gravity_action_type;
HPX_REGISTER_ACTION(solve_gravity_action_type);

hpx::future<void> node_client::solve_gravity(bool ene) const {
    return hpx::async<typename node_server::solve_gravity_action>(get_unmanaged_gid(), ene);
}

void node_server::solve_gravity(bool ene) {
    if (!gravity_on) {
        return;
    }
    std::array<hpx::future<void>, NCHILD> child_futs;
    if (is_refined)
    {
        integer index = 0;;
        for (auto& child : children) {
            child_futs[index++] = child.solve_gravity(ene);
        }
    }
    compute_fmm(RHO, ene);
    if( is_refined ) {
    	wait_all_and_propagate_exceptions(child_futs);
    }
}
