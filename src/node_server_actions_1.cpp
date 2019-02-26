/*
 * node_server_actions_1.cpp
 *
 *  Created on: Sep 23, 2016
 *      Author: dmarce1
 */

#include "octotiger/config.hpp"

#include "octotiger/container_device.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/diagnostics.hpp"
#include "octotiger/future.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_registry.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/profiler.hpp"
#include "octotiger/taylor.hpp"

#include <hpx/include/lcos.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/lcos/broadcast.hpp>

#include <boost/iostreams/stream.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <vector>

typedef node_server::regrid_gather_action regrid_gather_action_type;
HPX_REGISTER_ACTION(regrid_gather_action_type);

future<node_count_type> node_client::regrid_gather(bool rb) const {
	return hpx::async<typename node_server::regrid_gather_action>(get_unmanaged_gid(), rb);
}

node_count_type node_server::regrid_gather(bool rebalance_only) {
	node_registry::delete_(my_location);
	node_count_type count;
	count.total = 1;
	count.leaf = is_refined ? 0 : 1;
	std::vector<hpx::future<void>> kfuts;
	if (is_refined) {
		if (!rebalance_only) {
			/* Turning refinement off */
			if (refinement_flag == 0) {
				for( int i = 0; i < NCHILD; i++ ) {
					kfuts.push_back(children[i].kill());
				}
				std::fill_n(children.begin(), NCHILD, node_client());
				is_refined = false;
			}
		}

		if (is_refined) {
			std::array<future<node_count_type>, NCHILD> futs;
			integer index = 0;
			for (auto& child : children) {
				futs[index++] = child.regrid_gather(rebalance_only);
			}
			auto futi = futs.begin();
			for (auto const& ci : geo::octant::full_set()) {
				const auto child_cnt = futi->get();
				++futi;
				child_descendant_count[ci] = child_cnt.total;
				count.leaf += child_cnt.leaf;
				count.total += child_cnt.total;
			}
		} else {
			count.leaf = 1;
			for (auto const& ci : geo::octant::full_set()) {
				child_descendant_count[ci] = 0;
			}
		}
	} else if (!rebalance_only) {
		if (refinement_flag != 0) {
			refinement_flag = 0;
			count.total += NCHILD;
			count.leaf += NCHILD - 1;

			/* Turning refinement on*/
			is_refined = true;

			for (auto& ci : geo::octant::full_set()) {
				child_descendant_count[ci] = 1;

			}
		}
	}
	grid_ptr->set_leaf(!is_refined);
	hpx::wait_all(kfuts);
	return count;
}

future<hpx::id_type> node_server::create_child(hpx::id_type const& locality, integer ci) {
	return hpx::new_<node_server>(hpx::find_here(), my_location.get_child(ci), me, current_time, rotational_time, step_num,
			hcycle, rcycle, gcycle).then([this, ci](future<hpx::id_type>&& child_idf)
	{
		hpx::id_type child_id = child_idf.get();
		node_client child = child_id;
		{
			std::array<integer, NDIM> lb = {2 * H_BW, 2 * H_BW, 2 * H_BW};
			std::array<integer, NDIM> ub;
			lb[XDIM] += (1 & (ci >> 0)) * (INX);
			lb[YDIM] += (1 & (ci >> 1)) * (INX);
			lb[ZDIM] += (1 & (ci >> 2)) * (INX);
			for (integer d = 0; d != NDIM; ++d) {
				ub[d] = lb[d] + (INX);
			}
			std::vector<real> outflows(opts().n_fields, ZERO);
			if (ci == 0) {
				outflows = grid_ptr->get_outflows_raw();
			}
			if (current_time > ZERO)
			{
				std::vector<real> prolong;
				{
					std::unique_lock<hpx::lcos::local::spinlock> lk(prolong_mtx);
					prolong = grid_ptr->get_prolong(lb, ub);
				}
				GET(child.set_grid(std::move(prolong), std::move(outflows)));
			}
		}
		if( opts().radiation ) {
			std::array<integer, NDIM> lb = {2 * R_BW, 2 * R_BW, 2 * R_BW};
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
		return child_id;
	});
}

typedef node_server::regrid_scatter_action regrid_scatter_action_type;
HPX_REGISTER_ACTION(regrid_scatter_action_type);

future<void> node_client::regrid_scatter(integer a, integer b) const {
	return hpx::async<typename node_server::regrid_scatter_action>(get_unmanaged_gid(), a, b);
}

void node_server::regrid_scatter(integer a_, integer total) {
	position = a_;
	refinement_flag = 0;
	std::array<future<void>, geo::octant::count()> futs;
	if (is_refined) {
		integer a = a_;
		++a;
		integer index = 0;
		for (auto& ci : geo::octant::full_set()) {
			const integer loc_index = a * options::all_localities.size() / total;
			const auto child_loc = options::all_localities[loc_index];
			if (children[ci].empty()) {
				futs[index++] = create_child(child_loc, ci).then([this, ci, a, total](future<hpx::id_type>&& child)
				{
					children[ci] = GET(child);
					GET(children[ci].regrid_scatter(a, total));
				});
			} else {
				const hpx::id_type id = children[ci].get_gid();
				integer current_child_id = hpx::naming::get_locality_id_from_gid(id.get_gid());
				auto current_child_loc = options::all_localities[current_child_id];
				if (child_loc != current_child_loc) {
					futs[index++] = children[ci].copy_to_locality(child_loc).then(
							[this, ci, a, total](future<hpx::id_type>&& child)
							{
								children[ci] = GET(child);
								GET(children[ci].regrid_scatter(a, total));
							});
				} else {
					futs[index++] = children[ci].regrid_scatter(a, total);
				}
			}
			a += child_descendant_count[ci];
		}
	}
	clear_family();
	if (is_refined) {
		for (auto& f : futs) {
			GET(f);
		}
	}
}

node_count_type node_server::regrid(const hpx::id_type& root_gid, real omega, real new_floor, bool rb,bool grav_energy_comp) {
	timings::scope ts(timings_, timings::time_regrid);
	hpx::util::high_resolution_timer timer;
	assert(grid_ptr != nullptr);
	printf("-----------------------------------------------\n");
	if (!rb) {
		printf("checking for refinement\n");
		check_for_refinement(omega, new_floor);
	} else {
		node_registry::clear();
	}
	printf("regridding\n");
	real tstart = timer.elapsed();
	auto a = regrid_gather(rb);
	real tstop = timer.elapsed();
	printf("Regridded tree in %f seconds\n", real(tstop - tstart));
	printf("rebalancing %i nodes with %i leaves\n", int(a.total), int(a.leaf));
	tstart = timer.elapsed();
	regrid_scatter(0, a.total);
	tstop = timer.elapsed();
	printf("Rebalanced tree in %f seconds\n", real(tstop - tstart));
	assert(grid_ptr != nullptr);
	tstart = timer.elapsed();
	printf("forming tree connections\n");
	a.amr_bnd = form_tree(hpx::unmanaged(root_gid));
	printf( "%i amr boundaries\n", a.amr_bnd);
	tstop = timer.elapsed();
	printf("Formed tree in %f seconds\n", real(tstop - tstart));
	printf("solving gravity\n");
	solve_gravity(grav_energy_comp, !opts().output_filename.empty());
	double elapsed = timer.elapsed();
	printf("regrid done in %f seconds\n---------------------------------------\n", elapsed);
	return a;
}

typedef node_server::set_aunt_action set_aunt_action_type;
HPX_REGISTER_ACTION(set_aunt_action_type);

future<void> node_client::set_aunt(const hpx::id_type& aunt, const geo::face& f) const {
	return hpx::async<typename node_server::set_aunt_action>(get_unmanaged_gid(), aunt, f);
}

void node_server::set_aunt(const hpx::id_type& aunt, const geo::face& face) {
	if (aunts[face].get_gid() != hpx::invalid_id) {
		printf("AUNT ALREADY SET\n");
		abort();
	}
	aunts[face] = aunt;
}

typedef node_server::set_grid_action set_grid_action_type;
HPX_REGISTER_ACTION(set_grid_action_type);

future<void> node_client::set_grid(std::vector<real>&& g, std::vector<real>&& o) const {
	return hpx::async<typename node_server::set_grid_action>(get_unmanaged_gid(), std::move(g), std::move(o));
}

void node_server::set_grid(const std::vector<real>& data, std::vector<real>&& outflows) {
	grid_ptr->set_prolong(data, std::move(outflows));
}

typedef node_server::solve_gravity_action solve_gravity_action_type;
HPX_REGISTER_ACTION(solve_gravity_action_type);

future<void> node_client::solve_gravity(bool ene, bool aonly) const {
	return hpx::async<typename node_server::solve_gravity_action>(get_unmanaged_gid(), ene, aonly);
}

void node_server::solve_gravity(bool ene, bool aonly) {
	if (!opts().gravity) {
		return;
	}
	std::array<future<void>, NCHILD> child_futs;
	if (is_refined) {
		integer index = 0;
		;
		for (auto& child : children) {
			child_futs[index++] = child.solve_gravity(ene, aonly);
		}
	}
	compute_fmm(RHO, ene, aonly);
	if (is_refined) {
		//	wait_all_and_propagate_exceptions(child_futs);
		for (auto& f : child_futs) {
			GET(f);
		}
	}
}
