//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
#include <hpx/collectives/broadcast.hpp>

#include <boost/iostreams/stream.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <vector>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

using amr_error_action_type = node_server::amr_error_action;
HPX_REGISTER_ACTION(amr_error_action_type);

future<std::pair<real, real>> node_client::amr_error() const {
	return hpx::async<typename node_server::amr_error_action>(get_unmanaged_gid());
}

std::pair<real, real> node_server::amr_error() {
	std::vector<hpx::future<std::pair<real, real>>> kfuts;
	auto sum = std::make_pair(0.0, 0.0);
	if (is_refined) {
		for (int i = 0; i < NCHILD; i++) {
			kfuts.push_back(children[i].amr_error());
		}
		hpx::wait_all(kfuts);
		for (int i = 0; i < NCHILD; i++) {
			auto tmp = kfuts[i].get();
			sum.first += tmp.first;
			sum.second += tmp.second;
		}
	} else {
		sum = grid_ptr->amr_error();
	}
	return sum;
}

using regrid_gather_action_type = node_server::regrid_gather_action;
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
				for (int i = 0; i < NCHILD; i++) {
					kfuts.push_back(children[i].kill());
				}
				std::fill_n(children.begin(), NCHILD, node_client());
				is_refined = false;
			}
		}

		if (is_refined) {
			std::array<future<node_count_type>, NCHILD> futs;
			integer index = 0;
			for (auto &child : children) {
				futs[index++] = child.regrid_gather(rebalance_only);
			}
			auto futi = futs.begin();
			for (auto const &ci : geo::octant::full_set()) {
				const auto child_cnt = futi->get();
				++futi;
				child_descendant_count[ci] = child_cnt.total;
				count.leaf += child_cnt.leaf;
				count.total += child_cnt.total;
			}
		} else {
			count.leaf = 1;
			for (auto const &ci : geo::octant::full_set()) {
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

			for (auto &ci : geo::octant::full_set()) {
				child_descendant_count[ci] = 1;

			}
		}
	}
	grid_ptr->set_leaf(!is_refined);
	hpx::wait_all(kfuts);
	return count;
}

future<hpx::id_type> node_server::create_child(hpx::id_type const &locality, integer ci) {
	return hpx::async(hpx::util::annotated_function([ci, this](hpx::id_type const locality) {

		return hpx::new_<node_server>(locality, my_location.get_child(ci), me, current_time, rotational_time, step_num, hcycle, rcycle, gcycle).then([this, ci](future<hpx::id_type> &&child_idf) {
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
			if (current_time > ZERO || opts().restart_filename != "") {
				std::vector<real> prolong;
				{
					std::unique_lock < hpx::lcos::local::spinlock > lk(prolong_mtx);
					prolong = grid_ptr->get_prolong(lb, ub);
				}
				GET(child.set_grid(std::move(prolong), std::move(outflows)));
			}
		}
		if (opts().radiation) {
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
					std::unique_lock < hpx::lcos::local::spinlock > lk(prolong_mtx);
					prolong = rad_grid_ptr->get_prolong(lb, ub);
				}
				child.set_rad_grid(std::move(prolong)/*, std::move(outflows)*/).get();
			}
		}
		return child_id;
	});}, "node_server::create_child::lambda"), locality);
}

using regrid_scatter_action_type = node_server::regrid_scatter_action;
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
		for (auto &ci : geo::octant::full_set()) {
			const integer loc_index = a * options::all_localities.size() / total;
			const auto child_loc = options::all_localities[loc_index];
			if (children[ci].empty()) {
				futs[index++] = create_child(child_loc, ci).then([this, ci, a, total](future<hpx::id_type> &&child) {
					children[ci] = GET(child);
					GET(children[ci].regrid_scatter(a, total));
				});
			} else {
				const hpx::id_type id = children[ci].get_gid();
				integer current_child_id = hpx::naming::get_locality_id_from_gid(id.get_gid());
				auto current_child_loc = options::all_localities[current_child_id];
				if (child_loc != current_child_loc) {
					futs[index++] = children[ci].copy_to_locality(child_loc).then([this, ci, a, total](future<hpx::id_type> &&child) {
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
	if (is_refined) {
		for (auto &f : futs) {
			GET(f);
		}
	}
	clear_family();
}

node_count_type node_server::regrid(const hpx::id_type &root_gid, real omega, real new_floor, bool rb, bool grav_energy_comp) {
	timings::scope ts(timings_, timings::time_regrid);
	hpx::util::high_resolution_timer timer;
	assert(grid_ptr != nullptr);
	print("-----------------------------------------------\n");
	if (!rb) {
		print("checking for refinement\n");
		check_for_refinement(omega, new_floor);
	} else {
		node_registry::clear();
	}
	print("regridding\n");
	real tstart = timer.elapsed();
	auto a = regrid_gather(rb);
	real tstop = timer.elapsed();
	print("Regridded tree in %f seconds\n", real(tstop - tstart));
	print("rebalancing %i nodes with %i leaves\n", int(a.total), int(a.leaf));
	tstart = timer.elapsed();
	regrid_scatter(0, a.total);
	tstop = timer.elapsed();
	print("Rebalanced tree in %f seconds\n", real(tstop - tstart));
	assert(grid_ptr != nullptr);
	tstart = timer.elapsed();
	print("forming tree connections\n");
	a.amr_bnd = form_tree(hpx::unmanaged(root_gid));
	print("%i amr boundaries\n", a.amr_bnd);
	tstop = timer.elapsed();
	print("Formed tree in %f seconds\n", real(tstop - tstart));
	print("solving gravity\n");
	solve_gravity(grav_energy_comp, false);
	double elapsed = timer.elapsed();
	print("regrid done in %f seconds\n---------------------------------------\n", elapsed);
	return a;
}

using set_aunt_action_type = node_server::set_aunt_action;
HPX_REGISTER_ACTION(set_aunt_action_type);

future<void> node_client::set_aunt(const hpx::id_type &aunt, const geo::face &f) const {
	return hpx::async<typename node_server::set_aunt_action>(get_unmanaged_gid(), aunt, f);
}

void node_server::set_aunt(const hpx::id_type &aunt, const geo::face &face) {
	if (aunts[face].get_gid() != hpx::invalid_id) {
		print("AUNT ALREADY SET\n");
		abort();
	}
	aunts[face] = aunt;
}

using set_grid_action_type = node_server::set_grid_action;
HPX_REGISTER_ACTION(set_grid_action_type);

future<void> node_client::set_grid(std::vector<real> &&g, std::vector<real> &&o) const {
	return hpx::async<typename node_server::set_grid_action>(get_unmanaged_gid(), std::move(g), std::move(o));
}

void node_server::set_grid(const std::vector<real> &data, std::vector<real> &&outflows) {
	grid_ptr->set_prolong(data, std::move(outflows));
}

using solve_gravity_action_type = node_server::solve_gravity_action;
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
		for (auto &child : children) {
			child_futs[index++] = child.solve_gravity(ene, aonly);
		}
	}
	compute_fmm(RHO, ene, aonly);
	if (is_refined) {
		//	wait_all_and_propagate_exceptions(child_futs);
		for (auto &f : child_futs) {
			GET(f);
		}
	}
}
#endif
